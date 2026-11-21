"""
train_cxr_concepts.py  —  CXR Concept Prediction with TorchXRayVision ResNet50

Key differences from previous version:
  - Uses xrv.models.ResNet(weights="resnet50-res512-all") — pretrained on
    NIH + RSNA + SIIM + VinDr CXR datasets. These weights have already
    learned to distinguish pleural_effusion, lung_opacity, cardiomegaly etc.
    This is the strongest available public CXR backbone for this task.

  - XRV ResNet expects:
      * Single-channel (grayscale) input — shape (B, 1, H, W)
      * Pixel values in [-1024, 1024] range  (NOT ImageNet normalisation)
      * Square images (it auto-resizes to 512x512 internally)
    Your dataloader_cxr.py currently produces 3-channel RGB with ImageNet
    normalisation — the transforms section below overrides this correctly.

  - backbone.features(x) already includes avgpool + flatten internally,
    returning (B, 2048) directly. No separate avgpool call needed.

  - Two-stage training preserved: freeze backbone for finetune_epochs,
    then unfreeze with 10x lower LR on backbone vs concept head.
"""

import os
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from torch.utils.data import DataLoader
import torchxrayvision as xrv

# We reuse the Dataset class from dataloader_cxr but supply our own transforms
from dataloader_cxr import CXRConceptDataset, CXR_CONCEPTS


# ============================================================
# XRV-COMPATIBLE TRANSFORMS
# ============================================================
# XRV requires:
#   - Grayscale (1 channel)
#   - Values in [-1024, 1024]
# Standard pipeline: load as grayscale PIL → tensor [0,1] → scale to [-1024,1024]

def xrv_train_transforms():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=1),   # XRV needs 1 channel
        transforms.ToTensor(),                          # [0, 1]
        transforms.Lambda(lambda x: x * 2048.0 - 1024.0),  # → [-1024, 1024]
    ])

def xrv_eval_transforms():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2048.0 - 1024.0),
    ])

def cxr_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def get_xrv_dataloaders(train_csv, val_csv, test_csv, batch_size=32):
    train_ds = CXRConceptDataset(train_csv, transform=xrv_train_transforms())
    val_ds   = CXRConceptDataset(val_csv,   transform=xrv_eval_transforms())
    test_ds  = CXRConceptDataset(test_csv,  transform=xrv_eval_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        prefetch_factor=2, persistent_workers=True,
        collate_fn=cxr_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=cxr_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=cxr_collate
    )
    return train_loader, val_loader, test_loader


# ============================================================
# MODEL
# ============================================================

class CXRConceptXRV(nn.Module):
    """
    TorchXRayVision ResNet50 backbone + deep concept head.

    backbone.features(x):
        - Handles grayscale (1-channel) input
        - Internally does conv1 → bn1 → relu → maxpool → layer1-4 → avgpool → flatten
        - Returns (B, 2048) directly — no separate avgpool needed
        - Auto-resizes to 512x512 if input is different size

    Concept head: 2048 → 256 → 128 → num_concepts
        Same architecture that worked for the MobileNet isolation experiment.
    """
    def __init__(self, num_concepts=15):
        super().__init__()

        # Load XRV ResNet50 pretrained on CXR datasets
        self.backbone = xrv.models.ResNet(weights="resnet50-res512-all")
        # backbone.features(x) returns (B, 2048) — used in forward below

        self.concept_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, num_concepts)   # raw logits
        )

    def forward(self, x):
        # backbone.features() already does avgpool + flatten → (B, 2048)
        feats = self.backbone.features(x)     # (B, 2048)
        return self.concept_head(feats)       # (B, num_concepts)


# ============================================================
# POS-WEIGHT
# ============================================================

def build_pos_weight(train_loader, num_concepts, device):
    print("Computing per-concept pos_weight from training data...")
    pos_counts = torch.zeros(num_concepts)
    total = 0

    for batch in train_loader:
        if batch is None:
            continue
        labels = batch['concepts']
        pos_counts += labels.sum(dim=0)
        total += labels.shape[0]

    neg_counts = total - pos_counts
    pos_weight = torch.where(
        pos_counts > 0,
        neg_counts / pos_counts.clamp(min=1),
        torch.ones(num_concepts)
    ).clamp(max=20.0).to(device)

    print("Per-concept pos_weight:")
    for c, w in zip(CXR_CONCEPTS, pos_weight.tolist()):
        print(f"  {c:<30s}: {w:.2f}")
    return pos_weight


# ============================================================
# METRICS
# ============================================================

def compute_per_concept_auroc(y_true, y_scores):
    aurocs = []
    for i in range(y_true.shape[1]):
        try:
            aurocs.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
        except ValueError:
            aurocs.append(0.5)
    return aurocs


# ============================================================
# LOGGER
# ============================================================

def setup_logger(save_dir, exp_name):
    os.makedirs(os.path.join(save_dir, exp_name), exist_ok=True)
    log_file = os.path.join(save_dir, exp_name, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


# ============================================================
# ARGS
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(description="CXR Concept Prediction — XRV ResNet50")

    parser.add_argument('--train_csv', type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv')
    parser.add_argument('--val_csv', type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv')
    parser.add_argument('--test_csv', type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv')

    parser.add_argument('--save_dir',  type=str, default='./cxr_concept_checkpoints')
    parser.add_argument('--exp_name',  type=str, default='xrv_resnet50_concept')

    parser.add_argument('--batch_size',          type=int,   default=16)
    parser.add_argument('--lr',                  type=float, default=1e-4)
    parser.add_argument('--weight_decay',        type=float, default=1e-4)
    parser.add_argument('--epochs',              type=int,   default=25)
    parser.add_argument('--patience_scheduler',  type=int,   default=3)
    parser.add_argument('--patience_early_stop', type=int,   default=3)
    parser.add_argument('--finetune_epochs',     type=int,   default=10,
                        help="Epochs to train concept head only before unfreezing backbone.")

    return parser.parse_args()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        if batch is None:
            continue

        images   = batch['image'].to(device)      # (B, 1, 512, 512)
        concepts = batch['concepts'].to(device)   # (B, 15)

        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(images)
            loss   = criterion(logits, concepts)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ============================================================
# VALIDATE
# ============================================================

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            images   = batch['image'].to(device)
            concepts = batch['concepts'].to(device)

            logits = model(images)
            loss   = criterion(logits, concepts)
            total_loss += loss.item()

            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(concepts.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    per_auroc  = compute_per_concept_auroc(all_labels, all_probs)
    mean_auroc = float(np.mean(per_auroc))
    avg_loss   = total_loss / max(len(loader), 1)

    return avg_loss, mean_auroc, per_auroc


# ============================================================
# LOG HELPER
# ============================================================

def log_per_concept(logger, per_auroc):
    for concept, auroc in zip(CXR_CONCEPTS, per_auroc):
        flag = "  <<< LOW" if auroc < 0.65 else ""
        logger.info(f"    {concept:<30s}: {auroc:.4f}{flag}")


# ============================================================
# MAIN
# ============================================================

def main():
    args   = get_args()
    device = torch.device('cuda')

    logger = setup_logger(args.save_dir, args.exp_name)
    logger.info(f"Arguments: {vars(args)}")

    # --- Data (XRV-compatible transforms applied here) ---
    train_loader, val_loader, test_loader = get_xrv_dataloaders(
        args.train_csv, args.val_csv, args.test_csv,
        batch_size=args.batch_size
    )
    logger.info("Dataloaders ready with XRV normalisation (grayscale, [-1024, 1024]).")

    # --- Model ---
    model = CXRConceptXRV(num_concepts=len(CXR_CONCEPTS)).to(device)
    logger.info(f"Backbone: XRV ResNet50 (resnet50-res512-all)")
    logger.info(f"Concept head: 2048 → 256 → 128 → {len(CXR_CONCEPTS)}")

    # --- Loss ---
    pos_weight = build_pos_weight(train_loader, len(CXR_CONCEPTS), device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    #load_path = os.path.join(args.save_dir, args.exp_name, "best_weights.pth")
    state_dict = torch.load("/home/azwad/Works/Multimodal-CBM/cxr_concept_checkpoints/xrv_resnet50_concept/best_weights.pth", map_location='cpu')
    model.load_state_dict(state_dict, strict=True)
    save_path = os.path.join(args.save_dir, args.exp_name, "best_weights.pth")
    scaler    = GradScaler()
    best_val_auroc    = 0.0
    epochs_no_improve = 0

    # ---- Stage 1: freeze backbone, warm up concept head ----
    logger.info(f"\n=== Stage 1: Head-only warm-up for {args.finetune_epochs} epochs ===")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', patience=args.patience_scheduler, factor=0.5
    )

    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, mean_auroc, per_auroc = validate(model, val_loader, criterion, device)
        scheduler.step(mean_auroc)

        logger.info(
            f"[Stage1] Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Mean AUROC: {mean_auroc:.4f}"
        )
        log_per_concept(logger, per_auroc)

        if mean_auroc > best_val_auroc:
            best_val_auroc = mean_auroc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  [*] Saved best model (AUROC: {best_val_auroc:.4f})")

    # ---- Stage 2: unfreeze backbone, end-to-end with differential LR ----
    logger.info(f"\n=== Stage 2: End-to-end fine-tuning for {args.epochs - args.finetune_epochs} epochs ===")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer = AdamW([
        {'params': model.backbone.parameters(),     'lr': args.lr * 0.1},  # 10x lower
        {'params': model.concept_head.parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', patience=args.patience_scheduler, factor=0.5
    )
    epochs_no_improve = 0

    for epoch in range(args.finetune_epochs + 1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, mean_auroc, per_auroc = validate(model, val_loader, criterion, device)
        scheduler.step(mean_auroc)

        logger.info(
            f"[Stage2] Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Mean AUROC: {mean_auroc:.4f}"
        )
        log_per_concept(logger, per_auroc)

        if mean_auroc > best_val_auroc:
            best_val_auroc    = mean_auroc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"  [*] Saved best model (AUROC: {best_val_auroc:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"  [!] No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= args.patience_early_stop:
            logger.info("Early stopping triggered.")
            break

    # ---- Final test evaluation ----
    logger.info("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss, test_auroc, test_per_auroc = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Mean AUROC: {test_auroc:.4f}")
    log_per_concept(logger, test_per_auroc)


if __name__ == "__main__":
    main()