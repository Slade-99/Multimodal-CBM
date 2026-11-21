"""
train_cxr_concepts.py

Isolated CXR concept prediction training.
- MobileNetV3-Large backbone (fast, ~3x less compute than ResNet50)
- Deeper concept head with dropout (replaces the single linear layer)
- No downstream task (mortality / AHF) — concept loss only
- Per-concept AUROC logged every epoch so you can see exactly which concepts are failing
- Pos-weight from training data computed automatically (no manual tuning needed)
"""

import os
import logging
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import numpy as np
from dataloader_ecg import get_ecg_concept_dataloaders, ECG_CONCEPTS


# ================= LOSS =================

def build_pos_weight(train_loader, num_concepts, device):
    """
    Compute per-concept pos_weight = (# negatives) / (# positives) from
    the training set. Much better than guessing a fixed weight.
    Caps at 20 to avoid exploding gradients on extremely rare concepts.
    """
    print("Computing per-concept pos_weight from training data...")
    pos_counts = torch.zeros(num_concepts)
    total      = 0

    for batch in train_loader:
        if batch is None:
            continue
        labels = batch['concepts']   # (B, 15)
        pos_counts += labels.sum(dim=0)
        total      += labels.shape[0]

    neg_counts = total - pos_counts
    # Avoid divide-by-zero for any concept with zero positives
    pos_weight = torch.where(
        pos_counts > 0,
        neg_counts / pos_counts.clamp(min=1),
        torch.ones(num_concepts)
    ).clamp(max=20.0).to(device)

    print("Per-concept pos_weight:")
    for i, (c, w) in enumerate(zip(ECG_CONCEPTS, pos_weight.tolist())):
        print(f"  {c:<30s}: {w:.2f}")
    return pos_weight


# ================= METRICS =================

def compute_per_concept_auroc(y_true, y_scores):
    """
    y_true, y_scores: np.ndarray of shape (N, 15)
    Returns list of per-concept AUROC (0.5 fallback if no positives in split).
    """
    aurocs = []
    for i in range(y_true.shape[1]):
        try:
            aurocs.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
        except ValueError:
            aurocs.append(0.5)
    return aurocs


# ================= LOGGING =================

def setup_logger(save_dir, exp_name):
    os.makedirs(os.path.join(save_dir, exp_name), exist_ok=True)
    log_file = os.path.join(save_dir, exp_name, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()   # Also print to console for this debug script
        ]
    )
    return logging.getLogger()


# ================= ARGS =================

def get_args():
    parser = argparse.ArgumentParser(description="ECG Concept Prediction — Isolation Debug")

    parser.add_argument('--train_csv', type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv')
    parser.add_argument('--val_csv',   type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv')
    parser.add_argument('--test_csv',  type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv')

    parser.add_argument('--save_dir',  type=str, default='./ecg_concept_checkpoints')
    parser.add_argument('--exp_name',  type=str, default='ecg_concept_debug')

    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--patience_scheduler',  type=int, default=3)
    parser.add_argument('--patience_early_stop', type=int, default=7)

    # Two-stage training: first train only the head, then unfreeze backbone
    parser.add_argument('--finetune_epochs', type=int, default=5,
                        help="Epochs to train only the concept head before unfreezing backbone.")

    return parser.parse_args()


# ================= TRAIN ONE EPOCH =================

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch, logger):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        if batch is None:
            continue

        images   = batch['waveform'].to(device)      # (B, 12, 5000)
        concepts = batch['concepts'].to(device)   # (B, 14)

        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(images)                # (B, 14)
            loss   = criterion(logits, concepts)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ================= VALIDATE =================

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            images   = batch['waveform'].to(device)
            concepts = batch['concepts'].to(device)

            logits = model(images)
            loss   = criterion(logits, concepts)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(concepts.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)   # (N, 14)
    all_labels = np.concatenate(all_labels, axis=0)   # (N, 14)

    per_concept_auroc = compute_per_concept_auroc(all_labels, all_probs)
    mean_auroc        = np.mean(per_concept_auroc)
    avg_loss          = total_loss / max(len(loader), 1)

    return avg_loss, mean_auroc, per_concept_auroc


# ================= MAIN =================

def main():
    args   = get_args()
    device = torch.device('cuda')

    logger = setup_logger(args.save_dir, args.exp_name)
    logger.info(f"Arguments: {vars(args)}")

    # --- Data ---
    train_loader, val_loader, test_loader = get_ecg_concept_dataloaders(
        args.train_csv, args.val_csv, args.test_csv, batch_size=args.batch_size
    )

    # --- Model ---
    from test import model
    model = model.to(device)

    
    pos_weight = build_pos_weight(train_loader, len(ECG_CONCEPTS), device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Two-stage training ---
    # Stage 1: freeze backbone, train only concept head (fast warm-up)
    logger.info(f"\n=== Stage 1: Training concept head only for {args.finetune_epochs} epochs ===")
    for param in model.parameters():
        param.requires_grad = False

    for param in model.dense.parameters():
        param.requires_grad = True

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scaler    = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience_scheduler, factor=0.5)

    best_val_auroc   = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(args.save_dir, args.exp_name, "best_weights.pth")

    for epoch in range(1, args.finetune_epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger)
        val_loss, mean_auroc, per_auroc = validate(model, val_loader, criterion, device)
        scheduler.step(mean_auroc)

        logger.info(f"[Stage1] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean AUROC: {mean_auroc:.4f}")
        _log_per_concept(logger, per_auroc)

        if mean_auroc > best_val_auroc:
            best_val_auroc = mean_auroc
            torch.save(model.state_dict(), save_path)
            logger.info(f"  [*] Saved best model (AUROC: {best_val_auroc:.4f})")

    # Stage 2: unfreeze backbone, train end-to-end with lower LR
    logger.info(f"\n=== Stage 2: Full end-to-end fine-tuning for {args.epochs - args.finetune_epochs} epochs ===")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr*0.1, weight_decay=args.weight_decay
    )

    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience_scheduler, factor=0.5)
    epochs_no_improve = 0

    for epoch in range(args.finetune_epochs + 1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, logger)
        val_loss, mean_auroc, per_auroc = validate(model, val_loader, criterion, device)
        scheduler.step(mean_auroc)

        logger.info(f"[Stage2] Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean AUROC: {mean_auroc:.4f}")
        _log_per_concept(logger, per_auroc)

        if mean_auroc > best_val_auroc:
            best_val_auroc = mean_auroc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"  [*] Saved best model (AUROC: {best_val_auroc:.4f})")
        else:
            epochs_no_improve += 1
            logger.info(f"  [!] No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= args.patience_early_stop:
            logger.info("Early stopping triggered.")
            break

    # --- Final Test Evaluation ---
    logger.info("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss, test_auroc, test_per_auroc = validate(model, test_loader, criterion, device)
    logger.info(f"Test Loss: {test_loss:.4f} | Mean AUROC: {test_auroc:.4f}")
    _log_per_concept(logger, test_per_auroc)


def _log_per_concept(logger, per_auroc):
    """Logs per-concept AUROC so you can see exactly which concepts are failing."""
    for concept, auroc in zip(ECG_CONCEPTS, per_auroc):
        flag = "  <<< LOW" if auroc < 0.65 else ""
        logger.info(f"    {concept:<30s}: {auroc:.4f}{flag}")


if __name__ == "__main__":
    main()