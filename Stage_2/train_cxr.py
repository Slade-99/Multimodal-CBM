"""
train_cxr_stage2.py  —  Stage 2: CXR Task Head Training

Flow:
  1. Define CXR model inline (XRV encoder + task heads)
  2. Load Stage 1 encoder weights and freeze encoder
  3. Train only mortality + AHF task heads on sigmoid concept probabilities
  4. Concept AUROC logged every epoch to confirm encoder stays healthy
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
from sklearn.metrics import roc_auc_score, average_precision_score
import torchxrayvision as xrv
from dataloader_cxr import get_cxr_stage2_dataloaders, CXR_CONCEPTS


# ============================================================
# PATHS
# ============================================================

TRAIN_CSV = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
VAL_CSV   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
TEST_CSV  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

STAGE1_WEIGHTS = "/home/azwad/Works/Multimodal-CBM/cxr_concept_checkpoints/xrv_resnet50_concept/best_weights.pth"

MORT_POS_WEIGHT = 14.44
AHF_POS_WEIGHT  = 2.89


# ============================================================
# MODEL
# ============================================================

class CXRConceptEncoder(nn.Module):
    """
    XRV ResNet50 backbone + concept head.
    Identical to CXRConceptXRV from Stage 1 — must match exactly
    so Stage 1 weights load correctly.
    """
    def __init__(self, num_concepts=15):
        super().__init__()
        self.backbone     = xrv.models.ResNet(weights="resnet50-res512-all")
        self.concept_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_concepts)
        )

    def forward(self, x):
        feats = self.backbone.features(x)   # (B, 2048)
        return self.concept_head(feats)     # (B, 15) raw logits


def _build_task_head(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(16, 1)
    )


class UniModalCXR(nn.Module):
    """
    Full CXR CBM model for Stage 2.
    encoder  : frozen after Stage 1 weights loaded
    task heads: trainable — learn to predict mortality/AHF from concept probs
    """
    def __init__(self, num_concepts=15):
        super().__init__()
        self.encoder        = CXRConceptEncoder(num_concepts=num_concepts)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, image):
        concept_logits = self.encoder(image)          # (B, 15) raw logits
        concept_probs  = torch.sigmoid(concept_logits) # (B, 15) in [0,1]

        # Task heads receive probabilities, not logits — this is the CBM bottleneck.
        # The task head weights are directly interpretable as concept importances.
        return {
            'concept_logits':   concept_logits,
            'mortality_logits': self.mortality_head(concept_probs).squeeze(-1),
            'ahf_logits':       self.ahf_head(concept_probs).squeeze(-1),
        }


# ============================================================
# LOGGER
# ============================================================

def setup_logger(save_dir, exp_name):
    os.makedirs(os.path.join(save_dir, exp_name), exist_ok=True)
    log_file = os.path.join(save_dir, exp_name, "stage2_training_log.txt")
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
    parser = argparse.ArgumentParser(description="CXR Stage 2 — Task Head Training")

    parser.add_argument('--save_dir',            type=str,   default='./cxr_stage2_checkpoints')
    parser.add_argument('--exp_name',            type=str,   default='cxr_stage2')
    parser.add_argument('--batch_size',          type=int,   default=32)
    parser.add_argument('--lr',                  type=float, default=1e-3)
    parser.add_argument('--weight_decay',        type=float, default=1e-4)
    parser.add_argument('--epochs',              type=int,   default=30)
    parser.add_argument('--patience_scheduler',  type=int,   default=5)
    parser.add_argument('--patience_early_stop', type=int,   default=3)

    return parser.parse_args()


# ============================================================
# METRICS
# ============================================================

def compute_auroc(y_true, y_probs):
    try:
        return roc_auc_score(y_true, y_probs)
    except ValueError:
        return 0.5

def compute_auprc(y_true, y_probs):
    try:
        return average_precision_score(y_true, y_probs)
    except ValueError:
        return 0.0

def compute_per_concept_auroc(y_true, y_scores):
    aurocs = []
    for i in range(y_true.shape[1]):
        try:
            aurocs.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
        except ValueError:
            aurocs.append(0.5)
    return np.array(aurocs)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(model, loader, optimizer, scaler, device, loss_mort, loss_ahf):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False, desc="Train"):
        if batch is None:
            continue

        image  = batch['image'].to(device)             # (B, 1, 512, 512)
        y_mort = batch['target_mortality'].float().to(device)
        y_ahf  = batch['target_ahf'].float().to(device)

        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(image)
            loss = (loss_mort(outputs['mortality_logits'], y_mort) +
                    loss_ahf(outputs['ahf_logits'],        y_ahf)) / 2.0

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

def evaluate(model, loader, device, loss_mort, loss_ahf):
    model.eval()

    mort_probs,    mort_targets    = [], []
    ahf_probs,     ahf_targets     = [], []
    concept_probs, concept_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            image  = batch['image'].to(device)
            y_mort = batch['target_mortality'].float().to(device)
            y_ahf  = batch['target_ahf'].float().to(device)

            outputs = model(image)

            loss = (loss_mort(outputs['mortality_logits'], y_mort) +
                    loss_ahf(outputs['ahf_logits'],        y_ahf)) / 2.0
            total_loss += loss.item()

            mort_probs.extend(
                torch.sigmoid(outputs['mortality_logits']).cpu().numpy())
            mort_targets.extend(batch['target_mortality'].numpy())

            ahf_probs.extend(
                torch.sigmoid(outputs['ahf_logits']).cpu().numpy())
            ahf_targets.extend(batch['target_ahf'].numpy())

            # Concept AUROC — confirms encoder is still healthy while heads train
            concept_probs.append(
                torch.sigmoid(outputs['concept_logits']).cpu().numpy())
            concept_targets.append(batch['concepts'].numpy())

    mort_auroc = compute_auroc(np.array(mort_targets), np.array(mort_probs))
    mort_auprc = compute_auprc(np.array(mort_targets), np.array(mort_probs))
    ahf_auroc  = compute_auroc(np.array(ahf_targets),  np.array(ahf_probs))
    ahf_auprc  = compute_auprc(np.array(ahf_targets),  np.array(ahf_probs))

    cp = np.concatenate(concept_probs,   axis=0)
    ct = np.concatenate(concept_targets, axis=0)
    per_concept        = compute_per_concept_auroc(ct, cp)
    mean_concept_auroc = float(np.mean(per_concept))
    avg_loss           = total_loss / max(len(loader), 1)

    return {
        "mort_auroc":        mort_auroc,
        "mort_auprc":        mort_auprc,
        "ahf_auroc":         ahf_auroc,
        "ahf_auprc":         ahf_auprc,
        "combined":          (mort_auroc + ahf_auroc) / 2.0,
        "concept_auroc":     mean_concept_auroc,
        "per_concept_auroc": per_concept,
        "val_loss":          avg_loss,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args   = get_args()
    device = torch.device('cuda')

    logger = setup_logger(args.save_dir, args.exp_name)
    logger.info(f"CXR Stage 2 — Task Head Training")
    logger.info(f"Arguments: {vars(args)}")

    # --- Data ---
    train_loader, val_loader, test_loader = get_cxr_stage2_dataloaders(
        TRAIN_CSV, VAL_CSV, TEST_CSV, batch_size=args.batch_size
    )
    logger.info("Dataloaders ready. Transform: XRV grayscale [-1024, 1024]")

    # --- Model ---
    model = UniModalCXR(num_concepts=len(CXR_CONCEPTS)).to(device)

    # --- Load Stage 1 encoder weights ---
    if not os.path.exists(STAGE1_WEIGHTS):
        raise FileNotFoundError(
            f"Stage 1 weights not found at {STAGE1_WEIGHTS}. "
            f"Run train_cxr_concepts.py first."
        )

    stage1_sd = torch.load(STAGE1_WEIGHTS, map_location=device, weights_only=True)

    # Stage 1 saved CXRConceptXRV directly — keys match model.encoder exactly
    missing, unexpected = model.encoder.load_state_dict(stage1_sd, strict=False)
    logger.info(f"Loaded Stage 1 weights from {STAGE1_WEIGHTS}")
    if not missing and not unexpected:
        logger.info("  All keys matched perfectly.")
    if missing:
        logger.warning(f"  Missing keys ({len(missing)}): "
                       f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        logger.warning(f"  Unexpected keys ({len(unexpected)}): "
                       f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # --- Freeze encoder ---
    for param in model.encoder.parameters():
        param.requires_grad = False

    frozen    = sum(p.numel() for p in model.encoder.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Encoder frozen: {frozen:,} params | Task heads trainable: {trainable:,} params")

    # --- Loss ---
    loss_mort = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([MORT_POS_WEIGHT], device=device))
    loss_ahf  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([AHF_POS_WEIGHT],  device=device))

    # --- Optimizer: task heads only ---
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        patience=args.patience_scheduler, factor=0.5
    )
    scaler = GradScaler()

    best_combined     = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(args.save_dir, args.exp_name, "best_stage2_weights.pth")

    # --- Epoch loop ---
    for epoch in range(1, args.epochs + 1):

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device,
            loss_mort, loss_ahf
        )
        metrics = evaluate(model, val_loader, device, loss_mort, loss_ahf)

        scheduler.step(metrics['combined'])

        logger.info(f"Epoch {epoch}/{args.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}")
        logger.info(f"  Mortality  AUROC: {metrics['mort_auroc']:.4f}  "
                    f"AUPRC: {metrics['mort_auprc']:.4f}")
        logger.info(f"  AHF        AUROC: {metrics['ahf_auroc']:.4f}  "
                    f"AUPRC: {metrics['ahf_auprc']:.4f}")
        logger.info(f"  Combined: {metrics['combined']:.4f} | "
                    f"Concept AUROC: {metrics['concept_auroc']:.4f}")

        # Per-concept breakdown every epoch
        for name, auroc in zip(CXR_CONCEPTS, metrics['per_concept_auroc']):
            flag = "  <<< LOW" if auroc < 0.65 else ""
            logger.info(f"    {name:<30s}: {auroc:.4f}{flag}")

        if metrics['combined'] > best_combined:
            best_combined     = metrics['combined']
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"  [*] Best model saved (combined={best_combined:.4f})\n")
        else:
            epochs_no_improve += 1
            logger.info(f"  [!] No improvement for {epochs_no_improve} epochs.\n")

        if epochs_no_improve >= args.patience_early_stop:
            logger.info("Early stopping triggered.")
            break

    # --- Final test evaluation ---
    logger.info("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_metrics = evaluate(model, test_loader, device, loss_mort, loss_ahf)
    logger.info(f"Test Loss: {test_metrics['val_loss']:.4f}")
    logger.info(f"Mortality  AUROC: {test_metrics['mort_auroc']:.4f}  "
                f"AUPRC: {test_metrics['mort_auprc']:.4f}")
    logger.info(f"AHF        AUROC: {test_metrics['ahf_auroc']:.4f}  "
                f"AUPRC: {test_metrics['ahf_auprc']:.4f}")
    logger.info(f"Mean Concept AUROC: {test_metrics['concept_auroc']:.4f}")
    logger.info("Per-concept AUROC:")
    for name, auroc in zip(CXR_CONCEPTS, test_metrics['per_concept_auroc']):
        flag = "  <<< LOW" if auroc < 0.65 else ""
        logger.info(f"  {name:<30s}: {auroc:.4f}{flag}")


if __name__ == "__main__":
    main()