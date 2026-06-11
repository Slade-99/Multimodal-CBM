"""
train_ehr_concepts.py

Isolated EHR concept prediction training.
- Deeper MLP encoder: 21 -> 128 -> 64 -> 32 -> 13
- No downstream task (mortality / AHF) — concept loss only
- ehr_mask (B, 21) used in loss to ignore concepts whose source
  feature was missing — prevents penalizing predictions on imputed zeros
- Per-concept AUROC logged every epoch
- Pos-weight computed automatically from training data (mask-aware)
"""

import os
import logging
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

from dataloader_ehr import get_ehr_concept_dataloaders, EHR_CONCEPTS


# ============================================================
# CONCEPT → FEATURE MAPPING
#
# Maps each of the 13 EHR concept indices to the raw feature
# index (in the 21-dim EHR_FEATURES vector) it is derived from.
# If that feature is missing (mask=0), the concept label is
# unreliable and must be excluded from the loss.
#
# EHR_FEATURES order (from dataloader_ehr.py):
#   Continuous (0-18): age, heart_rate, mean_bp, resp_rate, spo2,
#     temperature, bmi, anion_gap, bilirubin, bnp, bun, creatinine,
#     lactate, platelets, rdw, troponin, wbc, fluid_balance, gcs_total
#   Binary (19-20): admission_emergency, vasopressor_usage
#
# EHR_CONCEPTS order:
#   0:tachycardia, 1:tachypnea, 2:abnormal_temp, 3:hypotension,
#   4:hypoxia, 5:altered_mental, 6:elevated_creatinine,
#   7:elevated_bilirubin, 8:thrombocytopenia, 9:abnormal_wbc,
#   10:hyperlactatemia, 11:elevated_bnp, 12:elevated_troponin
# ============================================================

EHR_CONCEPT_TO_FEATURE = {
    0:  1,   # tachycardia         <- heart_rate     (idx 1)
    1:  3,   # tachypnea           <- resp_rate       (idx 3)
    2:  5,   # abnormal_temp       <- temperature     (idx 5)
    3:  2,   # hypotension         <- mean_bp         (idx 2)
    4:  4,   # hypoxia             <- spo2            (idx 4)
    5:  18,  # altered_mental      <- gcs_total       (idx 18)
    6:  11,  # elevated_creatinine <- creatinine      (idx 11)
    7:  8,   # elevated_bilirubin  <- bilirubin       (idx 8)
    8:  13,  # thrombocytopenia    <- platelets       (idx 13)
    9:  16,  # abnormal_wbc        <- wbc             (idx 16)
    10: 12,  # hyperlactatemia     <- lactate         (idx 12)
    11: 9,   # elevated_bnp        <- bnp             (idx 9)
    12: 15,  # elevated_troponin   <- troponin        (idx 15)
}


# ============================================================
# MODEL
# ============================================================

class EHRConceptEncoder(nn.Module):
    """
    Wider and deeper than original 21->64->32->13.
    First layer expanded to 128 to give the model room to learn
    pairwise feature interactions before compressing to concept space.
    """
    def __init__(self, input_dim=21, num_concepts=13):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, num_concepts)   # raw logits
        )

    def forward(self, x):
        return self.network(x)   # (B, 13)


# ============================================================
# MASK-AWARE LOSS
# ============================================================

def masked_ehr_concept_loss(logits, labels, feat_mask, pos_weight):
    """
    BCEWithLogitsLoss computed only on concepts whose source feature
    was actually observed (feat_mask == 1) for each sample.

    Args:
        logits:     (B, 13)  raw model output
        labels:     (B, 13)  binary concept ground truth
        feat_mask:  (B, 21)  1=observed, 0=missing/zero-imputed
        pos_weight: (13,)    per-concept class imbalance weight

    Why this matters:
        EHR features are zero-imputed when missing. The concept labels
        derived from those features (e.g. tachycardia from heart_rate)
        are unreliable when the source feature was absent. Training on
        these entries teaches the model wrong signal — it sees a zeroed
        input and an arbitrary label. The mask excludes them entirely.

    Returns:
        Scalar loss averaged over valid (observed) entries only.
        Returns 0.0 tensor (with grad) if entire batch is missing.
    """
    B, C = logits.shape

    # Build (B, 13) boolean validity mask from (B, 21) feature mask
    concept_valid = torch.ones(B, C, dtype=torch.bool, device=logits.device)
    for c_idx, f_idx in EHR_CONCEPT_TO_FEATURE.items():
        concept_valid[:, c_idx] = feat_mask[:, f_idx].bool()

    if not concept_valid.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Unreduced loss: (B, 13) — keeps individual entry losses
    loss_per_entry = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        pos_weight=pos_weight,
        reduction='none'
    )

    # Mask out invalid entries, normalise by number of valid entries only
    masked = loss_per_entry * concept_valid.float()
    return masked.sum() / concept_valid.sum().clamp(min=1)


# ============================================================
# POS-WEIGHT (mask-aware)
# ============================================================

def build_pos_weight(train_loader, num_concepts, device):
    """
    Compute pos_weight = #negatives / #positives per concept,
    counting only samples where the source feature was observed.
    Caps at 20 to prevent gradient explosion on extremely rare concepts.
    """
    print("Computing per-concept pos_weight from training data...")
    pos_counts   = torch.zeros(num_concepts)
    valid_counts = torch.zeros(num_concepts)

    for batch in train_loader:
        if batch is None:
            continue
        labels    = batch['ehr_concepts']   # (B, 13)
        feat_mask = batch['ehr_mask']       # (B, 21)
        B = labels.shape[0]

        concept_valid = torch.ones(B, num_concepts, dtype=torch.bool)
        for c_idx, f_idx in EHR_CONCEPT_TO_FEATURE.items():
            concept_valid[:, c_idx] = feat_mask[:, f_idx].bool()

        pos_counts   += (labels * concept_valid.float()).sum(dim=0)
        valid_counts += concept_valid.float().sum(dim=0)

    neg_counts = valid_counts - pos_counts
    pos_weight = torch.where(
        pos_counts > 0,
        neg_counts / pos_counts.clamp(min=1),
        torch.ones(num_concepts)
    ).clamp(max=20.0).to(device)

    print("Per-concept pos_weight (valid samples only):")
    for c, w in zip(EHR_CONCEPTS, pos_weight.tolist()):
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
    parser = argparse.ArgumentParser(description="EHR Concept Prediction — Isolation")

    parser.add_argument('--train_csv', type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv')
    parser.add_argument('--val_csv', type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv')
    parser.add_argument('--test_csv', type=str,
                        default='/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv')

    parser.add_argument('--save_dir',          type=str, default='./ehr_concept_checkpoints')
    parser.add_argument('--exp_name',          type=str, default='ehr_concept_debug')
    parser.add_argument('--batch_size',        type=int,   default=256)
    parser.add_argument('--lr',                type=float, default=1e-3)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--epochs',            type=int,   default=50)
    parser.add_argument('--patience_scheduler',  type=int, default=5)
    parser.add_argument('--patience_early_stop', type=int, default=10)

    return parser.parse_args()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(model, loader, pos_weight, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        if batch is None:
            continue

        ehr      = batch['ehr_features'].to(device)   # (B, 21)
        concepts = batch['ehr_concepts'].to(device)   # (B, 13)
        mask     = batch['ehr_mask'].to(device)       # (B, 21) ← used here

        optimizer.zero_grad()
        with autocast('cuda'):
            logits = model(ehr)
            loss   = masked_ehr_concept_loss(logits, concepts, mask, pos_weight)

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

def validate(model, loader, pos_weight, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            ehr      = batch['ehr_features'].to(device)
            concepts = batch['ehr_concepts'].to(device)
            mask     = batch['ehr_mask'].to(device)

            logits = model(ehr)
            loss   = masked_ehr_concept_loss(logits, concepts, mask, pos_weight)
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
# LOG PER CONCEPT
# ============================================================

def log_per_concept(logger, per_auroc):
    for concept, auroc in zip(EHR_CONCEPTS, per_auroc):
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

    # --- Data ---
    train_loader, val_loader, test_loader = get_ehr_concept_dataloaders(
        args.train_csv, args.val_csv, args.test_csv,
        batch_size=args.batch_size
    )

    # --- Model ---
    model = EHRConceptEncoder(input_dim=21, num_concepts=len(EHR_CONCEPTS)).to(device)

    # --- Pos-weight computed from training data with mask awareness ---
    pos_weight = build_pos_weight(train_loader, len(EHR_CONCEPTS), device)

    save_path = os.path.join(args.save_dir, args.exp_name, "best_weights.pth")

    # --- Optimizer and scheduler ---
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        patience=args.patience_scheduler, factor=0.5
    )
    scaler = GradScaler()

    best_val_auroc    = 0.0
    epochs_no_improve = 0

    # --- Epoch loop ---
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, pos_weight, optimizer, scaler, device, epoch
        )
        val_loss, mean_auroc, per_auroc = validate(
            model, val_loader, pos_weight, device
        )
        scheduler.step(mean_auroc)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
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

    # --- Final test evaluation ---
    logger.info("\n=== Test Set Evaluation ===")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss, test_auroc, test_per_auroc = validate(
        model, test_loader, pos_weight, device
    )
    logger.info(f"Test Loss: {test_loss:.4f} | Mean AUROC: {test_auroc:.4f}")
    log_per_concept(logger, test_per_auroc)


if __name__ == "__main__":
    main()