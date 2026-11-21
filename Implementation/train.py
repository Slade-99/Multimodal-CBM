import os
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
from config import get_args
from cbm_model import EHREncoder, UniModalCXR, UniModalEHR, UniModalECG
from dataloader import get_dataloaders, ModalityDropoutConfig
from modality_dropout import (
    TRIMODAL_CONFIG, BIMODAL_CXR_EHR_CONFIG,
    BIMODAL_CXR_ECG_CONFIG, BIMODAL_EHR_ECG_CONFIG,
    ModalityDropoutScheduler
)

torch.manual_seed(7)
np.random.seed(7)


def setup_logger(save_dir, exp_name):
    log_file = os.path.join(save_dir, exp_name, "training_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
        ]
    )
    return logging.getLogger()


# ================= JOINT LOSS FUNCTION =================
import torch.nn.functional as F

class JointCBMLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

        self.ehr_concept_map = {
            0: 1, 1: 3, 2: 5, 3: 2, 4: 4, 6: 11,
            7: 8, 8: 13, 9: 16, 10: 12, 11: 9, 12: 15
        }

        weight_path = "/home/azwad/Works/Multimodal-CBM/Implementation/concept_weights.pt"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing {weight_path}. Run get_concept_weights.py first!")

        concept_weights = torch.load(weight_path, map_location=device, weights_only=True)

        self.bce_cxr = nn.BCEWithLogitsLoss(pos_weight=concept_weights['cxr'])
        self.bce_ecg = nn.BCEWithLogitsLoss(pos_weight=concept_weights['ecg'])
        self.ehr_pos_weight = concept_weights['ehr']

        self.bce_mortality = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.44], device=device))
        self.bce_ahf       = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.89],  device=device))

    def forward(self, outputs, targets, cxr_mod_mask, ecg_mod_mask, ehr_mod_mask, ehr_feature_mask):
        concept_losses = []

        def masked_concept_loss(loss_fn, logits, labels, mask):
            present = mask.bool()
            if present.any():
                return loss_fn(logits[present], labels[present].float())
            return None

        def masked_ehr_loss(logits, labels, mod_mask, feat_mask):
            mod_present = mod_mask.bool()
            if not mod_present.any():
                return None
            concept_mask = torch.ones((feat_mask.shape[0], 13), device=feat_mask.device).bool()
            for c_idx, f_idx in self.ehr_concept_map.items():
                concept_mask[:, c_idx] = feat_mask[:, f_idx].bool()
            final_mask = mod_present.unsqueeze(1) & concept_mask
            if not final_mask.any():
                return None
            loss = F.binary_cross_entropy_with_logits(
                logits, labels.float(),
                pos_weight=self.ehr_pos_weight,
                reduction='none'
            )
            masked_loss = loss * final_mask.float()
            return masked_loss.sum() / final_mask.sum()

        l_cxr = masked_concept_loss(self.bce_cxr, outputs['cxr_concept_logits'], targets['cxr_concepts'], cxr_mod_mask)
        l_ecg = masked_concept_loss(self.bce_ecg, outputs['ecg_concept_logits'], targets['ecg_concepts'], ecg_mod_mask)
        l_ehr = masked_ehr_loss(outputs['ehr_concept_logits'], targets['ehr_concepts'], ehr_mod_mask, ehr_feature_mask)

        for l in [l_cxr, l_ecg, l_ehr]:
            if l is not None:
                concept_losses.append(l)

        concept_loss = torch.stack(concept_losses).mean() if concept_losses else torch.tensor(0.0, device=ehr_feature_mask.device)

        loss_mortality = self.bce_mortality(outputs['mortality_logits'], targets['target_mortality'].float())
        loss_ahf       = self.bce_ahf(outputs['ahf_logits'], targets['target_ahf'].float())
        target_loss    = (loss_mortality + loss_ahf) / 2.0

        total_loss = (self.alpha * concept_loss) + (self.beta * target_loss)
        return total_loss, concept_loss, target_loss


# ================= METRICS =================

def compute_metrics(y_true, y_scores):
    try:
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
    except ValueError:
        auroc, auprc = 0.5, 0.0
    return auroc, auprc


def compute_concept_auroc(y_true, y_scores):
    """
    Computes per-concept AUROC and returns the mean.
    y_true, y_scores: np.ndarray of shape (N, num_concepts)

    This replaces the threshold-based accuracy metric which was always
    reporting ~30% because rare concepts are almost never predicted
    as positive at a fixed 0.5 threshold on imbalanced data.
    """
    aurocs = []
    for i in range(y_true.shape[1]):
        try:
            aurocs.append(roc_auc_score(y_true[:, i], y_scores[:, i]))
        except ValueError:
            aurocs.append(0.5)
    return np.array(aurocs), np.mean(aurocs)


# ================= MAIN =================

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger = setup_logger(args.save_dir, args.exp_name)
    logger.info(f"Starting Experiment: {args.experiment} on {device}")
    logger.info(f"Arguments: {vars(args)}")

    # --- 1. Active Modalities ---
    active_mods = []
    if 'cxr' in args.experiment or args.experiment == 'trimodal': active_mods.append('cxr')
    if 'ecg' in args.experiment or args.experiment == 'trimodal': active_mods.append('ecg')
    if 'ehr' in args.experiment or args.experiment == 'trimodal': active_mods.append('ehr')

    drop_config = None
    dropout_scheduler = None

    # --- 2. Data ---
    train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

    train_loader, val_loader, test_loader, train_ds = get_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=args.batch_size,
        dropout_config=drop_config
    )
    logger.info("Dataloaders initialized successfully.")

    # --- 3. Model and Loss ---
    if args.experiment == "cxr_only":
        model = UniModalCXR().to(device)
    elif args.experiment == "ecg_only":
        model = UniModalECG().to(device)
    elif args.experiment == "ehr_only":
        model = UniModalEHR().to(device)
    else:
        raise ValueError(f"Unknown experiment: {args.experiment}")

    criterion = JointCBMLoss(alpha=args.alpha_concept, beta=args.beta_target, device=device)

    # --- 4. Two-Stage Setup ---
    FINETUNE_EPOCHS = 5

    def get_backbone(m):
        if args.experiment == "cxr_only":
            return m.encoder.features
        elif args.experiment == "ecg_only":
            return nn.Sequential(m.encoder.conv1, m.encoder.bn1,
                                 m.encoder.layer1, m.encoder.layer2,
                                 m.encoder.layer3, m.encoder.layer4)
        return None

    backbone = get_backbone(model)

    if backbone is not None:
        logger.info(f"=== Stage 1: Freezing backbone for {FINETUNE_EPOCHS} epochs ===")
        for param in backbone.parameters():
            param.requires_grad = False
    else:
        logger.info("=== EHR: no backbone to freeze — single-stage training ===")
        FINETUNE_EPOCHS = 0

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_finetune, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience_scheduler, factor=0.5)

    # --- 5. State ---
    best_combined_score = 0.0   # Now tracks concept + downstream jointly
    epochs_no_improve   = 0
    save_path = os.path.join(args.save_dir, args.exp_name, "best_final_weights.pth")
    scaler    = GradScaler()

    # Which concept key and concept count to track for this experiment
    concept_key_map = {
        'cxr_only': ('cxr_concept_logits', 'cxr_concepts', 15),
        'ecg_only': ('ecg_concept_logits', 'ecg_concepts', 14),
        'ehr_only': ('ehr_concept_logits', 'ehr_concepts', 13),
    }
    logits_key, target_key, num_concepts = concept_key_map[args.experiment]

    # --- 6. Epoch Loop ---
    for epoch in range(args.epochs):

        # Stage transition
        if epoch == FINETUNE_EPOCHS and backbone is not None:
            logger.info("=== Stage 2: Unfreezing backbone — end-to-end training ===")
            for param in backbone.parameters():
                param.requires_grad = True

            if args.experiment == "cxr_only":
                param_groups = [
                    {'params': model.encoder.features.parameters(),     'lr': args.lr_e2e},
                    {'params': model.encoder.concept_head.parameters(), 'lr': args.lr_finetune},
                    {'params': model.mortality_head.parameters(),        'lr': args.lr_finetune},
                    {'params': model.ahf_head.parameters(),             'lr': args.lr_finetune},
                ]
            elif args.experiment == "ecg_only":
                param_groups = [
                    {'params': model.encoder.conv1.parameters(),        'lr': args.lr_e2e},
                    {'params': model.encoder.bn1.parameters(),          'lr': args.lr_e2e},
                    {'params': model.encoder.layer1.parameters(),       'lr': args.lr_e2e},
                    {'params': model.encoder.layer2.parameters(),       'lr': args.lr_e2e},
                    {'params': model.encoder.layer3.parameters(),       'lr': args.lr_e2e},
                    {'params': model.encoder.layer4.parameters(),       'lr': args.lr_e2e},
                    {'params': model.encoder.concept_head.parameters(), 'lr': args.lr_finetune},
                    {'params': model.mortality_head.parameters(),        'lr': args.lr_finetune},
                    {'params': model.ahf_head.parameters(),             'lr': args.lr_finetune},
                ]

            optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience_scheduler, factor=0.5)
            logger.info(f"Backbone LR: {args.lr_e2e} | Heads LR: {args.lr_finetune}")

        # ---- Train ----
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for batch in loop:
            image        = batch['image'].to(device)
            waveform     = batch['waveform'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            ehr_feat_mask = batch['ehr_mask'].to(device)
            cxr_mask      = batch['cxr_mod_mask'].to(device)
            ecg_mask      = batch['ecg_mod_mask'].to(device)
            ehr_mask      = batch['ehr_mod_mask'].to(device)

            targets = {
                'cxr_concepts':     batch['cxr_concepts'].to(device),
                'ecg_concepts':     batch['ecg_concepts'].to(device),
                'ehr_concepts':     batch['ehr_concepts'].to(device),
                'target_mortality': batch['target_mortality'].to(device),
                'target_ahf':       batch['target_ahf'].to(device)
            }

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)
                loss, _, _ = criterion(outputs, targets, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0

        all_mort_preds,    all_mort_targets    = [], []
        all_ahf_preds,     all_ahf_targets     = [], []
        all_concept_preds, all_concept_targets = [], []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False):
                image        = batch['image'].to(device)
                waveform     = batch['waveform'].to(device)
                ehr_features = batch['ehr_features'].to(device)
                cxr_mask      = batch['cxr_mod_mask'].to(device)
                ecg_mask      = batch['ecg_mod_mask'].to(device)
                ehr_mask      = batch['ehr_mod_mask'].to(device)
                ehr_feat_mask = batch['ehr_mask'].to(device)

                targets = {
                    'cxr_concepts':     batch['cxr_concepts'].to(device),
                    'ecg_concepts':     batch['ecg_concepts'].to(device),
                    'ehr_concepts':     batch['ehr_concepts'].to(device),
                    'target_mortality': batch['target_mortality'].to(device),
                    'target_ahf':       batch['target_ahf'].to(device)
                }

                outputs = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)
                loss, _, _ = criterion(outputs, targets, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)
                val_loss += loss.item()

                all_mort_preds.extend(torch.sigmoid(outputs['mortality_logits']).cpu().numpy())
                all_mort_targets.extend(targets['target_mortality'].cpu().numpy())
                all_ahf_preds.extend(torch.sigmoid(outputs['ahf_logits']).cpu().numpy())
                all_ahf_targets.extend(targets['target_ahf'].cpu().numpy())

                # Collect concept predictions for the active modality only
                concept_probs = torch.sigmoid(outputs[logits_key]).cpu().numpy()
                concept_lbls  = targets[target_key].cpu().numpy()
                all_concept_preds.append(concept_probs)
                all_concept_targets.append(concept_lbls)

        mort_auroc, mort_auprc = compute_metrics(all_mort_targets, all_mort_preds)
        ahf_auroc,  ahf_auprc  = compute_metrics(all_ahf_targets,  all_ahf_preds)
        avg_task_auroc = (mort_auroc + ahf_auroc) / 2.0

        concept_preds_arr   = np.concatenate(all_concept_preds,   axis=0)
        concept_targets_arr = np.concatenate(all_concept_targets,  axis=0)
        per_concept_auroc, mean_concept_auroc = compute_concept_auroc(concept_targets_arr, concept_preds_arr)

        avg_val_loss = val_loss / len(val_loader)

        # Combined score: equal weight on concept and downstream AUROC.
        # This is the key fix — model selection now cares about BOTH objectives.
        # Previously best model was saved on downstream AUROC only, so concept
        # loss had zero influence on which checkpoint was kept.
        combined_score = 0.5 * mean_concept_auroc + 0.5 * avg_task_auroc

        stage_tag = "Stage1" if epoch < FINETUNE_EPOCHS else "Stage2"
        logger.info(f"[{stage_tag}] Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Mortality  - AUROC: {mort_auroc:.4f}, AUPRC: {mort_auprc:.4f}")
        logger.info(f"AHF        - AUROC: {ahf_auroc:.4f}, AUPRC: {ahf_auprc:.4f}")
        logger.info(f"Task AUROC: {avg_task_auroc:.4f} | Concept AUROC: {mean_concept_auroc:.4f} | Combined: {combined_score:.4f}")
        logger.info(f"Per-concept AUROC: { {k: round(float(v),4) for k,v in zip(range(num_concepts), per_concept_auroc)} }")

        if dropout_scheduler is not None:
            dropout_scheduler.step()

        # Scheduler now tracks combined score, not just task AUROC
        scheduler.step(combined_score)

        if combined_score > best_combined_score:
            best_combined_score = combined_score
            epochs_no_improve   = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"[*] New best model saved (combined={best_combined_score:.4f})\n")
        else:
            epochs_no_improve += 1
            logger.info(f"[!] No improvement for {epochs_no_improve} epochs.\n")

        if epochs_no_improve >= args.patience_early_stop:
            logger.info("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()