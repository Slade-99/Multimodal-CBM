"""
train_fusion.py  —  Stage 2: HealNet-CBM Fusion Training
=========================================================
Assumes Stage 1 (concept encoder pretraining) is already complete and
saved weights exist for UniModalCXR, UniModalECG, UniModalEHR.

Architecture:
  - Frozen encoders produce sigmoid concept probability vectors
  - Each concept becomes its own token → 42 tokens max (15 CXR + 14 ECG + 13 EHR)
  - Multi-head cross-attention over concept tokens (cross-modal, cross-concept)
  - Task heads predict Mortality and AHF from attended representation
  - Attention weights are saved per-batch for all 4 explainability layers

Explainability outputs per forward pass:
  1. Modality contribution  — mean attention received per modality block
  2. Concept contribution   — mean attention received per concept token
  3. Cross-modal attention  — attention from concept i (mod A) to concept j (mod B)
  4. Counterfactuals        — handled in test/eval script by flipping concept bits

Experiments handled: cxr_ecg, cxr_ehr, ecg_ehr, trimodal
"""

import os
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from cbm_model import UniModalCXR, UniModalECG, UniModalEHR
from dataloader import get_dataloaders
from config_2 import get_args, CXR_CONCEPTS, ECG_CONCEPTS, EHR_CONCEPTS

# ============================================================
# CONSTANTS
# ============================================================

ALL_CONCEPTS = CXR_CONCEPTS + ECG_CONCEPTS + EHR_CONCEPTS   # 15 + 14 + 13 = 42

# Maps experiment name → active modalities and their concept slices
# Slice indices into the 42-concept concatenated vector
EXPERIMENT_CONFIG = {
    "cxr_ecg":  {"modalities": ["cxr", "ecg"],       "slices": {"cxr": slice(0,15),  "ecg": slice(15,29)}},
    "cxr_ehr":  {"modalities": ["cxr", "ehr"],       "slices": {"cxr": slice(0,15),  "ehr": slice(29,42)}},
    "ecg_ehr":  {"modalities": ["ecg", "ehr"],       "slices": {"ecg": slice(15,29), "ehr": slice(29,42)}},
    "trimodal": {"modalities": ["cxr", "ecg", "ehr"],"slices": {"cxr": slice(0,15),  "ecg": slice(15,29), "ehr": slice(29,42)}},
}

CONCEPT_COUNTS = {"cxr": 15, "ecg": 14, "ehr": 13}

# Encoder weight paths — update these to wherever Stage 1 saved them
ENCODER_WEIGHTS = {
    "cxr": "/home/azwad/Works/Multimodal-CBM/checkpoints_stage1/cxr_only/best_final_weights.pth",
    "ecg": "/home/azwad/Works/Multimodal-CBM/checkpoints_stage1/ecg_only/best_final_weights.pth",
    "ehr": "/home/azwad/Works/Multimodal-CBM/checkpoints_stage1/ehr_only/best_final_weights.pth",
}


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
# MODEL
# ============================================================

class ConceptTokenEmbedding(nn.Module):
    """
    Lifts a vector of N concept probabilities into N separate d_model-dim tokens.
    Each concept gets its own learned linear embedding + a learned positional
    embedding that encodes both (a) which concept it is and (b) which modality
    it came from. This lets the attention distinguish cross-concept vs
    cross-modal relationships.

    Input:  (B, N_concepts)  float in [0, 1]
    Output: (B, N_concepts, d_model)
    """
    def __init__(self, num_concepts: int, d_model: int, modality_id: int, num_modalities: int = 3):
        super().__init__()
        # Per-concept value projection: scalar → d_model
        self.value_proj = nn.Linear(1, d_model)

        # Learned concept position embeddings: one per concept
        self.concept_pos = nn.Embedding(num_concepts, d_model)

        # Learned modality embeddings: shared across all concepts of this modality
        self.modality_emb = nn.Embedding(num_modalities, d_model)
        self.modality_id  = modality_id

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, N)
        B, N = x.shape

        # Project each concept value: (B, N, 1) → (B, N, d_model)
        val_emb = self.value_proj(x.unsqueeze(-1))

        # Concept position: (N,) → (N, d_model) → broadcast to (B, N, d_model)
        pos_idx = torch.arange(N, device=x.device)
        pos_emb = self.concept_pos(pos_idx).unsqueeze(0).expand(B, -1, -1)

        # Modality embedding: scalar → (1, 1, d_model) → broadcast
        mod_idx = torch.tensor([self.modality_id], device=x.device)
        mod_emb = self.modality_emb(mod_idx).unsqueeze(0).expand(B, N, -1)

        return self.norm(val_emb + pos_emb + mod_emb)   # (B, N, d_model)


def _build_task_head(d_model: int) -> nn.Sequential:
    """Shared factory for mortality / AHF prediction heads."""
    return nn.Sequential(
        nn.Linear(d_model, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1)
    )


class HealNetCBM(nn.Module):
    """
    Concept-Bottlenecked HealNet Fusion Model.

    Flow:
      1. Frozen encoders → sigmoid concept probabilities per modality
      2. ConceptTokenEmbedding lifts each concept to a d_model token
      3. Multi-head self-attention over all concept tokens jointly
         (cross-modal attention emerges naturally — token from CXR concept i
          can attend to ECG concept j)
      4. Mean-pool attended tokens → task heads

    Attention weights are returned for explainability layers 1–3.
    Layer 4 (counterfactuals) is handled at inference time by
    directly manipulating the concept probability vectors.
    """
    def __init__(self, active_modalities: list, d_model: int = 128, num_heads: int = 4):
        super().__init__()
        self.active_modalities = active_modalities
        self.d_model = d_model

        # ----- Frozen Encoders -----
        modality_id_map = {"cxr": 0, "ecg": 1, "ehr": 2}

        self.encoders   = nn.ModuleDict()
        self.embedders  = nn.ModuleDict()
        self.concept_ranges = {}   # {modality: (start_idx, end_idx)} in the full token sequence

        token_offset = 0
        for mod in active_modalities:
            # Load UniModal wrapper — we only need the encoder sub-module
            if mod == "cxr":
                wrapper = UniModalCXR()
            elif mod == "ecg":
                wrapper = UniModalECG()
            else:
                wrapper = UniModalEHR()

            encoder = wrapper.encoder
            for p in encoder.parameters():
                p.requires_grad = False

            self.encoders[mod]  = encoder
            self.embedders[mod] = ConceptTokenEmbedding(
                num_concepts=CONCEPT_COUNTS[mod],
                d_model=d_model,
                modality_id=modality_id_map[mod]
            )

            n = CONCEPT_COUNTS[mod]
            self.concept_ranges[mod] = (token_offset, token_offset + n)
            token_offset += n

        self.total_tokens = token_offset   # e.g. 42 for trimodal

        # ----- Transformer Encoder (cross-concept attention) -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True    # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # We need raw attention weights for explainability, so we also keep
        # a single MHA layer to extract them explicitly at inference time
        self.attn_for_explain = nn.MultiheadAttention(
            d_model, num_heads=num_heads, batch_first=True, dropout=0.0
        )

        # ----- Task Heads -----
        self.mort_head = _build_task_head(d_model)
        self.ahf_head  = _build_task_head(d_model)

    def encode_concepts(self, image, waveform, ehr_features):
        """
        Runs all frozen encoders and returns a dict of sigmoid concept probs.
        Separated so inference-time counterfactual interventions can directly
        supply modified concept vectors and skip this step.
        """
        concept_probs = {}
        with torch.no_grad():
            if "cxr" in self.active_modalities:
                concept_probs["cxr"] = torch.sigmoid(self.encoders["cxr"](image))
            if "ecg" in self.active_modalities:
                concept_probs["ecg"] = torch.sigmoid(self.encoders["ecg"](waveform))
            if "ehr" in self.active_modalities:
                concept_probs["ehr"] = torch.sigmoid(self.encoders["ehr"](ehr_features))
        return concept_probs

    def fuse(self, concept_probs: dict, return_attention: bool = False):
        """
        Takes concept probability dicts and runs the fusion transformer.
        Separating encode_concepts() and fuse() enables counterfactual
        interventions: caller can modify concept_probs before calling fuse().

        Args:
            concept_probs:    dict mod → (B, N_concepts) sigmoid probs
            return_attention: if True, also return raw attention weights
                              shape (B, total_tokens, total_tokens)

        Returns:
            outputs dict with mortality_logits, ahf_logits
            optionally: attention_weights (B, total_tokens, total_tokens)
        """
        # Build token sequence: (B, total_tokens, d_model)
        token_list = []
        for mod in self.active_modalities:
            tokens = self.embedders[mod](concept_probs[mod])   # (B, N_mod, d_model)
            token_list.append(tokens)

        x = torch.cat(token_list, dim=1)   # (B, total_tokens, d_model)

        # Pass through transformer
        x_attended = self.transformer(x)   # (B, total_tokens, d_model)

        # Mean pool over all concept tokens
        pooled = x_attended.mean(dim=1)    # (B, d_model)

        outputs = {
            "mortality_logits": self.mort_head(pooled).squeeze(-1),
            "ahf_logits":       self.ahf_head(pooled).squeeze(-1),
            "concept_probs":    concept_probs,   # pass through for logging
        }

        if return_attention:
            # Extract attention weights explicitly using the dedicated MHA layer
            # This is separate from the transformer so we get clean weights
            with torch.no_grad():
                _, attn_weights = self.attn_for_explain(x, x, x, need_weights=True, average_attn_weights=True)
            outputs["attention_weights"] = attn_weights   # (B, total_tokens, total_tokens)

        return outputs

    def forward(self, image, waveform, ehr_features,
                cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask,
                return_attention: bool = False):
        concept_probs = self.encode_concepts(image, waveform, ehr_features)
        return self.fuse(concept_probs, return_attention=return_attention)

    def get_modality_attention(self, attn_weights: torch.Tensor) -> dict:
        """
        Explainability Layer 1: per-modality attention contribution.
        Aggregates token-level attention weights into modality-level scores.

        attn_weights: (B, total_tokens, total_tokens)
        Returns: dict mod → scalar mean attention received (averaged over batch)
        """
        mod_attn = {}
        for mod in self.active_modalities:
            start, end = self.concept_ranges[mod]
            # Mean attention received by all tokens of this modality
            mod_attn[mod] = attn_weights[:, :, start:end].mean(dim=(1, 2)).mean().item()
        return mod_attn

    def get_concept_attention(self, attn_weights: torch.Tensor) -> dict:
        """
        Explainability Layer 2: per-concept attention contribution.
        Returns attention received by each concept token.

        attn_weights: (B, total_tokens, total_tokens)
        Returns: dict mod → np.ndarray of shape (N_concepts,)
        """
        concept_attn = {}
        for mod in self.active_modalities:
            start, end = self.concept_ranges[mod]
            # Mean attention received by each concept token across batch
            concept_attn[mod] = attn_weights[:, :, start:end].mean(dim=(0, 1)).cpu().numpy()
        return concept_attn

    def get_cross_modal_attention(self, attn_weights: torch.Tensor, mod_a: str, mod_b: str) -> np.ndarray:
        """
        Explainability Layer 3: cross-modal concept-to-concept attention.
        Returns how much each concept in mod_a attends to each concept in mod_b.

        Returns: np.ndarray of shape (N_concepts_a, N_concepts_b)
        """
        sa, ea = self.concept_ranges[mod_a]
        sb, eb = self.concept_ranges[mod_b]
        # attn[b, i, j] = how much token i attends to token j
        return attn_weights[:, sa:ea, sb:eb].mean(dim=0).cpu().numpy()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(model, loader, optimizer, scaler, device,
                bce_mort, bce_ahf, logger, epoch):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        image        = batch['image'].to(device)
        waveform     = batch['waveform'].to(device)
        ehr_features = batch['ehr_features'].to(device)
        cxr_mask     = batch['cxr_mod_mask'].to(device)
        ecg_mask     = batch['ecg_mod_mask'].to(device)
        ehr_mask     = batch['ehr_mod_mask'].to(device)
        ehr_feat_mask= batch['ehr_mask'].to(device)

        y_mort = batch['target_mortality'].float().to(device)
        y_ahf  = batch['target_ahf'].float().to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(image, waveform, ehr_features,
                            cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)
            loss = (bce_mort(outputs['mortality_logits'], y_mort) +
                    bce_ahf(outputs['ahf_logits'],        y_ahf)) / 2.0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader, device, bce_mort, bce_ahf):
    model.eval()

    mort_probs,  mort_targets  = [], []
    ahf_probs,   ahf_targets   = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            image        = batch['image'].to(device)
            waveform     = batch['waveform'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            cxr_mask     = batch['cxr_mod_mask'].to(device)
            ecg_mask     = batch['ecg_mod_mask'].to(device)
            ehr_mask     = batch['ehr_mod_mask'].to(device)
            ehr_feat_mask= batch['ehr_mask'].to(device)

            y_mort = batch['target_mortality'].float().to(device)
            y_ahf  = batch['target_ahf'].float().to(device)

            outputs = model(image, waveform, ehr_features,
                            cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)

            loss = (bce_mort(outputs['mortality_logits'], y_mort) +
                    bce_ahf(outputs['ahf_logits'],        y_ahf)) / 2.0
            total_loss += loss.item()

            mort_probs.extend(torch.sigmoid(outputs['mortality_logits']).cpu().numpy())
            mort_targets.extend(batch['target_mortality'].numpy())
            ahf_probs.extend(torch.sigmoid(outputs['ahf_logits']).cpu().numpy())
            ahf_targets.extend(batch['target_ahf'].numpy())

    mort_auroc = roc_auc_score(mort_targets, mort_probs)
    ahf_auroc  = roc_auc_score(ahf_targets,  ahf_probs)
    mort_auprc = average_precision_score(mort_targets, mort_probs)
    ahf_auprc  = average_precision_score(ahf_targets,  ahf_probs)
    avg_loss   = total_loss / max(len(loader), 1)

    return {
        "mort_auroc": mort_auroc, "mort_auprc": mort_auprc,
        "ahf_auroc":  ahf_auroc,  "ahf_auprc":  ahf_auprc,
        "combined":   (mort_auroc + ahf_auroc) / 2.0,
        "val_loss":   avg_loss
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args   = get_args()
    device = torch.device('cuda')

    if args.experiment not in EXPERIMENT_CONFIG:
        raise ValueError(
            f"This script handles bimodal/trimodal experiments only. "
            f"Got: {args.experiment}. "
            f"Valid: {list(EXPERIMENT_CONFIG.keys())}"
        )

    exp_cfg           = EXPERIMENT_CONFIG[args.experiment]
    active_modalities = exp_cfg["modalities"]

    logger = setup_logger(args.save_dir, args.exp_name)
    logger.info(f"Experiment : {args.experiment}")
    logger.info(f"Modalities : {active_modalities}")
    logger.info(f"Arguments  : {vars(args)}")
    train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"
    # --- Data ---
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=args.batch_size,
        dropout_config=None
    )
    logger.info("Dataloaders ready.")

    # --- Model ---
    model = HealNetCBM(active_modalities=active_modalities, d_model=128, num_heads=4).to(device)

    # Load pretrained encoder weights from Stage 1
    for mod in active_modalities:
        weight_path = ENCODER_WEIGHTS[mod]
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Stage 1 weights not found for {mod} at {weight_path}. "
                f"Run unimodal concept training first."
            )
        # Stage 1 saved the full UniModal wrapper state dict.
        # We only need the encoder sub-module weights.
        state_dict     = torch.load(weight_path, map_location=device, weights_only=True)
        encoder_sd     = {k.replace("encoder.", "", 1): v
                          for k, v in state_dict.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoders[mod].load_state_dict(encoder_sd, strict=True)
        logger.info(f"Loaded {mod} encoder weights from {weight_path}")
        if missing:
            logger.warning(f"  Missing keys: {missing}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {unexpected}")

    # Confirm encoders are frozen
    total_params    = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,} "
                f"(encoders frozen = {total_params - trainable_params:,} frozen)")

    # --- Loss with pos_weight for class imbalance ---
    # pos_weight = #negatives / #positives
    # Mortality ~7% positive → ~13x, AHF ~26% positive → ~2.89x (from your original config)
    bce_mort = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.44], device=device))
    bce_ahf  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.89],  device=device))

    # --- Optimizer: only train fusion + task head params ---
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_finetune,
        weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        patience=args.patience_scheduler, factor=0.5
    )
    scaler = GradScaler()

    # --- Training state ---
    best_combined     = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(args.save_dir, args.exp_name, "best_fusion_weights.pth")

    # --- Epoch loop ---
    for epoch in range(1, args.epochs + 1):

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device,
            bce_mort, bce_ahf, logger, epoch
        )

        metrics = evaluate(model, val_loader, device, bce_mort, bce_ahf)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {metrics['val_loss']:.4f}"
        )
        logger.info(
            f"  Mortality  AUROC: {metrics['mort_auroc']:.4f}  AUPRC: {metrics['mort_auprc']:.4f}"
        )
        logger.info(
            f"  AHF        AUROC: {metrics['ahf_auroc']:.4f}  AUPRC: {metrics['ahf_auprc']:.4f}"
        )
        logger.info(f"  Combined AUROC: {metrics['combined']:.4f}")

        scheduler.step(metrics['combined'])

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

    logger.info(f"\nTraining complete. Best combined AUROC: {best_combined:.4f}")
    logger.info(f"Weights saved to: {save_path}")


if __name__ == "__main__":
    main()