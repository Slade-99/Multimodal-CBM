"""
train_fusion_healnet.py  —  Stage 2: True HEALNet-CBM Fusion Training
======================================================================
Implements the genuine HEALNet architecture principle:

  "A shared latent bottleneck S is iteratively updated through d
   attention-based fusion layers. Each layer uses the shared S as
   the Query and each modality's own concept tokens as Keys/Values.
   Modality-specific Q/K/V projection weights are kept separate."

   — Hemker et al., NeurIPS 2024 (arXiv:2311.09115)

Combined with CBM (Concept Bottleneck Model) principle:
  - Frozen Stage-1 encoders produce interpretable sigmoid concept probs
  - Concept probs become the modality token sequences X_m
  - encode_concepts() and fuse() are separated so Layer-4 counterfactual
    interventions can modify concept_probs before fusion runs

Architecture Flow:
  1. Frozen encoders → sigmoid concept probability vectors per modality
  2. Per-modality linear projectors lift concept probs → d_model tokens  (X_m)
  3. Learned shared latent S  shape (B, n_latent, d_model)               (S)
  4. For each of d fusion layers:
       a. For each active modality m:
            Q_m  = S  @ W_m^Q          (shared latent as query)
            K_m  = X_m @ W_m^K         (modality concepts as keys)
            V_m  = X_m @ W_m^V         (modality concepts as values)
            A_m  = softmax(Q_m K_m^T / sqrt(d)) V_m   ← cross-attention
            attn_weights_m stored for explainability
       b. S ← S + masked_mean_over_modalities(A_m)
            — missing-modality mask zeros out absent modalities PER SAMPLE
            — mean is taken only over present modalities per sample
       c. S ← SNN(S)                           (Self-Normalising Network)
  5. Mean-pool final S → task heads → Mortality logit, AHF logit

Missing Modality Handling (per-sample, not per-experiment):
  - Each sample in a batch carries binary modality masks from the dataloader
    (cxr_mod_mask, ecg_mod_mask, ehr_mod_mask) where 0 = modality absent.
  - In encode_concepts(): absent samples get zero concept probs (encoder
    output is zeroed by the mask) so no garbage propagates forward.
  - In HEALNetFusionLayer.forward(): the attended output A_m for each
    modality is multiplied by its per-sample mask before being averaged
    into S. Samples missing modality m get zero contribution from it.
  - The mean divisor is the count of PRESENT modalities per sample, so
    S is always on the same scale regardless of how many modalities exist.
  - Explainability layers 1-3 receive the same masks and report
    attention only over present modalities.

Explainability Layers (all 4):
  1. Modality contribution  — mean cross-attention weight S → X_m per modality
                              (only over samples where that modality is present)
  2. Concept contribution   — mean cross-attention weight per concept token
                              (masked to present samples only)
  3. Cross-modal attention  — co-attention between mod_a and mod_b concepts
                              (only over samples where BOTH are present)
  4. Counterfactuals        — caller modifies concept_probs dict, calls fuse()
                              directly; encode_concepts() is bypassed

Experiments handled: cxr_ecg, cxr_ehr, ecg_ehr, trimodal
"""

import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
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

EXPERIMENT_CONFIG = {
    "cxr_ecg":  {"modalities": ["cxr", "ecg"],        "slices": {"cxr": slice(0, 15),  "ecg": slice(15, 29)}},
    "cxr_ehr":  {"modalities": ["cxr", "ehr"],        "slices": {"cxr": slice(0, 15),  "ehr": slice(29, 42)}},
    "ecg_ehr":  {"modalities": ["ecg", "ehr"],        "slices": {"ecg": slice(15, 29), "ehr": slice(29, 42)}},
    "trimodal": {"modalities": ["cxr", "ecg", "ehr"], "slices": {"cxr": slice(0, 15),  "ecg": slice(15, 29), "ehr": slice(29, 42)}},
}

CONCEPT_COUNTS = {"cxr": 15, "ecg": 14, "ehr": 13}

ENCODER_WEIGHTS = {
    "cxr": "/home/azwad/Works/Multimodal-CBM/cxr_concept_checkpoints/xrv_resnet50_concept/best_weights.pth",
    "ecg": "/home/azwad/Works/Multimodal-CBM/ecg_concept_checkpoints/ecg_concepts/best.pth",
    "ehr": "/home/azwad/Works/Multimodal-CBM/ehr_concept_checkpoints/ehr_concept_debug/best_weights.pth",
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
# BUILDING BLOCKS
# ============================================================

class ConceptProjector(nn.Module):
    """
    Projects a modality's concept probability vector into a sequence of
    d_model-dimensional tokens — one token per concept.

    Input:  (B, N_concepts)          float in [0, 1]
    Output: (B, N_concepts, d_model)

    Note: when a sample is missing this modality entirely, its concept_probs
    will already be zeroed by encode_concepts() before reaching here, so the
    projected tokens will also be near-zero. The fusion layer then gates
    these tokens out with the modality mask anyway, providing double safety.
    """
    def __init__(self, num_concepts: int, d_model: int, modality_id: int, num_modalities: int = 3):
        super().__init__()
        self.value_proj   = nn.Linear(1, d_model)
        self.concept_pos  = nn.Embedding(num_concepts, d_model)
        self.modality_emb = nn.Embedding(num_modalities, d_model)
        self.modality_id  = modality_id
        self.norm         = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N    = x.shape
        val_emb = self.value_proj(x.unsqueeze(-1))                           # (B, N, d_model)
        pos_idx = torch.arange(N, device=x.device)
        pos_emb = self.concept_pos(pos_idx).unsqueeze(0).expand(B, -1, -1)  # (B, N, d_model)
        mod_idx = torch.tensor([self.modality_id], device=x.device)
        mod_emb = self.modality_emb(mod_idx).unsqueeze(0).expand(B, N, -1)  # (B, N, d_model)
        return self.norm(val_emb + pos_emb + mod_emb)                        # (B, N, d_model)


class SNN(nn.Module):
    """
    Self-Normalising Network layer — HEALNet's stabilisation block.

    Uses SELU activations with LeCun normal init + AlphaDropout.
    SELU preserves zero mean and unit variance between layers without
    explicit LayerNorm, provided the weights are LeCun-initialised.

    Applied as a residual:  S ← S + SNN(S)

    Input / Output: (B, n_latent, d_model)
    """
    def __init__(self, d_model: int, expansion: int = 2, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.SELU(),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ModalityCrossAttention(nn.Module):
    """
    Per-modality cross-attention: shared latent S queries modality tokens X_m.

    Core HEALNet equation:
        Q_m = S  @ W_m^Q    (shared latent asks questions)
        K_m = X_m @ W_m^K   (modality concepts hold keys)
        V_m = X_m @ W_m^V   (modality concepts hold values)
        out = softmax(Q_m K_m^T / sqrt(d_k)) @ V_m

    W_m^{Q,K,V} are MODALITY-SPECIFIC — separate weights for each modality.
    This is fundamentally different from a standard transformer where all
    tokens share the same Q/K/V projections.

    Input:
        S   : (B, n_latent,   d_model)
        X_m : (B, N_concepts, d_model)
    Output:
        attended    : (B, n_latent, d_model)
        attn_weights: (B, n_latent, N_concepts)  heads-averaged, for explainability
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.d_model   = d_model
        self.scale     = math.sqrt(self.d_k)

        self.W_q      = nn.Linear(d_model, d_model, bias=False)
        self.W_k      = nn.Linear(d_model, d_model, bias=False)
        self.W_v      = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, S: torch.Tensor, X_m: torch.Tensor):
        B, Ls, D = S.shape
        _, Lx, _ = X_m.shape

        Q = self.W_q(S)    # (B, Ls, D)
        K = self.W_k(X_m)  # (B, Lx, D)
        V = self.W_v(X_m)  # (B, Lx, D)

        def split_heads(t, seq_len):
            return t.view(B, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        Q = split_heads(Q, Ls)   # (B, H, Ls, d_k)
        K = split_heads(K, Lx)   # (B, H, Lx, d_k)
        V = split_heads(V, Lx)   # (B, H, Lx, d_k)

        scores       = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, Ls, Lx)
        attn         = F.softmax(scores, dim=-1)                           # (B, H, Ls, Lx)
        attn_weights = attn.mean(dim=1)                                    # (B, Ls, Lx)

        out = torch.matmul(attn, V)                            # (B, H, Ls, d_k)
        out = out.transpose(1, 2).contiguous().view(B, Ls, D)  # (B, Ls, D)
        out = self.out_proj(out)

        return out, attn_weights


class HEALNetFusionLayer(nn.Module):
    """
    One complete HEALNet fusion layer with per-sample missing-modality masking.

    For each active modality m:
        1. Run ModalityCrossAttention(S, X_m) → attended_m (B, n_latent, d_model)
                                                 attn_w_m   (B, n_latent, N_m)
        2. Apply modality presence mask per sample:
               mask_m shape: (B,) → reshaped to (B, 1, 1) for broadcasting
               attended_m *= mask_m   — zero out rows for absent samples

    Then fuse into S:
        3. Stack attended outputs: (n_mods, B, n_latent, d_model)
        4. Sum over modalities and divide by count of PRESENT modalities
           per sample (not total modalities) → safe masked mean
        5. S ← S + fused
        6. S ← SNN(S)

    The masked mean in step 4 is the critical fix: a sample missing modality m
    contributes 0 from that modality to the update of its own S row, and the
    denominator correctly reflects how many modalities it actually has.

    Input:
        S              : (B, n_latent, d_model)
        modality_tokens: dict[mod → (B, N_m, d_model)]
        mod_masks      : dict[mod → (B,) float32  0=absent 1=present]
    Output:
        S_updated  : (B, n_latent, d_model)
        layer_attn : dict[mod → (B, n_latent, N_m)]  — raw attention for explainability
    """
    def __init__(self, active_modalities: list, d_model: int, num_heads: int, snn_dropout: float = 0.05):
        super().__init__()
        self.active_modalities = active_modalities
        self.cross_attns = nn.ModuleDict({
            mod: ModalityCrossAttention(d_model, num_heads)
            for mod in active_modalities
        })
        self.snn = SNN(d_model, expansion=2, dropout=snn_dropout)

    def forward(self, S: torch.Tensor, modality_tokens: dict, mod_masks: dict):
        B = S.shape[0]
        attended_stack = []
        mask_stack     = []
        layer_attn     = {}

        for mod in self.active_modalities:
            X_m    = modality_tokens[mod]   # (B, N_m, d_model)
            mask_m = mod_masks[mod]         # (B,) float 0/1

            attended_m, attn_w = self.cross_attns[mod](S, X_m)
            # attended_m : (B, n_latent, d_model)
            # attn_w     : (B, n_latent, N_m)

            # Gate attended output: zero out absent samples for this modality.
            # mask_m (B,) → (B, 1, 1) broadcasts across n_latent and d_model.
            gate       = mask_m.view(B, 1, 1)
            attended_m = attended_m * gate   # absent samples → zero vector
            attn_w     = attn_w * gate       # gate explainability signal too

            attended_stack.append(attended_m)
            mask_stack.append(mask_m)
            layer_attn[mod] = attn_w

        # Sum attended outputs across modalities (absent ones contribute zero)
        attended_tensor = torch.stack(attended_stack, dim=0)  # (n_mods, B, n_latent, d_model)
        fused_sum       = attended_tensor.sum(dim=0)          # (B, n_latent, d_model)

        # Count present modalities per sample: (B,) → (B, 1, 1)
        # Clamp to 1 to avoid divide-by-zero for any fully-missing edge case.
        n_present = torch.stack(mask_stack, dim=0).sum(dim=0)  # (B,)
        n_present = n_present.clamp(min=1.0).view(B, 1, 1)

        # Masked mean: each sample is divided by its OWN present modality count.
        fused = fused_sum / n_present   # (B, n_latent, d_model)

        S = S + fused    # residual update of shared latent
        S = self.snn(S)  # SNN stabilisation

        return S, layer_attn


# ============================================================
# TASK HEAD
# ============================================================

def _build_task_head(d_model: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(32, 1)
    )


# ============================================================
# MAIN MODEL
# ============================================================

class HEALNetCBM(nn.Module):
    """
    True HEALNet-CBM Fusion Model with per-sample missing modality handling.

    CBM Stage  : frozen encoders → sigmoid concept probs (interpretable bottleneck)
    HEALNet Stage: iterative shared-latent cross-attention fusion with masking

    Missing modality contract:
        - cxr_mod_mask / ecg_mod_mask / ehr_mod_mask are (B,) float tensors
          where 1.0 = modality present for this sample, 0.0 = absent.
        - Absent samples have their encoder output zeroed before projection.
        - Fusion layer gates each modality's attended output per sample.
        - The shared latent S for a sample is only updated by its present modalities.
        - Explainability methods accept the same masks and exclude absent samples.

    Parameters:
        active_modalities : list of modality names active for this experiment
        d_model           : shared latent dimension (default 128)
        num_heads         : attention heads (default 4)
        num_fusion_layers : depth d — how many times S is updated (default 3)
        n_latent          : number of shared latent query slots (default 8)
    """

    def __init__(
        self,
        active_modalities: list,
        d_model:           int = 128,
        num_heads:         int = 4,
        num_fusion_layers: int = 3,
        n_latent:          int = 8,
    ):
        super().__init__()
        self.active_modalities = active_modalities
        self.d_model           = d_model
        self.num_fusion_layers = num_fusion_layers
        self.n_latent          = n_latent

        modality_id_map = {"cxr": 0, "ecg": 1, "ehr": 2}

        # ------------------------------------------------------------------
        # 1. Frozen Concept Encoders
        # ------------------------------------------------------------------
        self.encoders   = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        for mod in active_modalities:
            if mod == "cxr":
                wrapper = UniModalCXR()
            elif mod == "ecg":
                wrapper = UniModalECG()
            else:
                wrapper = UniModalEHR()

            encoder = wrapper.encoder
            for p in encoder.parameters():
                p.requires_grad = False

            self.encoders[mod]   = encoder
            self.projectors[mod] = ConceptProjector(
                num_concepts   = CONCEPT_COUNTS[mod],
                d_model        = d_model,
                modality_id    = modality_id_map[mod],
                num_modalities = 3,
            )

        # ------------------------------------------------------------------
        # 2. Shared Latent S — learned, broadcast to batch at runtime
        # ------------------------------------------------------------------
        self.shared_latent = nn.Parameter(torch.randn(1, n_latent, d_model) * 0.02)

        # ------------------------------------------------------------------
        # 3. HEALNet Fusion Layers
        # ------------------------------------------------------------------
        self.fusion_layers = nn.ModuleList([
            HEALNetFusionLayer(
                active_modalities = active_modalities,
                d_model           = d_model,
                num_heads         = num_heads,
                snn_dropout       = 0.05,
            )
            for _ in range(num_fusion_layers)
        ])

        # ------------------------------------------------------------------
        # 4. Task Heads
        # ------------------------------------------------------------------
        self.mort_head = _build_task_head(d_model)
        self.ahf_head  = _build_task_head(d_model)

    # ------------------------------------------------------------------
    # ENCODE — Step 1: frozen encoders + missing-modality zeroing
    # ------------------------------------------------------------------

    def encode_concepts(self, image, waveform, ehr_features, mod_masks: dict) -> dict:
        """
        Runs frozen encoders → sigmoid concept probabilities.
        Zeroes out concept probs for samples where that modality is absent.

        Args:
            image, waveform, ehr_features : raw inputs (B, ...)
            mod_masks : dict[mod → (B,) float32]  1=present 0=absent

        Returns:
            concept_probs : dict[mod → (B, N_concepts)]
                            absent samples have all-zero concept vectors.

        Separated from fuse() to enable Explainability Layer 4
        (counterfactual interventions via direct concept_probs manipulation).
        """
        concept_probs = {}
        with torch.no_grad():
            if "cxr" in self.active_modalities:
                probs = torch.sigmoid(self.encoders["cxr"](image))        # (B, 15)
                probs = probs * mod_masks["cxr"].unsqueeze(1)              # zero absent
                concept_probs["cxr"] = probs

            if "ecg" in self.active_modalities:
                probs = torch.sigmoid(self.encoders["ecg"](waveform))     # (B, 14)
                probs = probs * mod_masks["ecg"].unsqueeze(1)
                concept_probs["ecg"] = probs

            if "ehr" in self.active_modalities:
                probs = torch.sigmoid(self.encoders["ehr"](ehr_features)) # (B, 13)
                probs = probs * mod_masks["ehr"].unsqueeze(1)
                concept_probs["ehr"] = probs

        return concept_probs

    # ------------------------------------------------------------------
    # FUSE — Steps 2–4: HEALNet fusion with missing-modality masking
    # ------------------------------------------------------------------

    def fuse(self, concept_probs: dict, mod_masks: dict,
             return_attention: bool = False) -> dict:
        """
        Runs the HEALNet fusion from concept probability dicts.

        Args:
            concept_probs    : dict[mod → (B, N_m)]
            mod_masks        : dict[mod → (B,) float32]  1=present 0=absent
                               Used by fusion layers to gate per-sample contributions.
                               For Layer-4 counterfactuals: pass the ORIGINAL masks
                               (you're only changing concept values, not presence).
            return_attention : if True, include all_layer_attn in output

        Returns dict with:
            mortality_logits : (B,)
            ahf_logits       : (B,)
            concept_probs    : passed through (for logging / Layer 4)
            all_layer_attn   : list[dict[mod → (B, n_latent, N_m)]]
                               only present if return_attention=True
        """
        B = next(iter(concept_probs.values())).shape[0]

        # Project concept probs → modality token sequences X_m
        modality_tokens = {
            mod: self.projectors[mod](concept_probs[mod])   # (B, N_m, d_model)
            for mod in self.active_modalities
        }

        # Initialise shared latent: broadcast learned S to batch
        # .clone() so in-place SNN ops don't reach back into the Parameter
        S = self.shared_latent.expand(B, -1, -1).clone()   # (B, n_latent, d_model)

        # Iterative fusion: d HEALNet layers, each updating S
        all_layer_attn = []
        for layer in self.fusion_layers:
            S, layer_attn = layer(S, modality_tokens, mod_masks)
            all_layer_attn.append(layer_attn)
            # modality_tokens (X_m) stay fixed across layers —
            # only S is updated, which matches the HEALNet design.

        # Mean-pool final shared latent → task predictions
        pooled = S.mean(dim=1)   # (B, d_model)

        outputs = {
            "mortality_logits": self.mort_head(pooled).squeeze(-1),
            "ahf_logits":       self.ahf_head(pooled).squeeze(-1),
            "concept_probs":    concept_probs,
        }

        if return_attention:
            outputs["all_layer_attn"] = all_layer_attn

        return outputs

    # ------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------

    def forward(self, image, waveform, ehr_features,
                cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask,
                return_attention: bool = False) -> dict:
        """
        Full forward pass.

        cxr_mask / ecg_mask / ehr_mask : (B,) float  1=present 0=absent
            Per-sample modality presence flags — now actually used.
        ehr_feat_mask : within-EHR feature mask, handled internally by
            the EHR encoder; not used for modality-level gating here.
        """
        mod_masks = {
            mod: mask.float()
            for mod, mask in zip(
                self.active_modalities,
                [cxr_mask, ecg_mask, ehr_mask][:len(self.active_modalities)]
            )
        }
        # Build mask dict keyed by actual active modality names
        mod_masks = {}
        all_mask_vals = {"cxr": cxr_mask, "ecg": ecg_mask, "ehr": ehr_mask}
        for mod in self.active_modalities:
            mod_masks[mod] = all_mask_vals[mod].float()

        concept_probs = self.encode_concepts(
            image, waveform, ehr_features, mod_masks
        )
        return self.fuse(concept_probs, mod_masks, return_attention=return_attention)

    # ------------------------------------------------------------------
    # EXPLAINABILITY LAYER 1: Modality Contribution
    # ------------------------------------------------------------------

    def get_modality_attention(self, all_layer_attn: list, mod_masks: dict) -> dict:
        """
        Explainability Layer 1: How much did each modality contribute?

        Reports mean attention S → X_m, averaged only over samples where
        that modality was present (mask=1). Absent samples would have zero
        attention (gated out in fusion) and must not pollute the mean.

        Args:
            all_layer_attn : list[dict[mod → (B, n_latent, N_m)]]
            mod_masks      : dict[mod → (B,) float]

        Returns:
            dict[mod → float]
        """
        mod_scores = {}
        for mod in self.active_modalities:
            mask      = mod_masks[mod]
            n_present = mask.sum().item()

            if n_present == 0:
                mod_scores[mod] = 0.0
                continue

            total = 0.0
            for layer_attn in all_layer_attn:
                attn_w = layer_attn[mod]              # (B, n_latent, N_m)
                # Select only rows where modality was present, then mean
                total += attn_w[mask.bool()].mean().item()

            mod_scores[mod] = total / len(all_layer_attn)

        return mod_scores

    # ------------------------------------------------------------------
    # EXPLAINABILITY LAYER 2: Concept Contribution
    # ------------------------------------------------------------------

    def get_concept_attention(self, all_layer_attn: list, mod_masks: dict) -> dict:
        """
        Explainability Layer 2: Which individual concepts were most attended?

        Returns mean attention per concept token, averaged over present
        samples only and over all fusion layers.

        Args:
            all_layer_attn : list[dict[mod → (B, n_latent, N_m)]]
            mod_masks      : dict[mod → (B,) float]

        Returns:
            dict[mod → np.ndarray shape (N_concepts,)]
        """
        concept_attn = {}
        for mod in self.active_modalities:
            mask       = mod_masks[mod]
            n_present  = mask.sum().item()
            n_concepts = CONCEPT_COUNTS[mod]

            if n_present == 0:
                concept_attn[mod] = np.zeros(n_concepts)
                continue

            accum = np.zeros(n_concepts, dtype=np.float64)
            for layer_attn in all_layer_attn:
                attn_w = layer_attn[mod]                          # (B, n_latent, N_m)
                # Select present samples → average over batch and latent dims
                per_concept = attn_w[mask.bool()].mean(dim=(0, 1)).cpu().numpy()
                accum += per_concept

            concept_attn[mod] = accum / len(all_layer_attn)

        return concept_attn

    # ------------------------------------------------------------------
    # EXPLAINABILITY LAYER 3: Cross-Modal Attention
    # ------------------------------------------------------------------

    def get_cross_modal_attention(self, all_layer_attn: list, mod_masks: dict,
                                  mod_a: str, mod_b: str) -> np.ndarray:
        """
        Explainability Layer 3: Co-attention between mod_a and mod_b concepts.

        Computes the outer product of per-concept attention vectors for mod_a
        and mod_b, yielding a (N_a, N_b) co-attention matrix. Only samples
        where BOTH modalities are present are included.

        Args:
            all_layer_attn : list[dict[mod → (B, n_latent, N_m)]]
            mod_masks      : dict[mod → (B,) float]
            mod_a, mod_b   : modality names to compare

        Returns:
            np.ndarray of shape (N_concepts_a, N_concepts_b)
        """
        both_present = (mod_masks[mod_a].bool() & mod_masks[mod_b].bool())
        n_both = both_present.sum().item()

        if n_both == 0:
            return np.zeros((CONCEPT_COUNTS[mod_a], CONCEPT_COUNTS[mod_b]))

        cross_accum = None
        for layer_attn in all_layer_attn:
            attn_a = layer_attn[mod_a][both_present]   # (n_both, n_latent, N_a)
            attn_b = layer_attn[mod_b][both_present]   # (n_both, n_latent, N_b)

            vec_a = attn_a.mean(dim=(0, 1)).cpu().numpy()   # (N_a,)
            vec_b = attn_b.mean(dim=(0, 1)).cpu().numpy()   # (N_b,)

            cross       = np.outer(vec_a, vec_b)
            cross_accum = cross if cross_accum is None else cross_accum + cross

        return cross_accum / len(all_layer_attn)

    # ------------------------------------------------------------------
    # EXPLAINABILITY LAYER 4: Counterfactuals (usage pattern)
    # ------------------------------------------------------------------
    # Layer 4 is a USAGE PATTERN at inference time, not a model method.
    #
    # Example — "what if this patient's CXR showed no pleural effusion?":
    #
    #   mod_masks = {"cxr": cxr_mask, "ecg": ecg_mask, "ehr": ehr_mask}
    #   concept_probs = model.encode_concepts(img, wav, ehr, mod_masks)
    #
    #   modified = {mod: p.clone() for mod, p in concept_probs.items()}
    #   modified["cxr"][:, 3] = 0.0   # zero concept 3 = pleural effusion
    #
    #   out_orig = model.fuse(concept_probs, mod_masks)
    #   out_intv = model.fuse(modified,      mod_masks)  # same masks!
    #
    #   delta = out_intv["mortality_logits"] - out_orig["mortality_logits"]
    #   # delta[i] = causal effect of removing that concept for patient i
    #
    # Pass the ORIGINAL mod_masks to fuse() — you are only modifying concept
    # values, not whether a modality was present for the patient.


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(model, loader, optimizer, scaler, device,
                bce_mort, bce_ahf, logger, epoch):
    model.train()
    total_loss = 0.0

    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        image         = batch['image'].to(device)
        waveform      = batch['waveform'].to(device)
        ehr_features  = batch['ehr_features'].to(device)
        cxr_mask      = batch['cxr_mod_mask'].to(device)
        ecg_mask      = batch['ecg_mod_mask'].to(device)
        ehr_mask      = batch['ehr_mod_mask'].to(device)
        ehr_feat_mask = batch['ehr_mask'].to(device)

        y_mort = batch['target_mortality'].float().to(device)
        y_ahf  = batch['target_ahf'].float().to(device)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(
                image, waveform, ehr_features,
                cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask
            )
            loss = (bce_mort(outputs['mortality_logits'], y_mort) +
                    bce_ahf(outputs['ahf_logits'],        y_ahf)) / 2.0

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / max(len(loader), 1)


# ============================================================
# VALIDATION
# ============================================================

def evaluate(model, loader, device, bce_mort, bce_ahf):
    model.eval()

    mort_probs, mort_targets = [], []
    ahf_probs,  ahf_targets  = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            image         = batch['image'].to(device)
            waveform      = batch['waveform'].to(device)
            ehr_features  = batch['ehr_features'].to(device)
            cxr_mask      = batch['cxr_mod_mask'].to(device)
            ecg_mask      = batch['ecg_mod_mask'].to(device)
            ehr_mask      = batch['ehr_mod_mask'].to(device)
            ehr_feat_mask = batch['ehr_mask'].to(device)

            y_mort = batch['target_mortality'].float().to(device)
            y_ahf  = batch['target_ahf'].float().to(device)

            outputs = model(
                image, waveform, ehr_features,
                cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask
            )

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
        "val_loss":   avg_loss,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args   = get_args()
    device = torch.device('cuda')

    if args.experiment not in EXPERIMENT_CONFIG:
        raise ValueError(
            f"Valid experiments: {list(EXPERIMENT_CONFIG.keys())}. Got: {args.experiment}"
        )

    active_modalities = EXPERIMENT_CONFIG[args.experiment]["modalities"]

    logger = setup_logger(args.save_dir, args.exp_name)
    logger.info(f"Experiment       : {args.experiment}")
    logger.info(f"Active modalities: {active_modalities}")
    logger.info(f"Arguments        : {vars(args)}")
    train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"
    # --- Data ---
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=args.batch_size,
        dropout_config=None,
    )
    logger.info("Dataloaders ready.")

    # --- Model ---
    model = HEALNetCBM(
        active_modalities = active_modalities,
        d_model           = 128,
        num_heads         = 4,
        num_fusion_layers = 3,
        n_latent          = 8,
    ).to(device)

    # --- Load Stage 1 encoder weights ---
    # --- Parameter audit ---
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params    = total_params - trainable_params
    logger.info(
        f"Parameters — Total: {total_params:,} | "
        f"Trainable: {trainable_params:,} | Frozen (encoders): {frozen_params:,}"
    )

    # --- Loss ---
    bce_mort = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([14.44], device=device))
    bce_ahf  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.89],  device=device))

    # --- Optimizer ---
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = args.lr_finetune,
        weight_decay = args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        patience=args.patience_scheduler, factor=0.5,
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
            bce_mort, bce_ahf, logger, epoch,
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
        logger.info(f"  Combined AUROC : {metrics['combined']:.4f}")

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
    logger.info(f"Weights saved to : {save_path}")


if __name__ == "__main__":
    main()