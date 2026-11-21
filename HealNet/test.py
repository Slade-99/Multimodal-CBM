import os
import math
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from scipy import stats

from cbm_model import (
    UniModalCXR,
    UniModalECG,
    UniModalEHR
)

from dataloader import get_dataloaders

from config_2 import (
    CXR_CONCEPTS,
    ECG_CONCEPTS,
    EHR_CONCEPTS
)


# ============================================================
# CONSTANTS
# ============================================================

ALL_CONCEPTS = (
    CXR_CONCEPTS +
    ECG_CONCEPTS +
    EHR_CONCEPTS
)

EXPERIMENT_CONFIG = {

    "cxr_ecg": {
        "modalities": ["cxr", "ecg"]
    },

    "cxr_ehr": {
        "modalities": ["cxr", "ehr"]
    },

    "ecg_ehr": {
        "modalities": ["ecg", "ehr"]
    },

    "trimodal": {
        "modalities": ["cxr", "ecg", "ehr"]
    }
}

CONCEPT_COUNTS = {
    "cxr": 15,
    "ecg": 14,
    "ehr": 13
}

N_BOOTSTRAP = 1000
CI_LEVEL     = 95
RANDOM_SEED  = 42


# ============================================================
# LOGGER
# ============================================================

def setup_logger(save_dir, exp_name):

    os.makedirs(
        os.path.join(save_dir, exp_name),
        exist_ok=True
    )

    log_file = os.path.join(
        save_dir,
        exp_name,
        "test_results.txt"
    )

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

    def __init__(
        self,
        num_concepts,
        d_model,
        modality_id,
        num_modalities=3
    ):
        super().__init__()

        self.value_proj = nn.Linear(
            1,
            d_model
        )

        self.concept_pos = nn.Embedding(
            num_concepts,
            d_model
        )

        self.modality_emb = nn.Embedding(
            num_modalities,
            d_model
        )

        self.modality_id = modality_id

        self.norm = nn.LayerNorm(
            d_model
        )

    def forward(self, x):

        B, N = x.shape

        val_emb = self.value_proj(
            x.unsqueeze(-1)
        )

        pos_idx = torch.arange(
            N,
            device=x.device
        )

        pos_emb = self.concept_pos(
            pos_idx
        ).unsqueeze(0).expand(B, -1, -1)

        mod_idx = torch.tensor(
            [self.modality_id],
            device=x.device
        )

        mod_emb = self.modality_emb(
            mod_idx
        ).unsqueeze(0).expand(B, N, -1)

        return self.norm(
            val_emb +
            pos_emb +
            mod_emb
        )


class SNN(nn.Module):

    def __init__(
        self,
        d_model,
        expansion=2,
        dropout=0.05
    ):
        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                d_model,
                d_model * expansion
            ),

            nn.SELU(),

            nn.AlphaDropout(dropout),

            nn.Linear(
                d_model * expansion,
                d_model
            ),

            nn.SELU()
        )

    def forward(self, x):
        return x + self.net(x)


class ModalityCrossAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = math.sqrt(self.d_k)

        self.W_q = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_k = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.W_v = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

        self.out_proj = nn.Linear(
            d_model,
            d_model,
            bias=False
        )

    def forward(self, S, X_m):

        B, Ls, D = S.shape
        _, Lx, _ = X_m.shape

        Q = self.W_q(S)
        K = self.W_k(X_m)
        V = self.W_v(X_m)

        def split_heads(t, seq_len):

            return t.view(
                B,
                seq_len,
                self.num_heads,
                self.d_k
            ).transpose(1, 2)

        Q = split_heads(Q, Ls)
        K = split_heads(K, Lx)
        V = split_heads(V, Lx)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / self.scale

        attn = F.softmax(
            scores,
            dim=-1
        )

        out = torch.matmul(attn, V)

        out = out.transpose(
            1,
            2
        ).contiguous().view(B, Ls, D)

        return self.out_proj(out)


class HEALNetFusionLayer(nn.Module):

    def __init__(
        self,
        active_modalities,
        d_model,
        num_heads
    ):
        super().__init__()

        self.active_modalities = active_modalities

        self.cross_attns = nn.ModuleDict({

            mod: ModalityCrossAttention(
                d_model,
                num_heads
            )

            for mod in active_modalities
        })

        self.snn = SNN(d_model)

    def forward(
        self,
        S,
        modality_tokens,
        mod_masks
    ):

        B = S.shape[0]

        attended_stack = []
        mask_stack = []

        for mod in self.active_modalities:

            X_m = modality_tokens[mod]

            mask_m = mod_masks[mod]

            attended_m = self.cross_attns[mod](
                S,
                X_m
            )

            attended_m = attended_m * \
                mask_m.view(B, 1, 1)

            attended_stack.append(
                attended_m
            )

            mask_stack.append(mask_m)

        attended_tensor = torch.stack(
            attended_stack,
            dim=0
        )

        fused_sum = attended_tensor.sum(
            dim=0
        )

        n_present = torch.stack(
            mask_stack,
            dim=0
        ).sum(dim=0)

        n_present = n_present.clamp(
            min=1.0
        ).view(B, 1, 1)

        fused = fused_sum / n_present

        S = S + fused

        S = self.snn(S)

        return S


# ============================================================
# TASK HEAD
# ============================================================

def build_task_head(d_model):

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
# MODEL
# ============================================================

class HEALNetCBM(nn.Module):

    def __init__(
        self,
        active_modalities,
        d_model=128,
        num_heads=4,
        num_fusion_layers=3,
        n_latent=8
    ):
        super().__init__()

        self.active_modalities = active_modalities

        modality_id_map = {
            "cxr": 0,
            "ecg": 1,
            "ehr": 2
        }

        self.encoders = nn.ModuleDict()
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

            self.encoders[mod] = encoder

            self.projectors[mod] = ConceptProjector(
                num_concepts=CONCEPT_COUNTS[mod],
                d_model=d_model,
                modality_id=modality_id_map[mod],
                num_modalities=3
            )

        self.shared_latent = nn.Parameter(
            torch.randn(
                1,
                n_latent,
                d_model
            ) * 0.02
        )

        self.fusion_layers = nn.ModuleList([

            HEALNetFusionLayer(
                active_modalities,
                d_model,
                num_heads
            )

            for _ in range(num_fusion_layers)
        ])

        self.mort_head = build_task_head(
            d_model
        )

        self.ahf_head = build_task_head(
            d_model
        )

    def encode_concepts(
        self,
        image,
        waveform,
        ehr_features,
        mod_masks
    ):

        concept_probs = {}

        with torch.no_grad():

            if "cxr" in self.active_modalities:

                probs = torch.sigmoid(
                    self.encoders["cxr"](image)
                )

                probs = probs * \
                    mod_masks["cxr"].unsqueeze(1)

                concept_probs["cxr"] = probs

            if "ecg" in self.active_modalities:

                probs = torch.sigmoid(
                    self.encoders["ecg"](waveform)
                )

                probs = probs * \
                    mod_masks["ecg"].unsqueeze(1)

                concept_probs["ecg"] = probs

            if "ehr" in self.active_modalities:

                probs = torch.sigmoid(
                    self.encoders["ehr"](ehr_features)
                )

                probs = probs * \
                    mod_masks["ehr"].unsqueeze(1)

                concept_probs["ehr"] = probs

        return concept_probs

    def fuse(
        self,
        concept_probs,
        mod_masks
    ):

        B = next(
            iter(concept_probs.values())
        ).shape[0]

        modality_tokens = {

            mod: self.projectors[mod](
                concept_probs[mod]
            )

            for mod in self.active_modalities
        }

        S = self.shared_latent.expand(
            B,
            -1,
            -1
        ).clone()

        for layer in self.fusion_layers:

            S = layer(
                S,
                modality_tokens,
                mod_masks
            )

        pooled = S.mean(dim=1)

        return {

            "mortality_logits":
                self.mort_head(
                    pooled
                ).squeeze(-1),

            "ahf_logits":
                self.ahf_head(
                    pooled
                ).squeeze(-1)
        }

    def forward(
        self,
        image,
        waveform,
        ehr_features,
        cxr_mask,
        ecg_mask,
        ehr_mask,
        ehr_feat_mask
    ):

        mod_masks = {}

        all_masks = {

            "cxr": cxr_mask.float(),
            "ecg": ecg_mask.float(),
            "ehr": ehr_mask.float()
        }

        for mod in self.active_modalities:
            mod_masks[mod] = all_masks[mod]

        concept_probs = self.encode_concepts(
            image,
            waveform,
            ehr_features,
            mod_masks
        )

        return self.fuse(
            concept_probs,
            mod_masks
        )


# ============================================================
# THRESHOLD SELECTION
# Finds the optimal decision threshold on the VALIDATION set
# using the F1-maximisation strategy, which is robust to class
# imbalance and avoids the default 0.5 bias.
# ============================================================

def find_optimal_threshold(y_true, y_probs, strategy="f1"):
    """
    strategy options:
        'f1'      – maximise F1 score (default; good for imbalance)
        'youden'  – maximise Youden's J = sensitivity + specificity - 1
        'min_dist'– minimise distance to (0,1) on ROC curve
    Returns the optimal threshold as a float.
    """

    if strategy == "f1":
        precisions, recalls, thresholds = \
            precision_recall_curve(y_true, y_probs)

        f1_scores = np.where(
            (precisions + recalls) == 0,
            0.0,
            2 * precisions * recalls / (precisions + recalls)
        )

        # precision_recall_curve returns one fewer threshold
        best_idx = np.argmax(f1_scores[:-1])
        return float(thresholds[best_idx])

    elif strategy in ("youden", "min_dist"):
        fprs, tprs, thresholds = roc_curve(y_true, y_probs)

        if strategy == "youden":
            j_scores = tprs - fprs
            best_idx  = np.argmax(j_scores)
        else:
            dists     = np.sqrt(fprs ** 2 + (1 - tprs) ** 2)
            best_idx  = np.argmin(dists)

        return float(thresholds[best_idx])

    else:
        raise ValueError(
            f"Unknown threshold strategy: {strategy}"
        )


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# All intervals are BCa (bias-corrected and accelerated) when
# possible; falls back to percentile if acceleration fails.
# ============================================================

def _jackknife_accel(y_true, y_probs, metric_fn):
    """Compute the acceleration constant for BCa intervals."""
    n      = len(y_true)
    jk     = np.array([
        metric_fn(
            np.delete(y_true, i),
            np.delete(y_probs, i)
        )
        for i in range(n)
    ])
    jk_mean = jk.mean()
    num     = np.sum((jk_mean - jk) ** 3)
    denom   = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
    return num / denom if denom != 0 else 0.0


def bootstrap_ci(
    y_true,
    y_probs,
    metric_fn,
    n_boot=N_BOOTSTRAP,
    ci=CI_LEVEL,
    seed=RANDOM_SEED,
    use_bca=True,
    threshold=None
):
    """
    Returns (point_estimate, lower_ci, upper_ci).

    metric_fn signature:
        fn(y_true, y_probs)            – for threshold-free metrics
        fn(y_true, y_probs, threshold) – pass threshold kwarg

    When threshold is not None the bootstrap re-uses the FIXED
    threshold (found on the val set) so we don't leak test labels.
    """

    rng   = np.random.RandomState(seed)
    n     = len(y_true)
    alpha = (100 - ci) / 2.0

    if threshold is not None:
        def _metric(yt, yp):
            return metric_fn(yt, yp, threshold)
    else:
        _metric = metric_fn

    point = _metric(y_true, y_probs)

    boot_scores = []
    for _ in range(n_boot):
        idx   = rng.choice(n, n, replace=True)
        yt_b  = y_true[idx]
        yp_b  = y_probs[idx]

        # Skip degenerate bootstrap samples
        if len(np.unique(yt_b)) < 2:
            continue

        try:
            boot_scores.append(_metric(yt_b, yp_b))
        except Exception:
            continue

    boot_scores = np.array(boot_scores)

    if use_bca and len(boot_scores) >= 200:
        try:
            # Bias-correction constant z0
            prop_less = np.mean(boot_scores < point)
            prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
            z0        = stats.norm.ppf(prop_less)

            # Acceleration constant via jackknife
            accel = _jackknife_accel(y_true, y_probs, _metric)

            z_alpha_lo = stats.norm.ppf(alpha / 100.0)
            z_alpha_hi = stats.norm.ppf(1 - alpha / 100.0)

            def _adj_pct(z_alpha):
                num   = z0 + z_alpha
                denom = 1 - accel * (z0 + z_alpha)
                return stats.norm.cdf(z0 + num / denom) * 100

            pct_lo = _adj_pct(z_alpha_lo)
            pct_hi = _adj_pct(z_alpha_hi)

            lower  = np.percentile(boot_scores, pct_lo)
            upper  = np.percentile(boot_scores, pct_hi)

        except Exception:
            # Fallback to percentile
            lower = np.percentile(boot_scores, alpha)
            upper = np.percentile(boot_scores, 100 - alpha)
    else:
        lower = np.percentile(boot_scores, alpha)
        upper = np.percentile(boot_scores, 100 - alpha)

    return float(point), float(lower), float(upper)


# ============================================================
# DELONG TEST FOR AUC COMPARISON
# Implements the DeLong et al. (1988) method for comparing two
# correlated ROC curves on the same test set.
# ============================================================

def _compute_midrank(x):
    """Compute midranks of x."""
    J    = np.argsort(x)
    Z    = x[J]
    N    = len(x)
    T    = np.zeros(N, dtype=float)
    i    = 0
    while i < N:
        j = i
        while j < N - 1 and Z[j] == Z[j + 1]:
            j += 1
        T[i:j + 1] = (i + j + 1) / 2.0 + 1
        i = j + 1
    T2       = np.empty(N, dtype=float)
    T2[J]    = T
    return T2


def delong_auc_variance(y_true, y_probs):
    """
    Compute AUC and its variance using the DeLong method.
    Returns (auc, auc_variance).
    """
    pos_idx  = np.where(y_true == 1)[0]
    neg_idx  = np.where(y_true == 0)[0]
    n_pos    = len(pos_idx)
    n_neg    = len(neg_idx)

    pos_prob = y_probs[pos_idx]
    neg_prob = y_probs[neg_idx]

    # Placement values
    V_pos = np.array([
        np.mean((neg_prob < p) + 0.5 * (neg_prob == p))
        for p in pos_prob
    ])
    V_neg = np.array([
        np.mean((pos_prob > n) + 0.5 * (pos_prob == n))
        for n in neg_prob
    ])

    auc   = V_pos.mean()
    s_pos = np.var(V_pos, ddof=1) / n_pos
    s_neg = np.var(V_neg, ddof=1) / n_neg

    return auc, s_pos + s_neg


def delong_test(
    y_true,
    y_probs_a,
    y_probs_b
):
    """
    Two-sided DeLong test comparing AUC(model_a) vs AUC(model_b)
    on the same test set.

    Returns:
        auc_a  : float
        auc_b  : float
        z      : float  – z-statistic
        p_value: float  – two-sided p-value
    """

    auc_a, var_a = delong_auc_variance(y_true, y_probs_a)
    auc_b, var_b = delong_auc_variance(y_true, y_probs_b)

    # Covariance term (DeLong eq. 6)
    pos_idx  = np.where(y_true == 1)[0]
    neg_idx  = np.where(y_true == 0)[0]
    n_pos    = len(pos_idx)
    n_neg    = len(neg_idx)

    Va_pos = np.array([
        np.mean(
            (y_probs_a[neg_idx] < p) +
            0.5 * (y_probs_a[neg_idx] == p)
        )
        for p in y_probs_a[pos_idx]
    ])
    Vb_pos = np.array([
        np.mean(
            (y_probs_b[neg_idx] < p) +
            0.5 * (y_probs_b[neg_idx] == p)
        )
        for p in y_probs_b[pos_idx]
    ])
    Va_neg = np.array([
        np.mean(
            (y_probs_a[pos_idx] > n) +
            0.5 * (y_probs_a[pos_idx] == n)
        )
        for n in y_probs_a[neg_idx]
    ])
    Vb_neg = np.array([
        np.mean(
            (y_probs_b[pos_idx] > n) +
            0.5 * (y_probs_b[pos_idx] == n)
        )
        for n in y_probs_b[neg_idx]
    ])

    cov_pos = np.cov(Va_pos, Vb_pos, ddof=1)[0, 1] / n_pos
    cov_neg = np.cov(Va_neg, Vb_neg, ddof=1)[0, 1] / n_neg
    cov     = cov_pos + cov_neg

    var_diff = var_a + var_b - 2 * cov
    var_diff = max(var_diff, 1e-10)

    z       = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z))

    return float(auc_a), float(auc_b), float(z), float(p_value)


# ============================================================
# METRIC HELPERS (threshold-aware versions for bootstrap)
# ============================================================

def _auroc(yt, yp):
    return roc_auc_score(yt, yp)


def _auprc(yt, yp):
    return average_precision_score(yt, yp)


def _f1_at_threshold(yt, yp, thr):
    preds = (yp >= thr).astype(int)
    return f1_score(yt, preds, zero_division=0)


def _sensitivity_at_threshold(yt, yp, thr):
    preds = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, preds).ravel()
    return tp / (tp + fn + 1e-8)


def _specificity_at_threshold(yt, yp, thr):
    preds = (yp >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, preds).ravel()
    return tn / (tn + fp + 1e-8)


# ============================================================
# FULL METRICS WITH CI
# ============================================================

def compute_all_metrics_with_ci(
    y_true,
    y_probs,
    threshold,
    task_name,
    logger,
    n_boot=N_BOOTSTRAP
):
    """
    Compute all metrics with BCa bootstrap CIs at the fixed
    threshold found on the validation set.
    Logs a formatted block and returns a dict.
    """

    results = {}

    logger.info(f"\n  [{task_name}]  threshold = {threshold:.4f}")

    # ---- AUROC ----
    auc, auc_lo, auc_hi = bootstrap_ci(
        y_true, y_probs, _auroc, n_boot=n_boot
    )
    results["auroc"] = (auc, auc_lo, auc_hi)
    logger.info(
        f"  AUROC       : {auc:.4f}  "
        f"(95% CI {auc_lo:.4f}–{auc_hi:.4f})"
    )

    # ---- AUPRC ----
    prc, prc_lo, prc_hi = bootstrap_ci(
        y_true, y_probs, _auprc, n_boot=n_boot
    )
    results["auprc"] = (prc, prc_lo, prc_hi)
    logger.info(
        f"  AUPRC       : {prc:.4f}  "
        f"(95% CI {prc_lo:.4f}–{prc_hi:.4f})"
    )

    # ---- F1 ----
    f1, f1_lo, f1_hi = bootstrap_ci(
        y_true, y_probs,
        _f1_at_threshold,
        n_boot=n_boot,
        threshold=threshold
    )
    results["f1"] = (f1, f1_lo, f1_hi)
    logger.info(
        f"  F1          : {f1:.4f}  "
        f"(95% CI {f1_lo:.4f}–{f1_hi:.4f})"
    )

    # ---- Sensitivity ----
    sens, sens_lo, sens_hi = bootstrap_ci(
        y_true, y_probs,
        _sensitivity_at_threshold,
        n_boot=n_boot,
        threshold=threshold
    )
    results["sensitivity"] = (sens, sens_lo, sens_hi)
    logger.info(
        f"  Sensitivity : {sens:.4f}  "
        f"(95% CI {sens_lo:.4f}–{sens_hi:.4f})"
    )

    # ---- Specificity ----
    spec, spec_lo, spec_hi = bootstrap_ci(
        y_true, y_probs,
        _specificity_at_threshold,
        n_boot=n_boot,
        threshold=threshold
    )
    results["specificity"] = (spec, spec_lo, spec_hi)
    logger.info(
        f"  Specificity : {spec:.4f}  "
        f"(95% CI {spec_lo:.4f}–{spec_hi:.4f})"
    )

    # ---- Class distribution ----
    n_pos  = int(y_true.sum())
    n_neg  = len(y_true) - n_pos
    prevalence = n_pos / len(y_true)
    logger.info(
        f"  Prevalence  : {prevalence:.3f}  "
        f"(pos={n_pos}, neg={n_neg}, total={len(y_true)})"
    )

    return results


# ============================================================
# INFERENCE LOOP
# ============================================================

def run_inference(model, loader, device):
    """
    Collect raw predicted probabilities and ground-truth labels
    from the test loader. Returns numpy arrays.
    """

    model.eval()

    mort_probs   = []
    mort_targets = []
    ahf_probs    = []
    ahf_targets  = []

    with torch.no_grad():

        for batch in tqdm(loader, desc="Inference"):

            image        = batch["image"].to(device)
            waveform     = batch["waveform"].to(device)
            ehr_features = batch["ehr_features"].to(device)
            cxr_mask     = batch["cxr_mod_mask"].to(device)
            ecg_mask     = batch["ecg_mod_mask"].to(device)
            ehr_mask     = batch["ehr_mod_mask"].to(device)
            ehr_feat_mask= batch["ehr_mask"].to(device)

            outputs = model(
                image, waveform, ehr_features,
                cxr_mask, ecg_mask,
                ehr_mask, ehr_feat_mask
            )

            mort_probs.extend(
                torch.sigmoid(
                    outputs["mortality_logits"]
                ).cpu().numpy()
            )
            mort_targets.extend(
                batch["target_mortality"].numpy()
            )

            ahf_probs.extend(
                torch.sigmoid(
                    outputs["ahf_logits"]
                ).cpu().numpy()
            )
            ahf_targets.extend(
                batch["target_ahf"].numpy()
            )

    return (
        np.array(mort_probs),
        np.array(mort_targets),
        np.array(ahf_probs),
        np.array(ahf_targets)
    )


# ============================================================
# VALIDATION-SET THRESHOLD FINDER
# Runs a lightweight forward pass over the val loader to find
# optimal thresholds WITHOUT touching test labels.
# ============================================================

def find_thresholds_on_val(model, val_loader, device, strategy="f1"):
    """
    Returns (mort_threshold, ahf_threshold) derived from the
    validation set so no test-set information is used.
    """

    model.eval()

    mort_probs_v   = []
    mort_targets_v = []
    ahf_probs_v    = []
    ahf_targets_v  = []

    with torch.no_grad():

        for batch in tqdm(val_loader, desc="Val threshold search"):

            image        = batch["image"].to(device)
            waveform     = batch["waveform"].to(device)
            ehr_features = batch["ehr_features"].to(device)
            cxr_mask     = batch["cxr_mod_mask"].to(device)
            ecg_mask     = batch["ecg_mod_mask"].to(device)
            ehr_mask     = batch["ehr_mod_mask"].to(device)
            ehr_feat_mask= batch["ehr_mask"].to(device)

            outputs = model(
                image, waveform, ehr_features,
                cxr_mask, ecg_mask,
                ehr_mask, ehr_feat_mask
            )

            mort_probs_v.extend(
                torch.sigmoid(
                    outputs["mortality_logits"]
                ).cpu().numpy()
            )
            mort_targets_v.extend(
                batch["target_mortality"].numpy()
            )

            ahf_probs_v.extend(
                torch.sigmoid(
                    outputs["ahf_logits"]
                ).cpu().numpy()
            )
            ahf_targets_v.extend(
                batch["target_ahf"].numpy()
            )

    mort_thr = find_optimal_threshold(
        np.array(mort_targets_v),
        np.array(mort_probs_v),
        strategy=strategy
    )

    ahf_thr = find_optimal_threshold(
        np.array(ahf_targets_v),
        np.array(ahf_probs_v),
        strategy=strategy
    )

    return mort_thr, ahf_thr


# ============================================================
# ARGUMENTS
# ============================================================

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        required=True
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "cxr_ecg",
            "cxr_ehr",
            "ecg_ehr",
            "trimodal"
        ]
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./fusion_test_results"
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="fusion_test"
    )

    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="Number of bootstrap resamples for CIs (default 1000)"
    )

    parser.add_argument(
        "--threshold_strategy",
        type=str,
        default="f1",
        choices=["f1", "youden", "min_dist"],
        help=(
            "Threshold selection strategy on val set. "
            "'f1' maximises F1 (recommended for imbalanced data); "
            "'youden' maximises sensitivity+specificity-1; "
            "'min_dist' minimises ROC distance to (0,1)."
        )
    )

    parser.add_argument(
        "--compare_experiment",
        type=str,
        default=None,
        choices=[
            "cxr_ecg",
            "cxr_ehr",
            "ecg_ehr",
            "trimodal",
            None
        ],
        help=(
            "Optional second experiment to run DeLong AUC "
            "comparison against the primary experiment."
        )
    )

    parser.add_argument(
        "--compare_weights",
        type=str,
        default=None,
        help="Checkpoint path for --compare_experiment."
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args   = get_args()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logger = setup_logger(args.save_dir, args.exp_name)

    active_modalities = \
        EXPERIMENT_CONFIG[args.experiment]["modalities"]

    logger.info(f"Experiment        : {args.experiment}")
    logger.info(f"Modalities        : {active_modalities}")
    logger.info(f"Threshold strategy: {args.threshold_strategy}")
    logger.info(f"Bootstrap samples : {args.n_bootstrap}")
    logger.info(f"Device            : {device}")

    # ---- Data ----
    train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

    _, val_loader, test_loader, _ = get_dataloaders(
        train_csv,
        val_csv,
        test_csv,
        batch_size=args.batch_size,
        dropout_config=None
    )

    # ---- Primary model ----
    model = HEALNetCBM(
        active_modalities=active_modalities,
        d_model=128,
        num_heads=4,
        num_fusion_layers=3,
        n_latent=8
    ).to(device)

    model.load_state_dict(
        torch.load(
            args.weights,
            map_location=device,
            weights_only=True
        )
    )

    logger.info(f"Weights loaded from: {args.weights}")

    # ---- Find optimal thresholds on validation set ----
    logger.info(
        "\nFinding optimal thresholds on validation set..."
    )

    mort_thr, ahf_thr = find_thresholds_on_val(
        model,
        val_loader,
        device,
        strategy=args.threshold_strategy
    )

    logger.info(
        f"  Mortality threshold : {mort_thr:.4f}"
    )
    logger.info(
        f"  AHF threshold       : {ahf_thr:.4f}"
    )

    # ---- Test inference ----
    logger.info("\nRunning inference on test set...")

    (
        mort_probs,
        mort_targets,
        ahf_probs,
        ahf_targets
    ) = run_inference(model, test_loader, device)

    # ---- Metrics with CI ----
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS WITH 95% CI (BCa Bootstrap)")
    logger.info("=" * 60)

    mort_metrics = compute_all_metrics_with_ci(
        mort_targets,
        mort_probs,
        threshold=mort_thr,
        task_name="MORTALITY",
        logger=logger,
        n_boot=args.n_bootstrap
    )

    ahf_metrics = compute_all_metrics_with_ci(
        ahf_targets,
        ahf_probs,
        threshold=ahf_thr,
        task_name="AHF",
        logger=logger,
        n_boot=args.n_bootstrap
    )

    combined_auroc = (
        mort_metrics["auroc"][0] +
        ahf_metrics["auroc"][0]
    ) / 2.0

    logger.info(
        f"\n  Combined AUROC (mean): {combined_auroc:.4f}"
    )

    # ---- Optional DeLong comparison ----
    if args.compare_experiment and args.compare_weights:

        logger.info("\n" + "=" * 60)
        logger.info(
            f"DELONG TEST vs {args.compare_experiment.upper()}"
        )
        logger.info("=" * 60)

        cmp_modalities = \
            EXPERIMENT_CONFIG[
                args.compare_experiment
            ]["modalities"]

        cmp_model = HEALNetCBM(
            active_modalities=cmp_modalities,
            d_model=128,
            num_heads=4,
            num_fusion_layers=3,
            n_latent=8
        ).to(device)

        cmp_model.load_state_dict(
            torch.load(
                args.compare_weights,
                map_location=device,
                weights_only=True
            )
        )

        logger.info(
            f"Compare weights loaded: {args.compare_weights}"
        )

        (
            cmp_mort_probs,
            _,
            cmp_ahf_probs,
            _
        ) = run_inference(cmp_model, test_loader, device)

        # Mortality DeLong
        auc_a, auc_b, z, p = delong_test(
            mort_targets,
            mort_probs,
            cmp_mort_probs
        )

        sig = "✓ significant" if p < 0.05 else "✗ not significant"

        logger.info(
            f"\n  [Mortality]"
            f"\n  {args.experiment} AUC = {auc_a:.4f}"
            f"\n  {args.compare_experiment} AUC = {auc_b:.4f}"
            f"\n  z = {z:.3f},  p = {p:.4f}  ({sig})"
        )

        # AHF DeLong
        auc_a, auc_b, z, p = delong_test(
            ahf_targets,
            ahf_probs,
            cmp_ahf_probs
        )

        sig = "✓ significant" if p < 0.05 else "✗ not significant"

        logger.info(
            f"\n  [AHF]"
            f"\n  {args.experiment} AUC = {auc_a:.4f}"
            f"\n  {args.compare_experiment} AUC = {auc_b:.4f}"
            f"\n  z = {z:.3f},  p = {p:.4f}  ({sig})"
        )

    logger.info("\n" + "=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()


# ============================================================
# RUN COMMANDS
# ============================================================
#
# Basic test (F1-optimal threshold + 1000 bootstrap CIs):
#
# python test.py \
#     --experiment trimodal \
#     --weights /home/azwad/Works/Multimodal-CBM/HealNet/checkpoints_5/run_trimodal/best_fusion_weights.pth
#
# python test.py \
#     --experiment cxr_ecg \
#     --weights /home/azwad/Works/Multimodal-CBM/HealNet/checkpoints_5/run_cxr_ecg/best_fusion_weights.pth
#
# python test.py \
#     --experiment cxr_ehr \
#     --weights /home/azwad/Works/Multimodal-CBM/HealNet/checkpoints_5/run_cxr_ehr/best_fusion_weights.pth
#
# python test.py \
#     --experiment ecg_ehr \
#     --weights /home/azwad/Works/Multimodal-CBM/HealNet/checkpoints_5/run_ecg_ehr/best_fusion_weights.pth
#
# With Youden threshold instead of F1:
#
# python test.py \
#     --experiment trimodal \
#     --weights ./fusion_checkpoints/trimodal/best_fusion_weights.pth \
#     --threshold_strategy youden
#
# DeLong comparison (trimodal vs cxr_ehr):
#
# python test.py \
#     --experiment trimodal \
#     --weights /home/azwad/Works/Multimodal-CBM/HealNet/checkpoints_5/run_trimodal/best_fusion_weights.pth
#     --compare_experiment cxr_ehr \
#     --compare_weights /home/azwad/Works/Multimodal-CBM/HealNet/checkpoints_5/run_cxr_ehr/best_fusion_weights.pth
#
# Faster run with fewer bootstrap samples (for debugging):
#
# python test_fusion_healnet.py \
#     --experiment cxr_ehr \
#     --weights ./fusion_checkpoints/cxr_ehr/best_fusion_weights.pth \
#     --n_bootstrap 200

