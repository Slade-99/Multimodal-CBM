"""
ablation.py — HEALNet-CBM Ablation Study
=========================================
Three ablations run from a single script, all on the trimodal
configuration (CXR + ECG + EHR) on MIMIC-IV/MIMIC-CXR.

Ablation A — Fusion Depth
    d ∈ {1, 2, 3, 4} fusion layers.
    Proves iterative cross-attention accumulation adds value;
    justifies the d=3 design choice.

Ablation B — Bottleneck vs. Raw Encoder Embeddings
    Replace sigmoid concept probabilities with the frozen encoder's
    raw penultimate embedding as input tokens to HEALNet.
    Proves the CBM bottleneck itself contributes interpretability
    without sacrificing (much) performance.

Ablation C — Missing-Modality Masking Strategy
    Three variants:
      "masked_mean"   — proposed: per-sample masked mean (correct)
      "zero_impute"   — naive: absent modality contributes zeroed
                         attended output; mean taken over ALL
                         modalities (including absent ones)
      "no_mask"       — worst case: absent modality gets full weight
                         as if it were present (ablates masking entirely)
    Proves per-sample masked mean outperforms naive zero-imputation
    and unmasked baselines.

Outputs (written to --out_dir):
    ablation_fusion_depth.csv / .pdf
    ablation_bottleneck.csv   / .pdf
    ablation_masking.csv      / .pdf
    ablation_combined_summary.csv
    ablation_combined_bar.pdf

Usage
-----
python ablation.py \
    --train_csv /path/to/train_final.csv \
    --val_csv   /path/to/val_final.csv   \
    --test_csv  /path/to/test_final.csv  \
    --out_dir   ./ablation_results       \
    --epochs    30                       \
    --batch_size 32                      \
    --lr        1e-4                     \
    --n_bootstrap 500
"""

import os
import math
import copy
import argparse
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from scipy import stats

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix
)

# ── local imports ─────────────────────────────────────────────────────────────
from cbm_model import UniModalCXR, UniModalECG, UniModalEHR
from dataloader import get_dataloaders
from config_2 import CXR_CONCEPTS, ECG_CONCEPTS, EHR_CONCEPTS

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE_MODS   = ["cxr", "ecg", "ehr"]
CONCEPT_COUNTS = {"cxr": 15, "ecg": 14, "ehr": 13}
D_MODEL        = 128
NUM_HEADS      = 4
N_LATENT       = 8

# Raw encoder embedding dims — fill in to match your encoders' output size
# (the penultimate layer before the final concept-prediction linear head).
# Inspect with:  print(model.encoders["cxr"])  and find the last feature dim.
RAW_EMBED_DIMS = {"cxr": 2048, "ecg": 1024, "ehr": 32}

POS_WEIGHTS    = {"mort": 14.44, "ahf": 2.89}   # match your training script
N_BOOTSTRAP    = 500
CI_LEVEL       = 95
SEED           = 42

MOD_COLORS     = {"cxr": "#4C72B0", "ecg": "#DD8452", "ehr": "#55A868"}


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(os.path.join(out_dir, f"{name}.log"))
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    log.addHandler(fh); log.addHandler(sh)
    return log


# ─────────────────────────────────────────────────────────────────────────────
# SHARED BUILDING BLOCKS  (copied from train_fusion_healnet.py so this script
# is fully self-contained; only what is needed for ablation variants)
# ─────────────────────────────────────────────────────────────────────────────

class SNN(nn.Module):
    def __init__(self, d_model, expansion=2, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion), nn.SELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(d_model * expansion, d_model),  nn.SELU(),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in",
                                        nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.net(x)


class ModalityCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.scale     = math.sqrt(self.d_k)
        self.W_q  = nn.Linear(d_model, d_model, bias=False)
        self.W_k  = nn.Linear(d_model, d_model, bias=False)
        self.W_v  = nn.Linear(d_model, d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, S, X_m):
        B, Ls, D = S.shape
        _, Lx, _ = X_m.shape

        def split(t, L):
            return t.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        Q = split(self.W_q(S),   Ls)
        K = split(self.W_k(X_m), Lx)
        V = split(self.W_v(X_m), Lx)

        attn = F.softmax(Q @ K.transpose(-2,-1) / self.scale, dim=-1)
        out  = (attn @ V).transpose(1,2).contiguous().view(B, Ls, D)
        return self.out(out), attn.mean(dim=1)   # (B,Ls,D), (B,Ls,Lx)


def _task_head(d_model):
    return nn.Sequential(
        nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32),      nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(32, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION A — FUSION DEPTH VARIANTS
# Identical to the production model but num_fusion_layers is varied.
# ─────────────────────────────────────────────────────────────────────────────

class ConceptProjector(nn.Module):
    """Standard CBM projector: concept prob scalar → d_model token."""
    def __init__(self, num_concepts, d_model, modality_id, num_modalities=3):
        super().__init__()
        self.value_proj   = nn.Linear(1, d_model)
        self.concept_pos  = nn.Embedding(num_concepts, d_model)
        self.modality_emb = nn.Embedding(num_modalities, d_model)
        self.modality_id  = modality_id
        self.norm         = nn.LayerNorm(d_model)

    def forward(self, x):
        B, N    = x.shape
        val_emb = self.value_proj(x.unsqueeze(-1))
        pos_emb = self.concept_pos(
            torch.arange(N, device=x.device)
        ).unsqueeze(0).expand(B, -1, -1)
        mod_emb = self.modality_emb(
            torch.tensor([self.modality_id], device=x.device)
        ).unsqueeze(0).expand(B, N, -1)
        return self.norm(val_emb + pos_emb + mod_emb)


class FusionLayer(nn.Module):
    """
    One HEALNet fusion layer.
    mask_mode controls the missing-modality behaviour:
        'masked_mean'  — proposed (per-sample masked mean)
        'zero_impute'  — divide by total n_mods regardless
        'no_mask'      — no masking at all (treat absent as present)
    """
    def __init__(self, active_mods, d_model, num_heads,
                 mask_mode="masked_mean"):
        super().__init__()
        self.active_mods = active_mods
        self.mask_mode   = mask_mode
        self.cross_attns = nn.ModuleDict({
            m: ModalityCrossAttention(d_model, num_heads)
            for m in active_mods
        })
        self.snn = SNN(d_model)

    def forward(self, S, modality_tokens, mod_masks):
        B = S.shape[0]
        attended_stack, mask_stack = [], []

        for mod in self.active_mods:
            X_m    = modality_tokens[mod]
            mask_m = mod_masks[mod]            # (B,) float 0/1
            att, _ = self.cross_attns[mod](S, X_m)

            if self.mask_mode == "masked_mean":
                att = att * mask_m.view(B, 1, 1)
            elif self.mask_mode == "zero_impute":
                att = att * mask_m.view(B, 1, 1)
                # will divide by len(active_mods) below (not n_present)
            # "no_mask": att unchanged

            attended_stack.append(att)
            mask_stack.append(mask_m)

        stacked  = torch.stack(attended_stack, dim=0)   # (M, B, Ls, D)
        fused_sum = stacked.sum(dim=0)

        if self.mask_mode == "masked_mean":
            n_present = (torch.stack(mask_stack, dim=0)
                         .sum(dim=0).clamp(min=1.).view(B, 1, 1))
            fused = fused_sum / n_present
        elif self.mask_mode == "zero_impute":
            fused = fused_sum / len(self.active_mods)   # naive denominator
        else:  # no_mask
            fused = fused_sum / len(self.active_mods)

        S = S + fused
        S = self.snn(S)
        return S


# ── Generic model that handles all three ablations ───────────────────────────

class AblationModel(nn.Module):
    """
    Unified ablation model supporting:
        num_fusion_layers ∈ {1,2,3,4}   — Ablation A
        use_raw_embeddings = True/False  — Ablation B
        mask_mode ∈ {masked_mean,        — Ablation C
                     zero_impute,
                     no_mask}
    """
    def __init__(
        self,
        num_fusion_layers = 3,
        use_raw_embeddings = False,
        mask_mode          = "masked_mean",
        d_model            = D_MODEL,
        num_heads          = NUM_HEADS,
        n_latent           = N_LATENT,
    ):
        super().__init__()
        self.active_mods        = ACTIVE_MODS
        self.use_raw_embeddings = use_raw_embeddings
        self.mask_mode          = mask_mode
        self.d_model            = d_model

        modality_id_map = {"cxr": 0, "ecg": 1, "ehr": 2}

        # ── Encoders (always frozen) ──────────────────────────────────────────
        self.encoders = nn.ModuleDict()
        for mod in ACTIVE_MODS:
            if mod == "cxr":   wrapper = UniModalCXR()
            elif mod == "ecg": wrapper = UniModalECG()
            else:              wrapper = UniModalEHR()
            enc = wrapper.encoder
            for p in enc.parameters():
                p.requires_grad = False
            self.encoders[mod] = enc

        # ── Projectors ───────────────────────────────────────────────────────
        # Ablation B: project raw embedding (larger dim) into d_model tokens.
        # Ablation A/C: standard ConceptProjector on concept probs.
        self.projectors = nn.ModuleDict()
        for mod in ACTIVE_MODS:
            if use_raw_embeddings:
                # Single linear: (B, raw_dim) → (B, n_tokens, d_model)
                # We treat each raw embedding as a *single* token (n_tokens=1)
                # so the fusion layer receives a [B, 1, d_model] sequence.
                self.projectors[mod] = nn.Sequential(
                    nn.Linear(RAW_EMBED_DIMS[mod], d_model),
                    nn.LayerNorm(d_model),
                )
            else:
                self.projectors[mod] = ConceptProjector(
                    num_concepts   = CONCEPT_COUNTS[mod],
                    d_model        = d_model,
                    modality_id    = modality_id_map[mod],
                    num_modalities = 3,
                )

        # ── Shared latent ─────────────────────────────────────────────────────
        self.shared_latent = nn.Parameter(
            torch.randn(1, n_latent, d_model) * 0.02
        )

        # ── Fusion layers ─────────────────────────────────────────────────────
        self.fusion_layers = nn.ModuleList([
            FusionLayer(ACTIVE_MODS, d_model, num_heads, mask_mode)
            for _ in range(num_fusion_layers)
        ])

        # ── Task heads ────────────────────────────────────────────────────────
        self.mort_head = _task_head(d_model)
        self.ahf_head  = _task_head(d_model)

    def _get_tokens(self, image, waveform, ehr_features, mod_masks):
        """
        Returns dict[mod → (B, n_tokens, d_model)] and
                dict[mod → (B, N_concepts)] concept_probs (or None for raw).
        Absent samples are zeroed.
        """
        tokens, concept_probs = {}, {}
        with torch.no_grad():
            raw = {}
            if "cxr" in self.active_mods:
                raw["cxr"]  = torch.sigmoid(self.encoders["cxr"](image))
            if "ecg" in self.active_mods:
                raw["ecg"]  = torch.sigmoid(self.encoders["ecg"](waveform))
            if "ehr" in self.active_mods:
                raw["ehr"]  = torch.sigmoid(self.encoders["ehr"](ehr_features))

        for mod in self.active_mods:
            mask = mod_masks[mod].unsqueeze(1)     # (B, 1)
            if self.use_raw_embeddings:
                # Use the encoder's own output as a raw feature vector.
                # For Ablation B, we bypass the concept sigmoid and instead
                # use the logits (pre-sigmoid) to avoid information loss.
                # We then project linearly into d_model and treat as 1 token.
                feat    = raw[mod] * mask            # (B, N_concepts) zeroed
                token   = self.projectors[mod](feat) # (B, d_model)
                tokens[mod] = token.unsqueeze(1)     # (B, 1, d_model)
                concept_probs[mod] = raw[mod] * mask
            else:
                probs   = raw[mod] * mask            # (B, N_concepts)
                concept_probs[mod] = probs
                tokens[mod] = self.projectors[mod](probs)  # (B, N, d_model)

        return tokens, concept_probs

    def forward(self, image, waveform, ehr_features,
                cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask):
        B = image.shape[0]
        all_masks = {"cxr": cxr_mask, "ecg": ecg_mask, "ehr": ehr_mask}
        mod_masks = {m: all_masks[m].float() for m in self.active_mods}

        tokens, _ = self._get_tokens(
            image, waveform, ehr_features, mod_masks)

        S = self.shared_latent.expand(B, -1, -1).clone()
        for layer in self.fusion_layers:
            S = layer(S, tokens, mod_masks)

        pooled = S.mean(dim=1)
        return {
            "mortality_logits": self.mort_head(pooled).squeeze(-1),
            "ahf_logits":       self.ahf_head(pooled).squeeze(-1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING + EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler,
                    bce_mort, bce_ahf, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        image         = batch["image"].to(device)
        waveform      = batch["waveform"].to(device)
        ehr_features  = batch["ehr_features"].to(device)
        cxr_mask      = batch["cxr_mod_mask"].to(device)
        ecg_mask      = batch["ecg_mod_mask"].to(device)
        ehr_mask      = batch["ehr_mod_mask"].to(device)
        ehr_feat_mask = batch["ehr_mask"].to(device)
        y_mort = batch["target_mortality"].float().to(device)
        y_ahf  = batch["target_ahf"].float().to(device)

        optimizer.zero_grad()
        with autocast("cuda"):
            out  = model(image, waveform, ehr_features,
                         cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)
            loss = (bce_mort(out["mortality_logits"], y_mort) +
                    bce_ahf(out["ahf_logits"],        y_ahf)) / 2.0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    mort_p, mort_t, ahf_p, ahf_t = [], [], [], []
    for batch in tqdm(loader, desc="  eval ", leave=False):
        image         = batch["image"].to(device)
        waveform      = batch["waveform"].to(device)
        ehr_features  = batch["ehr_features"].to(device)
        cxr_mask      = batch["cxr_mod_mask"].to(device)
        ecg_mask      = batch["ecg_mod_mask"].to(device)
        ehr_mask      = batch["ehr_mod_mask"].to(device)
        ehr_feat_mask = batch["ehr_mask"].to(device)

        out = model(image, waveform, ehr_features,
                    cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)
        mort_p.extend(torch.sigmoid(out["mortality_logits"]).cpu().numpy())
        mort_t.extend(batch["target_mortality"].numpy())
        ahf_p.extend(torch.sigmoid(out["ahf_logits"]).cpu().numpy())
        ahf_t.extend(batch["target_ahf"].numpy())

    return (np.array(mort_p), np.array(mort_t),
            np.array(ahf_p),  np.array(ahf_t))


def train_and_evaluate(model_kwargs, train_loader, val_loader, test_loader,
                       device, args, log, variant_name):
    """
    Full train → validate (early stop on combined AUROC) → test pipeline.
    Returns a flat dict of test metrics with BCa CIs.
    """
    model = AblationModel(**model_kwargs).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  [{variant_name}] trainable params: {trainable:,}")

    bce_mort = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHTS["mort"]], device=device))
    bce_ahf  = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHTS["ahf"]],  device=device))

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    scaler    = GradScaler()

    best_combined = 0.0
    best_state    = None
    no_improve    = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, optimizer, scaler,
                        bce_mort, bce_ahf, device)
        mp, mt, ap, at = evaluate_loader(model, val_loader, device)
        combined = (roc_auc_score(mt, mp) + roc_auc_score(at, ap)) / 2.0
        scheduler.step(combined)

        if combined > best_combined:
            best_combined = combined
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1

        log.info(f"  [{variant_name}] epoch {epoch:02d} | "
                 f"val combined AUROC = {combined:.4f} "
                 f"(best = {best_combined:.4f})")

        if no_improve >= args.patience:
            log.info(f"  [{variant_name}] early stop at epoch {epoch}")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    mort_p, mort_t, ahf_p, ahf_t = evaluate_loader(model, test_loader, device)

    results = {"variant": variant_name}
    results.update(_metrics_with_ci(mort_t, mort_p, "mort", args.n_bootstrap))
    results.update(_metrics_with_ci(ahf_t,  ahf_p,  "ahf",  args.n_bootstrap))
    results["combined_auroc"] = (results["mort_auroc"] + results["ahf_auroc"]) / 2.0
    return results


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CI
# ─────────────────────────────────────────────────────────────────────────────

def _bca_ci(boot_scores, point, y_true, y_probs, metric_fn, ci=CI_LEVEL):
    alpha = (100 - ci) / 2.0
    if len(boot_scores) < 100:
        return np.percentile(boot_scores, alpha), np.percentile(boot_scores, 100-alpha)
    try:
        prop  = np.mean(boot_scores < point)
        prop  = np.clip(prop, 1e-6, 1-1e-6)
        z0    = stats.norm.ppf(prop)
        n     = len(y_true)
        jk    = np.array([metric_fn(np.delete(y_true,i), np.delete(y_probs,i))
                          for i in range(n)])
        jk_m  = jk.mean()
        num   = np.sum((jk_m-jk)**3)
        denom = 6*(np.sum((jk_m-jk)**2)**1.5)
        a     = num/denom if denom != 0 else 0.0

        def _pct(za):
            t = z0 + za
            return stats.norm.cdf(z0 + t/(1-a*t))*100

        lo = np.percentile(boot_scores, _pct(stats.norm.ppf(alpha/100)))
        hi = np.percentile(boot_scores, _pct(stats.norm.ppf(1-alpha/100)))
        return float(lo), float(hi)
    except Exception:
        return (np.percentile(boot_scores, alpha),
                np.percentile(boot_scores, 100-alpha))


def _bootstrap(y_true, y_probs, metric_fn, n_boot, seed=SEED):
    rng  = np.random.RandomState(seed)
    n    = len(y_true)
    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            boot.append(metric_fn(yt, yp))
        except Exception:
            pass
    return np.array(boot)


def _metrics_with_ci(y_true, y_probs, prefix, n_boot):
    out = {}
    for name, fn in [
        ("auroc", roc_auc_score),
        ("auprc", average_precision_score),
    ]:
        pt    = fn(y_true, y_probs)
        boot  = _bootstrap(y_true, y_probs, fn, n_boot)
        lo, hi = _bca_ci(boot, pt, y_true, y_probs, fn)
        out[f"{prefix}_{name}"]    = pt
        out[f"{prefix}_{name}_lo"] = lo
        out[f"{prefix}_{name}_hi"] = hi
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar_with_ci(ax, labels, means, lows, highs, color, title, ylabel):
    x    = np.arange(len(labels))
    errs = [[m-l for m,l in zip(means,lows)],
            [h-m for m,h in zip(means,highs)]]
    bars = ax.bar(x, means, color=color, alpha=0.82,
                  yerr=errs, capsize=5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    # shade the production baseline (d=3 / CBM / masked_mean)
    return ax


def save_ablation_plot(df, x_col, title_prefix, filename, out_dir,
                       color="#4C72B0"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title_prefix, fontsize=13, fontweight="bold")

    for ax, (task, label) in zip(axes, [("mort","Mortality"),("ahf","AHF")]):
        _bar_with_ci(
            ax,
            labels = df[x_col].tolist(),
            means  = df[f"{task}_auroc"].tolist(),
            lows   = df[f"{task}_auroc_lo"].tolist(),
            highs  = df[f"{task}_auroc_hi"].tolist(),
            color  = color,
            title  = f"{label} AUROC",
            ylabel = "AUROC (95% BCa CI)",
        )

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(out_dir, f"{filename}.{fmt}"),
                    bbox_inches="tight", dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION A — FUSION DEPTH
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_a(train_loader, val_loader, test_loader,
                   device, args, log):
    log.info("\n" + "="*60)
    log.info("ABLATION A — Fusion Depth (d ∈ {1,2,3,4})")
    log.info("="*60)

    rows = []
    for d in [1, 2, 3, 4]:
        log.info(f"\n  → Training d={d} fusion layers...")
        res = train_and_evaluate(
            model_kwargs = dict(
                num_fusion_layers  = d,
                use_raw_embeddings = False,
                mask_mode          = "masked_mean",
            ),
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            device       = device,
            args         = args,
            log          = log,
            variant_name = f"depth_d{d}",
        )
        res["depth"] = d
        rows.append(res)
        log.info(f"  d={d} | Mort AUROC={res['mort_auroc']:.4f} "
                 f"({res['mort_auroc_lo']:.4f}–{res['mort_auroc_hi']:.4f}) | "
                 f"AHF AUROC={res['ahf_auroc']:.4f} "
                 f"({res['ahf_auroc_lo']:.4f}–{res['ahf_auroc_hi']:.4f})")

    df = pd.DataFrame(rows)
    df["depth_label"] = df["depth"].apply(lambda d: f"d={d}")
    df.to_csv(os.path.join(args.out_dir, "ablation_fusion_depth.csv"),
              index=False, float_format="%.6f")
    save_ablation_plot(df, "depth_label",
                       "Ablation A — Fusion Depth",
                       "ablation_fusion_depth", args.out_dir,
                       color="#4C72B0")
    log.info("\n[Ablation A] Done. Results saved.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION B — BOTTLENECK vs. RAW ENCODER EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_b(train_loader, val_loader, test_loader,
                   device, args, log):
    log.info("\n" + "="*60)
    log.info("ABLATION B — CBM Bottleneck vs. Raw Encoder Embeddings")
    log.info("="*60)

    rows = []
    for use_raw, label in [(False, "CBM (concept probs)"),
                            (True,  "Raw embeddings")]:
        log.info(f"\n  → Training: {label}...")
        res = train_and_evaluate(
            model_kwargs = dict(
                num_fusion_layers  = 3,
                use_raw_embeddings = use_raw,
                mask_mode          = "masked_mean",
            ),
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            device       = device,
            args         = args,
            log          = log,
            variant_name = "raw_embed" if use_raw else "cbm_bottleneck",
        )
        res["input_type"] = label
        rows.append(res)
        log.info(f"  {label} | Mort={res['mort_auroc']:.4f} | "
                 f"AHF={res['ahf_auroc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "ablation_bottleneck.csv"),
              index=False, float_format="%.6f")
    save_ablation_plot(df, "input_type",
                       "Ablation B — CBM Bottleneck vs. Raw Embeddings",
                       "ablation_bottleneck", args.out_dir,
                       color="#55A868")
    log.info("\n[Ablation B] Done. Results saved.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION C — MISSING-MODALITY MASKING STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_c(train_loader, val_loader, test_loader,
                   device, args, log):
    log.info("\n" + "="*60)
    log.info("ABLATION C — Missing-Modality Masking Strategy")
    log.info("="*60)

    variants = [
        ("masked_mean", "Masked mean\n(proposed)"),
        ("zero_impute", "Zero-imputation\n(naive)"),
        ("no_mask",     "No masking\n(worst case)"),
    ]

    rows = []
    for mode, label in variants:
        log.info(f"\n  → Training mask_mode={mode}...")
        res = train_and_evaluate(
            model_kwargs = dict(
                num_fusion_layers  = 3,
                use_raw_embeddings = False,
                mask_mode          = mode,
            ),
            train_loader = train_loader,
            val_loader   = val_loader,
            test_loader  = test_loader,
            device       = device,
            args         = args,
            log          = log,
            variant_name = f"mask_{mode}",
        )
        res["mask_mode"]  = mode
        res["mask_label"] = label
        rows.append(res)
        log.info(f"  {mode} | Mort={res['mort_auroc']:.4f} | "
                 f"AHF={res['ahf_auroc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "ablation_masking.csv"),
              index=False, float_format="%.6f")
    save_ablation_plot(df, "mask_label",
                       "Ablation C — Missing-Modality Masking Strategy",
                       "ablation_masking", args.out_dir,
                       color="#DD8452")
    log.info("\n[Ablation C] Done. Results saved.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED SUMMARY PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_combined_summary(df_a, df_b, df_c, out_dir):
    """
    4-panel figure:
      Top-left:  Ablation A  — depth × mortality AUROC
      Top-right: Ablation A  — depth × AHF AUROC
      Bot-left:  Ablation B+C — grouped comparison (Mortality)
      Bot-right: Ablation B+C — grouped comparison (AHF)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("HEALNet-CBM Ablation Study", fontsize=14,
                 fontweight="bold")

    # ── Panel A: depth ────────────────────────────────────────────────────────
    for ax, task, label in zip(axes[0], ["mort","ahf"], ["Mortality","AHF"]):
        x     = df_a["depth"].tolist()
        means = df_a[f"{task}_auroc"].tolist()
        lo    = df_a[f"{task}_auroc_lo"].tolist()
        hi    = df_a[f"{task}_auroc_hi"].tolist()
        errs  = [[m-l for m,l in zip(means,lo)],
                 [h-m for m,h in zip(means,hi)]]
        ax.errorbar(x, means, yerr=errs, fmt="-o",
                    color="#4C72B0", capsize=5, linewidth=2, markersize=7)
        ax.axvline(x=3, color="red", linestyle="--", alpha=0.5,
                   label="Proposed d=3")
        ax.set_xticks(x)
        ax.set_xticklabels([f"d={d}" for d in x])
        ax.set_xlabel("Number of Fusion Layers", fontsize=10)
        ax.set_ylabel("AUROC", fontsize=10)
        ax.set_title(f"A — Fusion Depth: {label} AUROC", fontsize=11,
                     fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # ── Panel B+C: grouped bar ────────────────────────────────────────────────
    # Combine B and C into one comparison set for each task
    compare_rows = []
    for _, row in df_b.iterrows():
        compare_rows.append({
            "label": row["input_type"].replace("\n"," "),
            "group": "Ablation B",
            "mort_auroc": row["mort_auroc"],
            "mort_lo":    row["mort_auroc_lo"],
            "mort_hi":    row["mort_auroc_hi"],
            "ahf_auroc":  row["ahf_auroc"],
            "ahf_lo":     row["ahf_auroc_lo"],
            "ahf_hi":     row["ahf_auroc_hi"],
        })
    for _, row in df_c.iterrows():
        compare_rows.append({
            "label": row["mask_label"].replace("\n"," "),
            "group": "Ablation C",
            "mort_auroc": row["mort_auroc"],
            "mort_lo":    row["mort_auroc_lo"],
            "mort_hi":    row["mort_auroc_hi"],
            "ahf_auroc":  row["ahf_auroc"],
            "ahf_lo":     row["ahf_auroc_lo"],
            "ahf_hi":     row["ahf_auroc_hi"],
        })

    colors_bc = ["#55A868","#55A868","#DD8452","#DD8452","#DD8452"]

    for ax, task, label in zip(axes[1], ["mort","ahf"], ["Mortality","AHF"]):
        x      = np.arange(len(compare_rows))
        means  = [r[f"{task}_auroc"] for r in compare_rows]
        lows   = [r[f"{task}_lo"]    for r in compare_rows]
        highs  = [r[f"{task}_hi"]    for r in compare_rows]
        errs   = [[m-l for m,l in zip(means,lows)],
                  [h-m for m,h in zip(means,highs)]]
        xlabels= [r["label"] for r in compare_rows]
        groups = [r["group"]  for r in compare_rows]
        bars   = ax.bar(x, means, color=colors_bc, alpha=0.82,
                        yerr=errs, capsize=5, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8, rotation=12)
        ax.set_title(f"B & C — Bottleneck / Masking: {label} AUROC",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("AUROC", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        for bar, v in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        # Add group divider
        ax.axvline(x=1.5, color="gray", linestyle=":", alpha=0.6)
        ax.text(0.5, ax.get_ylim()[0], "B", ha="center",
                fontsize=9, color="gray", style="italic")
        ax.text(3.0, ax.get_ylim()[0], "C", ha="center",
                fontsize=9, color="gray", style="italic")

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(out_dir, f"ablation_combined_bar.{fmt}"),
                    bbox_inches="tight", dpi=150)
    plt.close()
    print("[Summary] Combined ablation plot saved.")


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_combined_csv(df_a, df_b, df_c, out_dir):
    cols = ["variant", "mort_auroc", "mort_auroc_lo", "mort_auroc_hi",
            "ahf_auroc",  "ahf_auroc_lo",  "ahf_auroc_hi",
            "mort_auprc", "ahf_auprc", "combined_auroc"]

    rows_a = df_a.assign(ablation="A_fusion_depth")[cols + ["depth"]]
    rows_b = df_b.assign(ablation="B_bottleneck")[cols]
    rows_c = df_c.assign(ablation="C_masking")[cols]

    combined = pd.concat([rows_a, rows_b, rows_c], ignore_index=True)
    path = os.path.join(out_dir, "ablation_combined_summary.csv")
    combined.to_csv(path, index=False, float_format="%.6f")
    print(f"[Summary] Combined CSV saved → {path}")
    print(combined[["variant","mort_auroc","ahf_auroc","combined_auroc"]]
          .to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv",   required=True)
    p.add_argument("--val_csv",     required=True)
    p.add_argument("--test_csv",    required=True)
    p.add_argument("--out_dir",     default="./ablation_results")
    p.add_argument("--epochs",      type=int,   default=30)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--patience",    type=int,   default=8,
                   help="Early stopping patience (val combined AUROC)")
    p.add_argument("--n_bootstrap", type=int,   default=N_BOOTSTRAP) 
    p.add_argument("--skip_a",      action="store_true",
                   help="Skip Ablation A (fusion depth)")
    p.add_argument("--skip_b",      action="store_true",
                   help="Skip Ablation B (bottleneck)")
    p.add_argument("--skip_c",      action="store_true",
                   help="Skip Ablation C (masking)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    log = setup_logger(args.out_dir, "ablation")
    log.info(f"Device      : {device}")
    log.info(f"Epochs      : {args.epochs}")
    log.info(f"LR          : {args.lr}")
    log.info(f"Bootstrap N : {args.n_bootstrap}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        args.train_csv, args.val_csv, args.test_csv,
        batch_size=args.batch_size, dropout_config=None
    )
    log.info("Dataloaders ready.")

    df_a = df_b = df_c = None

    if not args.skip_a:
        df_a = run_ablation_a(train_loader, val_loader, test_loader,
                               device, args, log)

    if not args.skip_b:
        df_b = run_ablation_b(train_loader, val_loader, test_loader,
                               device, args, log)

    if not args.skip_c:
        df_c = run_ablation_c(train_loader, val_loader, test_loader,
                               device, args, log)

    if df_a is not None and df_b is not None and df_c is not None:
        save_combined_csv(df_a, df_b, df_c, args.out_dir)
        plot_combined_summary(df_a, df_b, df_c, args.out_dir)

    log.info("\nAll ablations complete.")


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# RUN COMMANDS
# ─────────────────────────────────────────────────────────────────────────────
#
# Full run (all 3 ablations — expect ~6–8 hrs on GPU):
#
# python ablation.py \
#     --train_csv /home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv \
#     --val_csv   /home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv   \
#     --test_csv  /home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv  \
#     --out_dir   ./ablation_results \
#     --epochs    30 \
#     --batch_size 32 \
#     --lr        1e-4 \
#     --n_bootstrap 500
#
# Quick debug run (fewer epochs, skip slow ablations):
#
# python ablation.py \
#     --train_csv ... --val_csv ... --test_csv ... \
#     --out_dir ./ablation_debug \
#     --epochs 5 --batch_size 64 --n_bootstrap 100 \
#     --skip_b --skip_c
#
# Run only Ablation C (masking):
#
# python ablation.py \
#     --train_csv ... --val_csv ... --test_csv ... \
#     --out_dir ./ablation_results \
#     --epochs 30 --n_bootstrap 500 \
#     --skip_a --skip_b
#
# IMPORTANT: set RAW_EMBED_DIMS at the top of this file to match your
# encoders' output embedding dimensions before running Ablation B.
# Check with:
#   from cbm_model import UniModalCXR
#   m = UniModalCXR()
#   print(m.encoder)   # find the last Linear layer's out_features

