import os
import argparse
import logging
import torch
import torch.nn as nn
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

# Import your specific dataloader
from dataloader_ehr import get_ehr_concept_dataloaders, EHR_CONCEPTS

# ============================================================
# CONSTANTS & SETUP
# ============================================================
N_BOOTSTRAP = 1000
CI_LEVEL     = 95
RANDOM_SEED  = 42

def setup_logger(save_dir, exp_name):
    os.makedirs(os.path.join(save_dir, exp_name), exist_ok=True)
    log_file = os.path.join(save_dir, exp_name, "test_results.txt")
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
# MODEL ARCHITECTURE (From your training script)
# ============================================================
class EHRConceptEncoder(nn.Module):
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
        return self.network(x)

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

class UniModalEHR(nn.Module):
    def __init__(self, input_dim=21, num_concepts=13):
        super().__init__()
        self.encoder        = EHRConceptEncoder(input_dim=input_dim, num_concepts=num_concepts)
        # Note: Hardcoded checkpoint loading removed for the test script. 
        # Main() handles loading the final weights.
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, ehr_features):
        logits = self.encoder(ehr_features)   
        probs = torch.sigmoid(logits)
        return {
            'ehr_concept_logits': logits,
            'mortality_logits':   self.mortality_head(probs).squeeze(-1),
            'ahf_logits':         self.ahf_head(probs).squeeze(-1)
        }

# ============================================================
# STATISTICAL GRADING (Thresholds, CIs, DeLong)
# ============================================================
def find_optimal_threshold(y_true, y_probs, strategy="f1"):
    if strategy == "f1":
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        f1_scores = np.where((precisions + recalls) == 0, 0.0, 2 * precisions * recalls / (precisions + recalls))
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

def _jackknife_accel(y_true, y_probs, metric_fn):
    n      = len(y_true)
    jk     = np.array([metric_fn(np.delete(y_true, i), np.delete(y_probs, i)) for i in range(n)])
    jk_mean = jk.mean()
    num     = np.sum((jk_mean - jk) ** 3)
    denom   = 6.0 * (np.sum((jk_mean - jk) ** 2) ** 1.5)
    return num / denom if denom != 0 else 0.0

def bootstrap_ci(y_true, y_probs, metric_fn, n_boot=N_BOOTSTRAP, ci=CI_LEVEL, seed=RANDOM_SEED, threshold=None):
    rng   = np.random.RandomState(seed)
    n     = len(y_true)
    alpha = (100 - ci) / 2.0

    if threshold is not None:
        def _metric(yt, yp): return metric_fn(yt, yp, threshold)
    else:
        _metric = metric_fn

    point = _metric(y_true, y_probs)
    boot_scores = []
    
    for _ in range(n_boot):
        idx   = rng.choice(n, n, replace=True)
        yt_b  = y_true[idx]
        yp_b  = y_probs[idx]
        if len(np.unique(yt_b)) < 2: continue
        try: boot_scores.append(_metric(yt_b, yp_b))
        except Exception: continue

    boot_scores = np.array(boot_scores)

    if len(boot_scores) >= 200:
        try:
            prop_less = np.clip(np.mean(boot_scores < point), 1e-6, 1 - 1e-6)
            z0        = stats.norm.ppf(prop_less)
            accel = _jackknife_accel(y_true, y_probs, _metric)
            z_alpha_lo = stats.norm.ppf(alpha / 100.0)
            z_alpha_hi = stats.norm.ppf(1 - alpha / 100.0)

            def _adj_pct(z_alpha):
                num   = z0 + z_alpha
                denom = 1 - accel * (z0 + z_alpha)
                return stats.norm.cdf(z0 + num / denom) * 100

            lower  = np.percentile(boot_scores, _adj_pct(z_alpha_lo))
            upper  = np.percentile(boot_scores, _adj_pct(z_alpha_hi))
        except Exception:
            lower = np.percentile(boot_scores, alpha)
            upper = np.percentile(boot_scores, 100 - alpha)
    else:
        lower = np.percentile(boot_scores, alpha)
        upper = np.percentile(boot_scores, 100 - alpha)

    return float(point), float(lower), float(upper)

def delong_auc_variance(y_true, y_probs):
    pos_idx  = np.where(y_true == 1)[0]
    neg_idx  = np.where(y_true == 0)[0]
    n_pos    = len(pos_idx)
    n_neg    = len(neg_idx)
    pos_prob = y_probs[pos_idx]
    neg_prob = y_probs[neg_idx]

    V_pos = np.array([np.mean((neg_prob < p) + 0.5 * (neg_prob == p)) for p in pos_prob])
    V_neg = np.array([np.mean((pos_prob > n) + 0.5 * (pos_prob == n)) for n in neg_prob])

    auc   = V_pos.mean()
    s_pos = np.var(V_pos, ddof=1) / n_pos
    s_neg = np.var(V_neg, ddof=1) / n_neg
    return auc, s_pos + s_neg

def delong_test(y_true, y_probs_a, y_probs_b):
    auc_a, var_a = delong_auc_variance(y_true, y_probs_a)
    auc_b, var_b = delong_auc_variance(y_true, y_probs_b)

    pos_idx  = np.where(y_true == 1)[0]
    neg_idx  = np.where(y_true == 0)[0]
    n_pos    = len(pos_idx)
    n_neg    = len(neg_idx)

    Va_pos = np.array([np.mean((y_probs_a[neg_idx] < p) + 0.5 * (y_probs_a[neg_idx] == p)) for p in y_probs_a[pos_idx]])
    Vb_pos = np.array([np.mean((y_probs_b[neg_idx] < p) + 0.5 * (y_probs_b[neg_idx] == p)) for p in y_probs_b[pos_idx]])
    Va_neg = np.array([np.mean((y_probs_a[pos_idx] > n) + 0.5 * (y_probs_a[pos_idx] == n)) for n in y_probs_a[neg_idx]])
    Vb_neg = np.array([np.mean((y_probs_b[pos_idx] > n) + 0.5 * (y_probs_b[pos_idx] == n)) for n in y_probs_b[neg_idx]])

    cov_pos = np.cov(Va_pos, Vb_pos, ddof=1)[0, 1] / n_pos
    cov_neg = np.cov(Va_neg, Vb_neg, ddof=1)[0, 1] / n_neg
    cov     = cov_pos + cov_neg

    var_diff = max(var_a + var_b - 2 * cov, 1e-10)
    z        = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value  = 2 * stats.norm.sf(abs(z))

    return float(auc_a), float(auc_b), float(z), float(p_value)

# Metric Helpers
def _auroc(yt, yp): return roc_auc_score(yt, yp)
def _auprc(yt, yp): return average_precision_score(yt, yp)
def _f1_at_thr(yt, yp, thr): return f1_score(yt, (yp >= thr).astype(int), zero_division=0)
def _sens_at_thr(yt, yp, thr): 
    tn, fp, fn, tp = confusion_matrix(yt, (yp >= thr).astype(int)).ravel()
    return tp / (tp + fn + 1e-8)
def _spec_at_thr(yt, yp, thr): 
    tn, fp, fn, tp = confusion_matrix(yt, (yp >= thr).astype(int)).ravel()
    return tn / (tn + fp + 1e-8)

def compute_all_metrics_with_ci(y_true, y_probs, threshold, task_name, logger, n_boot=N_BOOTSTRAP):
    results = {}
    logger.info(f"\n  [{task_name}]  threshold = {threshold:.4f}")

    metrics = [
        ("AUROC", _auroc, None),
        ("AUPRC", _auprc, None),
        ("F1", _f1_at_thr, threshold),
        ("Sensitivity", _sens_at_thr, threshold),
        ("Specificity", _spec_at_thr, threshold)
    ]

    for name, fn, thr in metrics:
        val, lo, hi = bootstrap_ci(y_true, y_probs, fn, n_boot=n_boot, threshold=thr)
        results[name.lower()] = (val, lo, hi)
        logger.info(f"  {name:<12}: {val:.4f}  (95% CI {lo:.4f}–{hi:.4f})")

    prevalence = int(y_true.sum()) / len(y_true)
    logger.info(f"  Prevalence  : {prevalence:.3f}  (pos={int(y_true.sum())}, neg={len(y_true)-int(y_true.sum())}, total={len(y_true)})")
    return results

# ============================================================
# INFERENCE LOGIC
# ============================================================
def run_inference(model, loader, device):
    model.eval()
    mort_probs, mort_targets = [], []
    ahf_probs, ahf_targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            if batch is None: continue
            
            ehr_features = batch["ehr_features"].to(device)
            outputs = model(ehr_features)

            mort_probs.extend(torch.sigmoid(outputs["mortality_logits"]).cpu().numpy())
            mort_targets.extend(batch["target_mortality"].numpy())

            ahf_probs.extend(torch.sigmoid(outputs["ahf_logits"]).cpu().numpy())
            ahf_targets.extend(batch["target_ahf"].numpy())

    return np.array(mort_probs), np.array(mort_targets), np.array(ahf_probs), np.array(ahf_targets)

def find_thresholds_on_val(model, val_loader, device, strategy="f1"):
    model.eval()
    mort_probs_v, mort_targets_v = [], []
    ahf_probs_v, ahf_targets_v = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Val threshold search"):
            if batch is None: continue
            
            ehr_features = batch["ehr_features"].to(device)
            outputs = model(ehr_features)

            mort_probs_v.extend(torch.sigmoid(outputs["mortality_logits"]).cpu().numpy())
            mort_targets_v.extend(batch["target_mortality"].numpy())

            ahf_probs_v.extend(torch.sigmoid(outputs["ahf_logits"]).cpu().numpy())
            ahf_targets_v.extend(batch["target_ahf"].numpy())

    mort_thr = find_optimal_threshold(np.array(mort_targets_v), np.array(mort_probs_v), strategy=strategy)
    ahf_thr  = find_optimal_threshold(np.array(ahf_targets_v),  np.array(ahf_probs_v),  strategy=strategy)
    return mort_thr, ahf_thr

# ============================================================
# ARGUMENTS & MAIN
# ============================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/home/azwad/Works/Multimodal-CBM/Stage_2/checkpoints_4/run_ehr_only/best_stage2_weights.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="./ehr_test_results")
    parser.add_argument("--exp_name", type=str, default="ehr_unimodal_test")
    parser.add_argument("--n_bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--threshold_strategy", type=str, default="f1", choices=["f1", "youden", "min_dist"])
    
    # Optional parameters if you want to compare against another model's weights
    parser.add_argument("--compare_weights", type=str, default=None)
    return parser.parse_args()

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(args.save_dir, args.exp_name)

    logger.info("Starting EHR UniModal Test")
    logger.info(f"Threshold strategy: {args.threshold_strategy}")
    logger.info(f"Bootstrap samples : {args.n_bootstrap}")

    # 1. Load Data
    train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

    _, val_loader, test_loader = get_ehr_concept_dataloaders(
        train_csv, val_csv, test_csv, batch_size=args.batch_size
    )

    # 2. Setup Primary Model
    model = UniModalEHR(input_dim=21, num_concepts=len(EHR_CONCEPTS)).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    logger.info(f"Weights loaded from: {args.weights}")

    # 3. Find Thresholds (Practice Test)
    logger.info("\nFinding optimal thresholds on validation set...")
    mort_thr, ahf_thr = find_thresholds_on_val(model, val_loader, device, strategy=args.threshold_strategy)
    logger.info(f"  Mortality threshold : {mort_thr:.4f}")
    logger.info(f"  AHF threshold       : {ahf_thr:.4f}")

    # 4. Final Inference (The Real Exam)
    logger.info("\nRunning inference on test set...")
    mort_probs, mort_targets, ahf_probs, ahf_targets = run_inference(model, test_loader, device)

    # 5. Grading and Metrics
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST RESULTS WITH 95% CI (BCa Bootstrap)")
    logger.info("=" * 60)

    mort_metrics = compute_all_metrics_with_ci(
        mort_targets, mort_probs, threshold=mort_thr, task_name="MORTALITY", logger=logger, n_boot=args.n_bootstrap
    )

    ahf_metrics = compute_all_metrics_with_ci(
        ahf_targets, ahf_probs, threshold=ahf_thr, task_name="AHF", logger=logger, n_boot=args.n_bootstrap
    )

    combined_auroc = (mort_metrics["auroc"][0] + ahf_metrics["auroc"][0]) / 2.0
    logger.info(f"\n  Combined AUROC (mean): {combined_auroc:.4f}")

    # 6. Compare Models (Optional)
    if args.compare_weights:
        logger.info("\n" + "=" * 60)
        logger.info("DELONG TEST (Comparing Primary vs Secondary Model)")
        logger.info("=" * 60)

        cmp_model = UniModalEHR(input_dim=21, num_concepts=len(EHR_CONCEPTS)).to(device)
        cmp_model.load_state_dict(torch.load(args.compare_weights, map_location=device, weights_only=True))
        
        cmp_mort_probs, _, cmp_ahf_probs, _ = run_inference(cmp_model, test_loader, device)

        # Mortality DeLong
        auc_a, auc_b, z, p = delong_test(mort_targets, mort_probs, cmp_mort_probs)
        sig = "✓ significant" if p < 0.05 else "✗ not significant"
        logger.info(f"\n  [Mortality]\n  Primary AUC = {auc_a:.4f}\n  Compare AUC = {auc_b:.4f}\n  z = {z:.3f},  p = {p:.4f}  ({sig})")

        # AHF DeLong
        auc_a, auc_b, z, p = delong_test(ahf_targets, ahf_probs, cmp_ahf_probs)
        sig = "✓ significant" if p < 0.05 else "✗ not significant"
        logger.info(f"\n  [AHF]\n  Primary AUC = {auc_a:.4f}\n  Compare AUC = {auc_b:.4f}\n  z = {z:.3f},  p = {p:.4f}  ({sig})")

    logger.info("\nDone.")

if __name__ == "__main__":
    main()