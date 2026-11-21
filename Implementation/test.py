import os
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
from tqdm import tqdm
from cbm_model import UniModalCXR, UniModalEHR, UniModalECG
from config import get_args
from dataloader import get_dataloaders, ModalityDropoutConfig

# ================= SETUP LOGGING =================

def setup_logger(save_dir, exp_name):
    log_file = os.path.join(save_dir, exp_name, "evaluation_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger()

# ================= METRICS =================

def calculate_ece(y_true, y_probs, n_bins=10):
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_probs, bin_edges, right=False)
    bin_indices = np.clip(bin_indices, 1, n_bins)

    ece = 0.0
    total_samples = len(y_probs)

    for bin_idx in range(1, n_bins + 1):
        mask = (bin_indices == bin_idx)
        if np.any(mask):
            prob_true = np.mean(y_true[mask])
            prob_pred = np.mean(y_probs[mask])
            bin_weight = np.sum(mask) / total_samples
            ece += np.abs(prob_true - prob_pred) * bin_weight

    return ece

def find_best_threshold(y_true, y_probs):
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_f1, best_thresh = 0.0, 0.5

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh

def calculate_final_metrics(y_true, y_probs, threshold):
    y_preds = (y_probs >= threshold).astype(int)

    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    ece = calculate_ece(y_true, y_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        'AUROC': auroc,
        'AUPRC': auprc,
        'F1': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'ECE': ece
    }

# ================= NEW: CONCEPT METRICS =================

def compute_concept_metrics(y_true, y_probs):
    aurocs = []
    auprcs = []

    for i in range(y_true.shape[1]):
        try:
            aurocs.append(roc_auc_score(y_true[:, i], y_probs[:, i]))
            auprcs.append(average_precision_score(y_true[:, i], y_probs[:, i]))
        except ValueError:
            aurocs.append(0.5)
            auprcs.append(0.0)

    return {
        'per_concept_auroc': aurocs,
        'mean_concept_auroc': np.mean(aurocs),
        'per_concept_auprc': auprcs,
        'mean_concept_auprc': np.mean(auprcs)
    }

# ================= INFERENCE =================

def run_inference(model, dataloader, device):
    model.eval()

    results = {
        'mort_probs': [], 'mort_targets': [],
        'ahf_probs': [], 'ahf_targets': [],
        'concept_preds': [], 'concept_targets': []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Running Inference", leave=False):
            image = batch['image'].to(device)
            waveform = batch['waveform'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            cxr_mask = batch['cxr_mod_mask'].to(device)
            ecg_mask = batch['ecg_mod_mask'].to(device)
            ehr_mask = batch['ehr_mod_mask'].to(device)
            ehr_feat_mask = batch['ehr_mask'].to(device)

            outputs = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask)

            results['mort_probs'].extend(torch.sigmoid(outputs['mortality_logits']).cpu().numpy())
            results['mort_targets'].extend(batch['target_mortality'].numpy())

            results['ahf_probs'].extend(torch.sigmoid(outputs['ahf_logits']).cpu().numpy())
            results['ahf_targets'].extend(batch['target_ahf'].numpy())

            c_preds = torch.cat([
                torch.sigmoid(outputs['cxr_concept_logits']),
                torch.sigmoid(outputs['ecg_concept_logits']),
                torch.sigmoid(outputs['ehr_concept_logits'])
            ], dim=1).cpu().numpy()

            c_targets = torch.cat([
                batch['cxr_concepts'],
                batch['ecg_concepts'],
                batch['ehr_concepts']
            ], dim=1).numpy()

            results['concept_preds'].extend(c_preds)
            results['concept_targets'].extend(c_targets)

    for k in results:
        results[k] = np.array(results[k])

    return results

# ================= MAIN =================

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    logger = setup_logger(args.save_dir, args.exp_name)

    logger.info(f"--- EVALUATING EXPERIMENT: {args.experiment} ---")

    train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

    _, val_loader, test_loader, _ = get_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=args.batch_size, dropout_config=None
    )

    if args.experiment == "cxr_only":
        model = UniModalCXR().to(device)
    elif args.experiment == "ecg_only":
        model = UniModalECG().to(device)
    elif args.experiment == "ehr_only":
        model = UniModalEHR().to(device)
    else:
        raise ValueError

    model.load_state_dict(torch.load(os.path.join("/home/azwad/Works/Multimodal-CBM/Implementation/checkpoints_3/run_cxr_only", "best_final_weights.pth"), map_location=device))

    # Threshold tuning
    val_results = run_inference(model, val_loader, device)
    t_mort = find_best_threshold(val_results['mort_targets'], val_results['mort_probs'])
    t_ahf  = find_best_threshold(val_results['ahf_targets'], val_results['ahf_probs'])

    # Test evaluation
    test_results = run_inference(model, test_loader, device)

    mort_metrics = calculate_final_metrics(test_results['mort_targets'], test_results['mort_probs'], t_mort)
    ahf_metrics  = calculate_final_metrics(test_results['ahf_targets'], test_results['ahf_probs'], t_ahf)

    # NEW: concept metrics
    concept_metrics = compute_concept_metrics(
        test_results['concept_targets'],
        test_results['concept_preds']
    )

    # Logging
    logger.info("\n================ FINAL TEST RESULTS ================")

    logger.info("MORTALITY PREDICTION:")
    for k, v in mort_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("\nACUTE HEART FAILURE (AHF) PREDICTION:")
    for k, v in ahf_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("\nCONCEPT PREDICTION METRICS:")
    logger.info(f"  Mean AUROC: {concept_metrics['mean_concept_auroc']:.4f}")
    logger.info(f"  Mean AUPRC: {concept_metrics['mean_concept_auprc']:.4f}")

    logger.info("\nPer-Concept AUROC:")
    for i, v in enumerate(concept_metrics['per_concept_auroc']):
        logger.info(f"  Concept {i}: {v:.4f}")

    logger.info("====================================================\n")

if __name__ == "__main__":
    main()