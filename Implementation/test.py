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

from config import get_args
from cbm_model import MultimodalCBM
from dataloader import get_dataloaders, ModalityDropoutConfig

# ================= SETUP LOGGING =================

def setup_logger(save_dir, exp_name):
    log_file = os.path.join(save_dir, exp_name, "evaluation_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s', # Keep it clean for evaluation logs
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger()

# ================= METRIC CALCULATIONS =================

def calculate_ece(y_true, y_probs, n_bins=10):
    """Calculates Expected Calibration Error (ECE) robustly."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    # Place probabilities into bins (digitize returns values from 1 to n_bins)
    bin_indices = np.digitize(y_probs, bin_edges, right=False)
    # Ensure probabilities of exactly 1.0 don't create an out-of-bounds bin
    bin_indices = np.clip(bin_indices, 1, n_bins)
    
    ece = 0.0
    total_samples = len(y_probs)
    
    # Iterate through each bin safely
    for bin_idx in range(1, n_bins + 1):
        mask = (bin_indices == bin_idx)
        if np.any(mask): # Only process bins that actually have data
            bin_true = y_true[mask]
            bin_probs = y_probs[mask]
            
            prob_true = np.mean(bin_true)
            prob_pred = np.mean(bin_probs)
            bin_weight = len(bin_true) / total_samples
            
            ece += np.abs(prob_true - prob_pred) * bin_weight
            
    return ece

def find_best_threshold(y_true, y_probs):
    """Searches thresholds from 0.01 to 0.99 to find the best F1 score."""
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
    """Calculates all requested metrics given a locked threshold."""
    y_preds = (y_probs >= threshold).astype(int)
    
    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    ece = calculate_ece(y_true, y_probs)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # True Positive Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # True Negative Rate
    
    return {
        'AUROC': auroc, 'AUPRC': auprc, 'F1': f1, 
        'Sensitivity': sensitivity, 'Specificity': specificity, 'ECE': ece
    }

# ================= INFERENCE LOOP =================

def run_inference(model, dataloader, device):
    """Runs data through the model and collects probabilities and true labels."""
    model.eval()
    
    all_results = {
        'mort_probs': [], 'mort_targets': [],
        'ahf_probs': [], 'ahf_targets': [],
        'concept_preds': [], 'concept_targets': []
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Running Inference", leave=False)):
            image = batch['image'].to(device)
            waveform = batch['waveform'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            cxr_mask = batch['cxr_mod_mask'].to(device)
            ecg_mask = batch['ecg_mod_mask'].to(device)
            ehr_mask = batch['ehr_mod_mask'].to(device)

            outputs = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)
            
            # Target Probabilities
            all_results['mort_probs'].extend(torch.sigmoid(outputs['mortality_logits']).cpu().numpy())
            all_results['mort_targets'].extend(batch['target_mortality'].numpy())
            
            all_results['ahf_probs'].extend(torch.sigmoid(outputs['ahf_logits']).cpu().numpy())
            all_results['ahf_targets'].extend(batch['target_ahf'].numpy())
            
            # Concept Probabilities (Combined)
            c_preds = torch.cat([
                torch.sigmoid(outputs['cxr_concept_logits']),
                torch.sigmoid(outputs['ecg_concept_logits']),
                torch.sigmoid(outputs['ehr_concept_logits'])
            ], dim=1).cpu().numpy()
            
            c_targets = torch.cat([
                batch['cxr_concepts'], batch['ecg_concepts'], batch['ehr_concepts']
            ], dim=1).numpy()
            
            all_results['concept_preds'].extend(c_preds)
            all_results['concept_targets'].extend(c_targets)

    # Convert lists to numpy arrays
    for k in all_results:
        all_results[k] = np.array(all_results[k])
        
    return all_results

# ================= PLOTTING =================

def generate_and_save_plots(results, exp_dir, task_name, true_key, prob_key):
    """Generates ROC, PR, and Calibration curves and saves them."""
    plot_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    y_true = results[true_key]
    y_probs = results[prob_key]
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUROC = {roc_auc_score(y_true, y_probs):.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{task_name} - ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{task_name.lower()}_roc.png"))
    plt.close()

    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.figure()
    plt.plot(recall, precision, label=f'AUPRC = {average_precision_score(y_true, y_probs):.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{task_name} - PR Curve')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{task_name.lower()}_pr.png"))
    plt.close()

    # 3. Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'{task_name} - Reliability Diagram')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{task_name.lower()}_calibration.png"))
    plt.close()
    
    return fpr, tpr, precision, recall, prob_true, prob_pred

# ================= MAIN =================

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    logger = setup_logger(args.save_dir, args.exp_name)
    
    logger.info(f"--- EVALUATING EXPERIMENT: {args.experiment} ---")

    # --- 1. Load Data ---
    active_mods = []
    if 'cxr' in args.experiment or args.experiment == 'trimodal': active_mods.append('cxr')
    if 'ecg' in args.experiment or args.experiment == 'trimodal': active_mods.append('ecg')
    if 'ehr' in args.experiment or args.experiment == 'trimodal': active_mods.append('ehr')

    exp_config = ModalityDropoutConfig(
        p_max={'cxr': 0.05, 'ehr': 0.30, 'ecg': 0.30}, 
        active_modalities=active_mods
    )

    train_csv =    "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    _, val_loader, test_loader, _ = get_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=args.batch_size, dropout_config=exp_config
    )

    # --- 2. Load Model ---
    model = MultimodalCBM().to(device)
    weights_path = os.path.join(exp_dir, "best_final_weights.pth")
    if not os.path.exists(weights_path):
        logger.error(f"Could not find weights at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    logger.info("Model weights loaded successfully.")

    # --- 3. Validation Phase (Find Thresholds) ---
    logger.info("\n[Phase 1] Tuning Thresholds on Validation Set...")
    val_results = run_inference(model, val_loader, device)
    
    best_thresh_mort = find_best_threshold(val_results['mort_targets'], val_results['mort_probs'])
    best_thresh_ahf = find_best_threshold(val_results['ahf_targets'], val_results['ahf_probs'])
    
    logger.info(f"Locked Mortality Threshold: {best_thresh_mort:.2f}")
    logger.info(f"Locked AHF Threshold: {best_thresh_ahf:.2f}")

    # --- 4. Test Phase (Final Evaluation) ---
    logger.info("\n[Phase 2] Evaluating on Test Set...")
    test_results = run_inference(model, test_loader, device)
    
    mort_metrics = calculate_final_metrics(test_results['mort_targets'], test_results['mort_probs'], best_thresh_mort)
    ahf_metrics = calculate_final_metrics(test_results['ahf_targets'], test_results['ahf_probs'], best_thresh_ahf)
    
    # Calculate overall concept accuracy (using 0.5 as standard threshold for concepts)
    concept_preds_binary = (test_results['concept_preds'] >= 0.5).astype(int)
    concept_accuracy = (concept_preds_binary == test_results['concept_targets']).mean()

    # Log Final Results
    logger.info("\n================ FINAL TEST RESULTS ================")
    logger.info("MORTALITY PREDICTION:")
    for k, v in mort_metrics.items(): logger.info(f"  {k}: {v:.4f}")
        
    logger.info("\nACUTE HEART FAILURE (AHF) PREDICTION:")
    for k, v in ahf_metrics.items(): logger.info(f"  {k}: {v:.4f}")
        
    logger.info(f"\nOVERALL CONCEPT PREDICTION ACCURACY: {concept_accuracy:.4f}")
    logger.info("====================================================\n")

    # --- 5. Generate Plots & Save Data ---
    logger.info("[Phase 3] Generating Plots...")
    m_fpr, m_tpr, m_pr, m_re, m_ptrue, m_ppred = generate_and_save_plots(
        test_results, exp_dir, "Mortality", 'mort_targets', 'mort_probs'
    )
    
    a_fpr, a_tpr, a_pr, a_re, a_ptrue, a_ppred = generate_and_save_plots(
        test_results, exp_dir, "AHF", 'ahf_targets', 'ahf_probs'
    )
    
    # Save the raw arrays so you can plot all 7 experiments together later
    np.savez(
        os.path.join(exp_dir, "plots", "plot_data.npz"),
        mort_fpr=m_fpr, mort_tpr=m_tpr, mort_precision=m_pr, mort_recall=m_re, mort_ptrue=m_ptrue, mort_ppred=m_ppred,
        ahf_fpr=a_fpr, ahf_tpr=a_tpr, ahf_precision=a_pr, ahf_recall=a_re, ahf_ptrue=a_ptrue, ahf_ppred=a_ppred
    )
    logger.info(f"Plots and raw curve data saved to {os.path.join(exp_dir, 'plots')}")

if __name__ == "__main__":
    main()