import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc

# List of the exact experiments run
EXPERIMENTS = ["cxr_only", "ecg_only", "ehr_only", "cxr_ehr", "cxr_ecg", "ehr_ecg", "trimodal"]
CHECKPOINT_DIR = "./Implementation/checkpoints_2"
OUTPUT_DIR = "./combined_plots"

# Clean names for the graph legends
LABELS = {
    "cxr_only": "CXR Only",
    "ecg_only": "ECG Only",
    "ehr_only": "EHR Only",
    "cxr_ehr":  "CXR + EHR",
    "cxr_ecg":  "CXR + ECG",
    "ehr_ecg":  "EHR + ECG",
    "trimodal": "Trimodal (All)"
}

# Colors to easily distinguish the lines
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

def load_data():
    """Loads the .npz files for all 7 experiments."""
    data = {}
    for exp in EXPERIMENTS:
        filepath = os.path.join(CHECKPOINT_DIR, f"run_{exp}", "plots", "plot_data.npz")
        if os.path.exists(filepath):
            data[exp] = np.load(filepath)
        else:
            print(f"Warning: Data for {exp} not found at {filepath}")
    return data

def plot_combined_roc(data, task_prefix, task_name):
    """Overlays ROC curves for all models on a single plot."""
    plt.figure(figsize=(8, 6))
    
    for (exp, npz), color in zip(data.items(), COLORS):
        fpr = npz[f'{task_prefix}_fpr']
        tpr = npz[f'{task_prefix}_tpr']
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{LABELS[exp]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{task_name} - Receiver Operating Characteristic', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"{task_name.lower().replace(' ', '_')}_combined_roc.png"), dpi=300)
    plt.close()

def plot_combined_pr(data, task_prefix, task_name):
    """Overlays Precision-Recall curves for all models on a single plot."""
    plt.figure(figsize=(8, 6))
    
    for (exp, npz), color in zip(data.items(), COLORS):
        precision = npz[f'{task_prefix}_precision']
        recall = npz[f'{task_prefix}_recall']
        # Calculate Area Under PR curve
        # Note: PR curves are sorted backwards by default, so we sort them to compute area safely
        sorted_indices = np.argsort(recall)
        pr_auc = auc(recall[sorted_indices], precision[sorted_indices])
        plt.plot(recall, precision, color=color, lw=2, label=f'{LABELS[exp]} (AUC = {pr_auc:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'{task_name} - Precision-Recall Curve', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"{task_name.lower().replace(' ', '_')}_combined_pr.png"), dpi=300)
    plt.close()

def plot_combined_calibration(data, task_prefix, task_name):
    """Overlays Calibration curves (Reliability Diagrams) for all models."""
    plt.figure(figsize=(8, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    for (exp, npz), color in zip(data.items(), COLORS):
        prob_true = npz[f'{task_prefix}_ptrue']
        prob_pred = npz[f'{task_prefix}_ppred']
        plt.plot(prob_pred, prob_true, "s-", color=color, label=f'{LABELS[exp]}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'{task_name} - Reliability Diagram', fontsize=14)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"{task_name.lower().replace(' ', '_')}_combined_calibration.png"), dpi=300)
    plt.close()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_data()
    
    if not data:
        print("No data found. Ensure the experiments have finished running.")
        return

    print("Generating Combined Plots for Mortality...")
    plot_combined_roc(data, "mort", "Mortality")
    plot_combined_pr(data, "mort", "Mortality")
    plot_combined_calibration(data, "mort", "Mortality")
    
    print("Generating Combined Plots for AHF...")
    plot_combined_roc(data, "ahf", "Acute Heart Failure")
    plot_combined_pr(data, "ahf", "Acute Heart Failure")
    plot_combined_calibration(data, "ahf", "Acute Heart Failure")
    
    print(f"All combined plots successfully saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()