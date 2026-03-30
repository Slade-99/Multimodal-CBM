# faithfulness_eval.py
"""
Two standard faithfulness tests:

1. Concept removal test (sufficiency):
   Remove the top-k most important concepts and measure
   how much the prediction drops. A faithful explanation
   should cause large drops when important concepts are removed.

2. Random baseline comparison:
   Removing random concepts should cause smaller drops
   than removing the important ones identified by the model.
"""
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from cbm_model import MultimodalCBM
from dataloader import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultimodalCBM().to(device)
model.load_state_dict(torch.load('/home/azwad/Works/Multimodal-CBM/Implementation/checkpoints_2/run_trimodal/best_final_weights.pth', map_location=device))
model.eval()
train_csv =    "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"
val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
_, _, test_loader, _ = get_dataloaders(
    train_csv, val_csv, test_csv, batch_size=32, dropout_config=None
)

def evaluate_with_concept_mask(model, loader, mask_indices, task='ahf', device='cuda'):
    """
    Runs inference with specified concept indices zeroed out.
    Returns AUROC on the test set.
    """
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            if batch is None:
                continue
            image        = batch['image'].to(device)
            waveform     = batch['waveform'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            cxr_mask     = batch['cxr_mod_mask'].to(device)
            ecg_mask     = batch['ecg_mod_mask'].to(device)
            ehr_mask     = batch['ehr_mod_mask'].to(device)

            outputs = model(image, waveform, ehr_features,
                            cxr_mask, ecg_mask, ehr_mask)

            # Get fused concept vector and zero out specified indices
            cxr_r = outputs['cxr_attended']
            ecg_r = outputs['ecg_attended']
            ehr_r = outputs['ehr_attended']
            fused = torch.cat([cxr_r, ecg_r, ehr_r], dim=1)

            fused_masked = fused.clone()
            fused_masked[:, mask_indices] = 0.0

            if task == 'ahf':
                prob = torch.sigmoid(model.ahf_head(fused_masked))
                labels = batch['target_ahf']
            else:
                prob = torch.sigmoid(model.mortality_head(fused_masked))
                labels = batch['target_mortality']

            all_probs.extend(prob.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return roc_auc_score(all_labels, all_probs)


def run_faithfulness_evaluation(model, loader, task='ahf', k_values=[1, 3, 5, 10]):
    """
    For each k in k_values:
    - Remove top-k important concepts → measure AUROC drop
    - Remove random k concepts (average over 10 runs) → measure AUROC drop
    Faithfulness = important removal causes larger drop than random
    """
    # First get importance scores across the full test set
    all_importance = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing importance"):
            if batch is None:
                continue
            image        = batch['image'].to(device)
            waveform     = batch['waveform'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            cxr_mask     = batch['cxr_mod_mask'].to(device)
            ecg_mask     = batch['ecg_mod_mask'].to(device)
            ehr_mask     = batch['ehr_mod_mask'].to(device)

            outputs = model(image, waveform, ehr_features,
                            cxr_mask, ecg_mask, ehr_mask)
            cxr_r = outputs['cxr_attended']
            ecg_r = outputs['ecg_attended']
            ehr_r = outputs['ehr_attended']
            fused = torch.cat([cxr_r, ecg_r, ehr_r], dim=1)
            fused.requires_grad_(True)

            if task == 'ahf':
                logit = model.ahf_head(fused)
            else:
                logit = model.mortality_head(fused)

            logit.sum().backward()
            importance = (fused.grad * fused).abs().mean(0).detach().cpu().numpy()
            all_importance.append(importance)

    mean_importance = np.stack(all_importance).mean(0)
    ranked_indices  = np.argsort(mean_importance)[::-1]

    # Baseline AUROC (no masking)
    baseline_auroc = evaluate_with_concept_mask(model, loader, [], task)
    print(f"Baseline AUROC ({task}): {baseline_auroc:.4f}")

    results = []
    for k in k_values:
        # Important concepts removed
        top_k_idx    = ranked_indices[:k].tolist()
        imp_auroc    = evaluate_with_concept_mask(model, loader, top_k_idx, task)
        imp_drop     = baseline_auroc - imp_auroc

        # Random concepts removed (10 trials)
        rand_drops = []
        for _ in range(10):
            rand_idx  = np.random.choice(42, k, replace=False).tolist()
            rand_auroc = evaluate_with_concept_mask(model, loader, rand_idx, task)
            rand_drops.append(baseline_auroc - rand_auroc)
        rand_drop_mean = np.mean(rand_drops)
        rand_drop_std  = np.std(rand_drops)

        results.append({
            'k':             k,
            'important_drop': imp_drop,
            'random_drop_mean': rand_drop_mean,
            'random_drop_std':  rand_drop_std,
            'faithfulness_ratio': imp_drop / (rand_drop_mean + 1e-8)
        })

        print(f"k={k:2d}: important drop={imp_drop:.4f} | "
              f"random drop={rand_drop_mean:.4f}±{rand_drop_std:.4f} | "
              f"ratio={imp_drop/(rand_drop_mean+1e-8):.2f}x")

    return results, mean_importance


print("Running faithfulness evaluation for AHF task...")
results_ahf, importance_ahf = run_faithfulness_evaluation(
    model, test_loader, task='ahf'
)

print("\nRunning faithfulness evaluation for mortality task...")
results_mort, importance_mort = run_faithfulness_evaluation(
    model, test_loader, task='mortality'
)

torch.save({
    'results_ahf':   results_ahf,
    'results_mort':  results_mort,
    'importance_ahf':  importance_ahf,
    'importance_mort': importance_mort,
}, 'faithfulness_results.pt')