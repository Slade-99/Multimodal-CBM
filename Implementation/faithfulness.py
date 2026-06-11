# faithfulness_eval_full.py

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from cbm_model import MultimodalCBM
from dataloader import get_dataloaders

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= LOAD MODEL =================
model = MultimodalCBM().to(device)
model.load_state_dict(torch.load(
    '/home/azwad/Works/Multimodal-CBM/Implementation/checkpoints_2/run_trimodal/best_final_weights.pth',
    map_location=device
))
model.eval()

# ================= LOAD DATA =================
train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

_, _, test_loader, _ = get_dataloaders(
    train_csv, val_csv, test_csv,
    batch_size=32, dropout_config=None
)

# ================= CORE FUNCTIONS =================

def get_fused_concepts(model, batch):
    image        = batch['image'].to(device)
    waveform     = batch['waveform'].to(device)
    ehr_features = batch['ehr_features'].to(device)
    cxr_mask     = batch['cxr_mod_mask'].to(device)
    ecg_mask     = batch['ecg_mod_mask'].to(device)
    ehr_mask     = batch['ehr_mod_mask'].to(device)

    outputs = model(image, waveform, ehr_features,
                    cxr_mask, ecg_mask, ehr_mask)

    fused = torch.cat([
        outputs['cxr_attended'],
        outputs['ecg_attended'],
        outputs['ehr_attended']
    ], dim=1)

    return fused, outputs


# ================= IMPORTANCE =================

def compute_importance(model, loader, task='ahf'):
    all_importance = []

    for batch in tqdm(loader, desc="Computing importance"):
        if batch is None:
            continue

        fused, _ = get_fused_concepts(model, batch)

        # Make leaf tensor
        fused = fused.clone().detach().requires_grad_(True)

        if task == 'ahf':
            logit = model.ahf_head(fused)
        else:
            logit = model.mortality_head(fused)

        model.zero_grad()
        logit.sum().backward()

        # More stable than grad*input
        importance = fused.grad.abs().mean(0).detach().cpu().numpy()
        all_importance.append(importance)

    mean_importance = np.stack(all_importance).mean(0)
    ranked_indices  = np.argsort(mean_importance)[::-1]

    return ranked_indices, mean_importance


# ================= EVALUATION =================

def evaluate(model, loader, indices=None, mode='remove', task='ahf'):
    """
    mode:
        'remove' → comprehensiveness (zero top-k)
        'keep'   → sufficiency (keep only top-k)
    """
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            if batch is None:
                continue

            fused, outputs = get_fused_concepts(model, batch)

            if indices is not None:
                if mode == 'remove':
                    fused[:, indices] = 0.0
                elif mode == 'keep':
                    mask = torch.zeros_like(fused)
                    mask[:, indices] = fused[:, indices]
                    fused = mask

            if task == 'ahf':
                prob = torch.sigmoid(model.ahf_head(fused))
                labels = batch['target_ahf']
            else:
                prob = torch.sigmoid(model.mortality_head(fused))
                labels = batch['target_mortality']

            all_probs.extend(prob.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return roc_auc_score(all_labels, all_probs)


# ================= MAIN PIPELINE =================

def run_faithfulness(model, loader, task='ahf', k_values=[3, 5, 10]):

    print(f"\n=== Running Faithfulness for {task.upper()} ===")

    # 1. Importance
    ranked_indices, importance = compute_importance(model, loader, task)

    # 2. Baseline
    baseline = evaluate(model, loader, indices=None, task=task)
    print(f"Baseline AUROC: {baseline:.4f}")

    results = []

    for k in k_values:
        print(f"\n--- k = {k} ---")

        top_k = ranked_indices[:k].tolist()

        # ===== COMPREHENSIVENESS =====
        imp_removed = evaluate(model, loader, top_k, mode='remove', task=task)
        comp_drop = baseline - imp_removed

        # Random baseline
        rand_drops = []
        for _ in range(5):
            rand_idx = np.random.choice(42, k, replace=False).tolist()
            rand_removed = evaluate(model, loader, rand_idx, mode='remove', task=task)
            rand_drops.append(baseline - rand_removed)

        rand_mean = np.mean(rand_drops)
        rand_std  = np.std(rand_drops)

        # ===== SUFFICIENCY =====
        suff_auc = evaluate(model, loader, top_k, mode='keep', task=task)

        results.append({
            'k': k,
            'baseline': baseline,
            'comprehensiveness_drop': comp_drop,
            'random_drop_mean': rand_mean,
            'random_drop_std': rand_std,
            'faithfulness_ratio': comp_drop / (rand_mean + 1e-8),
            'sufficiency_auc': suff_auc
        })

        print(f"Comprehensiveness drop: {comp_drop:.4f}")
        print(f"Random drop: {rand_mean:.4f} ± {rand_std:.4f}")
        print(f"Faithfulness ratio: {comp_drop/(rand_mean+1e-8):.2f}x")
        print(f"Sufficiency AUC: {suff_auc:.4f}")

    return results, importance


# ================= RUN =================

results_ahf, importance_ahf = run_faithfulness(model, test_loader, task='ahf')
results_mort, importance_mort = run_faithfulness(model, test_loader, task='mortality')

torch.save({
    'results_ahf': results_ahf,
    'results_mort': results_mort,
    'importance_ahf': importance_ahf,
    'importance_mort': importance_mort,
}, 'faithfulness_results_full.pt')