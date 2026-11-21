# clinical_validity_full.py

import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import ttest_ind

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

# ================= CONCEPT NAMES =================
CXR_CONCEPTS = [
    "lung_opacity", "pleural_effusion", "support_tube",
    "heart_enlarged", "lung_atelectasis", "pulmonary_edema",
    "support_line", "vascular_congestion", "lungs_hyperinflated",
    "apical_pneumothorax", "hemidiaphragm_elevation", "rib_fractures",
    "interstitial_markings", "volume_loss", "atrium_leads"
]

ECG_CONCEPTS = [
    "myocardial_infarction", "left_axis_deviation", "myocardial_ischemia",
    "sinus_bradycardia", "atrial_fibrillation", "low_qrs_voltage",
    "sinus_tachycardia", "right_bundle_branch_block",
    "left_ventricular_hypertrophy", "prolonged_qt_interval",
    "left_bundle_branch_block", "iv_conduction_defect",
    "pacemaker_rhythm", "left_atrial_abnormality"
]

EHR_CONCEPTS = [
    'tachycardia', 'tachypnea', 'abnormal_temp', 'hypotension', 'hypoxia',
    'altered_mental', 'elevated_creatinine', 'elevated_bilirubin',
    'thrombocytopenia', 'abnormal_wbc', 'hyperlactatemia',
    'elevated_bnp', 'elevated_troponin'
]

ALL_CONCEPTS = CXR_CONCEPTS + ECG_CONCEPTS + EHR_CONCEPTS
MODALITY = (["CXR"] * 15) + (["ECG"] * 14) + (["EHR"] * 13)

# ================= COLLECT ACTIVATIONS =================
ahf_pos = []
ahf_neg = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Collecting activations"):
        if batch is None:
            continue

        image        = batch['image'].to(device)
        waveform     = batch['waveform'].to(device)
        ehr_features = batch['ehr_features'].to(device)
        cxr_mask     = batch['cxr_mod_mask'].to(device)
        ecg_mask     = batch['ecg_mod_mask'].to(device)
        ehr_mask     = batch['ehr_mod_mask'].to(device)
        labels       = batch['target_ahf'].numpy()

        outputs = model(image, waveform, ehr_features,
                        cxr_mask, ecg_mask, ehr_mask)

        # 🔥 IMPORTANT: convert to probabilities
        cxr_r = torch.sigmoid(outputs['cxr_attended'])
        ecg_r = torch.sigmoid(outputs['ecg_attended'])
        ehr_r = torch.sigmoid(outputs['ehr_attended'])

        fused = torch.cat([cxr_r, ecg_r, ehr_r], dim=1).cpu().numpy()

        ahf_pos.append(fused[labels == 1])
        ahf_neg.append(fused[labels == 0])

# Stack
pos = np.concatenate(ahf_pos, axis=0)
neg = np.concatenate(ahf_neg, axis=0)

# ================= METRICS =================
pos_mean = pos.mean(0)
neg_mean = neg.mean(0)
diff     = pos_mean - neg_mean

# Effect size (Cohen's d)
pooled_std = np.sqrt((pos.std(0)**2 + neg.std(0)**2) / 2 + 1e-8)
effect_size = diff / pooled_std

# Statistical significance
p_values = []
for i in range(42):
    _, p = ttest_ind(pos[:, i], neg[:, i], equal_var=False)
    p_values.append(p)
p_values = np.array(p_values)

# ================= SORT =================
indices = np.argsort(diff)[::-1]

# ================= PRINT TABLE =================
print("\nTop Clinically Relevant Concepts for AHF (Sorted by Δ Activation)")
print(f"{'Concept':<35} {'Mod':<5} {'AHF+':>8} {'AHF-':>8} {'Δ':>8} {'d':>8} {'p-value':>12}")
print("-" * 90)

TOP_K = 15

for idx in indices[:TOP_K]:
    print(f"{ALL_CONCEPTS[idx]:<35} {MODALITY[idx]:<5} "
          f"{pos_mean[idx]:>8.3f} {neg_mean[idx]:>8.3f} "
          f"{diff[idx]:>+8.3f} {effect_size[idx]:>8.3f} "
          f"{p_values[idx]:>12.3e}")

# ================= SAVE =================
np.savez("clinical_validity_results.npz",
         concepts=ALL_CONCEPTS,
         modality=MODALITY,
         pos_mean=pos_mean,
         neg_mean=neg_mean,
         diff=diff,
         effect_size=effect_size,
         p_values=p_values)