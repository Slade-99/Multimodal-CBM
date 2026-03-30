# xai_four_layers.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from cbm_model import MultimodalCBM

# ============================================================
# CONFIG
# ============================================================

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
    "left_ventricular_hypertrohypertrophy", "prolonged_qt_interval",
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Establish CUDA context immediately to prevent cuBLAS warnings
_ = torch.matmul(torch.randn(1, 1).to(device), torch.randn(1, 1).to(device))

model = MultimodalCBM().to(device)
model.load_state_dict(torch.load('/home/azwad/Works/Multimodal-CBM/Implementation/checkpoints_2/run_trimodal/best_final_weights.pth', map_location=device))
model.eval()

batch = torch.load('/home/azwad/Works/Multimodal-CBM/selected_patient_batch.pt')
image        = batch['image'].to(device)
waveform     = batch['waveform'].to(device)
ehr_features = batch['ehr_features'].to(device)
cxr_mask     = batch['cxr_mod_mask'].to(device)
ecg_mask     = batch['ecg_mod_mask'].to(device)
ehr_mask     = batch['ehr_mod_mask'].to(device)

# ============================================================
# LAYER 1 — MODALITY CONTRIBUTION
# ============================================================

def get_modality_contributions(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask):
    with torch.no_grad():
        base = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)
        base_m = torch.sigmoid(base['mortality_logits']).item()
        base_a = torch.sigmoid(base['ahf_logits']).item()

        z_cxr = model(torch.zeros_like(image), waveform, ehr_features, torch.zeros_like(cxr_mask), ecg_mask, ehr_mask)
        z_ecg = model(image, torch.zeros_like(waveform), ehr_features, cxr_mask, torch.zeros_like(ecg_mask), ehr_mask)
        z_ehr = model(image, waveform, torch.zeros_like(ehr_features), cxr_mask, ecg_mask, torch.zeros_like(ehr_mask))

    return {
        'CXR': {'mortality': base_m - torch.sigmoid(z_cxr['mortality_logits']).item(), 'ahf': base_a - torch.sigmoid(z_cxr['ahf_logits']).item()},
        'ECG': {'mortality': base_m - torch.sigmoid(z_ecg['mortality_logits']).item(), 'ahf': base_a - torch.sigmoid(z_ecg['ahf_logits']).item()},
        'EHR': {'mortality': base_m - torch.sigmoid(z_ehr['mortality_logits']).item(), 'ahf': base_a - torch.sigmoid(z_ehr['ahf_logits']).item()},
    }, base_m, base_a

# ============================================================
# LAYER 2 — CONCEPT FEATURE CONTRIBUTIONS
# ============================================================

def get_concept_contributions(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, task='mortality'):
    model.zero_grad()
    with torch.set_grad_enabled(True):
        c_logits = model.cxr_encoder(image)
        w_logits = model.ecg_encoder(waveform)
        e_logits = model.ehr_encoder(ehr_features)

        c_p = torch.sigmoid(c_logits) * cxr_mask.view(-1, 1)
        w_p = torch.sigmoid(w_logits) * ecg_mask.view(-1, 1)
        e_p = torch.sigmoid(e_logits) * ehr_mask.view(-1, 1)

        c_r, w_r, e_r, _, _, _ = model.cross_modal_attention(c_p, w_p, e_p, cxr_mask, ecg_mask, ehr_mask)
        fused = torch.cat([c_r, w_r, e_r], dim=1)
        fused.retain_grad()

        logit = model.mortality_head(fused) if task == 'mortality' else model.ahf_head(fused)
        logit.backward()

        importance = (fused.grad * fused).squeeze(0).detach().cpu().numpy()
        activations = fused.squeeze(0).detach().cpu().numpy()
    return importance, activations

# ============================================================
# LAYER 3 — CROSS-MODAL ATTENTION WEIGHTS (FIXED FOR 1D CASE)
# ============================================================

def get_attention_weights(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask):
    model.eval()
    with torch.no_grad():
        out = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)

    def process_attn(attn_tensor):
        # Shape: (B, heads, Q, K)
        # Average over heads: (B, Q, K)
        avg = attn_tensor.mean(dim=1) 
        # Select first batch: (Q, K)
        res = avg[0].cpu().numpy()
        # If Q=1, result is (K,). We must reshape to (1, K) to avoid IndexError
        if res.ndim == 1:
            res = res.reshape(1, -1)
        return res

    cxr_w = process_attn(out['cxr_attn_weights']) # (Q_cxr, 27)
    ecg_w = process_attn(out['ecg_attn_weights']) # (Q_ecg, 28)
    ehr_w = process_attn(out['ehr_attn_weights']) # (Q_ehr, 29)

    return {
        'cxr→ecg': cxr_w[:, :14], 'cxr→ehr': cxr_w[:, 14:],
        'ecg→cxr': ecg_w[:, :15], 'ecg→ehr': ecg_w[:, 15:],
        'ehr→cxr': ehr_w[:, :15], 'ehr→ecg': ehr_w[:, 15:],
    }

# ============================================================
# LAYER 4 — COUNTERFACTUAL EXPLANATION
# ============================================================

def get_counterfactual(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, task='ahf', top_k=3):
    imp, act = get_concept_contributions(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, task=task)
    with torch.no_grad():
        out = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)
        orig_p = torch.sigmoid(out['mortality_logits'] if task == 'mortality' else out['ahf_logits']).item()

    ranked = np.argsort(np.abs(imp))[::-1]
    results = []
    for idx in ranked[:top_k]:
        mod_act = torch.from_numpy(act).clone().to(device).unsqueeze(0)
        orig_val = act[idx]
        flip_val = 1.0 - orig_val
        mod_act[0, idx] = flip_val
        with torch.no_grad():
            head = model.mortality_head if task == 'mortality' else model.ahf_head
            cf_p = torch.sigmoid(head(mod_act)).item()
        results.append({'concept': ALL_CONCEPTS[idx], 'orig_val': orig_val, 'flip_val': flip_val, 'orig_p': orig_p, 'cf_p': cf_p, 'change': cf_p - orig_p})
    return results

# ============================================================
# EXECUTION
# ============================================================

print("Layer 1: Modality contributions...")
contribs, m_p, a_p = get_modality_contributions(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)

print("\nLayer 2: Concept contributions (AHF task)...")
imp_ahf, acts = get_concept_contributions(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, task='ahf')

print("\nLayer 3: Cross-modal attention weights...")
attn = get_attention_weights(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)

print("\nLayer 4: Counterfactual (AHF task)...")
cf = get_counterfactual(model, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, task='ahf')

torch.save({'contributions': contribs, 'importance_ahf': imp_ahf, 'activations': acts, 'attention': attn, 'counterfactuals': cf}, 'xai_results.pt')
print("\nSuccess: XAI results saved to xai_results.pt")