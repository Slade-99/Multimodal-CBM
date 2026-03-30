import torch
import numpy as np
import os

# ============================================================
# CONFIG & CONCEPT NAMES
# ============================================================
CXR_CONCEPTS = [
    "Lung Opacity", "Pleural Effusion", "Support Tube", "Heart Enlarged", 
    "Lung Atelectasis", "Pulmonary Edema", "Support Line", "Vascular Congestion", 
    "Lungs Hyperinflated", "Apical Pneumothorax", "Hemidiaphragm Elevation", 
    "Rib Fractures", "Interstitial Markings", "Volume Loss", "Atrium Leads"
]

ECG_CONCEPTS = [
    "Myocardial Infarction", "Left Axis Deviation", "Myocardial Ischemia",
    "Sinus Bradycardia", "Atrial Fibrillation", "Low QRS Voltage",
    "Sinus Tachycardia", "Right Bundle Branch Block", "LV Hypertrophy", 
    "Prolonged QT Interval", "Left Bundle Branch Block", "IV Conduction Defect",
    "Pacemaker Rhythm", "Left Atrial Abnormality"
]

EHR_CONCEPTS = [
    'Tachycardia', 'Tachypnea', 'Abnormal Temp', 'Hypotension', 'Hypoxia',
    'Altered Mental Status', 'Elevated Creatinine', 'Elevated Bilirubin',
    'Thrombocytopenia', 'Abnormal WBC', 'Hyperlactatemia',
    'Elevated BNP', 'Elevated Troponin'
]

ALL_CONCEPTS = CXR_CONCEPTS + ECG_CONCEPTS + EHR_CONCEPTS

def generate_textual_report(file_path):
    if not os.path.exists(file_path):
        return f"Error: {file_path} not found. Please run the XAI script first."

    # Load results - using weights_only=False because .pt contains dicts/numpy
    data = torch.load(file_path, weights_only=False)
    
    report = []
    report.append("="*70)
    report.append("          CLINICAL EXPLAINABILITY REPORT: MULTIMODAL-CBM")
    report.append("="*70)
    report.append(f"Source Data: {file_path}")
    report.append("")

    # --- Layer 1: Modality Contributions ---
    report.append("LAYER 1: MODALITY CONTRIBUTION ANALYSIS")
    report.append("-" * 45)
    contribs = data['contributions']
    for mod in ['CXR', 'ECG', 'EHR']:
        m_impact = contribs[mod]['mortality']
        a_impact = contribs[mod]['ahf']
        report.append(f"{mod:4} | Mortality Impact: {m_impact:+.4f} | AHF Impact: {a_impact:+.4f}")
    report.append("\nInterpretation: Higher positive values indicate the modality increased")
    report.append("the risk score for this specific patient.")
    report.append("")

    # --- Layer 2: Top Concept Importance ---
    report.append("LAYER 2: TOP 10 INFLUENTIAL CONCEPTS (AHF Task)")
    report.append("-" * 45)
    importance = data['importance_ahf']
    # Get top 10 indices by absolute magnitude
    top_idx = np.argsort(np.abs(importance))[::-1][:10]
    for i in top_idx:
        report.append(f" - {ALL_CONCEPTS[i]:30} Importance Score: {importance[i]:+.4f}")
    report.append("")

    # --- Layer 3: Cross-Modal Attention ---
    report.append("LAYER 3: KEY CROSS-MODAL ATTENTION (Strongest Weights)")
    report.append("-" * 45)
    attn = data['attention']
    
    for pair, matrix in attn.items():
        src_mod, tgt_mod = pair.split('→')
        src_list = CXR_CONCEPTS if src_mod == 'cxr' else ECG_CONCEPTS if src_mod == 'ecg' else EHR_CONCEPTS
        tgt_list = CXR_CONCEPTS if tgt_mod == 'cxr' else ECG_CONCEPTS if tgt_mod == 'ecg' else EHR_CONCEPTS
        
        # Matrix is (Q, K). Find max
        idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        weight = matrix[idx]
        report.append(f"{pair:7}: {src_list[idx[0]]:22} -> {tgt_list[idx[1]]:22} (W: {weight:.4f})")
    report.append("")

    # --- Layer 4: Counterfactual Analysis ---
    report.append("LAYER 4: COUNTERFACTUAL 'WHAT-IF' SCENARIOS")
    report.append("-" * 45)
    cf_results = data['counterfactuals']
    
    # Corrected keys: orig_p, orig_val, flip_val, change
    for cf in cf_results:
        direction = "REDUCING" if cf['change'] < 0 else "INCREASING"
        report.append(f"If Concept '{cf['concept']}' is flipped ({cf['orig_val']:.2f} -> {cf['flip_val']:.2f}):")
        report.append(f"   Probability Shift: {cf['change']:+.4f} ({direction} risk)")
    
    report.append("")
    report.append("="*70)
    report.append("SUMMARY: The model prediction is primarily driven by " + 
                  ("EHR" if contribs['EHR']['ahf'] > contribs['CXR']['ahf'] else "CXR") + 
                  " markers.")
    report.append("="*70)
    
    return "\n".join(report)

if __name__ == "__main__":
    # Update this path to your actual results file
    results_path = '/home/azwad/Works/Multimodal-CBM/xai_results.pt'
    
    final_report = generate_textual_report(results_path)
    print(final_report)
    
    # Save to file
    output_file = 'clinical_xai_summary.txt'
    with open(output_file, 'w') as f:
        f.write(final_report)
    print(f"\nReport saved to: {os.path.abspath(output_file)}")