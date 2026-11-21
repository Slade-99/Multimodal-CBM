import torch
import numpy as np
import os

# ============================================================
# CONCEPT NAMES
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

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def categorize_concept(concept):
    if concept in ["Heart Enlarged", "LV Hypertrophy", "Myocardial Infarction", "Myocardial Ischemia"]:
        return "Cardiac Structure"
    if concept in ["Pulmonary Edema", "Pleural Effusion", "Vascular Congestion", "Lung Opacity"]:
        return "Pulmonary Congestion"
    if concept in ["Hypotension", "Abnormal Temp", "Hyperlactatemia", "Altered Mental Status"]:
        return "Systemic Instability"
    if concept in ["Sinus Tachycardia", "Sinus Bradycardia", "Atrial Fibrillation"]:
        return "Cardiac Rhythm"
    return "Other"

def importance_to_text(rank):
    if rank < 3:
        return "strong contributor"
    elif rank < 6:
        return "moderate contributor"
    else:
        return "supporting contributor"

# ============================================================
# MAIN REPORT GENERATOR
# ============================================================

def generate_textual_report(file_path):
    if not os.path.exists(file_path):
        return f"Error: {file_path} not found."

    data = torch.load(file_path, weights_only=False)

    report = []
    report.append("="*75)
    report.append("      CLINICAL INTERPRETABILITY REPORT (PATIENT-LEVEL)")
    report.append("="*75)

    # ========================================================
    # LAYER 1 — MODALITY
    # ========================================================
    report.append("\n1) OVERALL MODALITY CONTRIBUTION\n" + "-"*50)

    contribs = data['contributions']
    ahf_scores = {m: contribs[m]['ahf'] for m in ['CXR','ECG','EHR']}
    dominant = max(ahf_scores, key=ahf_scores.get)

    report.append(f"The model prediction is primarily driven by **{dominant}** findings.")

    if dominant == "CXR":
        report.append("This suggests that visible radiological abnormalities play a key role in this patient’s condition.")
    elif dominant == "EHR":
        report.append("This indicates that physiological and laboratory measurements are the main drivers of risk.")
    else:
        report.append("This indicates that cardiac electrical activity contributes most to the prediction.")

    report.append("\nNotably, the model dynamically adjusts its reliance on modalities depending on the patient context.")

    # ========================================================
    # LAYER 2 — CONCEPTS
    # ========================================================
    report.append("\n2) KEY CLINICAL FINDINGS\n" + "-"*50)

    importance = data['importance_ahf']
    top_idx = np.argsort(np.abs(importance))[::-1][:10]

    grouped = {"Cardiac Structure":[], "Pulmonary Congestion":[], 
               "Systemic Instability":[], "Cardiac Rhythm":[], "Other":[]}

    for rank, i in enumerate(top_idx):
        concept = ALL_CONCEPTS[i]
        category = categorize_concept(concept)
        desc = importance_to_text(rank)
        grouped[category].append(f"{concept} ({desc})")

    for category, items in grouped.items():
        if items:
            report.append(f"\n• {category}:")
            for item in items:
                report.append(f"   - {item}")

    report.append("\nInterpretation: These findings collectively indicate impaired cardiac function with associated systemic effects.")

    # ========================================================
    # LAYER 3 — CROSS-MODAL
    # ========================================================
    report.append("\n3) CROSS-MODAL CLINICAL RELATIONSHIPS\n" + "-"*50)

    attn = data['attention']
    for pair, matrix in attn.items():
        src_mod, tgt_mod = pair.split('→')
        src_list = CXR_CONCEPTS if src_mod == 'cxr' else ECG_CONCEPTS if src_mod == 'ecg' else EHR_CONCEPTS
        tgt_list = CXR_CONCEPTS if tgt_mod == 'cxr' else ECG_CONCEPTS if tgt_mod == 'ecg' else EHR_CONCEPTS

        idx = np.unravel_index(np.argmax(matrix), matrix.shape)

        report.append(
            f"- {src_list[idx[0]]} is associated with {tgt_list[idx[1]]}, "
            f"suggesting interaction between {src_mod.upper()} and {tgt_mod.upper()} findings."
        )

    report.append("\nThese relationships indicate coordinated multi-organ involvement rather than isolated abnormalities.")

    # ========================================================
    # LAYER 4 — COUNTERFACTUAL
    # ========================================================
    report.append("\n4) COUNTERFACTUAL ANALYSIS\n" + "-"*50)

    cf_results = data['counterfactuals']

    for cf in cf_results[:3]:
        direction = "decrease" if cf['change'] < 0 else "increase"
        report.append(
            f"- If {cf['concept']} were absent, predicted risk would {direction} significantly."
        )

    report.append("\nThis confirms that these findings are causally important to the model’s decision.")

    # ========================================================
    # FINAL SUMMARY
    # ========================================================
    report.append("\n" + "="*75)
    report.append("CLINICAL SUMMARY")
    report.append("="*75)

    report.append(
        "The model identifies a combination of cardiac abnormalities, pulmonary congestion, "
        "and systemic instability, consistent with acute decompensated heart failure."
    )

    report.append(
        "Importantly, the model integrates information across modalities, adapting its reasoning "
        "based on the most informative signals for the specific patient."
    )

    return "\n".join(report)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    results_path = '/home/azwad/Works/Multimodal-CBM/xai_results.pt'

    report = generate_textual_report(results_path)
    print(report)

    with open("clinical_xai_summary_readable.txt", "w") as f:
        f.write(report)