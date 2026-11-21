import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict
from config_2 import CXR_CONCEPTS, ECG_CONCEPTS, EHR_CONCEPTS
from train_heal import HEALNetCBM
from dataloader import get_dataloaders

# Q1 Journal Styling for plots
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'

CONCEPT_NAMES = {
    "cxr": CXR_CONCEPTS,
    "ecg": ECG_CONCEPTS,
    "ehr": EHR_CONCEPTS
}

def run_q1_explainability_suite(model_weights_path, train_csv, val_csv, test_csv):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(">>> Loading HEALNet-CBM Model...")
    active_modalities = ["cxr", "ecg", "ehr"]
    model = HEALNetCBM(active_modalities=active_modalities).to(device)
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    print(">>> Loading Test Dataset (Batch Size = 32 for fast global processing)...")
    _, _, test_loader, _ = get_dataloaders(
        train_csv, val_csv, test_csv,
        batch_size=32, 
        dropout_config=None
    )

    # ==========================================
    # DATA AGGREGATORS FOR GLOBAL XAI
    # ==========================================
    global_mod_attn = {m: [] for m in active_modalities}
    global_concept_attn = {m: [] for m in active_modalities}
    global_co_attn_list = []
    
    # Counterfactual trackers: {mod: {concept_idx: [list of risk drops]}}
    cf_impacts = {m: defaultdict(list) for m in active_modalities}

    # Tracking a specific patient for the Local Case Study
    case_study_patient = None
    max_mortality_risk = -1.0

    print(">>> Processing Entire Test Set for Global Explainability...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Patients"):
            # Move to device
            image = batch['image'].to(device)
            waveform = batch['waveform'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            
            cxr_mask = batch['cxr_mod_mask'].float().to(device)
            ecg_mask = batch['ecg_mod_mask'].float().to(device)
            ehr_mask = batch['ehr_mod_mask'].float().to(device)
            mod_masks = {"cxr": cxr_mask, "ecg": ecg_mask, "ehr": ehr_mask}

            # Ground Truths
            y_mort = batch['target_mortality'].to(device)

            # --- Base Prediction ---
            concept_probs = model.encode_concepts(image, waveform, ehr_features, mod_masks)
            outputs = model.fuse(concept_probs, mod_masks, return_attention=True)
            
            orig_mort_risk = torch.sigmoid(outputs['mortality_logits'])
            all_layer_attn = outputs['all_layer_attn']

            # --- Extract Local Attentions ---
            # Model methods return batch-averaged attention; to do true global, 
            # we collect the batch averages weighted by batch size, or process directly.
            # Here we use the model's built-in explainability methods.
            mod_a = model.get_modality_attention(all_layer_attn, mod_masks)
            for m in active_modalities:
                if mod_a.get(m, 0) > 0:
                    global_mod_attn[m].append(mod_a[m])
            
            con_a = model.get_concept_attention(all_layer_attn, mod_masks)
            for m in active_modalities:
                if np.sum(con_a.get(m, 0)) > 0:
                    global_concept_attn[m].append(con_a[m])

            co_a = model.get_cross_modal_attention(all_layer_attn, mod_masks, "cxr", "ehr")
            if np.sum(co_a) > 0:
                global_co_attn_list.append(co_a)

            # --- Run Global Counterfactuals (Vectorized for Batch) ---
            for mod in active_modalities:
                probs_tensor = concept_probs[mod] # (B, N_concepts)
                
                for c_idx, c_name in enumerate(CONCEPT_NAMES[mod]):
                    # Find patients in this batch who actually HAVE this concept (>0.5 prob)
                    # AND the modality is present.
                    active_mask = (probs_tensor[:, c_idx] > 0.5) & (mod_masks[mod] == 1.0)
                    
                    if not active_mask.any():
                        continue # No one in this batch has this concept
                    
                    # Intervene: Erase concept from memory
                    modified_probs = {m: p.clone() for m, p in concept_probs.items()}
                    modified_probs[mod][:, c_idx] = 0.0
                    
                    # Rerun fusion
                    new_out = model.fuse(modified_probs, mod_masks)
                    new_mort_risk = torch.sigmoid(new_out['mortality_logits'])
                    
                    # Calculate causal drop in risk ONLY for patients who had the concept
                    risk_drop = orig_mort_risk[active_mask] - new_mort_risk[active_mask]
                    cf_impacts[mod][c_idx].extend(risk_drop.cpu().numpy().tolist())

            # --- Find Best Case Study ---
            # We want a True Positive mortality patient with high risk to show in the paper
            for i in range(len(y_mort)):
                if y_mort[i] == 1 and orig_mort_risk[i].item() > max_mortality_risk:
                    # Check if they have all modalities for a good demo
                    if cxr_mask[i] == 1 and ecg_mask[i] == 1 and ehr_mask[i] == 1:
                        max_mortality_risk = orig_mort_risk[i].item()
                        case_study_patient = {
                            "risk": orig_mort_risk[i].item(),
                            "concepts": {m: concept_probs[m][i].cpu().numpy() for m in active_modalities}
                        }

    # ==========================================
    # AGGREGATE AND PLOT GLOBAL METRICS
    # ==========================================
    print("\n>>> Generating Q1-Standard Visualizations...")

    # 1. Global Modality Contribution (Pie Chart)
    avg_mod_attn = {m: np.mean(vals) for m, vals in global_mod_attn.items() if len(vals) > 0}
    plt.figure(figsize=(6, 6))
    plt.pie(avg_mod_attn.values(), labels=[m.upper() for m in avg_mod_attn.keys()], 
            autopct='%1.1f%%', colors=sns.color_palette("muted"))
    plt.title("Global Modality Contribution\n(Test Set Average)", fontweight='bold')
    plt.savefig("fig_1_global_modality.pdf", bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Global Concept Contribution (Bar Chart)
    avg_con_attn = []
    for mod in active_modalities:
        if len(global_concept_attn[mod]) > 0:
            avg_array = np.mean(global_concept_attn[mod], axis=0)
            for name, score in zip(CONCEPT_NAMES[mod], avg_array):
                avg_con_attn.append((f"[{mod.upper()}] {name}", score))
                
    avg_con_attn.sort(key=lambda x: x[1], reverse=True)
    top_15_concepts = avg_con_attn[:15]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=[x[1] for x in top_15_concepts], y=[x[0] for x in top_15_concepts], palette="viridis")
    plt.title("Top 15 Most Attended Clinical Concepts (Global)", fontweight='bold')
    plt.xlabel("Average Cross-Modal Attention Weight")
    plt.savefig("fig_2_global_concepts.pdf", bbox_inches='tight', dpi=300)
    plt.close()

    # 3. Global Co-Attention (Heatmap)
    if len(global_co_attn_list) > 0:
        avg_co_attn = np.mean(global_co_attn_list, axis=0)
        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_co_attn, xticklabels=CONCEPT_NAMES["ehr"], yticklabels=CONCEPT_NAMES["cxr"], 
                    cmap="Blues", annot=False)
        plt.title("Global Cross-Modal Attention: CXR vs. EHR", fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.savefig("fig_3_global_heatmap.pdf", bbox_inches='tight', dpi=300)
        plt.close()

    # ==========================================
    # PRINT RESULTS FOR THE MANUSCRIPT
    # ==========================================
    
    print("\n" + "="*70)
    print(" TABLE 1: GLOBAL COUNTERFACTUAL IMPACT (Average Treatment Effect)")
    print(" (How much does Mortality Risk drop when this concept is removed?)")
    print("="*70)
    cf_summary = []
    for mod in active_modalities:
        for c_idx, drops in cf_impacts[mod].items():
            if len(drops) > 10: # Only report if it occurred in at least 10 patients
                mean_drop = np.mean(drops)
                cf_summary.append((f"[{mod.upper()}] {CONCEPT_NAMES[mod][c_idx]}", mean_drop, len(drops)))
                
    cf_summary.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Concept':<35} | {'Avg Risk Drop':<15} | {'Patient Count'}")
    print("-" * 70)
    for name, drop, count in cf_summary[:15]: # Top 15 causal concepts
        print(f"{name:<35} | -{drop:.2%}          | {count}")


    print("\n" + "="*70)
    print(" LOCAL CASE STUDY: High-Risk True Positive Patient")
    print(" (Use this in the text to show clinical reasoning)")
    print("="*70)
    if case_study_patient:
        print(f"Base Predicted Mortality Risk: {case_study_patient['risk']:.2%}\n")
        print("Key findings identified by model (Probability > 80%):")
        for mod in active_modalities:
            probs = case_study_patient["concepts"][mod]
            for name, prob in zip(CONCEPT_NAMES[mod], probs):
                if prob > 0.8:
                    print(f"  - [{mod.upper()}] {name} ({prob:.1%} certainty)")
    else:
        print("Could not find a perfect patient for the case study.")

    print("\n>>> All figures saved as high-res PDFs ready for LaTeX insertion!")

if __name__ == "__main__":
    # Update these to your exact server paths
    WEIGHTS_PATH = "/home/azwad/Works/Multimodal-CBM/HealNet/checkpoints_5/run_trimodal/best_fusion_weights.pth" 
    TRAIN_CSV = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    VAL_CSV = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    TEST_CSV = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"
    
    run_q1_explainability_suite(WEIGHTS_PATH, TRAIN_CSV, VAL_CSV, TEST_CSV)