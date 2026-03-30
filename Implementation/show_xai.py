import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved results
data = torch.load('xai_results.pt')

def plot_xai_dashboard(data):
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # --- 1. Modality Contributions (Layer 1) ---
    ax1 = fig.add_subplot(gs[0, 0])
    modalities = list(data['contributions'].keys())
    mort_vals = [data['contributions'][m]['mortality'] for m in modalities]
    ahf_vals = [data['contributions'][m]['ahf'] for m in modalities]
    
    x = np.arange(len(modalities))
    ax1.bar(x - 0.2, mort_vals, 0.4, label='Mortality', color='skyblue')
    ax1.bar(x + 0.2, ahf_vals, 0.4, label='AHF', color='salmon')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modalities)
    ax1.set_title("Layer 1: Modality Contribution to Prediction")
    ax1.legend()

    # --- 2. Top Concept Importance (Layer 2) ---
    ax2 = fig.add_subplot(gs[0, 1])
    # Import list of concept names from your config or define here
    concepts = [
        "lung_opacity", "pleural_effusion", "support_tube", "heart_enlarged", "lung_atelectasis", 
        "pulmonary_edema", "support_line", "vascular_congestion", "lungs_hyperinflated", 
        "apical_pneumothorax", "hemidiaphragm_elevation", "rib_fractures", "interstitial_markings", 
        "volume_loss", "atrium_leads", "mi", "lad", "ischemia", "bradycardia", "afib", "low_qrs", 
        "tachy", "rbbb", "lvh", "qt", "lbbb", "ivcd", "pacer", "laa", 'tachy', 'tachy_p', 'temp', 
        'hypotension', 'hypoxia', 'mental', 'creat', 'bili', 'platelet', 'wbc', 'lactate', 'bnp', 'trop'
    ]
    importance = data['importance_ahf']
    top_idx = np.argsort(np.abs(importance))[-10:]
    ax2.barh([concepts[i] for i in top_idx], importance[top_idx], color='teal')
    ax2.set_title("Layer 2: Top 10 Influential Concepts (AHF)")

    # --- 3. Cross-Modal Attention (Layer 3) ---
    # We'll plot the CXR -> EHR attention as an example
    ax3 = fig.add_subplot(gs[1, :])
    attn_matrix = data['attention']['cxr→ehr']
    sns.heatmap(attn_matrix, cmap='viridis', ax=ax3, 
                xticklabels=concepts[29:], # EHR names
                yticklabels=concepts[:15])  # CXR names
    ax3.set_title("Layer 3: Cross-Modal Attention (CXR Concepts attending to EHR Data)")

    # --- 4. Counterfactual Impact (Layer 4) ---
    ax4 = fig.add_subplot(gs[2, :])
    cf = data['counterfactuals']
    names = [c['concept'] for c in cf]
    changes = [c['change'] for c in cf]
    colors = ['red' if x < 0 else 'green' for x in changes]
    ax4.bar(names, changes, color=colors)
    ax4.set_ylabel("Change in Probability")
    ax4.set_title("Layer 4: Counterfactual Impact (Flip Concept -> Prediction Change)")

    plt.suptitle("Multimodal-CBM Explainability Report", fontsize=20, y=0.95)
    plt.savefig('xai_visual_report.png')
    plt.show()

plot_xai_dashboard(data)