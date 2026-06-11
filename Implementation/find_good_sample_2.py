# find_good_sample.py
import torch
import pandas as pd
import numpy as np
from dataloader import get_dataloaders, TRIMODAL_CONFIG
from cbm_model import MultimodalCBM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MultimodalCBM().to(device)
model.load_state_dict(torch.load('/home/azwad/Works/Multimodal-CBM/Implementation/checkpoints_2/run_trimodal/best_final_weights.pth', map_location=device))
model.eval()

train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"
val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"

# Ensure batch_size=1 so we can map indices directly to the CSV rows
train_loader, val_loader, test_loader, _ = get_dataloaders(
    train_csv, val_csv, test_csv, batch_size=1, dropout_config=None
)

candidates = []

print("Searching for the best candidate...")

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if batch is None:
            continue

        # Only consider samples with all three modalities
        if batch['ecg_mod_mask'].item() < 1.0:
            continue

        # Extract data for prediction
        image        = batch['image'].to(device)
        waveform     = batch['waveform'].to(device)
        ehr_features = batch['ehr_features'].to(device)
        cxr_mask     = batch['cxr_mod_mask'].to(device)
        ecg_mask     = batch['ecg_mod_mask'].to(device)
        ehr_mask     = batch['ehr_mod_mask'].to(device)

        # Forward pass
        outputs = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)

        # Probabilities
        mort_prob = torch.sigmoid(outputs['mortality_logits']).item()
        ahf_prob  = torch.sigmoid(outputs['ahf_logits']).item()

        # Ground Truths
        mort_true = batch['target_mortality'].item()
        ahf_true  = batch['target_ahf'].item()
        
        # --- NEW: Extract Identifiers ---
        # Note: Adjust 'subject_id' or 'stay_id' based on your CSV column names
        patient_id = batch.get('study_id', batch.get('subject_id', "Unknown"))
        if torch.is_tensor(patient_id):
            patient_id = patient_id.item()

        # Accuracy check
        mort_correct = (mort_prob > 0.5) == mort_true
        ahf_correct  = (ahf_prob  > 0.5) == ahf_true

        # Criteria: Correct, high confidence on AHF, and AHF positive
        if mort_correct and ahf_correct and ahf_true == 1 and ahf_prob > 0.7:
            candidates.append({
                'study_id': patient_id,
                'test_loader_idx': i,
                'mort_prob': mort_prob,
                'ahf_prob':  ahf_prob,
                'mort_true': mort_true,
                'ahf_true':  ahf_true,
                'confidence': ahf_prob, # Focus on AHF confidence
                'batch': batch
            })

if not candidates:
    print("No candidates found matching the criteria.")
else:
    # Pick the most confident correct sample
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    best = candidates[0]

    print("\n--- Selected Sample Details ---")
    print(f"Patient ID:     {best['patient_id']}")
    print(f"Test Index:     {best['test_loader_idx']}")
    print(f"Mortality:      Prob: {best['mort_prob']:.4f} | True: {best['mort_true']}")
    print(f"AHF:            Prob: {best['ahf_prob']:.4f} | True: {best['ahf_true']}")
    
    # Save the full batch for downstream analysis
    torch.save(best['batch'], 'selected_patient_batch_2.pt')
    
    # Save a small text report for quick reference
    with open('selected_patient_info.txt', 'w') as f:
        f.write(f"Patient ID: {best['patient_id']}\n")
        f.write(f"AHF Truth: {best['ahf_true']}\n")
        f.write(f"Mortality Truth: {best['mort_true']}\n")
        f.write(f"AHF Probability: {best['ahf_prob']}\n")

    print("\nFiles 'selected_patient_batch.pt' and 'selected_patient_info.txt' have been saved.")