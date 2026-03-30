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
train_csv =    "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"
val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
train_loader, val_loader, test_loader, _ = get_dataloaders(
    train_csv, val_csv, test_csv, batch_size=1, dropout_config=None
)

candidates = []

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        if batch is None:
            continue

        # Only consider samples with all three modalities
        if batch['ecg_mod_mask'].item() < 1.0:
            continue

        image        = batch['image'].to(device)
        waveform     = batch['waveform'].to(device)
        ehr_features = batch['ehr_features'].to(device)
        cxr_mask     = batch['cxr_mod_mask'].to(device)
        ecg_mask     = batch['ecg_mod_mask'].to(device)
        ehr_mask     = batch['ehr_mod_mask'].to(device)

        outputs = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)

        mort_prob = torch.sigmoid(outputs['mortality_logits']).item()
        ahf_prob  = torch.sigmoid(outputs['ahf_logits']).item()

        mort_true = batch['target_mortality'].item()
        ahf_true  = batch['target_ahf'].item()

        mort_correct = (mort_prob > 0.5) == mort_true
        ahf_correct  = (ahf_prob  > 0.5) == ahf_true

        # We want: correct on both, high confidence, AHF positive (more interesting)
        if mort_correct and ahf_correct and ahf_true == 1 and ahf_prob > 0.7:
            candidates.append({
                'batch_idx': i,
                'mort_prob': mort_prob,
                'ahf_prob':  ahf_prob,
                'mort_true': mort_true,
                'ahf_true':  ahf_true,
                'confidence': mort_prob * ahf_prob,
                'batch': batch
            })

# Pick the most confident correct sample
candidates.sort(key=lambda x: x['confidence'], reverse=True)
best = candidates[0]
print(f"Selected sample index: {best['batch_idx']}")
print(f"Mortality prob: {best['mort_prob']:.3f} (true: {best['mort_true']})")
print(f"AHF prob:       {best['ahf_prob']:.3f}  (true: {best['ahf_true']})")

torch.save(best['batch'], 'selected_patient_batch.pt')