import os
import torch
import pandas as pd
import numpy as np
import wfdb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Dict

# ================= CONFIG =================

CXR_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_cxr'
ECG_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_ecg'
METADATA_CSV = '/home/azwad/Works/Multimodal-CBM/Datasets/CXR/mimic-cxr-2.0.0-metadata.csv'

# ================= CONCEPTS =================


ECG_CONCEPTS = [
    "myocardial_infarction", "left_axis_deviation", "myocardial_ischemia",
    "sinus_bradycardia", "atrial_fibrillation", "low_qrs_voltage",
    "sinus_tachycardia", "right_bundle_branch_block",
    "left_ventricular_hypertrophy", "prolonged_qt_interval",
    "left_bundle_branch_block", "iv_conduction_defect",
    "pacemaker_rhythm", "left_atrial_abnormality"
]



# ================= DATASET =================

class ECGConceptDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        # ===== RESTORED: CXR VIEW METADATA =====
        try:
            meta = pd.read_csv(METADATA_CSV, usecols=['dicom_id', 'ViewPosition'])
            self.view_map = dict(zip(meta['dicom_id'], meta['ViewPosition']))
        except Exception:
            self.view_map = {}


        
        self.ecg_concepts_all = torch.tensor(self.data[ECG_CONCEPTS].values.astype(float), dtype=torch.float32)
        self.mort_targets = torch.tensor(
            self.data['label_mortality'].values, dtype=torch.long)
        self.ahf_targets  = torch.tensor(
            self.data['label_ahf'].values, dtype=torch.long)
        
        self._log_concept_prevalence()

    def _log_concept_prevalence(self):
        """Prints per-concept positive rates so you can spot severe imbalance immediately."""
        print("\n--- Concept Prevalence (positive rate) ---")
        for i, c in enumerate(ECG_CONCEPTS):
            rate = self.ecg_concepts_all[:, i].mean().item()
            print(f"  {c:<30s}: {rate:.3f}")
        print("------------------------------------------\n")



    def set_dropout_probs(self, probs: Dict[str, float]):
        self.current_p = probs




    def __len__(self):
        return len(self.data)



    # ================= __getitem__ =================

    def __getitem__(self, idx):
        

        # ===== ECG =====
        row = self.data.iloc[idx] # iloc once just to get IDs is okay, but avoid for math
        ecg_tensor = torch.zeros((12, 5000))
        

        FAST_ECG_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_ecg_npy'

        if pd.notna(row['ecg_study_id']):
            eid = str(int(row['ecg_study_id']))
            npy_path = os.path.join(FAST_ECG_DIR, f"{eid}.npy")
            
            if os.path.exists(npy_path):
                # np.load is nearly instantaneous
                signal = np.load(npy_path)
                

                ecg_tensor = torch.tensor(signal, dtype=torch.float32)
                

        # ===== EHR =====
        ecg_concepts = self.ecg_concepts_all[idx]





        return {
            
            'waveform': ecg_tensor,
            'concepts': ecg_concepts,
            'target_mortality': self.mort_targets[idx],  # scalar long
            'target_ahf':       self.ahf_targets[idx],   # scalar long
        }

# ================= COLLATE =================

def clean_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ================= DATALOADER FACTORY =================

def get_ecg_concept_dataloaders(train_csv, val_csv, test_csv, batch_size=32):



    train_ds = ECGConceptDataset(train_csv)
    val_ds   = ECGConceptDataset(val_csv)
    test_ds  = ECGConceptDataset(test_csv)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4,               # Hires more background assistants
        pin_memory=True,             # Puts data directly into fast lane for GPU transfer
        prefetch_factor=2,           # Tells assistants to grab 2 batches in advance
        persistent_workers=True,     # Keeps assistants alive between epochs
        collate_fn=clean_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=clean_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=clean_collate
    )

    return train_loader, val_loader, test_loader


# ================= QUICK SANITY CHECK =================

if __name__ == "__main__":
    train_csv_path = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv_path   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv_path  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"




    

    train_loader, val_loader, test_loader , train_ds = get_ecg_concept_dataloaders(
        train_csv_path, val_csv_path, test_csv_path, batch_size=32
    )

    #During testing
    #train_ds.set_dropout_probs({'cxr': 0.01, 'ehr': 0.90, 'ecg': 0.01})
    
    #In training loop
    #train_ds.set_dropout_probs(scheduler.current_probs)
    #scheduler.step()

    for batch in train_loader:
        if batch is None:
            print("Empty batch — all samples were None.")
            continue

        print("waveform:      ", batch['waveform'].shape)        # (B, 12, 5000)
        print("ecg_concepts:  ", batch['concepts'].shape)    # (B, 14)




        




        break