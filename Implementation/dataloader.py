import os
import torch
import pandas as pd
import numpy as np
import wfdb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from modality_dropout import ModalityDropoutConfig , ModalityDropoutScheduler, apply_modality_dropout
from typing import Dict

# ================= CONFIG =================

CXR_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_cxr'
ECG_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_ecg'
METADATA_CSV = '/home/azwad/Works/Multimodal-CBM/Datasets/CXR/mimic-cxr-2.0.0-metadata.csv'
TRIMODAL_CONFIG = ModalityDropoutConfig(
    p_max={'cxr': 0.05, 'ehr': 0.30, 'ecg': 0.30},
    active_modalities=['cxr', 'ehr', 'ecg']
)
# ================= CONCEPTS =================

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

# ================= FEATURES =================

# Continuous features — will be z-score normalised
CONTINUOUS_FEATURES = [
    'age', 'heart_rate', 'mean_bp', 'resp_rate', 'spo2', 'temperature',
    'bmi', 'anion_gap', 'bilirubin', 'bnp', 'bun', 'creatinine', 'lactate',
    'platelets', 'rdw', 'troponin', 'wbc', 'fluid_balance', 'gcs_total'
]

# Binary flags — kept as 0/1, not z-scored
BINARY_FEATURES = [
    'admission_emergency', 'vasopressor_usage'
]

# Full feature list (order matters for mask alignment)
EHR_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES

# ================= PHYSIOLOGICAL CLIP RANGES =================
# Based on standard MIMIC preprocessing literature.
# Values outside these ranges are treated as erroneous and set to NaN.

CLIP_RANGES = {
    'heart_rate':    (0, 350),
    'mean_bp':       (0, 330),
    'resp_rate':     (0, 80),
    'spo2':          (0, 100),
    'temperature':   (78.8, 113),
    'bmi':           (10, 50),
    'creatinine':    (0.1, 60),
    'bilirubin':     (0.1, 60),
    'troponin':      (0.01, 50),
    'wbc':           (0, 200),
    'platelets':     (0, 2000),
    'bnp':           (0, 50000),
    'bun':           (0, 200),
    'lactate':       (0.4, 30),
    'anion_gap':     (5, 50),
    'rdw':           (0, 40),
    'fluid_balance': (-10000, 10000),
    'gcs_total':     (3, 15), 
    'age':           (0, 120),
}

# ================= HELPERS =================

def clip_to_physiological(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace values outside physiological plausible ranges with NaN.
    Operates only on columns present in both df and CLIP_RANGES.
    """
    df = df.copy()
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), other=np.nan)
    return df


def compute_scaler(csv_path: str):
    """
    Fit mean and std on continuous features from the training CSV only.
    Applies physiological clipping before computing statistics so that
    outliers do not bias the scaler.
    Returns (mean: pd.Series, std: pd.Series).
    """
    df = pd.read_csv(csv_path, usecols=CONTINUOUS_FEATURES)

    # Replace inf/-inf before clipping
    df = df.replace([np.inf, -np.inf], np.nan)

    # Clip to physiological ranges
    df = clip_to_physiological(df)

    mean = df.mean(skipna=True)
    std  = df.std(skipna=True) + 1e-8
    return mean, std

# ================= DATASET =================

class MultimodalCBMDataset(Dataset):
    def __init__(self, csv_path, transform=None, is_train=False, scaler=None , dropout_config: ModalityDropoutConfig = None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.is_train = is_train
        self.dropout_config = dropout_config
        self.current_p = {m: 0.0 for m in (dropout_config.active_modalities if dropout_config else [])}

        if scaler is None:
            self.mean, self.std = compute_scaler(csv_path)
        else:
            self.mean, self.std = scaler

        # ===== RESTORED: CXR VIEW METADATA =====
        try:
            meta = pd.read_csv(METADATA_CSV, usecols=['dicom_id', 'ViewPosition'])
            self.view_map = dict(zip(meta['dicom_id'], meta['ViewPosition']))
        except Exception:
            self.view_map = {}

        # ===== FAST: Pre-process all EHR data at once =====
        print("Pre-processing EHR data into RAM...")
        self.ehr_features_all, self.ehr_masks_all = self._bulk_process_ehr(self.data)
        
        # Pre-extract concepts and targets to pure numpy/tensors for fast indexing
        self.cxr_concepts_all = torch.tensor(self.data[CXR_CONCEPTS].values.astype(float), dtype=torch.float32)
        self.ecg_concepts_all = torch.tensor(self.data[ECG_CONCEPTS].values.astype(float), dtype=torch.float32)
        self.ehr_concepts_all = torch.tensor(self.data[EHR_CONCEPTS].values.astype(float), dtype=torch.float32)
        self.mort_targets_all = torch.tensor(self.data['label_mortality'].values, dtype=torch.long)
        self.ahf_targets_all = torch.tensor(self.data['label_ahf'].values, dtype=torch.long)



    def _bulk_process_ehr(self, df):
        raw_cont = df[CONTINUOUS_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
        raw_bin = df[BINARY_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
        
        # Clip
        for col, (lo, hi) in CLIP_RANGES.items():
            if col in raw_cont.columns:
                raw_cont[col] = raw_cont[col].where((raw_cont[col] >= lo) & (raw_cont[col] <= hi), np.nan)
                
        # Masks
        mask_cont = (~raw_cont.isna()).astype(float)
        mask_bin = (~raw_bin.isna()).astype(float)
        
        # Normalize & Impute
        norm_cont = ((raw_cont - self.mean) / self.std).fillna(0.0)
        norm_bin = raw_bin.fillna(0.0)
        
        all_features = pd.concat([norm_cont, norm_bin], axis=1).values
        all_masks = pd.concat([mask_cont, mask_bin], axis=1).values
        
        return torch.tensor(all_features, dtype=torch.float32), torch.tensor(all_masks, dtype=torch.float32)





    def set_dropout_probs(self, probs: Dict[str, float]):
        self.current_p = probs




    def __len__(self):
        return len(self.data)

    # ================= CXR =================

    def _get_best_cxr(self, path):
        if not os.path.exists(path):
            return None
        imgs = [f for f in os.listdir(path) if f.endswith('.jpg')]
        if not imgs:
            return None

        best = imgs[0]
        for img in imgs:
            dicom = img.replace('.jpg', '')
            view = self.view_map.get(dicom, '')
            if view == 'PA':
                return img
            elif view == 'AP':
                best = img
        return best

    # ================= EHR =================

    def _process_ehr(self, row):
        """
        Full EHR preprocessing pipeline:
          1. Extract raw values
          2. Replace inf/-inf with NaN
          3. Clip continuous features to physiological ranges
          4. Build missingness mask (1 = observed, 0 = missing)
          5. Z-score normalise continuous features using train scaler
          6. Zero-impute remaining NaNs  (0 ≈ population mean in z-score space)
          7. Leave binary features as 0/1 (not z-scored)
          8. Return (ehr_tensor, mask_tensor)
        """
        # --- 1. Extract raw ---
        raw_continuous = row[CONTINUOUS_FEATURES].astype(float)
        raw_binary     = row[BINARY_FEATURES].astype(float)

        # --- 2. Replace inf/-inf ---
        raw_continuous = raw_continuous.replace([np.inf, -np.inf], np.nan)
        raw_binary     = raw_binary.replace([np.inf, -np.inf], np.nan)

        # --- 3. Clip continuous to physiological ranges ---
        raw_continuous_clipped = raw_continuous.copy()
        for col, (lo, hi) in CLIP_RANGES.items():
            if col in raw_continuous_clipped.index:
                val = raw_continuous_clipped[col]
                if pd.notna(val) and not (lo <= val <= hi):
                    raw_continuous_clipped[col] = np.nan

        # --- 4. Build mask before any imputation ---
        mask_continuous = (~raw_continuous_clipped.isna()).astype(float)
        mask_binary     = (~raw_binary.isna()).astype(float)

        # --- 5. Z-score continuous (NaNs propagate through arithmetic) ---
        norm_continuous = (raw_continuous_clipped - self.mean) / self.std

        # --- 6. Zero-impute NaNs (0 = mean in z-score space) ---
        norm_continuous = norm_continuous.fillna(0.0)

        # --- 7. Binary: fill missing with 0 (absent/unknown → 0) ---
        norm_binary = raw_binary.fillna(0.0)

        # --- 8. Concatenate in fixed order: continuous then binary ---
        all_values = pd.concat([norm_continuous, norm_binary])
        all_mask   = pd.concat([mask_continuous, mask_binary])

        ehr_tensor  = torch.tensor(all_values.values, dtype=torch.float32)
        mask_tensor = torch.tensor(all_mask.values,   dtype=torch.float32)

        return ehr_tensor, mask_tensor

    # ================= __getitem__ =================

    def __getitem__(self, idx):
        

        # ===== CXR =====
        row = self.data.iloc[idx] # iloc once just to get IDs is okay, but avoid for math
        cxr_folder = str(row['report_path']).replace('.txt', '')
        
        # Initialize HDF5 file object lazily per worker
        if not hasattr(self, 'h5_file'):
            import h5py
            self.h5_file = h5py.File('/home/azwad/Works/Multimodal-CBM/Datasets/Data/cxr_images.h5', 'r')
            
        if cxr_folder in self.h5_file:
            image_np = self.h5_file[cxr_folder][:]
            image = Image.fromarray(image_np)
            if self.transform: image = self.transform(image)
        else:
            image = torch.zeros((3, 224, 224))

        # ===== ECG =====
        ecg_tensor = torch.zeros((12, 5000))
        ecg_mask   = torch.tensor(0.0)

        FAST_ECG_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_ecg_npy'

        if pd.notna(row['ecg_study_id']):
            eid = str(int(row['ecg_study_id']))
            npy_path = os.path.join(FAST_ECG_DIR, f"{eid}.npy")
            
            if os.path.exists(npy_path):
                # np.load is nearly instantaneous
                signal = np.load(npy_path)
                
                # Apply data augmentation only during training
                if self.is_train:
                    signal += np.random.normal(0, 0.05, signal.shape)

                ecg_tensor = torch.tensor(signal, dtype=torch.float32)
                ecg_mask   = torch.tensor(1.0)

        # ===== EHR =====
        ehr_features = self.ehr_features_all[idx]
        ehr_mask = self.ehr_masks_all[idx]
        
        cxr_concepts = self.cxr_concepts_all[idx]
        ecg_concepts = self.ecg_concepts_all[idx]
        ehr_concepts = self.ehr_concepts_all[idx]
        
        y_mort = self.mort_targets_all[idx]
        y_ahf = self.ahf_targets_all[idx]



        if self.is_train and self.dropout_config is not None:
            has = {'cxr': True, 'ehr': True,
                'ecg': ecg_mask.item() == 1.0}
            (image, ehr_features, ehr_mask, ecg_tensor,
            cxr_mod_mask, ehr_mod_mask, ecg_mod_mask) = apply_modality_dropout(
                self.current_p, has,
                ehr_features, ehr_mask, ecg_tensor, ecg_mask, image
            )
        else:
            cxr_mod_mask = torch.tensor(1.0)
            ehr_mod_mask = torch.tensor(1.0)
            ecg_mod_mask = ecg_mask



        return {
            'image':    image,
            'waveform': ecg_tensor,
            'ecg_mod_mask': ecg_mod_mask,

            'ehr_features': ehr_features,   # (len(EHR_FEATURES),)  normalised
            'ehr_mask':     ehr_mask,        # (len(EHR_FEATURES),)  1=observed
            'ehr_mod_mask': ehr_mod_mask,    # scalar 1=keep ehr, 0=drop ehr
            'cxr_mod_mask': cxr_mod_mask,    # scalar 1=keep cxr, 0=drop cxr

            'ehr_concepts': ehr_concepts,
            'cxr_concepts': cxr_concepts,
            'ecg_concepts': ecg_concepts,

            'target_mortality': y_mort,
            'target_ahf':       y_ahf,
        }

# ================= COLLATE =================

def clean_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# ================= DATALOADER FACTORY =================

def get_dataloaders(train_csv, val_csv, test_csv, batch_size=32, dropout_config: ModalityDropoutConfig = None):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Fit scaler on training data only — no leakage into val/test
    scaler = compute_scaler(train_csv)

    train_ds = MultimodalCBMDataset(train_csv, train_tf, is_train=True,  scaler=scaler, dropout_config=dropout_config)
    val_ds   = MultimodalCBMDataset(val_csv,   test_tf,  is_train=False, scaler=scaler, dropout_config=None)
    test_ds  = MultimodalCBMDataset(test_csv,  test_tf,  is_train=False, scaler=scaler, dropout_config=None)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=8,               # Hires more background assistants
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

    return train_loader, val_loader, test_loader , train_ds


# ================= QUICK SANITY CHECK =================

if __name__ == "__main__":
    train_csv_path = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv_path   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv_path  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

    TRIMODAL_CONFIG = ModalityDropoutConfig(
    p_max={'cxr': 0.05, 'ehr': 0.30, 'ecg': 0.30},
    active_modalities=['cxr', 'ehr', 'ecg']
    )

    scheduler = ModalityDropoutScheduler(
        config=TRIMODAL_CONFIG,   
        warmup_epochs=20
    )
    

    train_loader, val_loader, test_loader , train_ds = get_dataloaders(
        train_csv_path, val_csv_path, test_csv_path, batch_size=32, dropout_config=TRIMODAL_CONFIG
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

        print("image:         ", batch['image'].shape)           # (B, 3, 224, 224)
        print("waveform:      ", batch['waveform'].shape)        # (B, 12, 5000)
        print("ehr_features:  ", batch['ehr_features'].shape)    # (B, 21)
        print("ehr_mask:      ", batch['ehr_mask'].shape)        # (B, 21)
        print("cxr_concepts:  ", batch['cxr_concepts'].shape)    # (B, 15)
        print("ecg_concepts:  ", batch['ecg_concepts'].shape)    # (B, 14)
        print("ehr_concepts:  ", batch['ehr_concepts'].shape)    # (B, 13)
        print("target_mort:   ", batch['target_mortality'].shape) # (B,)
        print("target_ahf:    ", batch['target_ahf'].shape)       # (B,)
        print("cxr_mod_mask:  ", batch['cxr_mod_mask'].shape)    # (B,)
        print("ehr_mod_mask:  ", batch['ehr_mod_mask'].shape)    # (B,)
        print("ecg_mod_mask:  ", batch['ecg_mod_mask'].shape)    # (B,)



        if(batch['ehr_mod_mask'][0].item() < 1.0):
            print("\nEHR modality dropped for this batch.")
        else:
            print("\nEHR modality kept for this batch.")

        if(batch['cxr_mod_mask'][0].item() < 1.0):
            print("CXR modality dropped for this batch.")
        else:
            print("CXR modality kept for this batch.")

        if(batch['ecg_mod_mask'][0].item() < 1.0):
            print("ECG modality dropped for this batch.")
        else:
            print("ECG modality kept for this batch.")
        # Verify mask is working correctly
        mask = batch['ehr_mask']
        print(f"\nMean observed rate: {mask.mean().item():.3f}  "
              f"(1.0 = all features observed, 0.0 = all missing)")
        




        break