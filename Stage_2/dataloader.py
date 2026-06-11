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

CXR_DIR      = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_cxr'
ECG_DIR      = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_ecg'
METADATA_CSV = '/home/azwad/Works/Multimodal-CBM/Datasets/CXR/mimic-cxr-2.0.0-metadata.csv'

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

CONTINUOUS_FEATURES = [
    'age', 'heart_rate', 'mean_bp', 'resp_rate', 'spo2', 'temperature',
    'bmi', 'anion_gap', 'bilirubin', 'bnp', 'bun', 'creatinine', 'lactate',
    'platelets', 'rdw', 'troponin', 'wbc', 'fluid_balance', 'gcs_total'
]

BINARY_FEATURES = [
    'admission_emergency', 'vasopressor_usage'
]

EHR_FEATURES = CONTINUOUS_FEATURES + BINARY_FEATURES

# ================= PHYSIOLOGICAL CLIP RANGES =================

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
    df = df.copy()
    for col, (lo, hi) in CLIP_RANGES.items():
        if col in df.columns:
            df[col] = df[col].where((df[col] >= lo) & (df[col] <= hi), other=np.nan)
    return df


def compute_scaler(csv_path: str):
    df = pd.read_csv(csv_path, usecols=CONTINUOUS_FEATURES)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = clip_to_physiological(df)
    mean = df.mean(skipna=True)
    std  = df.std(skipna=True) + 1e-8
    return mean, std


# ================= DATASET =================

class MultimodalCBMDataset(Dataset):
    def __init__(self, csv_path, transform=None, is_train=False, scaler=None):
        self.data      = pd.read_csv(csv_path)
        self.transform = transform
        self.is_train  = is_train

        if scaler is None:
            self.mean, self.std = compute_scaler(csv_path)
        else:
            self.mean, self.std = scaler

        # CXR view metadata for PA/AP preference
        try:
            meta = pd.read_csv(METADATA_CSV, usecols=['dicom_id', 'ViewPosition'])
            self.view_map = dict(zip(meta['dicom_id'], meta['ViewPosition']))
        except Exception:
            self.view_map = {}

        # Pre-process all EHR data into RAM at init time
        self.ehr_features_all, self.ehr_masks_all = self._bulk_process_ehr(self.data)

        # Pre-extract all labels and concepts as tensors for fast indexing
        self.cxr_concepts_all = torch.tensor(
            self.data[CXR_CONCEPTS].values.astype(float), dtype=torch.float32)
        self.ecg_concepts_all = torch.tensor(
            self.data[ECG_CONCEPTS].values.astype(float), dtype=torch.float32)
        self.ehr_concepts_all = torch.tensor(
            self.data[EHR_CONCEPTS].values.astype(float), dtype=torch.float32)
        self.mort_targets_all = torch.tensor(
            self.data['label_mortality'].values, dtype=torch.long)
        self.ahf_targets_all  = torch.tensor(
            self.data['label_ahf'].values, dtype=torch.long)

    def _bulk_process_ehr(self, df):
        raw_cont = df[CONTINUOUS_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)
        raw_bin  = df[BINARY_FEATURES].astype(float).replace([np.inf, -np.inf], np.nan)

        for col, (lo, hi) in CLIP_RANGES.items():
            if col in raw_cont.columns:
                raw_cont[col] = raw_cont[col].where(
                    (raw_cont[col] >= lo) & (raw_cont[col] <= hi), np.nan)

        mask_cont = (~raw_cont.isna()).astype(float)
        mask_bin  = (~raw_bin.isna()).astype(float)

        norm_cont = ((raw_cont - self.mean) / self.std).fillna(0.0)
        norm_bin  = raw_bin.fillna(0.0)

        all_features = pd.concat([norm_cont, norm_bin], axis=1).values
        all_masks    = pd.concat([mask_cont, mask_bin], axis=1).values

        return (torch.tensor(all_features, dtype=torch.float32),
                torch.tensor(all_masks,    dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    # ================= __getitem__ =================

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ===== CXR =====
        cxr_folder = str(row['report_path']).replace('.txt', '')

        if not hasattr(self, 'h5_file'):
            import h5py
            self.h5_file = h5py.File(
                '/home/azwad/Works/Multimodal-CBM/Datasets/Data/cxr_images.h5', 'r')

        if cxr_folder in self.h5_file:
            image_np = self.h5_file[cxr_folder][:]
            image    = Image.fromarray(image_np)
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.zeros((3, 224, 224))
        else:
            # Fallback must match the transform's output shape exactly.
            # Hardcoding (3, 224, 224) breaks XRV experiments which produce (1, 512, 512).
            if self.transform is not None:
                dummy = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
                image = self.transform(dummy)
            else:
                image = torch.zeros((3, 224, 224))

        # ===== ECG =====
        ecg_tensor = torch.zeros((12, 5000))

        FAST_ECG_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_ecg_npy'

        if pd.notna(row['ecg_study_id']):
            eid      = str(int(row['ecg_study_id']))
            npy_path = os.path.join(FAST_ECG_DIR, f"{eid}.npy")
            if os.path.exists(npy_path):
                signal = np.load(npy_path)
                if self.is_train:
                    signal += np.random.normal(0, 0.05, signal.shape)
                ecg_tensor = torch.tensor(signal, dtype=torch.float32)

        # ===== EHR =====
        ehr_features  = self.ehr_features_all[idx]
        ehr_feat_mask = self.ehr_masks_all[idx]

        cxr_concepts = self.cxr_concepts_all[idx]
        ecg_concepts = self.ecg_concepts_all[idx]
        ehr_concepts = self.ehr_concepts_all[idx]

        y_mort = self.mort_targets_all[idx]
        y_ahf  = self.ahf_targets_all[idx]

        return {
            'image':            image,
            'waveform':         ecg_tensor,
            'ehr_features':     ehr_features,    # (21,) normalised
            'ehr_mask':         ehr_feat_mask,   # (21,) 1=observed, 0=missing
            'cxr_concepts':     cxr_concepts,
            'ecg_concepts':     ecg_concepts,
            'ehr_concepts':     ehr_concepts,
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

def get_dataloaders(train_csv, val_csv, test_csv, batch_size=32):
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

    scaler = compute_scaler(train_csv)

    train_ds = MultimodalCBMDataset(train_csv, train_tf, is_train=True,  scaler=scaler)
    val_ds   = MultimodalCBMDataset(val_csv,   test_tf,  is_train=False, scaler=scaler)
    test_ds  = MultimodalCBMDataset(test_csv,  test_tf,  is_train=False, scaler=scaler)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        prefetch_factor=2, persistent_workers=True,
        collate_fn=clean_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=clean_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=clean_collate
    )

    return train_loader, val_loader, test_loader, train_ds


# ================= SANITY CHECK =================

if __name__ == "__main__":
    train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        train_csv, val_csv, test_csv, batch_size=32
    )

    for batch in train_loader:
        if batch is None:
            print("Empty batch.")
            continue
        print("image:        ", batch['image'].shape)
        print("waveform:     ", batch['waveform'].shape)
        print("ehr_features: ", batch['ehr_features'].shape)
        print("ehr_mask:     ", batch['ehr_mask'].shape)
        print("cxr_concepts: ", batch['cxr_concepts'].shape)
        print("ecg_concepts: ", batch['ecg_concepts'].shape)
        print("ehr_concepts: ", batch['ehr_concepts'].shape)
        print("target_mort:  ", batch['target_mortality'].shape)
        print("target_ahf:   ", batch['target_ahf'].shape)
        break