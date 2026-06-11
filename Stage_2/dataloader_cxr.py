import os
import torch
import pandas as pd
import numpy as np
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ================= CONCEPTS =================

CXR_CONCEPTS = [
    "lung_opacity", "pleural_effusion", "support_tube",
    "heart_enlarged", "lung_atelectasis", "pulmonary_edema",
    "support_line", "vascular_congestion", "lungs_hyperinflated",
    "apical_pneumothorax", "hemidiaphragm_elevation", "rib_fractures",
    "interstitial_markings", "volume_loss", "atrium_leads"
]

# ================= PATHS =================

METADATA_CSV = '/home/azwad/Works/Multimodal-CBM/Datasets/CXR/mimic-cxr-2.0.0-metadata.csv'
HDF5_PATH    = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/cxr_images.h5'

# ================= TRANSFORMS =================
# XRV ResNet50 requires:
#   - Single-channel grayscale input (1, H, W)
#   - Pixel values in [-1024, 1024] range
# Pipeline: PIL → Grayscale(1ch) → ToTensor [0,1] → scale to [-1024, 1024]

def xrv_train_transform():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2048.0 - 1024.0),
    ])

def xrv_eval_transform():
    return transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2048.0 - 1024.0),
    ])


# ================= DATASET =================

class CXRStage2Dataset(Dataset):
    """
    CXR dataset for Stage 2 task head training.
    Returns CXR image, concept labels, and downstream targets (mortality + AHF).
    Uses XRV-compatible transforms: grayscale, [-1024, 1024] range.
    """
    def __init__(self, csv_path, transform=None):
        self.data      = pd.read_csv(csv_path)
        self.transform = transform

        # Validate required columns
        missing_concepts = [c for c in CXR_CONCEPTS if c not in self.data.columns]
        if missing_concepts:
            raise ValueError(f"CSV missing concept columns: {missing_concepts}")
        for col in ['report_path', 'label_mortality', 'label_ahf']:
            if col not in self.data.columns:
                raise ValueError(f"CSV missing required column: '{col}'")

        # Pre-load all labels into tensors for fast indexing
        self.concepts_all = torch.tensor(
            self.data[CXR_CONCEPTS].values.astype(float), dtype=torch.float32)
        self.mort_targets = torch.tensor(
            self.data['label_mortality'].values, dtype=torch.long)
        self.ahf_targets  = torch.tensor(
            self.data['label_ahf'].values, dtype=torch.long)

        # Build dicom -> ViewPosition map for PA/AP preference
        try:
            meta = pd.read_csv(METADATA_CSV, usecols=['dicom_id', 'ViewPosition'])
            self.view_map = dict(zip(meta['dicom_id'], meta['ViewPosition']))
        except Exception:
            self.view_map = {}

        print(f"[CXRStage2Dataset] Loaded {len(self.data)} samples from {csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row     = self.data.iloc[idx]
        cxr_key = str(row['report_path']).replace('.txt', '')

        # Open HDF5 lazily per worker
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(HDF5_PATH, 'r')

        if cxr_key in self.h5_file:
            image_np = self.h5_file[cxr_key][:]
            image    = Image.fromarray(image_np)
            if self.transform:
                image = self.transform(image)
        else:
            # Missing HDF5 key — return None, filtered by collate
            return None

        return {
            'image':            image,                   # (1, 512, 512) XRV format
            'concepts':         self.concepts_all[idx],  # (15,) float binary
            'target_mortality': self.mort_targets[idx],  # scalar long
            'target_ahf':       self.ahf_targets[idx],   # scalar long
        }


# ================= COLLATE =================

def cxr_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ================= DATALOADER FACTORY =================

def get_cxr_stage2_dataloaders(train_csv, val_csv, test_csv, batch_size=32):
    """
    Returns train, val, test DataLoaders yielding:
      - 'image':            (B, 1, 512, 512)  XRV grayscale [-1024, 1024]
      - 'concepts':         (B, 15)           binary float concept labels
      - 'target_mortality': (B,)              long
      - 'target_ahf':       (B,)              long
    """
    train_ds = CXRStage2Dataset(train_csv, transform=xrv_train_transform())
    val_ds   = CXRStage2Dataset(val_csv,   transform=xrv_eval_transform())
    test_ds  = CXRStage2Dataset(test_csv,  transform=xrv_eval_transform())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
        prefetch_factor=2, persistent_workers=True,
        collate_fn=cxr_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=cxr_collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        collate_fn=cxr_collate
    )
    return train_loader, val_loader, test_loader


# ================= SANITY CHECK =================

if __name__ == "__main__":
    train_csv = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"

    train_loader, val_loader, _ = get_cxr_stage2_dataloaders(
        train_csv, val_csv, test_csv, batch_size=16)

    for batch in train_loader:
        if batch is None:
            print("Empty batch.")
            continue
        print("image:            ", batch['image'].shape)            # (B, 1, 512, 512)
        print("concepts:         ", batch['concepts'].shape)         # (B, 15)
        print("target_mortality: ", batch['target_mortality'].shape) # (B,)
        print("target_ahf:       ", batch['target_ahf'].shape)       # (B,)
        print("image min/max:    ", batch['image'].min().item(),
        batch['image'].max().item())                           # ~[-1024, 1024]
        break