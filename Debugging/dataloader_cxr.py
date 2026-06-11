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


# ================= DATASET =================

class CXRConceptDataset(Dataset):
    """
    Lightweight dataset that loads only CXR images and their concept labels.
    No ECG, no EHR, no downstream targets — purely for concept prediction debugging.
    """
    def __init__(self, csv_path, transform=None):
        self.data      = pd.read_csv(csv_path)
        self.transform = transform

        # Pre-load all concept labels into a tensor for fast indexing
        self.concepts_all = torch.tensor(
            self.data[CXR_CONCEPTS].values.astype(float), dtype=torch.float32
        )

        # Build dicom -> ViewPosition map for PA/AP preference
        try:
            meta = pd.read_csv(METADATA_CSV, usecols=['dicom_id', 'ViewPosition'])
            self.view_map = dict(zip(meta['dicom_id'], meta['ViewPosition']))
        except Exception:
            self.view_map = {}

        # Validate concept label columns exist
        missing = [c for c in CXR_CONCEPTS if c not in self.data.columns]
        if missing:
            raise ValueError(f"CSV is missing concept columns: {missing}")

        # Validate report_path column exists (used to locate HDF5 key)
        if 'report_path' not in self.data.columns:
            raise ValueError("CSV must have a 'report_path' column to locate CXR images.")

        print(f"[CXRConceptDataset] Loaded {len(self.data)} samples from {csv_path}")
        self._log_concept_prevalence()

    def _log_concept_prevalence(self):
        """Prints per-concept positive rates so you can spot severe imbalance immediately."""
        print("\n--- Concept Prevalence (positive rate) ---")
        for i, c in enumerate(CXR_CONCEPTS):
            rate = self.concepts_all[:, i].mean().item()
            print(f"  {c:<30s}: {rate:.3f}")
        print("------------------------------------------\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # ===== Load CXR from HDF5 =====
        cxr_key = str(row['report_path']).replace('.txt', '')

        # Open HDF5 lazily per worker (same pattern as your original dataloader)
        if not hasattr(self, 'h5_file'):
            self.h5_file = h5py.File(HDF5_PATH, 'r')

        if cxr_key in self.h5_file:
            image_np = self.h5_file[cxr_key][:]
            image    = Image.fromarray(image_np)
            if self.transform:
                image = self.transform(image)
            valid_cxr = True
        else:
            # Return None — will be filtered by collate_fn
            return None

        concepts = self.concepts_all[idx]

        return {
            'image':    image,       # (3, 224, 224)
            'concepts': concepts,    # (15,) float — binary labels
        }


# ================= COLLATE =================

def cxr_collate(batch):
    """Drops any None samples (missing HDF5 keys)."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


# ================= DATALOADER FACTORY =================

def get_cxr_concept_dataloaders(train_csv, val_csv, test_csv, batch_size=32):
    """
    Returns train, val, test DataLoaders that yield only:
      - 'image':    (B, 3, 224, 224)
      - 'concepts': (B, 15)  binary float labels
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )

    # Training: standard augmentation (no heavy augmentation — concepts are spatially sensitive)
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    # Val/Test: deterministic
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = CXRConceptDataset(train_csv, transform=train_tf)
    val_ds   = CXRConceptDataset(val_csv,   transform=eval_tf)
    test_ds  = CXRConceptDataset(test_csv,  transform=eval_tf)

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

    train_loader, val_loader, _ = get_cxr_concept_dataloaders(train_csv, val_csv, test_csv, batch_size=16)

    for batch in train_loader:
        if batch is None:
            print("Empty batch.")
            continue
        print("image shape:    ", batch['image'].shape)     # (B, 3, 224, 224)
        print("concepts shape: ", batch['concepts'].shape)  # (B, 15)
        print("concepts sample:", batch['concepts'][0])
        break