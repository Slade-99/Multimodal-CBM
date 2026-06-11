from dataclasses import dataclass, field
from typing import Dict, Optional
import torch

# ================= DROPOUT CONFIG =================

@dataclass
class ModalityDropoutConfig:
    """
    Per-modality maximum dropout probabilities.
    Set p_max = 0.0 to completely protect a modality from dropout.
    Set p_max = 1.0 to allow full dropout of that modality.

    active_modalities controls which modalities are even in play —
    use this to switch between trimodal and bimodal experiment variants.
    """
    p_max: Dict[str, float] = field(default_factory=lambda: {
        'cxr': 0.05,   # rarely dropped — anchor modality
        'ehr': 0.30,   # freely dropped
        'ecg': 0.30,   # freely dropped
    })
    # Which modalities your current experiment uses.
    # For bimodal CXR+EHR: active_modalities = ['cxr', 'ehr']
    # For bimodal CXR+ECG: active_modalities = ['cxr', 'ecg']
    # For trimodal:        active_modalities = ['cxr', 'ehr', 'ecg']
    active_modalities: list = field(
        default_factory=lambda: ['cxr', 'ehr', 'ecg']
    )

# Pre-built configs for your experiment variants
TRIMODAL_CONFIG = ModalityDropoutConfig(
    p_max={'cxr': 0.05, 'ehr': 0.30, 'ecg': 0.30},
    active_modalities=['cxr', 'ehr', 'ecg']
)

BIMODAL_CXR_EHR_CONFIG = ModalityDropoutConfig(
    p_max={'cxr': 0.05, 'ehr': 0.30, 'ecg': 0.0},
    active_modalities=['cxr', 'ehr']
)

BIMODAL_CXR_ECG_CONFIG = ModalityDropoutConfig(
    p_max={'cxr': 0.05, 'ehr': 0.0, 'ecg': 0.30},
    active_modalities=['cxr', 'ecg']
)

BIMODAL_EHR_ECG_CONFIG = ModalityDropoutConfig(
    p_max={'cxr': 0.0, 'ehr': 0.30, 'ecg': 0.30},
    active_modalities=['ehr', 'ecg']
)


# ================= DROPOUT SCHEDULER =================

class ModalityDropoutScheduler:
    """
    Linearly warms up a global scale factor from 0 → 1 over
    warmup_epochs. Each modality's effective prob =
    scale * config.p_max[modality].
    """
    def __init__(self, config: ModalityDropoutConfig,
                 warmup_epochs: int = 20):
        self.config        = config
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        self.scale         = 0.0   # starts at 0, reaches 1.0

    @property
    def current_probs(self) -> Dict[str, float]:
        return {
            m: self.scale * self.config.p_max[m]
            for m in self.config.active_modalities
        }

    def step(self):
        self.current_epoch += 1
        self.scale = min(1.0, self.current_epoch / self.warmup_epochs)

    def __repr__(self):
        probs = {m: f"{p:.3f}" for m, p in self.current_probs.items()}
        return f"DropoutScheduler(epoch={self.current_epoch}, probs={probs})"


# ================= DROPOUT LOGIC =================

def apply_modality_dropout(
    p_current: Dict[str, float],
    has: Dict[str, bool],
    ehr_features, ehr_mask,
    ecg_tensor, ecg_mask_val,
    image,
):
    """
    Safely applies modality dropout given current per-modality probabilities.

    Rules:
      1. Only present modalities are eligible for dropout.
      2. At most one modality is dropped per sample.
      3. At least one modality is always kept (no all-zero batches).

    Returns updated tensors + scalar modality presence masks:
      cxr_mod_mask, ehr_mod_mask, ecg_mod_mask  (each 0.0 or 1.0)
    """
    has_cxr = has.get('cxr', True)
    has_ehr = has.get('ehr', True)
    has_ecg = has.get('ecg', ecg_mask_val.item() == 1.0)

    # Build eligible candidate list — only present modalities
    # weighted by their current dropout probability
    candidates = []
    weights    = []
    if has_cxr and p_current.get('cxr', 0.0) > 0:
        candidates.append('cxr'); weights.append(p_current['cxr'])
    if has_ehr and p_current.get('ehr', 0.0) > 0:
        candidates.append('ehr'); weights.append(p_current['ehr'])
    if has_ecg and p_current.get('ecg', 0.0) > 0:
        candidates.append('ecg'); weights.append(p_current['ecg'])

    drop_choice = None
    if candidates:
        # Safety check: don't drop if only one modality is present
        present_count = sum([has_cxr, has_ehr, has_ecg])
        if present_count > 1:
            # Sample one candidate proportionally to its weight
            total = sum(weights)
            r = torch.rand(1).item() * total
            cumulative = 0.0
            for cand, w in zip(candidates, weights):
                cumulative += w
                if r <= cumulative:
                    drop_choice = cand
                    break

    # Apply the drop and build modality masks
    cxr_mod_mask = torch.tensor(1.0 if has_cxr else 0.0)
    ehr_mod_mask = torch.tensor(1.0 if has_ehr else 0.0)
    ecg_mod_mask = torch.tensor(1.0 if has_ecg else 0.0)

    if drop_choice == 'cxr':
        # Zero the image tensor
        image        = torch.zeros_like(image)
        cxr_mod_mask = torch.tensor(0.0)

    elif drop_choice == 'ehr':
        ehr_features = torch.zeros_like(ehr_features)
        ehr_mask     = torch.zeros_like(ehr_mask)
        ehr_mod_mask = torch.tensor(0.0)

    elif drop_choice == 'ecg':
        ecg_tensor   = torch.zeros_like(ecg_tensor)
        ecg_mod_mask = torch.tensor(0.0)

    return (image, ehr_features, ehr_mask, ecg_tensor,
            cxr_mod_mask, ehr_mod_mask, ecg_mod_mask)