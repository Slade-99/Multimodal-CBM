import torch
import torch.nn as nn
import torchvision.models as models
import torchxrayvision as xrv
from net1d import Net1D 
# ================= CXR ENCODER (ResNet50) =================
class CXRConceptXRV(nn.Module):
    """
    TorchXRayVision ResNet50 backbone + deep concept head.

    backbone.features(x):
        - Handles grayscale (1-channel) input
        - Internally does conv1 → bn1 → relu → maxpool → layer1-4 → avgpool → flatten
        - Returns (B, 2048) directly — no separate avgpool needed
        - Auto-resizes to 512x512 if input is different size

    Concept head: 2048 → 256 → 128 → num_concepts
        Same architecture that worked for the MobileNet isolation experiment.
    """
    def __init__(self, num_concepts=15):
        super().__init__()

        # Load XRV ResNet50 pretrained on CXR datasets
        self.backbone = xrv.models.ResNet(weights="resnet50-res512-all")
        # backbone.features(x) returns (B, 2048) — used in forward below
        state_dict = torch.load("/home/azwad/Works/Multimodal-CBM/cxr_concept_checkpoints/xrv_resnet50_concept/best_weights.pth", map_location='cpu')
        self.backbone.load_state_dict(state_dict, strict=False)
        self.concept_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, num_concepts)   # raw logits
        )

    def forward(self, x):
        # backbone.features() already does avgpool + flatten → (B, 2048)
        feats = self.backbone.features(x)     # (B, 2048)
        return self.concept_head(feats)       # (B, num_concepts)
# ================= ECG ENCODER (1D-ResNet) =================

class ECGEncoder(nn.Module):
    def __init__(self, num_concepts=14):
        super().__init__()

        enc = Net1D(      in_channels=12, 
        base_filters=64,
        ratio=1, 
        filter_list=[64,160,160,400,400,1024,1024],    
        m_blocks_list=[2,2,2,3,3,4,4],   
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose=True, 
        use_bn=False,
        use_do=False,
        n_classes=14)

        state_dict = torch.load("/home/azwad/Works/Multimodal-CBM/ecg_concept_checkpoints/ecg_concept_debug/best_weights.pth", map_location='cpu')

        if "module." in list(state_dict.keys())[0]:
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
        enc.load_state_dict(state_dict, strict=False)

        self.enc = enc

    def forward(self, x):
        return self.enc(x)

# ================= EHR ENCODER =================

class EHRConceptEncoder(nn.Module):
    """
    Wider and deeper than original 21->64->32->13.
    First layer expanded to 128 to give the model room to learn
    pairwise feature interactions before compressing to concept space.
    """
    def __init__(self, input_dim=21, num_concepts=13):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, num_concepts)   # raw logits
        )

    def forward(self, x):
        return self.network(x)   # (B, 13)

# ================= TASK HEADS =================
# Shared factory for mortality / AHF heads.
# Made deeper: num_concepts -> 32 -> 16 -> 1
# The original 2-layer head (concepts -> 16 -> 1) was symmetric with the
# concept head width and had limited capacity. A wider first layer (32 units)
# gives the task head more room to learn non-linear combinations of concepts
# before the final binary prediction, which is especially important when
# concept scores are continuous logits rather than hard 0/1 values.

def _build_task_head(in_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(16, 1)
    )


# ================= UNIMODAL WRAPPERS =================

class UniModalCXR(nn.Module):
    def __init__(self, num_concepts=15):
        super().__init__()
        self.encoder       = CXRConceptXRV(num_concepts=num_concepts)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, image, waveform, ehr_features):
        # No noise injection — removed entirely
        logits = self.encoder(image)   # (B, 15)
        probs = torch.sigmoid(logits)
        return {
            'cxr_concept_logits': logits,
            'ecg_concept_logits': torch.zeros((logits.shape[0], 14), device=logits.device),
            'ehr_concept_logits': torch.zeros((logits.shape[0], 13), device=logits.device),
            'mortality_logits':   self.mortality_head(probs).squeeze(-1),
            'ahf_logits':         self.ahf_head(probs).squeeze(-1)
        }


class UniModalECG(nn.Module):
    def __init__(self, num_concepts=14):
        super().__init__()
        self.encoder       = ECGEncoder(num_concepts=num_concepts)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, image, waveform, ehr_features):
        # No noise injection — removed entirely
        logits = self.encoder(waveform)   # (B, 14)
        probs = torch.sigmoid(logits)
        return {
            'cxr_concept_logits': torch.zeros((logits.shape[0], 15), device=logits.device),
            'ecg_concept_logits': logits,
            'ehr_concept_logits': torch.zeros((logits.shape[0], 13), device=logits.device),
            'mortality_logits':   self.mortality_head(probs).squeeze(-1),
            'ahf_logits':         self.ahf_head(probs).squeeze(-1)
        }


class UniModalEHR(nn.Module):
    def __init__(self, input_dim=21, num_concepts=13):
        super().__init__()
        self.encoder       = EHRConceptEncoder(input_dim=input_dim, num_concepts=num_concepts)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, image, waveform, ehr_features):
        # No noise injection — removed entirely
        logits = self.encoder(ehr_features)   # (B, 13)
        probs = torch.sigmoid(logits)
        return {
            'cxr_concept_logits': torch.zeros((logits.shape[0], 15), device=logits.device),
            'ecg_concept_logits': torch.zeros((logits.shape[0], 14), device=logits.device),
            'ehr_concept_logits': logits,
            'mortality_logits':   self.mortality_head(probs).squeeze(-1),
            'ahf_logits':         self.ahf_head(probs).squeeze(-1)
        }