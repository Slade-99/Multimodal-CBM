
import torch
import torch.nn as nn
from net1d import Net1D 
import torchxrayvision as xrv


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


class UniModalEHR(nn.Module):
    def __init__(self, input_dim=21, num_concepts=13):
        super().__init__()
        self.encoder       = EHRConceptEncoder(input_dim=input_dim, num_concepts=num_concepts)
        ckpt = torch.load("/home/azwad/Works/Multimodal-CBM/ehr_concept_checkpoints/ehr_concept_debug/best_weights.pth", map_location="cpu")
        self.encoder.load_state_dict(ckpt,strict=True)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, ehr_features):
        # No noise injection — removed entirely
        logits = self.encoder(ehr_features)   # (B, 13)
        probs = torch.sigmoid(logits)
        return {
            'ehr_concept_logits': logits,
            'mortality_logits':   self.mortality_head(probs).squeeze(-1),
            'ahf_logits':         self.ahf_head(probs).squeeze(-1)
        }


class ECGEncoder(nn.Module):
    def __init__(self, checkpoint_path=None, num_concepts=14):
        super().__init__()

        # Your original network
        self.enc = Net1D(
            in_channels=12,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=True,
            use_bn=False,
            use_do=False,
            n_classes=num_concepts
        )

        # Only load here if a path is provided during initialization
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            
            # Use our custom loader defined below
            self.load_state_dict(ckpt, strict=False)

    def forward(self, x):
        return self.enc(x)

    # ---------------- CUSTOM LOADER ---------------- #
    # This intercepts any attempt to load weights (both inside this class 
    # and from your main training script) and fixes the names automatically.
    def load_state_dict(self, state_dict, strict=True):
        cleaned_state_dict = {}

        for k, v in state_dict.items():
            # Remove "module." if it was saved using DataParallel
            if k.startswith("module."):
                k = k[len("module."):]
            
            # If the key already has "enc.", keep it. 
            # If it doesn't, add "enc." so PyTorch can find self.enc
            if not k.startswith("enc."):
                k = "enc." + k
                
            cleaned_state_dict[k] = v

        # Pass the cleaned dictionary to PyTorch's default loader
        return super().load_state_dict(cleaned_state_dict, strict=strict)


def _build_task_head(in_features: int):
    return nn.Sequential(
        nn.Linear(in_features, 32),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Dropout(0.1),

        nn.Linear(16, 1)
    )


class UniModalECG(nn.Module):
    def __init__(self, num_concepts=14):
        super().__init__()

        # Initialize the encoder (it will load the weights safely)
        self.encoder = ECGEncoder(
            checkpoint_path="/home/azwad/Works/Multimodal-CBM/ecg_concept_checkpoints/ecg_concept_debug/best_weights.pth",
            num_concepts=num_concepts
        )

        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head = _build_task_head(num_concepts)

    def forward(self, waveform):
        concept_logits = self.encoder(waveform)  # (B, 14)
        concept_probs = torch.sigmoid(concept_logits)

        return {
            "ecg_concept_logits": concept_logits,
            "mortality_logits": self.mortality_head(concept_probs).squeeze(-1),
            "ahf_logits": self.ahf_head(concept_probs).squeeze(-1)
        }
    

class CXRConceptEncoder(nn.Module):
    """
    XRV ResNet50 backbone + concept head.
    Identical to CXRConceptXRV from Stage 1 — must match exactly
    so Stage 1 weights load correctly.
    """
    def __init__(self, num_concepts=15):
        super().__init__()
        self.backbone     = xrv.models.ResNet(weights="resnet50-res512-all")
        self.concept_head = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_concepts)
        )

    def forward(self, x):
        feats = self.backbone.features(x)   # (B, 2048)
        return self.concept_head(feats)     # (B, 15) raw logits


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


class UniModalCXR(nn.Module):
    """
    Full CXR CBM model for Stage 2.
    encoder  : frozen after Stage 1 weights loaded
    task heads: trainable — learn to predict mortality/AHF from concept probs
    """
    def __init__(self, num_concepts=15):
        super().__init__()
        self.encoder        = CXRConceptEncoder(num_concepts=num_concepts)
        ckpt = torch.load("/home/azwad/Works/Multimodal-CBM/cxr_concept_checkpoints/xrv_resnet50_concept/best_weights.pth", map_location="cpu")
        self.encoder.load_state_dict(ckpt,strict=True)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, image):
        concept_logits = self.encoder(image)          # (B, 15) raw logits
        concept_probs  = torch.sigmoid(concept_logits) # (B, 15) in [0,1]

        # Task heads receive probabilities, not logits — this is the CBM bottleneck.
        # The task head weights are directly interpretable as concept importances.
        return {
            'concept_logits':   concept_logits,
            'mortality_logits': self.mortality_head(concept_probs).squeeze(-1),
            'ahf_logits':       self.ahf_head(concept_probs).squeeze(-1),
        }
