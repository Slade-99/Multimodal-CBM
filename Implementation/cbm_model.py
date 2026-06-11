import torch
import torch.nn as nn
import torchvision.models as models


# ================= CXR ENCODER (ResNet50) =================

class CXREncoder(nn.Module):
    def __init__(self, num_concepts=15, pretrained=True):
        """
        ResNet50 backbone with a proper multi-layer concept head.

        Changes from original:
        - Backbone feature extractor is separated from the concept head so
          the head can be deeper without modifying backbone internals.
        - Concept head: 2048 -> 256 -> 128 -> num_concepts with BN + Dropout.
          A single Linear(2048, 15) had insufficient capacity to learn the
          non-linear feature combinations required for clinical concept prediction.
        - No noise injection anywhere — noise was corrupting concept gradients.
        """
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Strip the original FC — we replace it with a deeper head
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 2048, 1, 1)

        # Deeper concept head — proven to fix concept accuracy in isolation experiments
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
        x = self.features(x)              # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)           # (B, 2048)
        return self.concept_head(x)       # (B, num_concepts)


# ================= ECG ENCODER (1D-ResNet) =================

class BasicBlock1D(nn.Module):
    """A standard 1D ResNet block with a skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class ECGEncoder(nn.Module):
    def __init__(self, num_concepts=14):
        """
        1D-ResNet for 12-lead ECG signals of length 5000.

        Changes from original:
        - Replaced single Linear(512, num_concepts) classifier with a deeper
          concept head: 512 -> 128 -> num_concepts with BN + Dropout.
          Consistent with the fix applied to CXREncoder — shallow heads on
          top of deep backbones underfit concept prediction.
        """
        super().__init__()
        self.in_channels = 64
        self.conv1   = nn.Conv1d(12, 64, kernel_size=15, stride=4, padding=7, bias=False)
        self.bn1     = nn.BatchNorm1d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1  = self._make_layer(64,  num_blocks=2, stride=1)
        self.layer2  = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3  = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4  = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Deeper concept head (replaces single Linear)
        self.concept_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, num_concepts)   # raw logits
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [BasicBlock1D(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)           # (B, 512)
        return self.concept_head(x)       # (B, num_concepts)


# ================= EHR ENCODER =================

class EHREncoder(nn.Module):
    def __init__(self, input_dim=21, num_concepts=13):
        """
        Tabular EHR encoder.

        Changes from original:
        - Added a third hidden layer (128 units) before the existing 64->32 path.
          Tabular data with 21 mixed continuous/binary inputs benefits from a
          wider first layer to learn feature interactions before compression.
        - Network is now: 21 -> 128 -> 64 -> 32 -> num_concepts.
        - Kept BN + Dropout at each layer as before.
        """
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

            nn.Linear(32, num_concepts)    # raw logits
        )

    def forward(self, x):
        return self.network(x)             # (B, num_concepts)


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
        self.encoder       = CXREncoder(num_concepts=num_concepts)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask):
        # No noise injection — removed entirely
        logits = self.encoder(image)   # (B, 15)

        return {
            'cxr_concept_logits': logits,
            'ecg_concept_logits': torch.zeros((logits.shape[0], 14), device=logits.device),
            'ehr_concept_logits': torch.zeros((logits.shape[0], 13), device=logits.device),
            'mortality_logits':   self.mortality_head(logits).squeeze(-1),
            'ahf_logits':         self.ahf_head(logits).squeeze(-1)
        }


class UniModalECG(nn.Module):
    def __init__(self, num_concepts=14):
        super().__init__()
        self.encoder       = ECGEncoder(num_concepts=num_concepts)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask):
        # No noise injection — removed entirely
        logits = self.encoder(waveform)   # (B, 14)

        return {
            'cxr_concept_logits': torch.zeros((logits.shape[0], 15), device=logits.device),
            'ecg_concept_logits': logits,
            'ehr_concept_logits': torch.zeros((logits.shape[0], 13), device=logits.device),
            'mortality_logits':   self.mortality_head(logits).squeeze(-1),
            'ahf_logits':         self.ahf_head(logits).squeeze(-1)
        }


class UniModalEHR(nn.Module):
    def __init__(self, input_dim=21, num_concepts=13):
        super().__init__()
        self.encoder       = EHREncoder(input_dim=input_dim, num_concepts=num_concepts)
        self.mortality_head = _build_task_head(num_concepts)
        self.ahf_head       = _build_task_head(num_concepts)

    def forward(self, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask):
        # No noise injection — removed entirely
        logits = self.encoder(ehr_features)   # (B, 13)

        return {
            'cxr_concept_logits': torch.zeros((logits.shape[0], 15), device=logits.device),
            'ecg_concept_logits': torch.zeros((logits.shape[0], 14), device=logits.device),
            'ehr_concept_logits': logits,
            'mortality_logits':   self.mortality_head(logits).squeeze(-1),
            'ahf_logits':         self.ahf_head(logits).squeeze(-1)
        }