import torch
import torch.nn as nn
import torchvision.models as models

class CXREncoder(nn.Module):
    def __init__(self, num_concepts=15, pretrained=True):
        """
        Processes standard (B, 3, 224, 224) RGB images.
        """
        super().__init__()
        # Use the best available ImageNet weights
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer to output exactly 15 concepts
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_concepts)

    def forward(self, x):
        # Output shape: (B, 15)
        return self.backbone(x)


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

        # The skip connection: if dimensions change, we use a 1x1 conv to match them
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
        out += identity  # Add the shortcut back in
        return self.relu(out)

class ECGEncoder(nn.Module):
    def __init__(self, num_concepts=14):
        """
        A 1D-ResNet adapted for 12-lead ECG signals of length 5000.
        """
        super().__init__()
        # Initial convolution to heavily downsample the 5000-length signal
        self.in_channels = 64
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15, stride=4, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet Blocks
        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(512, num_concepts)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride))
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
        x = torch.flatten(x, 1)
        return self.classifier(x)


class EHREncoder(nn.Module):
    def __init__(self, input_dim=21, num_concepts=13):
        """
        Processes (B, 21) tabular features.
        """
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, num_concepts)
        )

    def forward(self, x):
        # Output shape: (B, 13)
        return self.network(x)





class UniModalCXR(nn.Module):
    def __init__(self, num_concepts=15):
        super().__init__()
        self.encoder = CXREncoder(num_concepts=num_concepts)

        # Slightly stronger head for stability
        self.mortality_head = nn.Sequential(
            nn.Linear(num_concepts, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

        self.ahf_head = nn.Sequential(
            nn.Linear(num_concepts, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask):
        logits = self.encoder(image)  # (B, 15)

        #  Optional concept noise (SAFE version)
        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise

        return {
            'cxr_concept_logits': logits,
            'ecg_concept_logits': torch.zeros((logits.shape[0], 14), device=logits.device),
            'ehr_concept_logits': torch.zeros((logits.shape[0], 13), device=logits.device),

            #  IMPORTANT: pass LOGITS, not sigmoid
            'mortality_logits': self.mortality_head(logits).squeeze(-1),
            'ahf_logits': self.ahf_head(logits).squeeze(-1)
        }
    

class UniModalECG(nn.Module):
    def __init__(self, num_concepts=14):
        super().__init__()
        self.encoder = ECGEncoder(num_concepts=num_concepts)

        self.mortality_head = nn.Sequential(
            nn.Linear(num_concepts, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

        self.ahf_head = nn.Sequential(
            nn.Linear(num_concepts, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask):
        logits = self.encoder(waveform)  # (B, 14)

        if self.training:
            noise = torch.randn_like(logits) * 0.1
            logits = logits + noise

        return {
            'cxr_concept_logits': torch.zeros((logits.shape[0], 15), device=logits.device),
            'ecg_concept_logits': logits,
            'ehr_concept_logits': torch.zeros((logits.shape[0], 13), device=logits.device),

            'mortality_logits': self.mortality_head(logits).squeeze(-1),
            'ahf_logits': self.ahf_head(logits).squeeze(-1)
        }
    

class UniModalEHR(nn.Module):
    def __init__(self, input_dim=21, num_concepts=13):
        super().__init__()
        self.encoder = EHREncoder(input_dim=input_dim, num_concepts=num_concepts)

        self.mortality_head = nn.Sequential(
            nn.Linear(num_concepts, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

        self.ahf_head = nn.Sequential(
            nn.Linear(num_concepts, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask, ehr_feat_mask):
        logits = self.encoder(ehr_features)  # (B, 13)

        if self.training:
            noise = torch.randn_like(logits) * 0.05  # smaller noise for tabular
            logits = logits + noise

        return {
            'cxr_concept_logits': torch.zeros((logits.shape[0], 15), device=logits.device),
            'ecg_concept_logits': torch.zeros((logits.shape[0], 14), device=logits.device),
            'ehr_concept_logits': logits,

            'mortality_logits': self.mortality_head(logits).squeeze(-1),
            'ahf_logits': self.ahf_head(logits).squeeze(-1)
        }