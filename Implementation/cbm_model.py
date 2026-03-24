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

# ================= FULL MULTIMODAL CBM =================

class MultimodalCBM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Modality Encoders (Concept Predictors)
        self.cxr_encoder = CXREncoder(num_concepts=15)
        self.ecg_encoder = ECGEncoder(num_concepts=14)
        self.ehr_encoder = EHREncoder(input_dim=21, num_concepts=13)
        
        # 2. Downstream Task Classifiers
        # Total concepts = 15 (CXR) + 14 (ECG) + 13 (EHR) = 42
        total_concepts = 42
        
        # We predict two different tasks, so we have two separate output heads
        self.mortality_head = nn.Sequential(
            nn.Linear(total_concepts, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        self.ahf_head = nn.Sequential(
            nn.Linear(total_concepts, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask):
        """
        The forward pass uses the modality dropout masks. If a modality is dropped
        during training, its predicted concepts are zeroed out before fusion.
        """
        
        # 1. Predict Concepts (Raw Logits)
        cxr_concept_logits = self.cxr_encoder(image)
        ecg_concept_logits = self.ecg_encoder(waveform)
        ehr_concept_logits = self.ehr_encoder(ehr_features)
        
        # Apply Sigmoid to turn logits into probabilities (0 to 1) for the bottleneck
        cxr_concept_probs = torch.sigmoid(cxr_concept_logits)
        ecg_concept_probs = torch.sigmoid(ecg_concept_logits)
        ehr_concept_probs = torch.sigmoid(ehr_concept_logits)
        
        # 2. Apply Modality Dropouts
        # If mask is 0 (dropped), the probabilities become 0. 
        # We use .view(-1, 1) to ensure the mask shape aligns with the batch dimension
        cxr_concept_probs = cxr_concept_probs * cxr_mask.view(-1, 1)
        ecg_concept_probs = ecg_concept_probs * ecg_mask.view(-1, 1)
        ehr_concept_probs = ehr_concept_probs * ehr_mask.view(-1, 1)
        
        # 3. Fusion (Concatenation)
        # Shape: (B, 42)
        fused_concepts = torch.cat([cxr_concept_probs, ecg_concept_probs, ehr_concept_probs], dim=1)
        
        # 4. Downstream Predictions (Raw Logits)
        mortality_logits = self.mortality_head(fused_concepts).squeeze(-1) # Shape: (B,)
        ahf_logits = self.ahf_head(fused_concepts).squeeze(-1)             # Shape: (B,)
        
        return {
            'cxr_concept_logits': cxr_concept_logits,
            'ecg_concept_logits': ecg_concept_logits,
            'ehr_concept_logits': ehr_concept_logits,
            'mortality_logits': mortality_logits,
            'ahf_logits': ahf_logits
        }