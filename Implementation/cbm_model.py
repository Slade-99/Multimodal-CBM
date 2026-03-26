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






class CrossModalConceptAttention(nn.Module):
    """
    Applies cross-modal attention directly in concept space.
    Each modality's concepts attend to the concepts of every other modality,
    capturing inter-modal clinical correlations at the bottleneck level.
    """
    def __init__(self, cxr_dim=15, ecg_dim=14, ehr_dim=13):
        super().__init__()
        total = cxr_dim + ecg_dim + ehr_dim  # 42

        # One attention head per modality — query is that modality,
        # keys/values are the full concatenated concept vector
        self.attn_cxr = nn.MultiheadAttention(embed_dim=cxr_dim, num_heads=1, batch_first=True)
        self.attn_ecg = nn.MultiheadAttention(embed_dim=ecg_dim, num_heads=1, batch_first=True)
        self.attn_ehr = nn.MultiheadAttention(embed_dim=ehr_dim, num_heads=1, batch_first=True)

        # Project full concept vector down to each modality's dim for key/value
        self.kv_proj_cxr = nn.Linear(total, cxr_dim)
        self.kv_proj_ecg = nn.Linear(total, ecg_dim)
        self.kv_proj_ehr = nn.Linear(total, ehr_dim)

        # Layer norms for residual stability
        self.norm_cxr = nn.LayerNorm(cxr_dim)
        self.norm_ecg = nn.LayerNorm(ecg_dim)
        self.norm_ehr = nn.LayerNorm(ehr_dim)

    def forward(self, cxr_probs, ecg_probs, ehr_probs):
        # Full concept context for key/value
        context = torch.cat([cxr_probs, ecg_probs, ehr_probs], dim=1)  # (B, 42)

        # Reshape to (B, 1, dim) for MultiheadAttention
        q_cxr = cxr_probs.unsqueeze(1)
        q_ecg = ecg_probs.unsqueeze(1)
        q_ehr = ehr_probs.unsqueeze(1)

        kv_cxr = self.kv_proj_cxr(context).unsqueeze(1)
        kv_ecg = self.kv_proj_ecg(context).unsqueeze(1)
        kv_ehr = self.kv_proj_ehr(context).unsqueeze(1)

        # Each modality attends to cross-modal context
        cxr_attended, _ = self.attn_cxr(q_cxr, kv_cxr, kv_cxr)
        ecg_attended, _ = self.attn_ecg(q_ecg, kv_ecg, kv_ecg)
        ehr_attended, _ = self.attn_ehr(q_ehr, kv_ehr, kv_ehr)

        # Residual connection + layer norm
        cxr_out = self.norm_cxr(cxr_probs + cxr_attended.squeeze(1))
        ecg_out = self.norm_ecg(ecg_probs + ecg_attended.squeeze(1))
        ehr_out = self.norm_ehr(ehr_probs + ehr_attended.squeeze(1))

        return cxr_out, ecg_out, ehr_out
    


# ================= FULL MULTIMODAL CBM =================
class MultimodalCBM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Modality Encoders (Concept Predictors)
        self.cxr_encoder = CXREncoder(num_concepts=15)
        self.ecg_encoder = ECGEncoder(num_concepts=14)
        self.ehr_encoder = EHREncoder(input_dim=21, num_concepts=13)
        self.cross_modal_attention = CrossModalConceptAttention(
            cxr_dim=15, ecg_dim=14, ehr_dim=13
        )
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
        
        # Clean logits for the loss function
        cxr_concept_logits = self.cxr_encoder(image)
        ecg_concept_logits = self.ecg_encoder(waveform)
        ehr_concept_logits = self.ehr_encoder(ehr_features)

        # Probabilities for the bottleneck
        cxr_concept_probs = torch.sigmoid(cxr_concept_logits)
        ecg_concept_probs = torch.sigmoid(ecg_concept_logits)
        ehr_concept_probs = torch.sigmoid(ehr_concept_logits)

        # Add noise to probabilities only during training, after sigmoid
        if self.training:
            cxr_concept_probs = (cxr_concept_probs + torch.randn_like(cxr_concept_probs) * 0.1).clamp(0, 1)
            ecg_concept_probs = (ecg_concept_probs + torch.randn_like(ecg_concept_probs) * 0.1).clamp(0, 1)
            ehr_concept_probs = (ehr_concept_probs + torch.randn_like(ehr_concept_probs) * 0.1).clamp(0, 1)
        
        # 2. Apply Modality Dropouts
        # If mask is 0 (dropped), the probabilities become 0. 
        # We use .view(-1, 1) to ensure the mask shape aligns with the batch dimension
                # Apply modality dropout masks
        cxr_concept_probs = cxr_concept_probs * cxr_mask.view(-1, 1)
        ecg_concept_probs = ecg_concept_probs * ecg_mask.view(-1, 1)
        ehr_concept_probs = ehr_concept_probs * ehr_mask.view(-1, 1)

        # 3. Cross-modal attention in concept space
        cxr_attended, ecg_attended, ehr_attended = self.cross_modal_attention(
            cxr_concept_probs, ecg_concept_probs, ehr_concept_probs
        )

        # Re-apply masks after attention — attention can reintroduce signal
        # from dropped modalities through the context projection, so we
        # zero them out again to enforce the dropout contract
        cxr_attended = cxr_attended * cxr_mask.view(-1, 1)
        ecg_attended = ecg_attended * ecg_mask.view(-1, 1)
        ehr_attended = ehr_attended * ehr_mask.view(-1, 1)

        # 4. Fusion — now attention-refined concepts
        fused_concepts = torch.cat([cxr_attended, ecg_attended, ehr_attended], dim=1)  # (B, 42)

        # 5. Downstream predictions
        mortality_logits = self.mortality_head(fused_concepts).squeeze(-1)
        ahf_logits       = self.ahf_head(fused_concepts).squeeze(-1)

        return {
            'cxr_concept_logits': cxr_concept_logits,
            'ecg_concept_logits': ecg_concept_logits,
            'ehr_concept_logits': ehr_concept_logits,
            'cxr_attended':       cxr_attended,
            'ecg_attended':       ecg_attended,
            'ehr_attended':       ehr_attended,
            'mortality_logits':   mortality_logits,
            'ahf_logits':         ahf_logits,
        }