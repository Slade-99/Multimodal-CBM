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
    def __init__(self, cxr_dim=15, ecg_dim=14, ehr_dim=13, embed_dim=32):
        super().__init__()
        # Project each concept scalar to a shared embedding space
        # so concepts from different modalities are comparable
        self.proj_cxr = nn.Linear(1, embed_dim)
        self.proj_ecg = nn.Linear(1, embed_dim)
        self.proj_ehr = nn.Linear(1, embed_dim)

        # Cross-modal attention: each modality's concepts (as tokens)
        # attend to every other modality's concepts (as tokens)
        self.attn_cxr = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.attn_ecg = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.attn_ehr = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # Project back to scalar concept space
        self.out_cxr = nn.Linear(embed_dim, 1)
        self.out_ecg = nn.Linear(embed_dim, 1)
        self.out_ehr = nn.Linear(embed_dim, 1)

        self.norm_cxr = nn.LayerNorm(embed_dim)
        self.norm_ecg = nn.LayerNorm(embed_dim)
        self.norm_ehr = nn.LayerNorm(embed_dim)

    def forward(self, cxr_probs, ecg_probs, ehr_probs,
                cxr_mask, ecg_mask, ehr_mask):

        # Apply modality masks
        cxr_probs = cxr_probs * cxr_mask.view(-1, 1)
        ecg_probs = ecg_probs * ecg_mask.view(-1, 1)
        ehr_probs = ehr_probs * ehr_mask.view(-1, 1)

        # Each concept becomes a token: (B, num_concepts, embed_dim)
        # unsqueeze(-1) makes each concept scalar → (B, num_concepts, 1)
        q_cxr = self.proj_cxr(cxr_probs.unsqueeze(-1))  # (B, 15, embed_dim)
        q_ecg = self.proj_ecg(ecg_probs.unsqueeze(-1))  # (B, 14, embed_dim)
        q_ehr = self.proj_ehr(ehr_probs.unsqueeze(-1))  # (B, 13, embed_dim)

        # Cross-modal key/value: each modality queries the OTHER two
        kv_for_cxr = torch.cat([q_ecg, q_ehr], dim=1)  # (B, 27, embed_dim)
        kv_for_ecg = torch.cat([q_cxr, q_ehr], dim=1)  # (B, 28, embed_dim)
        kv_for_ehr = torch.cat([q_cxr, q_ecg], dim=1)  # (B, 29, embed_dim)

        # Attention — now seq_q=num_concepts, seq_k=other_concepts
        # weights shape: (B, heads, num_query_concepts, num_key_concepts)
        cxr_attended, cxr_weights = self.attn_cxr(q_cxr, kv_for_cxr, kv_for_cxr)
        ecg_attended, ecg_weights = self.attn_ecg(q_ecg, kv_for_ecg, kv_for_ecg)
        ehr_attended, ehr_weights = self.attn_ehr(q_ehr, kv_for_ehr, kv_for_ehr)

        # Residual + norm
        cxr_out = self.norm_cxr(q_cxr + cxr_attended)  # (B, 15, embed_dim)
        ecg_out = self.norm_ecg(q_ecg + ecg_attended)
        ehr_out = self.norm_ehr(q_ehr + ehr_attended)

        # Project back to concept scalars
        cxr_refined = self.out_cxr(cxr_out).squeeze(-1)  # (B, 15)
        ecg_refined = self.out_ecg(ecg_out).squeeze(-1)  # (B, 14)
        ehr_refined = self.out_ehr(ehr_out).squeeze(-1)  # (B, 13)

        return (cxr_refined, ecg_refined, ehr_refined,
                cxr_weights, ecg_weights, ehr_weights)
    


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
        # 1. Concept Prediction
        cxr_concept_logits = self.cxr_encoder(image)
        ecg_concept_logits = self.ecg_encoder(waveform)
        ehr_concept_logits = self.ehr_encoder(ehr_features)

        cxr_concept_probs = torch.sigmoid(cxr_concept_logits)
        ecg_concept_probs = torch.sigmoid(ecg_concept_logits)
        ehr_concept_probs = torch.sigmoid(ehr_concept_logits)

        # Training noise for robustness
        if self.training:
            cxr_concept_probs = (cxr_concept_probs + torch.randn_like(cxr_concept_probs) * 0.05).clamp(0, 1)
            # ... repeat for others ...

        # 2. Cross-modal attention (using the refined return values)
        # Note: I renamed 'refined' to 'attended' here to match your later logic
        cxr_attended, ecg_attended, ehr_attended, \
        cxr_attn_w, ecg_attn_w, ehr_attn_w = self.cross_modal_attention(
            cxr_concept_probs, ecg_concept_probs, ehr_concept_probs,
            cxr_mask, ecg_mask, ehr_mask
        )

        # 3. Re-apply masks to enforce the "Dropout Contract"
        cxr_attended = cxr_attended * cxr_mask.view(-1, 1)
        ecg_attended = ecg_attended * ecg_mask.view(-1, 1)
        ehr_attended = ehr_attended * ehr_mask.view(-1, 1)

        # 4. Fusion and Head
        fused_concepts = torch.cat([cxr_attended, ecg_attended, ehr_attended], dim=1)

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
            'cxr_attn_weights': cxr_attn_w,  # (B, heads, 15, 27)
            'ecg_attn_weights': ecg_attn_w,  # (B, heads, 14, 28)
            'ehr_attn_weights': ehr_attn_w,  # (B, heads, 13, 29)
        }