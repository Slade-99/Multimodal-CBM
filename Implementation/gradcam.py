# gradcam.py
from email.mime import image
from xml.parsers.expat import model
import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image, waveform, ehr_features,
                 cxr_mask, ecg_mask, ehr_mask,
                 task='ahf'):
        self.model.eval()
        image = image.requires_grad_(True)

        outputs = self.model(image, waveform, ehr_features,
                             cxr_mask, ecg_mask, ehr_mask)

        if task == 'ahf':
            score = outputs['ahf_logits']
        else:
            score = outputs['mortality_logits']

        self.model.zero_grad()
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()

        # Normalise and resize to input image size
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cam


def overlay_gradcam(original_image_tensor, cam, alpha=0.4):
    """
    Overlays the GradCAM heatmap on the original image.
    Returns a numpy array (H, W, 3) suitable for plt.imshow.
    """
    # Denormalise the image
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = original_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img  = (img * std + mean).clip(0, 1)

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = alpha * heatmap + (1 - alpha) * img
    return overlay.clip(0, 1), img


# Usage
# Target the last conv layer of ResNet (layer4)
gradcam = GradCAM(model, model.cxr_encoder.backbone.layer4[-1])

cam = gradcam.generate(image, waveform, ehr_features,
                        cxr_mask, ecg_mask, ehr_mask, task='ahf')
overlay, original = overlay_gradcam(image, cam)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(original)
axes[0].set_title('Original CXR', fontsize=11)
axes[0].axis('off')
axes[1].imshow(overlay)
axes[1].set_title('GradCAM — AHF prediction', fontsize=11)
axes[1].axis('off')
plt.tight_layout()
plt.savefig('gradcam.pdf', dpi=300, bbox_inches='tight')