import torch
import torchvision.models as models

# 1. Load the RadImageNet file
raw_checkpoint = torch.load("/home/azwad/Works/Multimodal-CBM/ResNet50.pt", map_location='cpu', weights_only=False)

# Extract the dictionary
if 'state_dict' in raw_checkpoint:
    state_dict = raw_checkpoint['state_dict']
elif 'model' in raw_checkpoint:
    state_dict = raw_checkpoint['model']
else:
    state_dict = raw_checkpoint

# 2. Load a blank PyTorch ResNet50
backbone = models.resnet50(weights=None)
pytorch_keys = list(backbone.state_dict().keys())

# 3. Print the comparison
print("\n--- WHAT RADIMAGENET NAMED THEIR LAYERS ---")
print(list(state_dict.keys())[:10])

print("\n--- WHAT PYTORCH IS EXPECTING ---")
print(pytorch_keys[:10])
print("\n")