import torch
import torch.nn as nn
from net1d import Net1D 

def load_ecgfounder_for_mimic(num_mimic_classes, checkpoint_path='ECGFounder/checkpoint/12_lead_ECGFounder.pth'):
    
    model = Net1D(      in_channels=12, 
      base_filters=64, #32 64
      ratio=1, 
      filter_list=[64,160,160,400,400,1024,1024],    #[16,32,32,80,80,256,256] [32,64,64,160,160,512,512] [64,160,160,400,400,1024,1024]
      m_blocks_list=[2,2,2,3,3,4,4],   #[2,2,2,2,2,2,2] [2,2,2,3,3,4,4]
      kernel_size=16, 
      stride=2, 
      groups_width=16,
      verbose=True, 
      use_bn=False,
      use_do=False,
      n_classes=num_mimic_classes)
    
    # 3. Load the pre-trained weights downloaded from Hugging Face
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle potential "module." prefix if the model was trained with DataParallel
    if "module." in list(state_dict.keys())[0]:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict, strict=False)
    #print("Pre-trained ECGFounder weights loaded successfully.")
    

    return model

# Example Usage: Assuming you are predicting 5 specific conditions in MIMIC
num_classes = 14
model = load_ecgfounder_for_mimic(num_classes)

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#print(model)