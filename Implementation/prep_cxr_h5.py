import os
import h5py
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

CXR_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_cxr'
CSV_PATHS = ["/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv",
             "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv",
             "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"]
OUTPUT_H5 = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/cxr_images.h5'

def main():
    # Gather all unique report paths
    all_paths = set()
    for csv in CSV_PATHS:
        df = pd.read_csv(csv)
        all_paths.update(df['report_path'].dropna().apply(lambda x: str(x).replace('.txt', '')).tolist())
    
    print(f"Found {len(all_paths)} CXR folders. Packing into HDF5...")
    
    with h5py.File(OUTPUT_H5, 'w') as h5f:
        for folder in tqdm(all_paths):
            full_path = os.path.join(CXR_DIR, folder)
            if not os.path.exists(full_path): continue
            
            imgs = [f for f in os.listdir(full_path) if f.endswith('.jpg')]
            if not imgs: continue
            
            # Simple fallback for best image (you can use your view_map logic here)
            img_name = imgs[0] 
            
            try:
                img = Image.open(os.path.join(full_path, img_name)).convert('RGB')
                # Save as raw numpy array inside the HDF5 file
                h5f.create_dataset(folder, data=np.array(img), compression="lzf")
            except Exception:
                pass

if __name__ == "__main__":
    main()