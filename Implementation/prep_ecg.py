import os
import pandas as pd
import numpy as np
import wfdb
from tqdm import tqdm

# ================= CONFIGURATION =================
ECG_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_ecg'
# We will save all the fast files in a single, flat directory for easy loading
FAST_ECG_DIR = '/home/azwad/Works/Multimodal-CBM/Datasets/Data/mimic_ecg_npy'

CSV_PATHS = [
    "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv",
    "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv",
    "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"
]

def main():
    os.makedirs(FAST_ECG_DIR, exist_ok=True)
    
    # 1. Collect all unique ECGs across all datasets
    print("Gathering unique ECG IDs from CSVs...")
    unique_ecgs = {}
    
    for csv_path in CSV_PATHS:
        df = pd.read_csv(csv_path)
        # Drop rows where ecg_study_id is missing
        df_valid = df.dropna(subset=['ecg_study_id'])
        
        for _, row in df_valid.iterrows():
            eid = str(int(row['ecg_study_id']))
            sid = str(int(row['subject_id']))
            if eid not in unique_ecgs:
                unique_ecgs[eid] = sid

    print(f"Found {len(unique_ecgs)} unique ECG records to process.")
    
    # 2. Convert and save
    success_count = 0
    fail_count = 0
    
    for eid, sid in tqdm(unique_ecgs.items(), desc="Converting to .npy"):
        out_path = os.path.join(FAST_ECG_DIR, f"{eid}.npy")
        
        # Skip if already processed (useful if the script gets interrupted)
        if os.path.exists(out_path):
            success_count += 1
            continue
            
        try:
            # Reconstruct the original MIMIC folder structure
            original_path = os.path.join(ECG_DIR, f"p{sid[:4]}", f"p{sid}", f"s{eid}", eid)
            
            # Read the slow wfdb file
            record = wfdb.rdrecord(original_path)
            signal = record.p_signal
            signal = np.nan_to_num(signal)
            
            # Perform the per-channel z-score normalisation offline to save training time
            signal = (signal - signal.mean(axis=0)) / (signal.std(axis=0) + 1e-8)
            signal = signal.T  # Transpose to (12, 5000)
            
            # Save as a fast, raw numpy array
            # Using float32 instead of float64 cuts the file size in half without losing necessary precision
            np.save(out_path, signal.astype(np.float32))
            success_count += 1
            
        except Exception as e:
            fail_count += 1
            
    print("\n--- Conversion Complete ---")
    print(f"Successfully converted: {success_count}")
    print(f"Failed to read/find: {fail_count}")

if __name__ == "__main__":
    main()