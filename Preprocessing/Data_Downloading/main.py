import pandas as pd

print("Loading Master Dataset and true ECG paths...")
df = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Preprocessing/MASTER_MULTIMODAL_DATASET.csv')

ecg_records = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Datasets/ECG/record_list.csv', usecols=['study_id', 'path'])
ecg_records['study_id'] = ecg_records['study_id'].astype(str)

seen_cxr = set()
seen_ecg = set()

with open('/home/azwad/Works/Multimodal-CBM/Preprocessing/gcs_cxr_urls.txt', 'w') as f_cxr, open('/home/azwad/Works/Multimodal-CBM/Preprocessing/wget_ecg_urls.txt', 'w') as f_ecg:
    for index, row in df.iterrows():
        
        # 1. CXR -> Google Cloud Storage Links
        if pd.notna(row['report_path']):
            folder_path = str(row['report_path']).replace('.txt', '')
            cxr_url = f"gs://mimic-cxr-jpg-2.1.0.physionet.org/files/{folder_path}/\n"
            if cxr_url not in seen_cxr:
                f_cxr.write(cxr_url)
                seen_cxr.add(cxr_url)
        
        # 2. ECG -> Direct PhysioNet Links (wget)
        if pd.notna(row['ecg_study_id']) and row['ecg_study_id'] != 0.0:
            ecg_id = str(int(row['ecg_study_id']))
            match = ecg_records[ecg_records['study_id'] == ecg_id]
            
            if not match.empty:
                exact_path = match.iloc[0]['path']
                base_url = f"https://physionet.org/files/mimic-iv-ecg/1.0/{exact_path}"
                
                dat_url = f"{base_url}.dat\n"
                hea_url = f"{base_url}.hea\n"
                
                if dat_url not in seen_ecg:
                    f_ecg.write(dat_url)
                    f_ecg.write(hea_url)
                    seen_ecg.add(dat_url)
                    seen_ecg.add(hea_url)

print("Success! Created 'gcs_cxr_urls.txt' and 'wget_ecg_urls.txt'.")