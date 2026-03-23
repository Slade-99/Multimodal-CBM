import pandas as pd

print("Loading data...")
# 1. Your CXR vectors generated earlier
cxr_df = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Preprocessing/Concept_Vector_Preparation/CXR/cxr_concept_vectors.csv')
cxr_df = cxr_df.rename(columns={'study_id': 'report_path'})

# 2. The metadata file you just downloaded from MIMIC-CXR-JPG
metadata_df = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Datasets/mimic-cxr-reports (1)/mimic-cxr-2.0.0-metadata.csv', usecols=['subject_id', 'study_id', 'StudyDate', 'StudyTime'])
metadata_df = metadata_df.drop_duplicates(subset=['subject_id', 'study_id'])

# 3. The admissions file from MIMIC-IV EHR
# Update this path to where your MIMIC-IV admissions.csv is located
adm_df = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Datasets/MIMIC IV /admissions.csv', usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'])

# Function to clean the text path (e.g., p10/p10659857/s59206984.txt)
def clean_ids(path_string):
    parts = path_string.split('/') 
    subject_id = int(parts[1].replace('p', ''))
    study_id = int(parts[2].replace('s', '').replace('.txt', ''))
    return pd.Series([subject_id, study_id])

print("Extracting IDs from CXR paths...")
cxr_df[['subject_id', 'study_id']] = cxr_df['report_path'].apply(clean_ids)

print("Attaching timestamps...")
# Join CXR with metadata to get the X-ray time
cxr_with_time = pd.merge(cxr_df, metadata_df, on=['subject_id', 'study_id'], how='left')

# Combine StudyDate and StudyTime into a single, usable timestamp
cxr_with_time['StudyDateTime'] = pd.to_datetime(
    cxr_with_time['StudyDate'].astype(str).str[:8] + ' ' + 
    cxr_with_time['StudyTime'].astype(str).str.zfill(6).str[:6], 
    format='%Y%m%d %H%M%S', errors='coerce'
)

print("Mapping to hospital admissions (hadm_id)...")
adm_df['admittime'] = pd.to_datetime(adm_df['admittime'])
adm_df['dischtime'] = pd.to_datetime(adm_df['dischtime'])

# Merge all admissions for each patient
merged_df = pd.merge(cxr_with_time, adm_df, on='subject_id', how='left')

# Filter: Only keep rows where the X-ray was taken during the hospital stay
# Note: We add a 24-hour buffer to admittime because X-rays are often taken in the ER right before formal admission
valid_merges = merged_df[
    (merged_df['StudyDateTime'] >= (merged_df['admittime'] - pd.Timedelta(hours=24))) & 
    (merged_df['StudyDateTime'] <= merged_df['dischtime'])
].copy()

# Drop the extra time columns so the spreadsheet remains lightweight
cols_to_drop = ['StudyDate', 'StudyTime', 'admittime', 'dischtime']
final_mapped_cxr = valid_merges.drop(columns=cols_to_drop)

# Save it safely
final_mapped_cxr.to_csv('cxr_mapped_to_hadm.csv', index=False)
print(f"Success! Mapped {len(final_mapped_cxr)} X-rays to exact hospital admissions.")