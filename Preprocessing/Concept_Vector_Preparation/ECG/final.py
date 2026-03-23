import pandas as pd
import string

# Update this path to where your MIMIC-IV-ECG file is located
ecg_file_path = '/home/azwad/Works/Multimodal-CBM/Datasets/ECG/machine_measurements.csv'

# Your finalized Top 14 ECG concepts
top_14_concepts = [
    'myocardial_infarction', 'left_axis_deviation', 'myocardial_ischemia',
    'sinus_bradycardia', 'atrial_fibrillation', 'low_qrs_voltage',
    'sinus_tachycardia', 'right_bundle_branch_block', 'left_ventricular_hypertrophy',
    'prolonged_qt_interval', 'left_bundle_branch_block', 'iv_conduction_defect',
    'pacemaker_rhythm', 'left_atrial_abnormality'
]

# Words that mean the phrase is healthy or useless noise
noise_keywords = [
    'normal', 'nonspecific', 'variant', 'borderline', 
    'abnormal ecg', 'sinus rhythm', 'artifact', 'sinus arrhythmia'
]

# The translation map to catch all variations
substring_map = {
    'infarct': 'myocardial_infarction',
    'ischemia': 'myocardial_ischemia',
    'lvh': 'left_ventricular_hypertrophy',
    'left ventricular hypertrophy': 'left_ventricular_hypertrophy',
    'left axis': 'left_axis_deviation',
    'leftward axis': 'left_axis_deviation',
    'atrial fibrillation': 'atrial_fibrillation',
    'right bundle branch': 'right_bundle_branch_block',
    'rbbb': 'right_bundle_branch_block',
    'left bundle branch': 'left_bundle_branch_block',
    'lbbb': 'left_bundle_branch_block',
    'pacemaker': 'pacemaker_rhythm',
    'low qrs': 'low_qrs_voltage',
    'prolonged qt': 'prolonged_qt_interval',
    'atrial abnormality': 'left_atrial_abnormality',
    'atrial enlargement': 'left_atrial_abnormality',
    'sinus tachycardia': 'sinus_tachycardia',
    'sinus bradycardia': 'sinus_bradycardia',
    'iv conduction defect': 'iv_conduction_defect'
}

print("Generating final ECG concept vectors in chunks...")
results = []

# Process in chunks to keep memory usage low
chunk_iterator = pd.read_csv(ecg_file_path, chunksize=100000, low_memory=False)

for i, chunk in enumerate(chunk_iterator):
    # Find the columns that contain the machine text
    report_cols = [col for col in chunk.columns if str(col).startswith('report_')]
    
    for index, row in chunk.iterrows():
        # Create a dictionary starting with 0 for all 14 concepts
        vector = {concept: 0 for concept in top_14_concepts}
        vector['subject_id'] = row['subject_id']
        vector['study_id'] = row['study_id']
        
        # Check each phrase in this specific ECG report
        for col in report_cols:
            if pd.isna(row[col]):
                continue
                
            clean_phrase = str(row[col]).lower().translate(str.maketrans('', '', string.punctuation)).strip()
            
            # Skip if it is a healthy or noisy phrase
            if any(noise in clean_phrase for noise in noise_keywords):
                continue
                
            # If we find a keyword, flip the concept's value to 1
            for keyword, master_concept in substring_map.items():
                if keyword in clean_phrase and master_concept in top_14_concepts:
                    vector[master_concept] = 1
        
        results.append(vector)
        
    print(f"Processed chunk {i+1}...")

# Convert results into a table and save it
df_concepts = pd.DataFrame(results)
# Rearrange columns to put the IDs at the front
cols = ['subject_id', 'study_id'] + top_14_concepts
df_concepts = df_concepts[cols]

df_concepts.to_csv('ecg_concept_vectors.csv', index=False)
print(f"Success! Saved {len(df_concepts)} ECG records to 'ecg_concept_vectors.csv'")