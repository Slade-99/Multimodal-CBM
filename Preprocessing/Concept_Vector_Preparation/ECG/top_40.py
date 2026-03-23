import pandas as pd
from collections import Counter
import string

# Update this path to where your MIMIC-IV-ECG file is located
ecg_file_path = '/home/azwad/Works/Multimodal-CBM/Datasets/ECG/machine_measurements.csv'

# 1. Substrings that mean the phrase is useless noise or normal
noise_keywords = [
    'normal', 'nonspecific', 'variant', 'borderline', 
    'abnormal ecg', 'sinus rhythm', 'artifact','sinus arrhythmia'
]

# 2. Substring Binning Map (If the raw phrase contains the Key, map it to the Value)
# We use this to catch all variations, acronyms, and locations.
substring_map = {
    'infarct': 'myocardial_infarction',
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
    'first degree': 'first_degree_av_block',
    '1st degree': 'first_degree_av_block',
    'ischemia': 'myocardial_ischemia',
    'iv conduction defect': 'iv_conduction_defect'
}

phrase_counts = Counter()

print("Scanning ECG machine measurements and applying binning...")

chunk_iterator = pd.read_csv(ecg_file_path, chunksize=100000, low_memory=False)

for i, chunk in enumerate(chunk_iterator):
    report_cols = [col for col in chunk.columns if str(col).startswith('report_')]
    
    for col in report_cols:
        phrases = chunk[col].dropna().astype(str)
        
        for raw_phrase in phrases:
            # Clean: lowercase and remove punctuation (like the rogue periods)
            clean_phrase = raw_phrase.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            
            # Skip if it contains any noise keywords
            if any(noise in clean_phrase for noise in noise_keywords):
                continue
                
            # Check if it matches any of our master concepts
            found_concept = False
            for keyword, master_concept in substring_map.items():
                if keyword in clean_phrase:
                    phrase_counts[master_concept] += 1
                    found_concept = True
                    break # Stop checking once we find a match
            
            # If it's a new abnormality we didn't map yet, count it as is
            if not found_concept and len(clean_phrase) > 3:
                phrase_counts[clean_phrase] += 1

print("\n--- Refined Top 15 ECG Master Concepts ---")
top_15 = phrase_counts.most_common(15)

for i, (concept, count) in enumerate(top_15):
    print(f"{i+1}. {concept} ({count})")