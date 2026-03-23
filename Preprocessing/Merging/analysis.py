import pandas as pd

print("Loading Master Dataset...")
df = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Datasets/Updated_Preprocessed/New_2/val_final.csv')

# 1. CXR is our anchor, so it is always present
df['has_CXR'] = True

# 2. EHR is present if we successfully mapped it to a hospital admission
df['has_EHR'] = df['hadm_id'].notna()

# 3. ECG is present if it successfully found an ECG within the 48-hour window
df['has_ECG'] = df['ecg_study_id'].notna() & (df['ecg_study_id'] != 0.0)

# 4. Create a label for the combination
def get_combination(row):
    mods = []
    if row['has_CXR']: mods.append('CXR')
    if row['has_EHR']: mods.append('EHR')
    if row['has_ECG']: mods.append('ECG')
    return " + ".join(mods)

df['Combination'] = df.apply(get_combination, axis=1)

print("\n--- Modality Availability Counts ---")
combination_counts = df['Combination'].value_counts()
for combo, count in combination_counts.items():
    percentage = (count / len(df)) * 100
    print(f"{combo}: {count} rows ({percentage:.1f}%)")