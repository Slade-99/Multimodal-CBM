import pandas as pd
import numpy as np

print("Loading data...")
vitals = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Preprocessing/Concept_Vector_Preparation/EHR/filtered_vitals.csv')
labs   = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Preprocessing/Concept_Vector_Preparation/EHR/filtered_labs.csv')
# Load CXR anchor times — one row per CXR with hadm_id and StudyDateTime
cxr_anchors = pd.read_csv('/home/azwad/Works/Multimodal-CBM/Datasets/mimic-cxr-reports (1)/cxr_mapped_to_hadm.csv',
                           usecols=['hadm_id', 'study_id', 'StudyDateTime'])
cxr_anchors['StudyDateTime'] = pd.to_datetime(cxr_anchors['StudyDateTime'])

df = pd.concat([vitals, labs], ignore_index=True)
df = df.dropna(subset=['valuenum', 'hadm_id'])
df['charttime'] = pd.to_datetime(df['charttime'])

print("Joining measurements with CXR anchor times...")
# Each CXR anchor gets all measurements for that admission
df_merged = pd.merge(df, cxr_anchors, on='hadm_id', how='inner')

# Keep only measurements within 48h BEFORE the CXR
window_start = df_merged['StudyDateTime'] - pd.Timedelta(hours=48)
df_merged = df_merged[
    (df_merged['charttime'] >= window_start) &
    (df_merged['charttime'] <= df_merged['StudyDateTime'])
]

print("Applying clinical thresholds per CXR anchor...")

def check_thresholds(patient_data):
    tests   = patient_data['itemid'].values
    values  = patient_data['valuenum'].values
    records = {}
    for t, v in zip(tests, values):
        records.setdefault(t, []).append(v)

    def flag_if(item_id, condition_func):
        return 1 if item_id in records and any(
            condition_func(v) for v in records[item_id]) else 0

    return pd.Series({
        'tachycardia':        flag_if(220045, lambda x: x > 100),
        'tachypnea':          flag_if(220210, lambda x: x > 20),
        'abnormal_temp':      flag_if(223761, lambda x: x < 96.8 or x > 100.4),
        'hypotension':        flag_if(220052, lambda x: x < 65),
        'hypoxia':            flag_if(220277, lambda x: x < 90),
        'altered_mental':     flag_if(223901, lambda x: x < 6),
        'elevated_creatinine':flag_if(50912,  lambda x: x > 1.2),
        'elevated_bilirubin': flag_if(50885,  lambda x: x > 1.3),
        'thrombocytopenia':   flag_if(51265,  lambda x: x < 150),
        'abnormal_wbc':       flag_if(51301,  lambda x: x < 4 or x > 12),
        'hyperlactatemia':    flag_if(50813,  lambda x: x > 2.0),
        'elevated_bnp':       flag_if(50963,  lambda x: x > 125),
        'elevated_troponin':  flag_if(51003,  lambda x: x > 0.04)
    })

# Now group by individual CXR (study_id) not just hadm_id
final_ehr_concepts = (df_merged
    .groupby(['hadm_id', 'study_id'])
    .apply(check_thresholds)
    .reset_index())

final_ehr_concepts.to_csv('/home/azwad/Works/Multimodal-CBM/Preprocessing/Concept_Vector_Preparation/EHR/ehr_concept_vectors.csv', index=False)
print(f"Done. {len(final_ehr_concepts)} CXR-anchored EHR concept vectors saved.")