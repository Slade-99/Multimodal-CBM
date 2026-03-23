import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================================================================
# PATHS  — edit these to match your environment
# ================================================================

BASE_DIR    = '/home/azwad/Works/Multimodal-CBM'
DATASET_DIR = f'{BASE_DIR}/Datasets'
MIMIC_DIR   = f'{DATASET_DIR}/MIMIC IV'
ECG_DIR     = f'{DATASET_DIR}/ECG'
CXR_DIR     = f'{DATASET_DIR}/CXR'
PREP_DIR    = f'{BASE_DIR}/Preprocessing'

# --- Inputs from earlier pipeline stages ---
CXR_HADM_CSV      = f'{CXR_DIR}/cxr_mapped_to_hadm.csv'
EHR_CONCEPTS_CSV  = f'{PREP_DIR}/Concept_Vector_Preparation/EHR/ehr_concept_vectors.csv'
ECG_CONCEPTS_CSV  = f'{PREP_DIR}/Concept_Vector_Preparation/ECG/ecg_concept_vectors.csv'
ECG_RECORD_LIST   = f'{ECG_DIR}/record_list.csv'

# --- MIMIC-IV source tables ---
ADMISSIONS_CSV    = f'{MIMIC_DIR}/admissions.csv.gz'
DIAGNOSES_CSV     = f'{MIMIC_DIR}/diagnoses_icd.csv.gz'
PATIENTS_CSV      = f'{MIMIC_DIR}/patients.csv.gz'
CHARTEVENTS_CSV   = f'{MIMIC_DIR}/chartevents.csv.gz'
LABEVENTS_CSV     = f'{MIMIC_DIR}/labevents.csv.gz'
INPUTEVENTS_CSV   = f'{MIMIC_DIR}/inputevents.csv.gz'
OUTPUTEVENTS_CSV  = f'{MIMIC_DIR}/outputevents.csv.gz'

# --- Output ---
SAVE_DIR = f'{DATASET_DIR}/Updated_Preprocessed/New_2'
os.makedirs(SAVE_DIR, exist_ok=True)

# ================================================================
# ITEM ID MAPPINGS
# ================================================================

CHART_ITEMS = {
    220045: 'heart_rate',
    220210: 'resp_rate',
    223761: 'temperature',
    220052: 'mean_bp',
    220277: 'spo2',
    223901: 'gcs_motor',
    223900: 'gcs_verbal',
    220739: 'gcs_eyes',
    226512: 'weight',
    226730: 'height',
}

LAB_ITEMS = {
    50912: 'creatinine',
    50885: 'bilirubin',
    51265: 'platelets',
    51301: 'wbc',
    50813: 'lactate',
    50963: 'bnp',
    51003: 'troponin',
    50868: 'anion_gap',
    51006: 'bun',
    51277: 'rdw',
}

VASO_ITEMS = [221906, 221289, 222315, 221662, 221749]

# ================================================================
# STAGE 1 — BUILD MASTER MULTIMODAL DATASET
# ================================================================

def build_master_dataset():
    print("\n" + "="*60)
    print("STAGE 1: Building master multimodal dataset")
    print("="*60)

    # --- Load CXR anchors ---
    print("Loading CXR anchors...")
    cxr_df = pd.read_csv(CXR_HADM_CSV)
    cxr_df['StudyDateTime'] = pd.to_datetime(cxr_df['StudyDateTime'])
    cxr_df = cxr_df.sort_values('StudyDateTime').reset_index(drop=True)

    # --- Load windowed EHR concept vectors (hadm_id + study_id keyed) ---
    print("Loading windowed EHR concept vectors...")
    ehr_concepts_df = pd.read_csv(EHR_CONCEPTS_CSV)
    # Rename study_id to avoid collision with CXR study_id
    ehr_concepts_df = ehr_concepts_df.rename(columns={'study_id': 'cxr_study_id_ehr'})

    # --- Load ECG concepts and timestamps ---
    print("Loading ECG concepts and timestamps...")
    ecg_concepts = pd.read_csv(ECG_CONCEPTS_CSV)
    ecg_times = pd.read_csv(
        ECG_RECORD_LIST,
        usecols=['subject_id', 'study_id', 'ecg_time']
    )
    ecg_times['EcgDateTime'] = pd.to_datetime(ecg_times['ecg_time'])
    ecg_df = pd.merge(
        ecg_concepts,
        ecg_times[['subject_id', 'study_id', 'EcgDateTime']],
        on=['subject_id', 'study_id']
    )
    ecg_df = ecg_df.sort_values('EcgDateTime').reset_index(drop=True)
    ecg_df = ecg_df.rename(columns={'study_id': 'ecg_study_id'})

    # --- Time-align ECG to CXR (48-hour backward window) ---
    print("Time-aligning ECG to CXR within 48-hour window...")
    multimodal_df = pd.merge_asof(
        cxr_df,
        ecg_df,
        by='subject_id',
        left_on='StudyDateTime',
        right_on='EcgDateTime',
        direction='backward',
        tolerance=pd.Timedelta('48h')
    )
    multimodal_df = multimodal_df.drop(columns=['EcgDateTime'], errors='ignore')

    # Fill missing ECG concepts with 0 (no ECG in window = no finding)
    ecg_concept_cols = [
        c for c in ecg_concepts.columns
        if c not in ['subject_id', 'study_id']
    ]
    multimodal_df[ecg_concept_cols] = (
        multimodal_df[ecg_concept_cols].fillna(0).astype(int)
    )

    # --- Merge windowed EHR concept vectors ---
    # EHR concepts are keyed by (hadm_id, cxr study_id) so each CXR
    # gets the concept vector computed from its own 48-hour window
    print("Merging windowed EHR concept vectors...")
    ehr_merge_key = ehr_concepts_df.rename(
        columns={'cxr_study_id_ehr': 'study_id'}
    )
    multimodal_df = pd.merge(
        multimodal_df,
        ehr_merge_key,
        on=['hadm_id', 'study_id'],
        how='left'
    )

    ehr_concept_cols = [
        c for c in ehr_concepts_df.columns
        if c not in ['hadm_id', 'cxr_study_id_ehr']
    ]
    multimodal_df[ehr_concept_cols] = (
        multimodal_df[ehr_concept_cols].fillna(0).astype(int)
    )

    # --- Mortality label ---
    print("Attaching mortality labels...")
    adm_df = pd.read_csv(
        ADMISSIONS_CSV,
        usecols=['hadm_id', 'hospital_expire_flag']
    )
    multimodal_df = pd.merge(multimodal_df, adm_df, on='hadm_id', how='left')
    multimodal_df = multimodal_df.rename(
        columns={'hospital_expire_flag': 'label_mortality'}
    )

    # --- AHF label (ICD-9 428* / ICD-10 I50*) ---
    print("Attaching AHF labels...")
    icd_df = pd.read_csv(
        DIAGNOSES_CSV,
        usecols=['hadm_id', 'icd_code', 'icd_version']
    )
    ahf_hadm_ids = icd_df[
        ((icd_df['icd_version'] == 9) &
         (icd_df['icd_code'].astype(str).str.startswith('428'))) |
        ((icd_df['icd_version'] == 10) &
         (icd_df['icd_code'].astype(str).str.startswith('I50')))
    ]['hadm_id'].unique()

    multimodal_df['label_ahf'] = (
        multimodal_df['hadm_id'].isin(ahf_hadm_ids).astype(int)
    )

    print(f"Master dataset: {len(multimodal_df)} rows")
    print(f"  AHF cases:       {multimodal_df['label_ahf'].sum()}")
    print(f"  Mortality cases: {multimodal_df['label_mortality'].sum()}")

    return multimodal_df


# ================================================================
# STAGE 2 — PATIENT-LEVEL TRAIN / VAL / TEST SPLIT
# ================================================================

def split_dataset(master_df, train_ratio=0.7, val_ratio=0.15, seed=42):
    print("\n" + "="*60)
    print("STAGE 2: Patient-level train/val/test split")
    print("="*60)

    patients = master_df['subject_id'].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_patients = set(patients[:n_train])
    val_patients   = set(patients[n_train:n_train + n_val])
    test_patients  = set(patients[n_train + n_val:])

    train_df = master_df[master_df['subject_id'].isin(train_patients)].copy()
    val_df   = master_df[master_df['subject_id'].isin(val_patients)].copy()
    test_df  = master_df[master_df['subject_id'].isin(test_patients)].copy()

    print(f"  Train: {len(train_df)} rows ({len(train_patients)} patients)")
    print(f"  Val:   {len(val_df)} rows ({len(val_patients)} patients)")
    print(f"  Test:  {len(test_df)} rows ({len(test_patients)} patients)")

    return train_df, val_df, test_df


# ================================================================
# STAGE 3 — EHR CONTINUOUS FEATURE EXTRACTION
# ================================================================

def extract_events(filepath, item_dict, hadm_ids):
    """
    Reads a large MIMIC events file in chunks.
    Returns one row per hadm_id with the latest observed value
    for each feature, then takes the median across chunks.
    """
    chunks = []
    for chunk in tqdm(
        pd.read_csv(
            filepath,
            chunksize=10**6,
            usecols=['hadm_id', 'itemid', 'valuenum', 'charttime'],
            low_memory=False
        ),
        desc=f"  Extracting {os.path.basename(filepath)}"
    ):
        chunk = chunk.dropna(subset=['valuenum', 'hadm_id'])
        chunk = chunk[chunk['itemid'].isin(item_dict)]
        chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]
        if chunk.empty:
            continue

        chunk['feature'] = chunk['itemid'].map(item_dict)
        chunk = chunk.sort_values('charttime')
        latest = chunk.groupby(['hadm_id', 'feature']).tail(1)
        chunks.append(latest[['hadm_id', 'feature', 'valuenum']])

    if not chunks:
        return pd.DataFrame(columns=['hadm_id'])

    df = pd.concat(chunks)
    df = (
        df.groupby(['hadm_id', 'feature'])['valuenum']
          .median()
          .unstack()
          .reset_index()
    )
    return df


def compute_bmi(df):
    df['height_m'] = df['height'] / 100.0
    df['bmi'] = df['weight'] / (df['height_m'] ** 2)
    df = df.drop(columns=['height_m', 'height', 'weight'], errors='ignore')
    return df


def get_demographics(hadm_ids, subject_ids):
    patients = pd.read_csv(PATIENTS_CSV, usecols=['subject_id', 'anchor_age'])
    admissions = pd.read_csv(ADMISSIONS_CSV, usecols=['hadm_id', 'admission_type'])

    patients = patients[patients['subject_id'].isin(subject_ids)]
    patients = patients.rename(columns={'anchor_age': 'age'})
    patients['age'] = patients['age'].astype(float)

    admissions = admissions[admissions['hadm_id'].isin(hadm_ids)]
    admissions['admission_emergency'] = admissions['admission_type'].isin(
        ['EMERGENCY', 'URGENT']
    ).astype(int)

    return patients[['subject_id', 'age']], admissions[['hadm_id', 'admission_emergency']]


def process_interventions(hadm_ids):
    fluid_in  = []
    fluid_out = []
    vaso      = []

    print("  Processing input events (fluid in + vasopressors)...")
    for chunk in tqdm(
        pd.read_csv(
            INPUTEVENTS_CSV,
            chunksize=10**6,
            usecols=['hadm_id', 'itemid', 'amount'],
            low_memory=False
        )
    ):
        chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]
        valid = chunk.dropna(subset=['amount'])
        if not valid.empty:
            fluid_in.append(valid.groupby('hadm_id')['amount'].sum())

        v = chunk[chunk['itemid'].isin(VASO_ITEMS)]
        if not v.empty:
            vaso.append(v[['hadm_id']].drop_duplicates())

    print("  Processing output events (fluid out)...")
    for chunk in tqdm(
        pd.read_csv(
            OUTPUTEVENTS_CSV,
            chunksize=10**6,
            usecols=['hadm_id', 'value'],
            low_memory=False
        )
    ):
        chunk = chunk.dropna()
        chunk = chunk[chunk['hadm_id'].isin(hadm_ids)]
        if not chunk.empty:
            fluid_out.append(chunk.groupby('hadm_id')['value'].sum())

    df_in  = pd.concat(fluid_in).groupby(level=0).sum().rename('fluid_in')
    df_out = pd.concat(fluid_out).groupby(level=0).sum().rename('fluid_out')

    df = pd.concat([df_in, df_out], axis=1).fillna(0)
    df['fluid_balance'] = df['fluid_in'] - df['fluid_out']
    df = df.drop(columns=['fluid_in', 'fluid_out'])

    if vaso:
        df_vaso = pd.concat(vaso).drop_duplicates()
        df_vaso['vasopressor_usage'] = 1
    else:
        df_vaso = pd.DataFrame(columns=['hadm_id', 'vasopressor_usage'])

    df = df.reset_index().merge(df_vaso, on='hadm_id', how='left')
    df['vasopressor_usage'] = df['vasopressor_usage'].fillna(0).astype(int)

    return df


def apply_concepts(df):
    """
    Derives binary EHR concept labels from continuous features.
    GCS components are summed; missing components assume normal score.
    """
    # GCS — fill missing components with normal values before summing
    gcs_df = df[['gcs_motor', 'gcs_verbal', 'gcs_eyes']].fillna(
        {'gcs_motor': 6, 'gcs_verbal': 5, 'gcs_eyes': 4}
    )
    df['gcs_total']      = gcs_df.sum(axis=1)
    df['altered_mental'] = (df['gcs_total'] < 15).astype(int)

    df['tachycardia']   = (df['heart_rate'] > 100).astype(int)
    df['tachypnea']     = (df['resp_rate'] > 20).astype(int)
    df['abnormal_temp'] = (
        (df['temperature'] < 96.8) | (df['temperature'] > 100.4)
    ).astype(int)
    df['hypotension']   = (df['mean_bp'] < 65).astype(int)
    df['hypoxia']       = (df['spo2'] < 90).astype(int)

    df['elevated_creatinine'] = (df['creatinine'] > 1.2).astype(int)
    df['elevated_bilirubin']  = (df['bilirubin'] > 1.3).astype(int)
    df['thrombocytopenia']    = (df['platelets'] < 150).astype(int)
    df['abnormal_wbc']        = (
        (df['wbc'] < 4) | (df['wbc'] > 12)
    ).astype(int)
    df['hyperlactatemia']     = (df['lactate'] > 2.0).astype(int)

    # Age-adjusted BNP threshold (ACC/AHA guideline)
    df['elevated_bnp'] = (
        ((df['age'] < 75)  & (df['bnp'] > 125)) |
        ((df['age'] >= 75) & (df['bnp'] > 450))
    ).astype(int)

    df['elevated_troponin'] = (df['troponin'] > 0.04).astype(int)

    # Drop raw GCS components — gcs_total is kept as a continuous feature
    df = df.drop(columns=['gcs_motor', 'gcs_verbal', 'gcs_eyes'], errors='ignore')

    return df


def enrich_with_ehr_features(split_df, chart_df, lab_df,
                              patients_df, admissions_df, interventions_df):
    """
    Merges all continuous EHR features into a split dataframe
    and derives concept labels.
    """
    df = split_df.copy()
    df = df.merge(patients_df,     on='subject_id', how='left')
    df = df.merge(admissions_df,   on='hadm_id',    how='left')
    df = df.merge(chart_df,        on='hadm_id',    how='left')
    df = df.merge(lab_df,          on='hadm_id',    how='left')
    df = df.merge(interventions_df,on='hadm_id',    how='left')
    df = compute_bmi(df)
    df = apply_concepts(df)
    return df


# ================================================================
# MAIN
# ================================================================

def main():
    # --- Stage 1: Build master multimodal dataset ---
    master_df = build_master_dataset()

    # --- Stage 2: Patient-level split ---
    train_df, val_df, test_df = split_dataset(master_df)

    # --- Stage 3: Extract continuous EHR features ---
    print("\n" + "="*60)
    print("STAGE 3: Extracting continuous EHR features")
    print("="*60)

    all_hadm_ids   = master_df['hadm_id'].dropna().astype(int).unique()
    all_subject_ids= master_df['subject_id'].dropna().astype(int).unique()

    print("Extracting chart events...")
    chart_df = extract_events(CHARTEVENTS_CSV, CHART_ITEMS, all_hadm_ids)

    print("Extracting lab events...")
    lab_df = extract_events(LABEVENTS_CSV, LAB_ITEMS, all_hadm_ids)

    print("Extracting demographics...")
    patients_df, admissions_df = get_demographics(all_hadm_ids, all_subject_ids)

    print("Extracting fluid balance and vasopressor usage...")
    interventions_df = process_interventions(all_hadm_ids)

    # --- Stage 4: Enrich each split and save ---
    print("\n" + "="*60)
    print("STAGE 4: Enriching splits and saving")
    print("="*60)

    train_final = enrich_with_ehr_features(
        train_df, chart_df, lab_df,
        patients_df, admissions_df, interventions_df
    )
    val_final = enrich_with_ehr_features(
        val_df, chart_df, lab_df,
        patients_df, admissions_df, interventions_df
    )
    test_final = enrich_with_ehr_features(
        test_df, chart_df, lab_df,
        patients_df, admissions_df, interventions_df
    )

    train_final.to_csv(f'{SAVE_DIR}/train_final.csv', index=False)
    val_final.to_csv(f'{SAVE_DIR}/val_final.csv',   index=False)
    test_final.to_csv(f'{SAVE_DIR}/test_final.csv',  index=False)

    print(f"\n  Train rows: {len(train_final)}")
    print(f"  Val rows:   {len(val_final)}")
    print(f"  Test rows:  {len(test_final)}")
    print("\nDone. Final datasets saved to:", SAVE_DIR)


if __name__ == '__main__':
    main()