├── Datasets
│   ├── Data : Contains the raw data of ECG, CXR and EHR
│   │
│   ├── ECG: Contains the csv files and reports
│   │
│   ├── CXR: Contains the csv files and reports 
│   │
│   ├── MIMIC IV : Contains the csv files and reports  
│   │   
│   └── Preprocessed: Contains the main, train, val and test split csv files
│       
├── Preprocessing
│   ├── Concept_Vector_Preparation: Scripts used to prepare the concept vectors
│   │
│   ├── Data_Downloading: Scripts used to prepapre for data download
│   │
│   ├── Merging: Scripts to merge the 3 modalities



Steps:
1. Prepare CXR and ECG Concept Vectors
2. Use cxr_ehr.py file to generate cxr_mapped_to_hadm.csv
3. Use the EHR scripts to prepare the Concept Vectors
4. Run new_merge.py file in Merging folder
5. Download the necessary files



