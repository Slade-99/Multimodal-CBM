import pandas as pd

# 1. Define the IDs we want to keep
vital_ids = [220045, 220210, 223761, 220052, 220277, 223901]
lab_ids = [50912, 50885, 51265, 51301, 50813, 50963, 51003]

# Define your file paths (change these to where you unzipped the files)
chartevents_path = '/home/azwad/Works/Multimodal-CBM/Datasets/MIMIC IV/chartevents.csv.gz'
labevents_path = '/home/azwad/Works/Multimodal-CBM/Datasets/MIMIC IV/labevents.csv.gz'

def extract_safe(file_path, target_ids, output_name):
    print(f"Starting extraction for {output_name}...")
    
    # Open the output file in write mode with headers
    first_chunk = True
    
    # Read the file in chunks of 1 million rows to save RAM
    chunk_iterator = pd.read_csv(file_path, chunksize=1000000, low_memory=False)
    
    for i, chunk in enumerate(chunk_iterator):
        # Filter: Only keep rows where the itemid is in our target list
        filtered_chunk = chunk[chunk['itemid'].isin(target_ids)]
        
        # We only need the patient ID, the test ID, the value, and the time
        cols_to_keep = ['subject_id', 'hadm_id', 'itemid', 'valuenum', 'charttime']
        
        # Some tables might use different time column names, handle gracefully
        available_cols = [col for col in cols_to_keep if col in filtered_chunk.columns]
        filtered_chunk = filtered_chunk[available_cols]
        
        # Append to our new small CSV file
        filtered_chunk.to_csv(output_name, mode='a', header=first_chunk, index=False)
        first_chunk = False
        
        # Print progress so you know it hasn't frozen
        print(f"Processed chunk {i+1}...")
        
    print(f"Finished! Saved to {output_name}")

# Run the extraction (this might take 10-20 minutes depending on your hard drive speed)
extract_safe(chartevents_path, vital_ids, '/home/azwad/Works/Multimodal-CBM/Preprocessing/Concept_Vector_Preparation/EHR/filtered_vitals.csv')
extract_safe(labevents_path, lab_ids, '/home/azwad/Works/Multimodal-CBM/Preprocessing/Concept_Vector_Preparation/EHR/filtered_labs.csv')