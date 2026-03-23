import json
import pandas as pd

file_path = '/home/azwad/Works/Multimodal-CBM/Datasets/MIMIC-CXR_graphs.json'

# The exact Top 15 list you just generated
top_15_concepts = [
    'lung_opacity', 'pleural_effusion', 'support_tube', 'heart_enlarged', 
    'lung_atelectasis', 'pulmonary_edema', 'support_line', 'vascular_congestion', 
    'lungs_hyperinflated', 'apical_pneumothorax', 'hemidiaphragm_elevation', 
    'rib_fractures', 'interstitial_markings', 'volume_loss', 'atrium_leads'
]

# The exact same filters you used to clean the list
filler_words = {
    'normal', 'unchanged', 'clear', 'stable', 'mild', 'moderate', 
    'small', 'large', 'intact', 'tip', 'essentially', 'unremarkable', 
    'low', 'free', 'negative', 'gross', 'grossly', 'expanded', 
    'degenerative', 'tortuous'
}

binning_map = {
    'opacities': 'lung_opacity', 'opacity': 'lung_opacity',
    'opacification': 'lung_opacity', 'consolidation': 'lung_opacity',
    'atelectasis': 'lung_atelectasis', 'effusions': 'pleural_effusion',
    'effusion': 'pleural_effusion', 'blunting': 'pleural_effusion',
    'edema': 'pulmonary_edema', 'congestion': 'vascular_congestion',
    'enlarged': 'heart_enlarged', 'cardiomegaly': 'heart_enlarged',
    'enlargement': 'heart_enlarged', 'hyperinflated': 'lungs_hyperinflated',
    'hyperinflation': 'lungs_hyperinflated', 'tube': 'support_tube',
    'line': 'support_line'
}

print("Loading JSON file...")
with open(file_path, 'r') as f:
    data = json.load(f)

print("Building binary concept vectors for each report...")
results = []

for study_id, report in data.items():
    # Start with a blank vector of 15 zeros
    vector = [0] * 15
    found_concepts = set() # Store all valid concepts found in this specific report
    
    entities = report.get('entities', {})
    
    # 1. Extract concepts from this report using the exact same logic
    for ent_id, entity in entities.items():
        if entity.get('label') == 'OBS-DP':
            obs_word = entity.get('tokens', '').lower().strip()
            
            if obs_word in filler_words:
                continue
                
            if obs_word in binning_map:
                found_concepts.add(binning_map[obs_word])
            else:
                for relation in entity.get('relations', []):
                    if relation[0] == 'located_at':
                        target_id = relation[1]
                        target_entity = entities.get(target_id, {})
                        if target_entity:
                            anat_word = target_entity.get('tokens', '').lower().strip()
                            if anat_word not in filler_words:
                                found_concepts.add(f"{anat_word}_{obs_word}")
    
    # 2. Check which of the Top 15 concepts are in this report's found concepts
    for i, concept in enumerate(top_15_concepts):
        if concept in found_concepts:
            vector[i] = 1
            
    # Save the study_id (the report path) and the 15 numbers
    results.append([study_id] + vector)

# 3. Save everything to a CSV file
columns = ['study_id'] + top_15_concepts
df_concepts = pd.DataFrame(results, columns=columns)
df_concepts.to_csv('cxr_concept_vectors.csv', index=False)

print(f"Success! Saved {len(df_concepts)} rows to 'cxr_concept_vectors.csv'")