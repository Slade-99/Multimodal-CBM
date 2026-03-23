import json
from collections import Counter

file_path = '/home/azwad/Works/Multimodal-CBM/Datasets/MIMIC-CXR_graphs.json'
concept_counts = Counter()

# 1. Expanded filler words to ignore non-acute noise and technician notes
filler_words = {
    'normal', 'unchanged', 'clear', 'stable', 'mild', 'moderate', 
    'small', 'large', 'intact', 'tip', 'essentially', 'unremarkable', 
    'low', 'free', 'negative', 'gross', 'grossly', 
    'expanded', 'degenerative', 'tortuous' # Added to remove non-emergency noise
}

# 2. Concept Binning Dictionary 
# Groups duplicates into single master concepts
binning_map = {
    'opacities': 'lung_opacity',
    'opacity': 'lung_opacity',
    'atelectasis': 'lung_atelectasis',
    'effusions': 'pleural_effusion',
    'effusion': 'pleural_effusion',
    'blunting': 'pleural_effusion', # Maps costophrenic blunting to effusion
    'edema': 'pulmonary_edema',
    'congestion': 'vascular_congestion',
    'enlarged': 'heart_enlarged',
    'cardiomegaly': 'heart_enlarged',
    'enlargement': 'heart_enlarged', # Fixes cardiac_enlargement duplicate
    'hyperinflated': 'lungs_hyperinflated',
    'hyperinflation': 'lungs_hyperinflated',
    'tube': 'support_tube',
    'line': 'support_line',
    'opacification': 'lung_opacity',
    'consolidation': 'lung_opacity'
}

print("Loading JSON file...")
with open(file_path, 'r') as f:
    data = json.load(f)

print("Extracting, binning, and cleaning specific concepts...")

for study_id, report in data.items():
    entities = report.get('entities', {})
    
    for ent_id, entity in entities.items():
        if entity.get('label') == 'OBS-DP':
            obs_word = entity.get('tokens', '').lower().strip()
            
            # Skip if it's a useless filler word
            if obs_word in filler_words:
                continue
            
            # Check if this observation is in our binning map
            if obs_word in binning_map:
                # Use the unified master concept
                unified_concept = binning_map[obs_word]
                concept_counts[unified_concept] += 1
            else:
                # If it's a new unique word, link it to its anatomy
                for relation in entity.get('relations', []):
                    if relation[0] == 'located_at':
                        target_id = relation[1]
                        target_entity = entities.get(target_id, {})
                        
                        if target_entity:
                            anat_word = target_entity.get('tokens', '').lower().strip()
                            # Only keep it if the anatomy isn't a filler word either
                            if anat_word not in filler_words:
                                specific_concept = f"{anat_word}_{obs_word}"
                                concept_counts[specific_concept] += 1

# Get the true top 15 low-level clinical concepts
top_15_concepts = [item[0] for item in concept_counts.most_common(15)]
for concept, count in concept_counts.most_common(40):
    print(concept, count)
"""
print("\nFinal Cleaned Top 15 Low-Level Concepts Identified:")
for i, concept in enumerate(top_15_concepts):
    print(f"{i+1}. {concept}")
    """