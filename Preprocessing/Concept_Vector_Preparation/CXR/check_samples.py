import json

# Path to your downloaded mimic_cxr_graphs.json
file_path = 'Datasets/MIMIC-CXR_graphs.json'

with open(file_path, 'r') as f:
    # We use a generator to avoid loading the whole 700MB into RAM at once
    # though 700MB usually fits in RAM, this is safer for 50GB systems
    data = json.load(f)


first_key = list(data.keys())[1]
print(f"Sample Study ID: {first_key}")
print(json.dumps(data[first_key], indent=4))