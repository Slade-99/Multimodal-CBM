import re
import pandas as pd

def parse_experiment_logs(file_paths):
    data = []

    # Regex patterns for the specific fields
    patterns = {
        "Experiment": r"EVALUATING EXPERIMENT: ([\w_]+)",
        #"Mortality_AUROC": r"MORTALITY PREDICTION:.*?AUROC: ([\d.]+)",
        #"Mortality_AUPRC": r"MORTALITY PREDICTION:.*?AUPRC: ([\d.]+)",
        #"Mortality_F1": r"MORTALITY PREDICTION:.*?F1: ([\d.]+)",
        "Mortality_Sensitivity": r"MORTALITY PREDICTION:.*?Sensitivity: ([\d.]+)",
        "Mortality_Specificity": r"MORTALITY PREDICTION:.*?Specificity: ([\d.]+)",
        #"Mortality_ECE": r"MORTALITY PREDICTION:.*?ECE: ([\d.]+)",
        #"AHF_AUROC": r"ACUTE HEART FAILURE \(AHF\) PREDICTION:.*?AUROC: ([\d.]+)",
        #"AHF_AUPRC": r"ACUTE HEART FAILURE \(AHF\) PREDICTION:.*?AUPRC: ([\d.]+)",
        #"AHF_F1": r"ACUTE HEART FAILURE \(AHF\) PREDICTION:.*?F1: ([\d.]+)",
        #"AHF_ECE": r"ACUTE HEART FAILURE \(AHF\) PREDICTION:.*?ECE: ([\d.]+)",
        "AHF_Sensitivity": r"ACUTE HEART FAILURE \(AHF\) PREDICTION:.*?Sensitivity: ([\d.]+)",
        "AHF_Specificity": r"ACUTE HEART FAILURE \(AHF\) PREDICTION:.*?Specificity: ([\d.]+)",
        #"Overall_Concept_Accuracy": r"OVERALL CONCEPT PREDICTION ACCURACY: ([\d.]+)",
        
    }

    for path in file_paths:
        try:
            with open(path, 'r') as f:
                content = f.read().replace('\n', ' ') # Flatten to search across lines easily
                
                results = {}
                for field, pattern in patterns.items():
                    match = re.search(pattern, content)
                    results[field] = match.group(1) if match else None
                
                data.append(results)
        except FileNotFoundError:
            print(f"Warning: File not found at {path}")

    return pd.DataFrame(data)

# --- USAGE ---
file_list = [
    "/home/azwad/Works/Results/run_cxr_only/evaluation_log.txt",
    "/home/azwad/Works/Results/run_ehr_only/evaluation_log.txt",
    "/home/azwad/Works/Results/run_ecg_only/evaluation_log.txt",
    "/home/azwad/Works/Results/run_cxr_ecg/evaluation_log.txt",
    "/home/azwad/Works/Results/run_cxr_ehr/evaluation_log.txt",
    "/home/azwad/Works/Results/run_ehr_ecg/evaluation_log.txt",
    "/home/azwad/Works/Results/run_trimodal/evaluation_log.txt"
    # Add your other paths here
]

df = parse_experiment_logs(file_list)

# Display the table
print(df.to_string(index=False))

# Optional: Save to CSV
# df.to_csv("model_comparison_results.csv", index=False)