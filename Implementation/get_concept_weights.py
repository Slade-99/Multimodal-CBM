import pandas as pd
import torch
import os

# Import your lists of column names. Adjust this import based on where they live!
from config import CXR_CONCEPTS, ECG_CONCEPTS, EHR_CONCEPTS

CSV_PATH = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
OUTPUT_PATH = "/home/azwad/Works/Multimodal-CBM/Implementation/concept_weights.pt"

def calculate_weights(df, concept_columns):
    """Calculates pos_weight = (Total Negatives) / (Total Positives)"""
    weights = []
    total_samples = len(df)
    
    for concept in concept_columns:
        # Count how many patients actually have this concept (value == 1)
        positives = df[concept].sum()
        
        # Safety net: If a concept literally never appears, give it a neutral weight of 1.0
        if positives == 0:
            weights.append(1.0)
            print(f"Warning: {concept} has 0 positive cases. Defaulting weight to 1.0")
        else:
            negatives = total_samples - positives
            weight = negatives / positives
            weights.append(weight)
            
    return torch.tensor(weights, dtype=torch.float32)

def main():
    print("Loading training data...")
    df = pd.read_csv(CSV_PATH)
    
    print("Calculating Concept Weights...")
    cxr_weights = calculate_weights(df, CXR_CONCEPTS)
    ecg_weights = calculate_weights(df, ECG_CONCEPTS)
    ehr_weights = calculate_weights(df, EHR_CONCEPTS)
    
    # Save the tensors to a file so the training script can load them instantly
    torch.save({
        'cxr': cxr_weights,
        'ecg': ecg_weights,
        'ehr': ehr_weights
    }, OUTPUT_PATH)
    
    print(f"\nSuccess! Weights saved to {OUTPUT_PATH}")
    print(f"Sample CXR Weights: {cxr_weights[:3]}")

if __name__ == "__main__":
    main()