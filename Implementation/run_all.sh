#!/bin/bash

# The 'set -e' command ensures that if any script crashes or fails, 
# the entire bash script will stop immediately instead of continuing to the next experiment.
set -e

# These are the 7 exact experiments we established
EXPERIMENTS=("cxr_only" "ecg_only" "ehr_only" "cxr_ehr" "cxr_ecg" "ehr_ecg" "trimodal")

echo "Starting the 7-stage Multimodal CBM pipeline..."

for EXP in "${EXPERIMENTS[@]}"; do
    # Create a clean, readable folder name for each run (e.g., "run_cxr_ehr")
    EXP_DIR_NAME="run_${EXP}"
    
    echo "======================================================="
    echo "          STARTING EXPERIMENT: $EXP                    "
    echo "======================================================="
    
    # Phase 1: Train the model
    echo "--> [1/2] Training $EXP..."
    python train.py --experiment "$EXP" --exp_name "$EXP_DIR_NAME"
    
    # Phase 2: Evaluate the model (Threshold tuning and test set metrics)
    echo "--> [2/2] Evaluating $EXP..."
    python test_and_evaluate.py --experiment "$EXP" --exp_name "$EXP_DIR_NAME"
    
    echo "Finished $EXP. Results and plots saved in checkpoints/$EXP_DIR_NAME"
    echo ""
done

echo "All 7 experiments have completed successfully!"