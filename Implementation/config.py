import argparse
import os


CXR_CONCEPTS = [
    "lung_opacity", "pleural_effusion", "support_tube",
    "heart_enlarged", "lung_atelectasis", "pulmonary_edema",
    "support_line", "vascular_congestion", "lungs_hyperinflated",
    "apical_pneumothorax", "hemidiaphragm_elevation", "rib_fractures",
    "interstitial_markings", "volume_loss", "atrium_leads"
]

ECG_CONCEPTS = [
    "myocardial_infarction", "left_axis_deviation", "myocardial_ischemia",
    "sinus_bradycardia", "atrial_fibrillation", "low_qrs_voltage",
    "sinus_tachycardia", "right_bundle_branch_block",
    "left_ventricular_hypertrophy", "prolonged_qt_interval",
    "left_bundle_branch_block", "iv_conduction_defect",
    "pacemaker_rhythm", "left_atrial_abnormality"
]

EHR_CONCEPTS = [
    'tachycardia', 'tachypnea', 'abnormal_temp', 'hypotension', 'hypoxia',
    'altered_mental', 'elevated_creatinine', 'elevated_bilirubin',
    'thrombocytopenia', 'abnormal_wbc', 'hyperlactatemia',
    'elevated_bnp', 'elevated_troponin'
]




def get_args():
    parser = argparse.ArgumentParser(description="Multimodal Concept Bottleneck Model Training")

    # ================= EXPERIMENT CONTROLS =================
    # This single flag controls which modalities are active in the forward pass
    parser.add_argument('--experiment', type=str, default='trimodal',
                        choices=['cxr_only', 'ecg_only', 'ehr_only', 
                                 'cxr_ehr', 'cxr_ecg', 'ehr_ecg', 'trimodal'],
                        help="Which experiment to run (determines active modalities).")
    
    parser.add_argument('--exp_name', type=str, default='baseline_run',
                        help="Name of the experiment for saving logs and weights.")
    
    parser.add_argument('--save_dir', type=str, default='./checkpoints_2',
                        help="Directory to save the best model weights and plots.")

    # ================= HYPERPARAMETERS =================
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training and evaluation.")
    
    parser.add_argument('--lr_finetune', type=float, default=1e-4,
                        help="Learning rate for fine-tuning concept layers.")
    
    parser.add_argument('--lr_e2e', type=float, default=1e-5,
                        help="Learning rate for end-to-end training.")
    
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help="Weight decay (L2 penalty) for AdamW optimizer.")
    
    parser.add_argument('--epochs', type=int, default=60,
                        help="Maximum number of training epochs.")

    # ================= SCHEDULER & EARLY STOPPING =================
    parser.add_argument('--patience_scheduler', type=int, default=5,
                        help="Patience for ReduceLROnPlateau scheduler.")
    
    parser.add_argument('--patience_early_stop', type=int, default=15,
                        help="Epochs to wait before early stopping if no improvement.")
    
    # ================= LOSS WEIGHTS =================
    # Because this is a CBM, you have concept loss and target loss. 
    # These flags let you tune how much the model cares about concepts vs targets.
    parser.add_argument('--alpha_concept', type=float, default=5.0,
                        help="Weight for the concept prediction loss.")
    
    parser.add_argument('--beta_target', type=float, default=1.0,
                        help="Weight for the downstream task (Mortality/AHF) loss.")

    args = parser.parse_args()
    
    # Ensure save directory exists
    os.makedirs(os.path.join(args.save_dir, args.exp_name), exist_ok=True)
    
    return args

if __name__ == "__main__":
    # Test the parser
    args = get_args()
    print("Parsed Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")