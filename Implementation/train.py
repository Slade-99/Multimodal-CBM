import os
import logging
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
from torch.amp import autocast, GradScaler
# Import your custom modules
from config import get_args
from cbm_model import MultimodalCBM
from dataloader import get_dataloaders, ModalityDropoutConfig

# ================= SETUP LOGGING =================

def setup_logger(save_dir, exp_name):
    """
    Creates a logger that writes all training output to a file.
    """
    log_file = os.path.join(save_dir, exp_name, "training_log.txt")
    
    # Configure the logging system
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            # You can uncomment the line below if you also want it to print to the console
            # logging.StreamHandler() 
        ]
    )
    return logging.getLogger()

# ================= JOINT LOSS FUNCTION =================

class JointCBMLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        # 1. Concept Losses
        loss_cxr = self.bce_loss(outputs['cxr_concept_logits'], targets['cxr_concepts'].float())
        loss_ecg = self.bce_loss(outputs['ecg_concept_logits'], targets['ecg_concepts'].float())
        loss_ehr = self.bce_loss(outputs['ehr_concept_logits'], targets['ehr_concepts'].float())
        
        concept_loss = (loss_cxr + loss_ecg + loss_ehr) / 3.0

        # 2. Target Losses
        loss_mortality = self.bce_loss(outputs['mortality_logits'], targets['target_mortality'].float())
        loss_ahf = self.bce_loss(outputs['ahf_logits'], targets['target_ahf'].float())
        
        target_loss = (loss_mortality + loss_ahf) / 2.0

        # 3. Total Weighted Loss
        total_loss = (self.alpha * concept_loss) + (self.beta * target_loss)
        return total_loss, concept_loss, target_loss

# ================= METRIC EVALUATION =================

def compute_metrics(y_true, y_scores):
    """Computes AUROC and AUPRC directly from raw probability scores."""
    try:
        auroc = roc_auc_score(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
    except ValueError:
        auroc, auprc = 0.5, 0.0
    return auroc, auprc

# ================= MAIN TRAINING LOOP =================

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set up the logger
    logger = setup_logger(args.save_dir, args.exp_name)
    logger.info(f"Starting Experiment: {args.experiment} on {device}")
    logger.info(f"Arguments: {vars(args)}")

    # --- 1. Determine Active Modalities ---
    active_mods = []
    if 'cxr' in args.experiment or args.experiment == 'trimodal': active_mods.append('cxr')
    if 'ecg' in args.experiment or args.experiment == 'trimodal': active_mods.append('ecg')
    if 'ehr' in args.experiment or args.experiment == 'trimodal': active_mods.append('ehr')

    exp_config = ModalityDropoutConfig(
        p_max={'cxr': 0.05, 'ehr': 0.30, 'ecg': 0.30}, 
        active_modalities=active_mods
    )

    # --- 2. Load Data ---
    train_csv =    "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"


    train_loader, val_loader, test_loader, train_ds = get_dataloaders(
        train_csv, val_csv, test_csv, 
        batch_size=args.batch_size, 
        dropout_config=exp_config
    )
    logger.info("Dataloaders initialized successfully.")

    # --- 3. Initialize Model, Loss, Optimizer ---
    model = MultimodalCBM().to(device)
    criterion = JointCBMLoss(alpha=args.alpha_concept, beta=args.beta_target)

    optimizer = AdamW([
        {'params': model.cxr_encoder.parameters(), 'lr': args.lr_finetune},
        {'params': model.ecg_encoder.parameters(), 'lr': args.lr_finetune},
        {'params': model.ehr_encoder.parameters(), 'lr': args.lr_finetune},
        {'params': model.mortality_head.parameters(), 'lr': args.lr_e2e},
        {'params': model.ahf_head.parameters(), 'lr': args.lr_e2e}
    ], weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience_scheduler, factor=0.5)

    # --- 4. Training State Variables ---
    best_val_auroc = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(args.save_dir, args.exp_name, "best_final_weights.pth")

    # --- 5. Epoch Loop ---
    scaler = GradScaler()
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        # We keep tqdm for console visualization, but it won't print metrics
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]", leave=False)
        for i, batch in enumerate(loop):
            image = batch['image'].to(device)
            waveform = batch['waveform'].to(device)
            ehr_features = batch['ehr_features'].to(device)
            
            cxr_mask = batch['cxr_mod_mask'].to(device)
            ecg_mask = batch['ecg_mod_mask'].to(device)
            ehr_mask = batch['ehr_mod_mask'].to(device)

            targets = {
                'cxr_concepts': batch['cxr_concepts'].to(device),
                'ecg_concepts': batch['ecg_concepts'].to(device),
                'ehr_concepts': batch['ehr_concepts'].to(device),
                'target_mortality': batch['target_mortality'].to(device),
                'target_ahf': batch['target_ahf'].to(device)
            }

            optimizer.zero_grad()


            # 1. Wrap forward pass and loss in autocast
            with autocast('cuda'):
                outputs = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)
                loss, _, _ = criterion(outputs, targets)
            
            # 2. Scale the loss and backward pass
            scaler.scale(loss).backward()
            
            # 3. Step the optimizer and update the scaler
            scaler.step(optimizer)
            scaler.update()


            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- 6. Validation Phase ---
        model.eval()
        val_loss = 0.0
        
        all_mort_preds, all_mort_targets = [], []
        all_ahf_preds, all_ahf_targets = [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)):
                image = batch['image'].to(device)
                waveform = batch['waveform'].to(device)
                ehr_features = batch['ehr_features'].to(device)
                cxr_mask = batch['cxr_mod_mask'].to(device)
                ecg_mask = batch['ecg_mod_mask'].to(device)
                ehr_mask = batch['ehr_mod_mask'].to(device)

                targets = {
                    'cxr_concepts': batch['cxr_concepts'].to(device),
                    'ecg_concepts': batch['ecg_concepts'].to(device),
                    'ehr_concepts': batch['ehr_concepts'].to(device),
                    'target_mortality': batch['target_mortality'].to(device),
                    'target_ahf': batch['target_ahf'].to(device)
                }

                outputs = model(image, waveform, ehr_features, cxr_mask, ecg_mask, ehr_mask)
                loss, _, _ = criterion(outputs, targets)
                val_loss += loss.item()

                mort_probs = torch.sigmoid(outputs['mortality_logits']).cpu().numpy()
                ahf_probs = torch.sigmoid(outputs['ahf_logits']).cpu().numpy()
                
                all_mort_preds.extend(mort_probs)
                all_mort_targets.extend(targets['target_mortality'].cpu().numpy())
                
                all_ahf_preds.extend(ahf_probs)
                all_ahf_targets.extend(targets['target_ahf'].cpu().numpy())


        mort_auroc, mort_auprc = compute_metrics(all_mort_targets, all_mort_preds)
        ahf_auroc, ahf_auprc = compute_metrics(all_ahf_targets, all_ahf_preds)
        
        avg_val_auroc = (mort_auroc + ahf_auroc) / 2.0
        avg_val_loss = val_loss / len(val_loader)

        # Log metrics to file instead of printing
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        logger.info(f"Mortality - AUROC: {mort_auroc:.4f}, AUPRC: {mort_auprc:.4f}")
        logger.info(f"AHF       - AUROC: {ahf_auroc:.4f}, AUPRC: {ahf_auprc:.4f}")
        logger.info(f"Mean Val AUROC: {avg_val_auroc:.4f}")

        # --- 7. Scheduler & Early Stopping ---
        scheduler.step(avg_val_auroc)

        if avg_val_auroc > best_val_auroc:
            best_val_auroc = avg_val_auroc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            logger.info(f"[*] New best model saved to {save_path}\n")
        else:
            epochs_no_improve += 1
            logger.info(f"[!] No improvement for {epochs_no_improve} epochs.\n")

        if epochs_no_improve >= args.patience_early_stop:
            logger.info("Early stopping triggered. Training halted.")
            break

if __name__ == "__main__":
    main()