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
from modality_dropout import (
    TRIMODAL_CONFIG, BIMODAL_CXR_EHR_CONFIG, 
    BIMODAL_CXR_ECG_CONFIG, BIMODAL_EHR_ECG_CONFIG, 
    ModalityDropoutScheduler
)

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
    def __init__(self, alpha=1.0, beta=1.0, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        
        # 1. Load the Concept Weights we just generated
        weight_path = "/home/azwad/Works/Multimodal-CBM/Implementation/concept_weights.pt"
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Missing {weight_path}. Run get_concept_weights.py first!")
            
        concept_weights = torch.load(weight_path, map_location=device, weights_only=True)
        
        # 2. Concept Losses (Now fully weighted against class imbalance!)
        self.bce_cxr = nn.BCEWithLogitsLoss(pos_weight=concept_weights['cxr'])
        self.bce_ecg = nn.BCEWithLogitsLoss(pos_weight=concept_weights['ecg'])
        self.bce_ehr = nn.BCEWithLogitsLoss(pos_weight=concept_weights['ehr'])
        
        # 3. Target Losses (From our previous fix)
        mortality_weight = torch.tensor([14.44], device=device)
        self.bce_mortality = nn.BCEWithLogitsLoss(pos_weight=mortality_weight)
        
        ahf_weight = torch.tensor([2.89], device=device)
        self.bce_ahf = nn.BCEWithLogitsLoss(pos_weight=ahf_weight)

    def forward(self, outputs, targets, cxr_mod_mask, ecg_mod_mask, ehr_mod_mask):

        # --- Concept losses: only supervise present modalities ---
        def masked_concept_loss(loss_fn, logits, labels, mask):
            present = mask.bool()
            if present.sum() == 0:
                return torch.tensor(0.0, device=logits.device)
            return loss_fn(logits[present], labels[present].float())

        loss_cxr = masked_concept_loss(
            self.bce_cxr,
            outputs['cxr_concept_logits'],
            targets['cxr_concepts'],
            cxr_mod_mask
        )
        loss_ecg = masked_concept_loss(
            self.bce_ecg,
            outputs['ecg_concept_logits'],
            targets['ecg_concepts'],
            ecg_mod_mask
        )
        loss_ehr = masked_concept_loss(
            self.bce_ehr,
            outputs['ehr_concept_logits'],
            targets['ehr_concepts'],
            ehr_mod_mask
        )

        concept_loss = (loss_cxr + loss_ecg + loss_ehr) / 3.0

        # --- Task losses: always computed regardless of dropout ---
        loss_mortality = self.bce_mortality(
            outputs['mortality_logits'],
            targets['target_mortality'].float()
        )
        loss_ahf = self.bce_ahf(
            outputs['ahf_logits'],
            targets['target_ahf'].float()
        )
        target_loss = (loss_mortality + loss_ahf) / 2.0

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

    

    if args.experiment == 'trimodal':
        drop_config = TRIMODAL_CONFIG
    elif args.experiment == 'cxr_ehr':
        drop_config = BIMODAL_CXR_EHR_CONFIG
    elif args.experiment == 'cxr_ecg':
        drop_config = BIMODAL_CXR_ECG_CONFIG
    elif args.experiment == 'ehr_ecg':
        drop_config = BIMODAL_EHR_ECG_CONFIG
    else:
        # Unimodal experiments don't use dropout
        drop_config = None
    
    if drop_config:
        print(f"Active Modalities: {active_mods} with Modality Dropout")
    else:
        print(f"Active Modalities: {active_mods} without Modality Dropout")

    if drop_config:
        dropout_scheduler = ModalityDropoutScheduler(config=drop_config, warmup_epochs=20)
    else:
        dropout_scheduler = None

    # --- 2. Load Data ---
    train_csv =    "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/train_final.csv"
    val_csv   = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/val_final.csv"
    test_csv  = "/home/azwad/Works/Multimodal-CBM/Datasets/Preprocessed/test_final.csv"


    train_loader, val_loader, test_loader, train_ds = get_dataloaders(
        train_csv, val_csv, test_csv, 
        batch_size=args.batch_size, 
        dropout_config=drop_config
    )
    logger.info("Dataloaders initialized successfully.")

    # --- 3. Initialize Model, Loss, Optimizer ---
    model = MultimodalCBM().to(device)
    criterion = JointCBMLoss(alpha=args.alpha_concept, beta=args.beta_target, device=device)
    
    encoder_params = (
        list(model.cxr_encoder.parameters()) +
        list(model.ecg_encoder.parameters()) +
        list(model.ehr_encoder.parameters())
    )
    encoder_ids = set(id(p) for p in encoder_params)
    other_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = AdamW([
        {'params': encoder_params, 'lr': args.lr_finetune},
        {'params': other_params,   'lr': args.lr_e2e},
    ], weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=args.patience_scheduler, factor=0.5)

    # --- 4. Training State Variables ---
    best_val_auroc = 0.0
    epochs_no_improve = 0
    save_path = os.path.join(args.save_dir, args.exp_name, "best_final_weights.pth")

    # --- 5. Epoch Loop ---
    scaler = GradScaler()
    for epoch in range(args.epochs):

        if dropout_scheduler and hasattr(train_ds, 'set_dropout_probs'):
            train_ds.set_dropout_probs(dropout_scheduler.current_probs)
            print(f"Epoch {epoch+1} Dropout Probs: {dropout_scheduler.current_probs}")
            



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
                loss, _, _ = criterion(outputs, targets, cxr_mask, ecg_mask, ehr_mask)
            
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
                loss, _, _ = criterion(outputs, targets, cxr_mask, ecg_mask, ehr_mask)
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

        if dropout_scheduler is not None:
            dropout_scheduler.step()
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