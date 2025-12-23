import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

# Add project root to path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


from core.data_loader import MIMOSAR_Dataset, load_dataset_auto
from core.models import DBPNet
from core.utils import complex_matmul
from core.visualization_utils import (plot_unrolled_iterations, 
                                 plot_iteration_comparison,
                                 plot_mse_progression,
                                 plot_measurement_domain_progression)
from core.real_prior import enforce_real_prior, measure_imaginary_magnitude

# -----------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------
#MAT_FILE = 'data/FL_MIMO_SAR_data.mat'
MAT_FILE = 'data/data_training_sar.mat'
MODEL_SAVE_PATH = 'checkpoints/dbp_model.pth'

# Training Hyperparameters
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# Model Hyperparameters
NUM_UNROLLS = 5      # N1 in the paper: total unrolled iterations
NUM_ADMM_STEPS = 2   # N2 in the paper: internal ADMM steps

# *** NEW: Training Mode Configuration ***
TRAINING_MODE = 'unsupervised'  # Options: 'supervised', 'unsupervised', 'hybrid'
SUPERVISED_WEIGHT = 1.0   # Weight for supervised loss (image domain)
UNSUPERVISED_WEIGHT = 1.0 # Weight for unsupervised loss (measurement domain)

# *** NEW: Two-Stage Training Configuration ***
TRAINING_STRATEGY = 'two_stage'  # Options: 'end_to_end', 'two_stage'
PRETRAINED_DENOISER_PATH = 'checkpoints/denoiser_curriculum.pth'  # Path to pre-trained denoiser (for two_stage)
FREEZE_DENOISER = True  # True: freeze denoiser (only train ADMM), False: fine-tune denoiser too

# *** NEW: Denoiser Architecture Selection ***
# Enforces real-valued outputs at the architectural level (best approach!)
DENOISER_TYPE = 'real'  # Options: 'real' (best for real targets, ~62K params)
                        #          'complex' (allows imaginary, ~62K params)
                        #          'original' (shallow residual, ~3.6K params, backward compatible)
# Note: 'real' architecture enforces imaginary=0 at the architecture level (2.46× better than 'complex')

# *** Positivity Enforcement ***
# Physical constraint: Reflectivity cannot be negative (enforced at architecture level)
ENFORCE_POSITIVITY = True  # True: Add ReLU to enforce output ≥ 0 (only for DENOISER_TYPE='real')
                           # False: Allow negative values
# Note: Only applies to 'real' denoiser. Ignored for 'complex' and 'original' types.

# *** ADMM Physical Constraints ***
# Enforce physical constraints directly in ADMM optimization (as per MATLAB implementation)
ADMM_ENFORCE_REAL = True        # True: Project ADMM x-update to real values (discard imaginary)
ADMM_ENFORCE_POSITIVITY = True  # True: Clamp ADMM x-update to non-negative values
# Note: These are HARD constraints in optimization (not learned). Independent from denoiser constraints.

# Note: If ground truth 'x' is not available, will automatically fall back to unsupervised mode

# -----------------------------------------------------------------
# 2. Setup
# -----------------------------------------------------------------
def main():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")
    print(f"--- Training Strategy: {TRAINING_STRATEGY.upper()} ---")
    print(f"--- Training Mode: {TRAINING_MODE.upper()} ---")
    print(f"--- Denoiser Type: {DENOISER_TYPE.upper()} ---")
    if DENOISER_TYPE == 'real':
        print(f"--- Enforce Positivity: {ENFORCE_POSITIVITY} ---")

    # -----------------------------------------------------------------
    # 3. Load Data
    # -----------------------------------------------------------------
    print("Loading dataset...")
    
    # Determine if we need ground truth
    need_ground_truth = (TRAINING_MODE in ['supervised', 'hybrid'])
    
    # Auto-detect single-position vs multi-position data
    dataset = load_dataset_auto(MAT_FILE, return_ground_truth=need_ground_truth)
    
    # Check if ground truth is actually available
    has_ground_truth = dataset.has_ground_truth
    
    # Adjust training mode if ground truth not available
    if need_ground_truth and not has_ground_truth:
        print(f"\n*** WARNING: {TRAINING_MODE} training requested but ground truth not available!")
        print(f"*** Falling back to UNSUPERVISED training mode ***\n")
        actual_training_mode = 'unsupervised'
    else:
        actual_training_mode = TRAINING_MODE
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2, # Use 2 worker processes for data loading
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Get steering matrix A and move it to the device
    A_tensor = dataset.A.to(device)
    # Add a 'batch' dimension for our matmul functions
    A_batch_tensor = A_tensor.unsqueeze(0) 

    # -----------------------------------------------------------------
    # 4. Initialize Model, Loss, and Optimizer
    # -----------------------------------------------------------------
    print("Initializing model...")
    model = DBPNet(
        A_tensor, 
        num_iterations=NUM_UNROLLS, 
        N_admm_steps=NUM_ADMM_STEPS,
        denoiser_type=DENOISER_TYPE,
        enforce_positivity=ENFORCE_POSITIVITY,
        admm_enforce_real=ADMM_ENFORCE_REAL,
        admm_enforce_positivity=ADMM_ENFORCE_POSITIVITY
    ).to(device)
    
    # Handle two-stage training: load pre-trained denoiser
    if TRAINING_STRATEGY == 'two_stage':
        print(f"\n*** TWO-STAGE TRAINING MODE ***")
        if os.path.exists(PRETRAINED_DENOISER_PATH):
            model.load_pretrained_denoiser(PRETRAINED_DENOISER_PATH)
            
            if FREEZE_DENOISER:
                model.freeze_denoiser()
                print(f"Training mode: ADMM parameters only (denoiser frozen)")
            else:
                print(f"Training mode: Fine-tuning denoiser + training ADMM parameters")
        else:
            print(f"\n*** ERROR: Pre-trained denoiser not found at {PRETRAINED_DENOISER_PATH}!")
            print(f"*** Please train the denoiser first using: python train_denoiser_only.py")
            print(f"*** Falling back to END-TO-END training ***\n")
    else:
        print(f"\n*** END-TO-END TRAINING MODE ***")
        print(f"Training all parameters from scratch (denoiser + ADMM)")

    # Loss functions
    criterion_image = nn.MSELoss()      # For supervised loss (image domain)
    criterion_measurement = nn.MSELoss() # For unsupervised loss (measurement domain)
    
    # Optimizer (Adam)
    # Use get_trainable_params() to only optimize unfrozen parameters
    trainable_params = model.get_trainable_params()
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
    
    print(f"\nModel Parameters:")
    print(f"  Num Unrolls (N1): {NUM_UNROLLS}")
    print(f"  Num ADMM Steps (N2): {NUM_ADMM_STEPS}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    print(f"Training Configuration:")
    print(f"  Mode: {actual_training_mode}")
    if actual_training_mode == 'hybrid':
        print(f"  Supervised weight: {SUPERVISED_WEIGHT}")
        print(f"  Unsupervised weight: {UNSUPERVISED_WEIGHT}")

    # -----------------------------------------------------------------
    # 5. Training Loop
    # -----------------------------------------------------------------
    print(f"\n--- Starting Training for {NUM_EPOCHS} Epochs ---\n")
    
    model.train() # Set the model to training mode
    
    # Track losses for plotting
    epoch_losses = []
    epoch_supervised_losses = []
    epoch_unsupervised_losses = []
    
    tic = time.time()
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(NUM_EPOCHS), desc="Training", unit="epoch", 
                      dynamic_ncols=True, leave=True)
    
    for epoch in epoch_pbar:
        epoch_total_loss = 0.0
        epoch_sup_loss = 0.0
        epoch_unsup_loss = 0.0
        
        # Create progress bar for batches
        batch_pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                         desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch",
                         leave=False, dynamic_ncols=True)
        
        for i, batch_data in batch_pbar:
            # Unpack batch data
            if has_ground_truth:
                y_batch, x_gt_batch = batch_data
                x_gt_batch = x_gt_batch.to(device)
            else:
                y_batch = batch_data
                x_gt_batch = None
            
            # Move data to the selected device
            y_batch = y_batch.to(device)
            
            # --- Forward Pass ---
            # 1. Get the estimated reflectivity x_hat from the model
            #    (real-valued prior is enforced at architecture level if DENOISER_TYPE='real')
            x_hat_batch = model(y_batch)
            
            # 2. Compute losses based on training mode
            total_loss = 0.0
            loss_supervised = torch.tensor(0.0)
            loss_unsupervised = torch.tensor(0.0)
            
            if actual_training_mode == 'supervised':
                # Supervised: only image domain loss
                loss_supervised = criterion_image(x_hat_batch, x_gt_batch)
                total_loss = SUPERVISED_WEIGHT * loss_supervised
                
            elif actual_training_mode == 'unsupervised':
                # Unsupervised: only measurement domain loss
                y_hat_batch = complex_matmul(A_batch_tensor, x_hat_batch)
                loss_unsupervised = criterion_measurement(y_hat_batch, y_batch)
                total_loss = UNSUPERVISED_WEIGHT * loss_unsupervised
                
            elif actual_training_mode == 'hybrid':
                # Hybrid: both losses
                loss_supervised = criterion_image(x_hat_batch, x_gt_batch)
                y_hat_batch = complex_matmul(A_batch_tensor, x_hat_batch)
                loss_unsupervised = criterion_measurement(y_hat_batch, y_batch)
                total_loss = (SUPERVISED_WEIGHT * loss_supervised + 
                             UNSUPERVISED_WEIGHT * loss_unsupervised)
            
            # --- Backward Pass and Optimization ---
            # 1. Clear previous gradients
            optimizer.zero_grad()
            
            # 2. Compute gradients
            total_loss.backward()
            
            # 3. Update model parameters
            optimizer.step()
            
            epoch_total_loss += total_loss.item()
            epoch_sup_loss += loss_supervised.item() if isinstance(loss_supervised, torch.Tensor) else 0.0
            epoch_unsup_loss += loss_unsupervised.item() if isinstance(loss_unsupervised, torch.Tensor) else 0.0
            
            # Update batch progress bar with current loss
            postfix_dict = {'loss': f'{total_loss.item():.6f}'}
            if actual_training_mode in ['supervised', 'hybrid']:
                postfix_dict['sup'] = f'{loss_supervised.item():.6f}'
            if actual_training_mode in ['unsupervised', 'hybrid']:
                postfix_dict['unsup'] = f'{loss_unsupervised.item():.6f}'
            batch_pbar.set_postfix(postfix_dict)

        # Calculate average loss for the epoch
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_sup_loss = epoch_sup_loss / len(dataloader)
        avg_unsup_loss = epoch_unsup_loss / len(dataloader)
        
        epoch_losses.append(avg_total_loss)
        epoch_supervised_losses.append(avg_sup_loss)
        epoch_unsupervised_losses.append(avg_unsup_loss)
        
        # Update epoch progress bar with average losses
        postfix_dict = {'avg_loss': f'{avg_total_loss:.6f}'}
        if actual_training_mode in ['supervised', 'hybrid']:
            postfix_dict['avg_sup'] = f'{avg_sup_loss:.6f}'
        if actual_training_mode in ['unsupervised', 'hybrid']:
            postfix_dict['avg_unsup'] = f'{avg_unsup_loss:.6f}'
        epoch_pbar.set_postfix(postfix_dict)

    toc = time.time()
    
    # Print final training summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total Training Time: {toc-tic:.2f} seconds ({(toc-tic)/60:.2f} minutes)")
    print(f"Final Average Loss: {avg_total_loss:.8f}")
    if actual_training_mode in ['supervised', 'hybrid']:
        print(f"  Supervised (Image Domain): {avg_sup_loss:.8f}")
    if actual_training_mode in ['unsupervised', 'hybrid']:
        print(f"  Unsupervised (Measurement Domain): {avg_unsup_loss:.8f}")
    print("="*80 + "\n")

    # -----------------------------------------------------------------
    # 6. Plot Training Curves
    # -----------------------------------------------------------------
    print("Generating training loss curves...")
    plt.figure(figsize=(12, 6))
    
    epochs_range = range(1, NUM_EPOCHS + 1)
    plt.plot(epochs_range, epoch_losses, 'b-', label='Total Loss', linewidth=2)
    
    if actual_training_mode in ['supervised', 'hybrid']:
        plt.plot(epochs_range, epoch_supervised_losses, 'r--', label='Supervised (Image Domain)', linewidth=1.5)
    if actual_training_mode in ['unsupervised', 'hybrid']:
        plt.plot(epochs_range, epoch_unsupervised_losses, 'g--', label='Unsupervised (Measurement Domain)', linewidth=1.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve ({actual_training_mode.upper()} mode)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale often better for visualizing loss
    
    loss_curve_path = f'training_loss_{actual_training_mode}.png'
    plt.savefig(loss_curve_path)
    print(f"Training loss curve saved to {loss_curve_path}")
    plt.close()

    # -----------------------------------------------------------------
    # 7. Debug Plots - Last Batch
    # -----------------------------------------------------------------
    print("Generating debug plots of the last training batch...")
    with torch.no_grad():
        # --- PLOT 1: Measurement Domain (y vs y_hat) ---

        # Select the first sample (index 0) from the last batch
        y_sample = y_batch[0]       # Shape [2, N_v]
        
        # Compute y_hat for this sample
        x_hat_sample = x_hat_batch[0:1]  # Keep batch dim
        y_hat_sample = complex_matmul(A_batch_tensor, x_hat_sample)[0]  # Shape [2, N_v]

        # Move to CPU, get the real part (index 0), and convert to numpy
        y_real = y_sample[0].cpu().numpy()
        y_hat_real = y_hat_sample[0].cpu().numpy()

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(y_real, 'r', label='Ground Truth (y_batch)[0, real]')
        plt.plot(y_hat_real, 'b--', label='Network Output (y_hat_batch)[0, real]')
        plt.title(f'Debug Plot - Last Batch (Epoch {NUM_EPOCHS}) - Measurement Domain')
        plt.xlabel('Virtual Antenna Index')
        plt.ylabel('Amplitude (Real Part)')
        plt.legend()
        plt.grid(True)

        debug_plot_path_y = 'train_debug_plot_y.png'
        plt.savefig(debug_plot_path_y)
        print(f"Measurement debug plot saved to {debug_plot_path_y}")
        plt.close() # Close the figure to start the next one

        # --- PLOT 2: Image Domain (x_hat vs x_gt if available) ---

        # Select the corresponding x_hat sample (index 0)
        x_hat_sample = x_hat_batch[0] # Shape [2, N_theta]

        # Get real part and convert to numpy
        x_hat_real = x_hat_sample[0].cpu().numpy()

        # *** FIX: Get NUM_ANGLE_BINS from actual data instead of hardcoding ***
        NUM_ANGLE_BINS = x_hat_real.shape[0]  # Dynamically get from data
        START_ANGLE_DEG = 25.0
        END_ANGLE_DEG = -25.0
        
        # Create the angle axis
        theta = np.linspace(START_ANGLE_DEG, END_ANGLE_DEG, NUM_ANGLE_BINS)

        # Plot
        plt.figure(figsize=(10, 6))
        
        # Plot ground truth if available
        if has_ground_truth:
            x_gt_sample = x_gt_batch[0]
            x_gt_real = x_gt_sample[0].cpu().numpy()
            plt.plot(theta, x_gt_real, 'r', label='Ground Truth (x_gt)[0, real]', linewidth=2)
        
        plt.plot(theta, x_hat_real / np.max(x_hat_real), 'b--', label='Network Output (x_hat_batch)[0, real]', linewidth=2)
        plt.title(f'Debug Plot - Last Batch (Epoch {NUM_EPOCHS}) - Image Domain')
        plt.xlabel('Angle (degree)')
        plt.ylabel('Amplitude (Real Part)')
        plt.legend()
        plt.grid(True)

        debug_plot_path_x = 'train_debug_plot_x.png'
        plt.savefig(debug_plot_path_x)
        print(f"Image debug plot saved to {debug_plot_path_x}")
        plt.close()

    # Debug code - end

    # -----------------------------------------------------------------
    # 8. Visualize Progressive Refinement Through Iterations
    # -----------------------------------------------------------------
    print("Generating progressive refinement visualizations...")
    with torch.no_grad():
        # Run forward pass with intermediate outputs
        intermediates = model(y_batch, return_intermediates=True)
        
        # Prepare ground truth if available
        x_gt_for_viz = x_gt_batch if has_ground_truth else None
        
        # Generate all visualization plots
        print("  Creating detailed iteration-by-iteration plot...")
        plot_unrolled_iterations(intermediates, x_gt=x_gt_for_viz, sample_idx=0,
                                save_path='train_unrolled_iterations.png')
        
        print("  Creating iteration comparison plot...")
        plot_iteration_comparison(intermediates, x_gt=x_gt_for_viz, sample_idx=0,
                                 save_path='train_iteration_comparison.png')
        
        if has_ground_truth:
            print("  Creating MSE progression plot...")
            plot_mse_progression(intermediates, x_gt=x_gt_for_viz, sample_idx=0,
                               save_path='train_mse_progression.png')
        
        print("  Creating measurement domain progression plot...")
        plot_measurement_domain_progression(intermediates, y_gt=y_batch, A_tensor=A_tensor, 
                                          sample_idx=0, 
                                          save_path='train_measurement_domain_progression.png')
    
    print("Progressive refinement visualizations complete!")

    # -----------------------------------------------------------------
    # 9. Quantitative Evaluation on Full Dataset
    # -----------------------------------------------------------------
    if has_ground_truth:
        print("\n" + "="*80)
        print("QUANTITATIVE EVALUATION ON FULL DATASET")
        print("="*80)
        
        model.eval()
        
        # Create a DataLoader for evaluation (no shuffling, to preserve order)
        eval_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=0, pin_memory=True)
        
        # Storage for errors
        all_mse = []
        all_nmse = []
        all_mae = []  # Mean Absolute Error
        
        eval_start_time = time.time()
        
        with torch.no_grad():
            # Create progress bar for evaluation
            eval_pbar = tqdm(enumerate(eval_loader), total=len(eval_loader),
                           desc="Evaluating", unit="batch", dynamic_ncols=True)
            
            for batch_idx, batch_data in eval_pbar:
                # Unpack batch data (DataLoader returns tuple/list when dataset returns tuple)
                try:
                    if isinstance(batch_data, (tuple, list)) and len(batch_data) == 2:
                        y_batch, x_gt_batch = batch_data
                    else:
                        # Single tensor returned (no ground truth)
                        print(f"Warning: Batch {batch_idx} does not contain ground truth, skipping...")
                        continue
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error unpacking batch {batch_idx}: {e}, skipping...")
                    continue
                
                y_batch = y_batch.to(device)
                x_gt_batch = x_gt_batch.to(device)
                
                # Forward pass
                x_hat_batch = model(y_batch)
                
                # Compute per-sample MSE (averaged over channels and spatial dimension)
                # Shape: [batch, 2, N_theta] -> MSE per sample: [batch]
                mse_per_sample = torch.mean((x_hat_batch - x_gt_batch)**2, dim=[1, 2])
                
                # Compute per-sample MAE
                mae_per_sample = torch.mean(torch.abs(x_hat_batch - x_gt_batch), dim=[1, 2])
                
                # Compute per-sample NMSE (normalized by ground truth power)
                x_gt_power = torch.mean(x_gt_batch**2, dim=[1, 2])
                nmse_per_sample = mse_per_sample / (x_gt_power + 1e-10)
                
                all_mse.append(mse_per_sample.cpu())
                all_nmse.append(nmse_per_sample.cpu())
                all_mae.append(mae_per_sample.cpu())
                
                # Update progress bar
                eval_pbar.set_postfix({'MSE': f'{mse_per_sample.mean().item():.6f}'})
        
        eval_time = time.time() - eval_start_time
        print(f"Evaluation completed in {eval_time:.2f} seconds\n")
        
        # Check if we have any data
        if len(all_mse) == 0:
            print("\n" + "="*80)
            print("ERROR: No samples were evaluated!")
            print("This likely means ground truth data is not being returned properly.")
            print("Check that the dataset has ground truth and __getitem__ returns (y, x_gt)")
            print("="*80)
            print("\nSkipping quantitative evaluation...")
        else:
            # Concatenate all batches
            all_mse = torch.cat(all_mse)
            all_nmse = torch.cat(all_nmse)
            all_mae = torch.cat(all_mae)
            
            # Compute statistics
            print(f"\n{'='*80}")
            print(f"DATASET STATISTICS ({len(all_mse)} samples evaluated):")
            print(f"{'='*80}")
            print(f"\nMean Squared Error (MSE):")
            print(f"  Mean:   {all_mse.mean():.8f}")
            print(f"  Std:    {all_mse.std():.8f}")
            print(f"  Median: {all_mse.median():.8f}")
            print(f"  Min:    {all_mse.min():.8f}")
            print(f"  Max:    {all_mse.max():.8f}")
            
            print(f"\nNormalized MSE (NMSE):")
            print(f"  Mean:   {all_nmse.mean():.8f}")
            print(f"  Std:    {all_nmse.std():.8f}")
            print(f"  Median: {all_nmse.median():.8f}")
            print(f"  Min:    {all_nmse.min():.8f}")
            print(f"  Max:    {all_nmse.max():.8f}")
            
            print(f"\nMean Absolute Error (MAE):")
            print(f"  Mean:   {all_mae.mean():.8f}")
            print(f"  Std:    {all_mae.std():.8f}")
            print(f"  Median: {all_mae.median():.8f}")
            print(f"  Min:    {all_mae.min():.8f}")
            print(f"  Max:    {all_mae.max():.8f}")
            
            # Find best and worst samples
            best_idx = all_mse.argmin().item()
            worst_idx = all_mse.argmax().item()
            print(f"\nBest sample:  #{best_idx} (MSE = {all_mse[best_idx]:.8f})")
            print(f"Worst sample: #{worst_idx} (MSE = {all_mse[worst_idx]:.8f})")
            print(f"{'='*80}\n")
            
            # Save results to file
            import pickle
            eval_results = {
                'mse_per_sample': all_mse.numpy(),
                'nmse_per_sample': all_nmse.numpy(),
                'mae_per_sample': all_mae.numpy(),
                'mse_mean': all_mse.mean().item(),
                'mse_std': all_mse.std().item(),
                'mse_median': all_mse.median().item(),
                'nmse_mean': all_nmse.mean().item(),
                'nmse_std': all_nmse.std().item(),
                'mae_mean': all_mae.mean().item(),
                'num_samples': len(all_mse),
                'best_sample_idx': best_idx,
                'worst_sample_idx': worst_idx,
                'training_mode': TRAINING_MODE,
                'denoiser_type': DENOISER_TYPE,
                'num_unrolls': NUM_UNROLLS,
                'num_admm_steps': NUM_ADMM_STEPS,
            }
            
            results_file = 'evaluation_results.pkl'
            with open(results_file, 'wb') as f:
                pickle.dump(eval_results, f)
            print(f"Evaluation results saved to '{results_file}'")
            
            # Generate error distribution plots
            print("Generating error distribution plots...")
            fig = plt.figure(figsize=(15, 5))
            
            # MSE histogram
            ax1 = plt.subplot(1, 3, 1)
            ax1.hist(all_mse.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(all_mse.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={all_mse.mean():.6f}')
            ax1.axvline(all_mse.median(), color='green', linestyle='--', linewidth=2, label=f'Median={all_mse.median():.6f}')
            ax1.set_xlabel('MSE', fontsize=11)
            ax1.set_ylabel('Number of Samples', fontsize=11)
            ax1.set_title('MSE Distribution', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # NMSE histogram
            ax2 = plt.subplot(1, 3, 2)
            ax2.hist(all_nmse.numpy(), bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.axvline(all_nmse.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={all_nmse.mean():.6f}')
            ax2.axvline(all_nmse.median(), color='orange', linestyle='--', linewidth=2, label=f'Median={all_nmse.median():.6f}')
            ax2.set_xlabel('NMSE', fontsize=11)
            ax2.set_ylabel('Number of Samples', fontsize=11)
            ax2.set_title('Normalized MSE Distribution', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # MAE histogram
            ax3 = plt.subplot(1, 3, 3)
            ax3.hist(all_mae.numpy(), bins=50, alpha=0.7, color='purple', edgecolor='black')
            ax3.axvline(all_mae.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={all_mae.mean():.6f}')
            ax3.axvline(all_mae.median(), color='orange', linestyle='--', linewidth=2, label=f'Median={all_mae.median():.6f}')
            ax3.set_xlabel('MAE', fontsize=11)
            ax3.set_ylabel('Number of Samples', fontsize=11)
            ax3.set_title('MAE Distribution', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            error_plot_path = 'train_error_distribution.png'
            plt.savefig(error_plot_path, dpi=150, bbox_inches='tight')
            print(f"Error distribution plot saved to '{error_plot_path}'")
            plt.close()
            
            # Generate per-sample error plot (line plot showing error across all samples)
            print("Generating per-sample error plot...")
            fig, ax = plt.subplots(figsize=(14, 6))
            sample_indices = np.arange(len(all_mse))
            ax.plot(sample_indices, all_mse.numpy(), 'b-', linewidth=1, alpha=0.6, label='MSE per sample')
            ax.axhline(all_mse.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean MSE = {all_mse.mean():.6f}')
            ax.scatter([best_idx], [all_mse[best_idx]], color='green', s=100, zorder=5, label=f'Best (#{best_idx})')
            ax.scatter([worst_idx], [all_mse[worst_idx]], color='red', s=100, zorder=5, label=f'Worst (#{worst_idx})')
            ax.set_xlabel('Sample Index', fontsize=12)
            ax.set_ylabel('MSE', fontsize=12)
            ax.set_title(f'Per-Sample MSE Across Dataset ({len(all_mse)} samples)', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            per_sample_plot_path = 'train_per_sample_errors.png'
            plt.savefig(per_sample_plot_path, dpi=150, bbox_inches='tight')
            print(f"Per-sample error plot saved to '{per_sample_plot_path}'")
            plt.close()
            
            print("\nQuantitative evaluation complete!")
    else:
        print("\n" + "="*80)
        print("Quantitative evaluation skipped (no ground truth available)")
        print("="*80)

    # -----------------------------------------------------------------
    # 10. Save the Trained Model
    # -----------------------------------------------------------------
    print("\n--- Training Complete ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()

