import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import time

# Add project root to path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


from core.data_loader import MIMOSAR_Dataset
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
MAT_FILE = '../../data/FL_MIMO_SAR_data.mat'
MODEL_SAVE_PATH = '../../checkpoints/dbp_model.pth'

# Training Hyperparameters
NUM_EPOCHS = 500
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
PRETRAINED_DENOISER_PATH = '../../checkpoints/denoiser_curriculum.pth'  # Path to pre-trained denoiser (for two_stage)
FREEZE_DENOISER = True  # True: freeze denoiser (only train ADMM), False: fine-tune denoiser too

# *** NEW: Real-Valued Prior Enforcement ***
# If target reflectivity is known to be real (no imaginary component)
ENFORCE_REAL_PRIOR = True  # True: enforce real-valued outputs, False: allow complex
REAL_PRIOR_STRATEGY = 'hybrid'  # Options: 'loss_penalty', 'hard_projection', 'hybrid'
REAL_PRIOR_WEIGHT = 0.1  # Weight for imaginary penalty (only for 'loss_penalty' and 'hybrid')

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
    if ENFORCE_REAL_PRIOR:
        print(f"--- Real Prior: {REAL_PRIOR_STRATEGY.upper()} (weight={REAL_PRIOR_WEIGHT}) ---")
    else:
        print(f"--- Real Prior: DISABLED (allowing complex outputs) ---")

    # -----------------------------------------------------------------
    # 3. Load Data
    # -----------------------------------------------------------------
    print("Loading dataset...")
    
    # Determine if we need ground truth
    need_ground_truth = (TRAINING_MODE in ['supervised', 'hybrid'])
    
    dataset = MIMOSAR_Dataset(MAT_FILE, return_ground_truth=need_ground_truth)
    
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
        N_admm_steps=NUM_ADMM_STEPS
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
    print(f"--- Starting Training for {NUM_EPOCHS} Epochs ---")
    
    model.train() # Set the model to training mode
    
    # Track losses for plotting
    epoch_losses = []
    epoch_supervised_losses = []
    epoch_unsupervised_losses = []
    
    tic = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_total_loss = 0.0
        epoch_sup_loss = 0.0
        epoch_unsup_loss = 0.0
        
        for i, batch_data in enumerate(dataloader):
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
            x_hat_batch = model(y_batch)
            
            # 2. Apply real-valued prior enforcement if enabled
            real_prior_penalty = torch.tensor(0.0)
            if ENFORCE_REAL_PRIOR:
                prior_strategy = REAL_PRIOR_STRATEGY if ENFORCE_REAL_PRIOR else 'none'
                x_hat_batch, real_prior_penalty = enforce_real_prior(
                    x_hat_batch, 
                    strategy=prior_strategy,
                    penalty_weight=REAL_PRIOR_WEIGHT
                )
            
            # 3. Compute losses based on training mode
            total_loss = 0.0
            loss_supervised = torch.tensor(0.0)
            loss_unsupervised = torch.tensor(0.0)
            
            if actual_training_mode == 'supervised':
                # Supervised: only image domain loss
                loss_supervised = criterion_image(x_hat_batch, x_gt_batch)
                total_loss = SUPERVISED_WEIGHT * loss_supervised + real_prior_penalty
                
            elif actual_training_mode == 'unsupervised':
                # Unsupervised: only measurement domain loss
                y_hat_batch = complex_matmul(A_batch_tensor, x_hat_batch)
                loss_unsupervised = criterion_measurement(y_hat_batch, y_batch)
                total_loss = UNSUPERVISED_WEIGHT * loss_unsupervised + real_prior_penalty
                
            elif actual_training_mode == 'hybrid':
                # Hybrid: both losses
                loss_supervised = criterion_image(x_hat_batch, x_gt_batch)
                y_hat_batch = complex_matmul(A_batch_tensor, x_hat_batch)
                loss_unsupervised = criterion_measurement(y_hat_batch, y_batch)
                total_loss = (SUPERVISED_WEIGHT * loss_supervised + 
                             UNSUPERVISED_WEIGHT * loss_unsupervised + 
                             real_prior_penalty)
            
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
            
            if (i + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], " + 
                      f"Total Loss: {total_loss.item():.6f}")
                if actual_training_mode in ['supervised', 'hybrid']:
                    print(f"    Supervised (image): {loss_supervised.item():.6f}")
                if actual_training_mode in ['unsupervised', 'hybrid']:
                    print(f"    Unsupervised (measurement): {loss_unsupervised.item():.6f}")

        # Print average loss for the epoch
        avg_total_loss = epoch_total_loss / len(dataloader)
        avg_sup_loss = epoch_sup_loss / len(dataloader)
        avg_unsup_loss = epoch_unsup_loss / len(dataloader)
        
        epoch_losses.append(avg_total_loss)
        epoch_supervised_losses.append(avg_sup_loss)
        epoch_unsupervised_losses.append(avg_unsup_loss)
        
        print(f"*** Epoch {epoch+1} Complete. Average Total Loss: {avg_total_loss:.6f} ***")
        if actual_training_mode in ['supervised', 'hybrid']:
            print(f"    Average Supervised Loss: {avg_sup_loss:.6f}")
        if actual_training_mode in ['unsupervised', 'hybrid']:
            print(f"    Average Unsupervised Loss: {avg_unsup_loss:.6f}")

    toc = time.time()
    print(f"Total Training time is: {toc-tic} seconds")

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
    # 9. Save the Trained Model
    # -----------------------------------------------------------------
    print("--- Training Complete ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained model saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()

