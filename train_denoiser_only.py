import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np
import time

from data_loader import MIMOSAR_Dataset
from models import CNNDenoiser
from utils import complex_matmul, complex_conj_transpose_matmul

# -----------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------
MAT_FILE = 'FL_MIMO_SAR_data.mat'
MODEL_SAVE_PATH = 'denoiser_pretrained.pth'

# Training Hyperparameters
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# *** Denoiser Training Mode ***
DENOISER_TRAINING_MODE = 'supervised'  # Options: 'supervised', 'unsupervised'

# Noise parameters (for supervised training)
NOISE_LEVEL = 0.0001  # Std dev of synthetic noise added to input

# -----------------------------------------------------------------
# 2. Setup
# -----------------------------------------------------------------
def main():
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")
    print(f"--- Denoiser Training Mode: {DENOISER_TRAINING_MODE.upper()} ---\n")

    # -----------------------------------------------------------------
    # 3. Load Data
    # -----------------------------------------------------------------
    print("Loading dataset...")
    
    # Determine if we need ground truth
    need_ground_truth = (DENOISER_TRAINING_MODE == 'supervised')
    
    dataset = MIMOSAR_Dataset(MAT_FILE, return_ground_truth=need_ground_truth)
    
    # Check if ground truth is actually available
    has_ground_truth = dataset.has_ground_truth
    
    # Adjust training mode if ground truth not available
    if need_ground_truth and not has_ground_truth:
        print(f"\n*** WARNING: Supervised training requested but ground truth not available!")
        print(f"*** Falling back to UNSUPERVISED training mode ***\n")
        actual_training_mode = 'unsupervised'
    else:
        actual_training_mode = DENOISER_TRAINING_MODE
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Get steering matrix A (needed for unsupervised training)
    A_tensor = dataset.A.to(device)
    A_batch_tensor = A_tensor.unsqueeze(0)

    # -----------------------------------------------------------------
    # 4. Initialize Model, Loss, and Optimizer
    # -----------------------------------------------------------------
    print("Initializing denoiser model...")
    model = CNNDenoiser().to(device)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Denoiser Training Configuration:")
    print(f"  Mode: {actual_training_mode}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    if actual_training_mode == 'supervised':
        print(f"  Noise level: {NOISE_LEVEL}")
    print()

    # -----------------------------------------------------------------
    # 5. Training Loop
    # -----------------------------------------------------------------
    print(f"--- Starting Denoiser Training for {NUM_EPOCHS} Epochs ---\n")
    
    model.train()
    
    epoch_losses = []
    
    tic = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        
        for i, batch_data in enumerate(dataloader):
            # Unpack batch data
            if has_ground_truth:
                y_batch, x_gt_batch = batch_data
                x_gt_batch = x_gt_batch.to(device)
            else:
                y_batch = batch_data
                x_gt_batch = None
            
            y_batch = y_batch.to(device)
            
            # --- Create input for denoiser ---
            # Input is always the matched filter output: x_noisy = A^H @ y
            with torch.no_grad():
                x_noisy = complex_conj_transpose_matmul(A_batch_tensor, y_batch)
            
            # For supervised training, add synthetic noise
            if actual_training_mode == 'supervised':
                noise = torch.randn_like(x_noisy) * NOISE_LEVEL
                x_noisy = x_noisy + noise
            
            # --- Forward Pass ---
            x_denoised = model(x_noisy)
            
            # --- Compute Loss ---
            if actual_training_mode == 'supervised':
                # Supervised: compare directly with ground truth
                loss = criterion(x_denoised, x_gt_batch)
                
            elif actual_training_mode == 'unsupervised':
                # Unsupervised: check measurement domain consistency
                # Forward model: y_hat = A @ x_denoised
                y_hat = complex_matmul(A_batch_tensor, x_denoised)
                loss = criterion(y_hat, y_batch)
            
            # --- Backward Pass and Optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}")

        # Print average loss for the epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        print(f"*** Epoch {epoch+1} Complete. Average Loss: {avg_epoch_loss:.6f} ***")

    toc = time.time()
    print(f"\nTotal Training time: {toc-tic:.2f} seconds")

    # -----------------------------------------------------------------
    # 6. Plot Training Loss Curve
    # -----------------------------------------------------------------
    print("Generating training loss curve...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Denoiser Training Loss ({actual_training_mode.upper()} mode)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    loss_curve_path = f'denoiser_training_loss_{actual_training_mode}.png'
    plt.savefig(loss_curve_path)
    print(f"Training loss curve saved to {loss_curve_path}")
    plt.close()

    # -----------------------------------------------------------------
    # 7. Visualize Denoiser Output on Last Batch
    # -----------------------------------------------------------------
    print("Generating visualization of denoiser output...")
    model.eval()
    
    with torch.no_grad():
        # Use last batch
        # Create noisy input
        x_noisy = complex_conj_transpose_matmul(A_batch_tensor, y_batch)
        if actual_training_mode == 'supervised':
            noise = torch.randn_like(x_noisy) * NOISE_LEVEL
            x_noisy = x_noisy + noise
        
        # Denoise
        x_denoised = model(x_noisy)
        
        # Select first sample
        x_noisy_sample = x_noisy[0]
        x_denoised_sample = x_denoised[0]
        
        # Convert to numpy (magnitude)
        x_noisy_np = torch.abs(torch.view_as_complex(x_noisy_sample.permute(1, 0).contiguous())).cpu().numpy()
        x_denoised_np = torch.abs(torch.view_as_complex(x_denoised_sample.permute(1, 0).contiguous())).cpu().numpy()
        
        if has_ground_truth:
            x_gt_sample = x_gt_batch[0]
            x_gt_np = torch.abs(torch.view_as_complex(x_gt_sample.permute(1, 0).contiguous())).cpu().numpy()
        else:
            x_gt_np = None
        
        # Create angle axis
        N_theta = x_noisy_np.shape[0]
        theta = np.linspace(25.0, -25.0, N_theta)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(theta, x_noisy_np, 'r-', label='Noisy Input (A^H @ y)', linewidth=2, alpha=0.7)
        plt.plot(theta, x_denoised_np, 'b-', label='Denoiser Output', linewidth=2)
        if x_gt_np is not None:
            plt.plot(theta, x_gt_np, 'g--', label='Ground Truth', linewidth=2, alpha=0.7)
        
        plt.xlabel('Angle (degrees)', fontsize=12)
        plt.ylabel('Magnitude', fontsize=12)
        plt.title(f'Denoiser Output Visualization ({actual_training_mode.upper()} mode)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([25, -25])
        
        viz_path = f'denoiser_output_{actual_training_mode}.png'
        plt.savefig(viz_path)
        print(f"Denoiser output visualization saved to {viz_path}")
        plt.close()
    
    # -----------------------------------------------------------------
    # 7b. Visualize Measurement Domain Consistency
    # -----------------------------------------------------------------
    print("Generating measurement domain visualization...")
    
    with torch.no_grad():
        # Forward pass through A: y_hat = A @ x_denoised
        y_hat = complex_matmul(A_batch_tensor, x_denoised)
        
        # Get first sample
        y_gt_sample = y_batch[0]  # Ground truth measurements
        y_hat_sample = y_hat[0]   # Predicted measurements from denoiser output
        
        # Convert to complex numpy
        y_gt_complex = torch.view_as_complex(y_gt_sample.permute(1, 0).contiguous()).cpu().numpy()
        y_hat_complex = torch.view_as_complex(y_hat_sample.permute(1, 0).contiguous()).cpu().numpy()
        
        # Get magnitude, real, imaginary
        y_gt_mag = np.abs(y_gt_complex)
        y_gt_real = np.real(y_gt_complex)
        y_gt_imag = np.imag(y_gt_complex)
        
        y_hat_mag = np.abs(y_hat_complex)
        y_hat_real = np.real(y_hat_complex)
        y_hat_imag = np.imag(y_hat_complex)
        
        # Measurement indices
        M_rx = y_gt_mag.shape[0]
        meas_idx = np.arange(M_rx)
        
        # Create plot with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Real Part
        axes[0].plot(meas_idx, y_gt_real, 'go-', label='Ground Truth y (Real)', linewidth=2, markersize=6)
        axes[0].plot(meas_idx, y_hat_real, 'b^--', label='Predicted y (Real)', linewidth=2, markersize=6, alpha=0.7)
        axes[0].set_xlabel('Measurement Index', fontsize=11)
        axes[0].set_ylabel('Real Part', fontsize=11)
        axes[0].set_title('Measurement Domain: Real Part', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Imaginary Part
        axes[1].plot(meas_idx, y_gt_imag, 'go-', label='Ground Truth y (Imag)', linewidth=2, markersize=6)
        axes[1].plot(meas_idx, y_hat_imag, 'b^--', label='Predicted y (Imag)', linewidth=2, markersize=6, alpha=0.7)
        axes[1].set_xlabel('Measurement Index', fontsize=11)
        axes[1].set_ylabel('Imaginary Part', fontsize=11)
        axes[1].set_title('Measurement Domain: Imaginary Part', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        meas_viz_path = f'denoiser_measurement_domain_{actual_training_mode}.png'
        plt.savefig(meas_viz_path)
        print(f"Measurement domain visualization saved to {meas_viz_path}")
        plt.close()
        
        # Calculate and print measurement domain error
        mse_measurement = torch.mean((y_hat - y_batch) ** 2).item()
        print(f"\nMeasurement Domain MSE: {mse_measurement:.6e}")

    # -----------------------------------------------------------------
    # 8. Save the Trained Denoiser
    # -----------------------------------------------------------------
    print("\n--- Training Complete ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained denoiser saved to: {MODEL_SAVE_PATH}")
    print(f"\nYou can now use this pre-trained denoiser with:")
    print(f"  python train_configurable.py --strategy two_stage")
    print(f"  (Set TRAINING_STRATEGY = 'two_stage' in train_configurable.py)")


if __name__ == '__main__':
    main()

