"""
Progressive Curriculum Training for Denoiser

This script implements curriculum learning to train a denoiser that can handle
inputs from any iteration depth in the unrolled network.

Strategy:
1. Stage 0: Train on A^H @ y
2. Stage 1: Generate x_after_ADMM_0, retrain on [A^H@y, x_after_ADMM_0]
3. Stage k: Generate x_after_ADMM_{k-1}, retrain on accumulated dataset
4. Continue until target number of iterations

This addresses the domain shift problem where later iterations see different
input distributions than the matched filter output.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from time import time

from core.models import CNNDenoiser, CNNDenoiser_RealOutput, CNNDenoiser_ComplexOutput, DCLayer_ADMM
from core.data_loader import MIMOSAR_Dataset
from core.utils import complex_matmul, complex_conj_transpose_matmul

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
MAT_FILE_PATH = 'data/FL_MIMO_SAR_data.mat'

# Curriculum Training Configuration
NUM_CURRICULUM_STAGES = 3  # Train for 3 iteration depths (0, 1, 2)
CURRICULUM_TRAINING_MODE = 'unsupervised'  # 'supervised' or 'unsupervised'
CURRICULUM_RETRAINING_STRATEGY = 'fine_tune'  # 'from_scratch' or 'fine_tune'

# Training Hyperparameters
EPOCHS_PER_STAGE = 1000  # Epochs for each curriculum stage
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# ADMM Configuration (for synthetic data generation)
NUM_ADMM_STEPS = 2  # Fixed ADMM steps per iteration

# Denoiser Architecture Selection
DENOISER_TYPE = 'complex'  # Options: 'real' (best for real targets, ~62K params)
                        #          'complex' (allows imaginary, ~62K params)
                        #          'original' (shallow residual, ~3.6K params)

# Positivity Enforcement
ENFORCE_POSITIVITY = False  # True: Add ReLU to enforce output ≥ 0 (only for DENOISER_TYPE='real')

# ADMM Physical Constraints (for synthetic data generation)
ADMM_ENFORCE_REAL = False        # True: Project ADMM x-update to real values (discard imaginary)
ADMM_ENFORCE_POSITIVITY = False  # True: Clamp ADMM x-update to non-negative values

# Output
MODEL_SAVE_PATH = 'checkpoints/denoiser_curriculum.pth'

# -----------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------

def generate_synthetic_data(denoiser, admm_layer, x_input_batch, y_batch, A_batch, device):
    """
    Generate synthetic data by passing through denoiser + ADMM
    
    Args:
        denoiser: Trained denoiser model
        admm_layer: ADMM layer with fixed parameters
        x_input_batch: Input to denoiser [B, 2, N_theta]
        y_batch: Original measurements for each sample [B, 2, M_rx]
        A_batch: Steering matrix [B, 2, M_rx, N_theta]
        device: torch device
    
    Returns:
        x_after_admm: Output after denoiser + ADMM [B, 2, N_theta]
    
    Note:
        y_batch contains the ORIGINAL measurements for each sample.
        ADMM enforces data consistency with these measurements, NOT synthetic ones.
        This ensures proper data fidelity throughout all curriculum stages.
    """
    denoiser.eval()
    with torch.no_grad():
        x_input_batch = x_input_batch.to(device)
        y_batch = y_batch.to(device)
        A_batch = A_batch.to(device)
        
        # Denoiser
        x_denoised = denoiser(x_input_batch)
        
        # ADMM uses ORIGINAL measurements (not synthetic!)
        u_init = torch.zeros_like(y_batch)
        
        # ADMM forward pass - enforces consistency with original y
        x_after_admm, _ = admm_layer(x_denoised, y_batch, u_init)
    
    return x_after_admm


def train_denoiser_one_stage(denoiser, dataloader, A_tensor, training_mode, 
                              epochs, lr, device, stage_num):
    """
    Train denoiser for one curriculum stage
    
    Args:
        denoiser: Denoiser model
        dataloader: DataLoader with (x_input, x_gt, y_gt) tuples
        A_tensor: Steering matrix [2, M_rx, N_theta]
        training_mode: 'supervised' or 'unsupervised'
        epochs: Number of epochs
        lr: Learning rate
        device: torch device
        stage_num: Current stage number (for logging)
    
    Returns:
        epoch_losses: List of average losses per epoch
    """
    optimizer = optim.Adam(denoiser.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    epoch_losses = []
    
    print(f"\n--- Stage {stage_num}: Training for {epochs} Epochs ---")
    print(f"  Training mode: {training_mode}")
    print(f"  Dataset size: {len(dataloader.dataset)} samples")
    
    for epoch in range(epochs):
        denoiser.train()
        batch_losses = []
        
        for batch_data in dataloader:
            x_input_batch, x_gt_batch, y_gt_batch = batch_data
            x_input_batch = x_input_batch.to(device)
            x_gt_batch = x_gt_batch.to(device)
            y_gt_batch = y_gt_batch.to(device)
            
            # Forward pass
            # (real-valued prior is enforced at architecture level if DENOISER_TYPE='real')
            x_denoised = denoiser(x_input_batch)
            
            # Compute loss
            if training_mode == 'supervised':
                loss = criterion(x_denoised, x_gt_batch)
            else:  # unsupervised
                # Reconstruct measurements
                A_batch = A_tensor.unsqueeze(0).expand(x_denoised.shape[0], -1, -1, -1).to(device)
                y_pred = complex_matmul(A_batch, x_denoised)
                loss = criterion(y_pred, y_gt_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return epoch_losses


def visualize_stage_losses(all_stage_losses, save_path='curriculum_stage_losses.png'):
    """Plot loss curves for all curriculum stages"""
    plt.figure(figsize=(14, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_stage_losses)))
    
    for stage_num, losses in enumerate(all_stage_losses):
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, color=colors[stage_num], 
                linewidth=2, label=f'Stage {stage_num}')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Curriculum Training: Loss per Stage', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Stage losses plot saved to {save_path}")
    plt.close()


def visualize_final_denoiser(denoiser, dataset, A_tensor, training_mode, device,
                             save_path='curriculum_final_output.png'):
    """Visualize final denoiser output in image domain"""
    denoiser.eval()
    
    # Get one sample
    if dataset.has_ground_truth:
        y_sample, x_gt_sample = dataset[20]
    else:
        y_sample = dataset[20]
        x_gt_sample = None
    
    # Create matched filter input
    A_batch = A_tensor.unsqueeze(0).to(device)
    y_batch = y_sample.unsqueeze(0).to(device)
    
    with torch.no_grad():
        x_noisy = complex_conj_transpose_matmul(A_batch, y_batch)
        x_denoised = denoiser(x_noisy)
    
    # Convert to numpy
    x_noisy_np = torch.abs(torch.view_as_complex(
        x_noisy[0].permute(1, 0).contiguous())).cpu().numpy()
    x_denoised_np = torch.abs(torch.view_as_complex(
        x_denoised[0].permute(1, 0).contiguous())).cpu().numpy()
    
    if x_gt_sample is not None:
        x_gt_np = torch.abs(torch.view_as_complex(
            x_gt_sample.permute(1, 0).contiguous())).cpu().numpy()
    else:
        x_gt_np = None
    
    # Create angle axis
    N_theta = x_noisy_np.shape[0]
    theta = np.linspace(25.0, -25.0, N_theta)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(theta, x_noisy_np, 'r-', label='Noisy Input (A^H @ y)', 
            linewidth=2, alpha=0.7)
    plt.plot(theta, x_denoised_np, 'b-', label='Curriculum Denoiser Output', 
            linewidth=2)
    if x_gt_np is not None:
        plt.plot(theta, x_gt_np, 'g--', label='Ground Truth', 
                linewidth=2, alpha=0.7)
    
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Magnitude', fontsize=12)
    plt.title(f'Curriculum-Trained Denoiser Output ({training_mode.upper()})', 
             fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([25, -25])
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Final denoiser output saved to {save_path}")
    plt.close()


def visualize_unrolled_network(denoiser, admm_layer, dataset, A_tensor, 
                                training_mode, device, num_iterations=5,
                                save_path='curriculum_unrolled_output.png'):
    """
    Visualize unrolled network output with multiple denoiser+ADMM iterations
    
    Shows the progression: Initial -> Iter1 -> Iter2 -> ... -> Final
    """
    denoiser.eval()
    admm_layer.eval()
    
    # Get one sample
    if dataset.has_ground_truth:
        y_sample, x_gt_sample = dataset[20]
    else:
        y_sample = dataset[20]
        x_gt_sample = None
    
    # Create matched filter input
    A_batch = A_tensor.unsqueeze(0).to(device)
    y_batch = y_sample.unsqueeze(0).to(device)
    
    # Run unrolled network
    with torch.no_grad():
        # Initial: matched filter output
        x_current = complex_conj_transpose_matmul(A_batch, y_batch)
        x_initial = x_current.clone()
        
        # Store outputs from each iteration
        iteration_outputs = [x_initial]
        
        # Run through iterations
        u = torch.zeros_like(y_batch)
        for it in range(num_iterations):
            # Denoiser
            x_denoised = denoiser(x_current)
            
            # ADMM (uses original measurements!)
            x_current, u = admm_layer(x_denoised, y_batch, u)
            
            iteration_outputs.append(x_current.clone())
    
    # Convert to numpy (magnitude)
    outputs_np = []
    for x_out in iteration_outputs:
        x_np = torch.abs(torch.view_as_complex(
            x_out[0].permute(1, 0).contiguous())).cpu().numpy()
        outputs_np.append(x_np)
    
    if x_gt_sample is not None:
        x_gt_np = torch.abs(torch.view_as_complex(
            x_gt_sample.permute(1, 0).contiguous())).cpu().numpy()
    else:
        x_gt_np = None
    
    # Create angle axis
    N_theta = outputs_np[0].shape[0]
    theta = np.linspace(25.0, -25.0, N_theta)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Show progression of all iterations
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(outputs_np)))
    
    for idx, (output, color) in enumerate(zip(outputs_np, colors)):
        if idx == 0:
            label = 'Initial (A^H @ y)'
            alpha = 0.5
            linestyle = '--'
        else:
            label = f'Iteration {idx}'
            alpha = 0.7 + (idx / len(outputs_np)) * 0.3  # Progressively more opaque
            linestyle = '-'
        
        ax.plot(theta, output, color=color, label=label, 
               linewidth=2, alpha=alpha, linestyle=linestyle)
    
    if x_gt_np is not None:
        ax.plot(theta, x_gt_np, 'r--', label='Ground Truth', 
               linewidth=2.5, alpha=0.9)
    
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title(f'Unrolled Network Progression ({training_mode.upper()})', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([25, -25])
    
    # Right plot: Compare initial vs final vs ground truth
    ax = axes[1]
    ax.plot(theta, outputs_np[0], 'r-', label='Initial (A^H @ y)', 
           linewidth=2, alpha=0.7)
    ax.plot(theta, outputs_np[-1], 'b-', label=f'Final (Iter {num_iterations})', 
           linewidth=2.5)
    if x_gt_np is not None:
        ax.plot(theta, x_gt_np, 'g--', label='Ground Truth', 
               linewidth=2.5, alpha=0.8)
    
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    ax.set_title('Initial vs Final Reconstruction', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([25, -25])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Unrolled network output saved to {save_path}")
    plt.close()
    
    # Print metrics
    if x_gt_np is not None:
        mse_initial = np.mean((outputs_np[0] - x_gt_np) ** 2)
        mse_final = np.mean((outputs_np[-1] - x_gt_np) ** 2)
        improvement = 100.0 * (mse_initial - mse_final) / mse_initial
        
        print(f"  Initial MSE: {mse_initial:.6e}")
        print(f"  Final MSE (Iter {num_iterations}): {mse_final:.6e}")
        print(f"  Improvement: {improvement:.2f}%")


def visualize_measurement_domain(denoiser, dataset, A_tensor, device,
                                 save_path='curriculum_measurement_domain.png'):
    """Visualize measurement domain consistency"""
    denoiser.eval()
    
    # Get one sample
    if dataset.has_ground_truth:
        y_sample, x_gt_sample = dataset[0]
    else:
        y_sample = dataset[0]
    
    # Create matched filter input
    A_batch = A_tensor.unsqueeze(0).to(device)
    y_batch = y_sample.unsqueeze(0).to(device)
    
    with torch.no_grad():
        x_noisy = complex_conj_transpose_matmul(A_batch, y_batch)
        x_denoised = denoiser(x_noisy)
        y_hat = complex_matmul(A_batch, x_denoised)
    
    # Convert to complex numpy
    y_gt_complex = torch.view_as_complex(
        y_batch[0].permute(1, 0).contiguous()).cpu().numpy()
    y_hat_complex = torch.view_as_complex(
        y_hat[0].permute(1, 0).contiguous()).cpu().numpy()
    
    # Extract real and imaginary
    y_gt_real = np.real(y_gt_complex)
    y_gt_imag = np.imag(y_gt_complex)
    y_hat_real = np.real(y_hat_complex)
    y_hat_imag = np.imag(y_hat_complex)
    
    # Measurement indices
    M_rx = y_gt_real.shape[0]
    meas_idx = np.arange(M_rx)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Real Part
    axes[0].plot(meas_idx, y_gt_real, 'go-', label='Ground Truth y (Real)', 
                linewidth=2, markersize=6)
    axes[0].plot(meas_idx, y_hat_real, 'b^--', label='Predicted y (Real)', 
                linewidth=2, markersize=6, alpha=0.7)
    axes[0].set_xlabel('Measurement Index', fontsize=11)
    axes[0].set_ylabel('Real Part', fontsize=11)
    axes[0].set_title('Curriculum Denoiser: Measurement Domain (Real)', 
                     fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Imaginary Part
    axes[1].plot(meas_idx, y_gt_imag, 'go-', label='Ground Truth y (Imag)', 
                linewidth=2, markersize=6)
    axes[1].plot(meas_idx, y_hat_imag, 'b^--', label='Predicted y (Imag)', 
                linewidth=2, markersize=6, alpha=0.7)
    axes[1].set_xlabel('Measurement Index', fontsize=11)
    axes[1].set_ylabel('Imaginary Part', fontsize=11)
    axes[1].set_title('Curriculum Denoiser: Measurement Domain (Imaginary)', 
                     fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Measurement domain plot saved to {save_path}")
    plt.close()
    
    # Calculate MSE
    mse_measurement = torch.mean((y_hat - y_batch) ** 2).item()
    print(f"Final Measurement Domain MSE: {mse_measurement:.6e}")


# -----------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------

def main():
    print("="*70)
    print("PROGRESSIVE CURRICULUM TRAINING FOR DENOISER")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Curriculum stages: {NUM_CURRICULUM_STAGES}")
    print(f"Training mode: {CURRICULUM_TRAINING_MODE}")
    print(f"Retraining strategy: {CURRICULUM_RETRAINING_STRATEGY}")
    print(f"Epochs per stage: {EPOCHS_PER_STAGE}")
    print(f"Denoiser type: {DENOISER_TYPE}")
    print("="*70)
    
    # -----------------------------------------------------------------
    # 1. Load Original Dataset
    # -----------------------------------------------------------------
    print("\n[1] Loading original dataset...")
    
    # Determine if we need ground truth
    need_ground_truth = (CURRICULUM_TRAINING_MODE == 'supervised')
    
    dataset = MIMOSAR_Dataset(MAT_FILE_PATH, return_ground_truth=need_ground_truth)
    A_tensor = dataset.A.to(DEVICE)
    
    print(f"  Original dataset size: {len(dataset)} samples")
    print(f"  Angle bins: {A_tensor.shape[2]}")
    print(f"  Virtual antennas: {A_tensor.shape[1]}")
    
    # Extract original data
    original_x_inputs = []  # A^H @ y for each sample
    original_x_gts = []
    original_ys = []
    
    A_batch = A_tensor.unsqueeze(0).to(DEVICE)
    
    for idx in range(len(dataset)):
        if dataset.has_ground_truth:
            y_sample, x_gt_sample = dataset[idx]
            original_x_gts.append(x_gt_sample)
        else:
            y_sample = dataset[idx]
            original_x_gts.append(None)
        
        original_ys.append(y_sample)
        
        # Compute matched filter output
        y_batch = y_sample.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            x_mf = complex_conj_transpose_matmul(A_batch, y_batch)
        original_x_inputs.append(x_mf.squeeze(0).cpu())
    
    print(f"  ✓ Original data prepared")
    
    # -----------------------------------------------------------------
    # 2. Initialize Models
    # -----------------------------------------------------------------
    print("\n[2] Initializing models...")
    
    # Create denoiser based on DENOISER_TYPE
    if DENOISER_TYPE == 'real':
        denoiser = CNNDenoiser_RealOutput(enforce_positivity=ENFORCE_POSITIVITY).to(DEVICE)
        positivity_str = " + Positivity" if ENFORCE_POSITIVITY else ""
        print(f"  Using CNNDenoiser_RealOutput (Deep, 1 channel, ~62K params{positivity_str})")
    elif DENOISER_TYPE == 'complex':
        denoiser = CNNDenoiser_ComplexOutput().to(DEVICE)
        print(f"  Using CNNDenoiser_ComplexOutput (Deep, 2 channels, ~62K params)")
    elif DENOISER_TYPE == 'original':
        denoiser = CNNDenoiser().to(DEVICE)
        print(f"  Using CNNDenoiser (Shallow residual, 2 channels, ~3.6K params)")
    else:
        raise ValueError(f"Unknown DENOISER_TYPE: {DENOISER_TYPE}")
    
    admm_layer = DCLayer_ADMM(A_tensor, N_admm_steps=NUM_ADMM_STEPS,
                             enforce_real=ADMM_ENFORCE_REAL,
                             enforce_positivity=ADMM_ENFORCE_POSITIVITY).to(DEVICE)
    
    # Freeze ADMM parameters
    for param in admm_layer.parameters():
        param.requires_grad = False
    
    print(f"  ✓ Denoiser initialized")
    print(f"  ✓ ADMM layer initialized (fixed parameters)")
    
    # -----------------------------------------------------------------
    # 3. Curriculum Training Loop
    # -----------------------------------------------------------------
    all_stage_losses = []
    accumulated_x_inputs = [torch.stack(original_x_inputs)]  # List of tensors
    accumulated_x_gts = [torch.stack(original_x_gts) if original_x_gts[0] is not None 
                        else None]
    accumulated_ys = [torch.stack(original_ys)]
    
    total_start_time = time()
    
    for stage in range(NUM_CURRICULUM_STAGES):
        print("\n" + "="*70)
        print(f"CURRICULUM STAGE {stage}")
        print("="*70)
        
        # Prepare dataset for this stage
        # Concatenate all accumulated data
        all_x_inputs = torch.cat(accumulated_x_inputs, dim=0)
        all_ys = torch.cat(accumulated_ys, dim=0)
        
        if accumulated_x_gts[0] is not None:
            # Repeat original ground truth for each synthetic sample
            all_x_gts = []
            for _ in range(len(accumulated_x_inputs)):
                all_x_gts.append(accumulated_x_gts[0])
            all_x_gts = torch.cat(all_x_gts, dim=0)
        else:
            all_x_gts = torch.zeros_like(all_x_inputs)  # Dummy (won't be used)
        
        # Create TensorDataset and DataLoader
        tensor_dataset = TensorDataset(all_x_inputs, all_x_gts, all_ys)
        dataloader = DataLoader(tensor_dataset, batch_size=BATCH_SIZE, shuffle=True)
        
        print(f"\nDataset for stage {stage}:")
        print(f"  Total samples: {len(tensor_dataset)}")
        print(f"  = {len(accumulated_x_inputs)} batches × {len(original_x_inputs)} original samples")
        
        # Retraining strategy
        if stage > 0 and CURRICULUM_RETRAINING_STRATEGY == 'from_scratch':
            print(f"  Reinitializing denoiser from scratch...")
            # Recreate denoiser based on type
            if DENOISER_TYPE == 'real':
                denoiser = CNNDenoiser_RealOutput(enforce_positivity=ENFORCE_POSITIVITY).to(DEVICE)
            elif DENOISER_TYPE == 'complex':
                denoiser = CNNDenoiser_ComplexOutput().to(DEVICE)
            elif DENOISER_TYPE == 'original':
                denoiser = CNNDenoiser().to(DEVICE)
        elif stage > 0:
            print(f"  Fine-tuning existing denoiser...")
        
        # Train denoiser for this stage
        stage_losses = train_denoiser_one_stage(
            denoiser=denoiser,
            dataloader=dataloader,
            A_tensor=A_tensor,
            training_mode=CURRICULUM_TRAINING_MODE,
            epochs=EPOCHS_PER_STAGE,
            lr=LEARNING_RATE,
            device=DEVICE,
            stage_num=stage
        )
        
        all_stage_losses.append(stage_losses)
        
        print(f"\n  Stage {stage} complete!")
        print(f"  Final loss: {stage_losses[-1]:.6e}")
        
        # Generate synthetic data for next stage (if not last stage)
        if stage < NUM_CURRICULUM_STAGES - 1:
            print(f"\n  Generating synthetic data for stage {stage + 1}...")
            
            synthetic_x_inputs = []
            A_batch_expanded = A_tensor.unsqueeze(0).to(DEVICE)
            
            # Use the most recent accumulated inputs
            current_inputs = accumulated_x_inputs[-1]
            
            # Generate in batches
            num_batches = (len(current_inputs) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(current_inputs))
                
                x_input_batch = current_inputs[start_idx:end_idx]
                # Get corresponding ORIGINAL measurements for this batch
                y_batch = accumulated_ys[0][start_idx:end_idx]
                A_batch = A_batch_expanded.expand(x_input_batch.shape[0], -1, -1, -1)
                
                x_after_admm = generate_synthetic_data(
                    denoiser=denoiser,
                    admm_layer=admm_layer,
                    x_input_batch=x_input_batch,
                    y_batch=y_batch,
                    A_batch=A_batch,
                    device=DEVICE
                )
                
                synthetic_x_inputs.append(x_after_admm.cpu())
            
            synthetic_x_inputs = torch.cat(synthetic_x_inputs, dim=0)
            
            # Accumulate
            accumulated_x_inputs.append(synthetic_x_inputs)
            accumulated_ys.append(accumulated_ys[0])  # Same original ys
            
            print(f"    Generated {len(synthetic_x_inputs)} synthetic samples")
            print(f"    Next stage will have {(stage + 2) * len(original_x_inputs)} total samples")
    
    total_time = time() - total_start_time
    
    print("\n" + "="*70)
    print("CURRICULUM TRAINING COMPLETE")
    print("="*70)
    print(f"Total training time: {total_time:.2f} seconds")
    print(f"Final dataset size: {len(tensor_dataset)} samples")
    
    # -----------------------------------------------------------------
    # 4. Visualizations
    # -----------------------------------------------------------------
    print("\n[4] Generating visualizations...")
    
    # Loss curves
    visualize_stage_losses(all_stage_losses, 
                          save_path='curriculum_stage_losses.png')
    
    # Final denoiser output (image domain)
    visualize_final_denoiser(denoiser, dataset, A_tensor, 
                             CURRICULUM_TRAINING_MODE, DEVICE,
                             save_path='curriculum_final_output.png')
    
    # Unrolled network output (multiple iterations)
    visualize_unrolled_network(denoiser, admm_layer, dataset, A_tensor,
                               CURRICULUM_TRAINING_MODE, DEVICE,
                               num_iterations=NUM_CURRICULUM_STAGES,
                               save_path='curriculum_unrolled_output.png')
    
    # Measurement domain
    visualize_measurement_domain(denoiser, dataset, A_tensor, DEVICE,
                                save_path='curriculum_measurement_domain.png')
    
    # -----------------------------------------------------------------
    # 5. Save Model
    # -----------------------------------------------------------------
    print(f"\n[5] Saving curriculum-trained denoiser...")
    torch.save(denoiser.state_dict(), MODEL_SAVE_PATH)
    print(f"  ✓ Saved to: {MODEL_SAVE_PATH}")
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Use the curriculum-trained denoiser in two-stage training:")
    print(f"     python train_configurable.py")
    print(f"       (Set PRETRAINED_DENOISER_PATH = '{MODEL_SAVE_PATH}')")
    print("  2. Compare with standard denoiser training:")
    print(f"     python train_denoiser_only.py")
    print("="*70)


if __name__ == '__main__':
    main()
