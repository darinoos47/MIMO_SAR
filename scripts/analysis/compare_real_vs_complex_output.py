"""
Compare Three Denoiser Architectures

This script trains three denoisers and compares their performance:
1. Real-Only Output: Deep encoder-decoder, 1 output channel (real only)
2. Complex Output: Deep encoder-decoder, 2 output channels (real + imag)
3. Original CNNDenoiser: Shallow residual network, 2 output channels

All are trained unsupervised using measurement domain loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from core.data_loader import MIMOSAR_Dataset
from core.utils import complex_matmul, complex_conj_transpose_matmul
from core.models import CNNDenoiser_RealOutput, CNNDenoiser_ComplexOutput, CNNDenoiser

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
MAT_FILE = 'data/FL_MIMO_SAR_data.mat'

# Training Hyperparameters
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Output
CHECKPOINT_REAL = 'checkpoints/denoiser_real_only.pth'
CHECKPOINT_COMPLEX = 'checkpoints/denoiser_complex.pth'
CHECKPOINT_ORIGINAL = 'checkpoints/denoiser_original.pth'
RESULTS_DIR = 'results/comparison_three_denoisers/'

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------
# Note: Using denoiser architectures from core.models:
#   - CNNDenoiser_RealOutput: Deep encoder-decoder, 1 output channel (real only)
#   - CNNDenoiser_ComplexOutput: Deep encoder-decoder, 2 output channels (real + imag)
#   - CNNDenoiser: Original shallow residual network, 2 output channels

def train_denoiser(denoiser, dataloader, A_tensor, num_epochs, lr, device, model_name):
    """
    Train a denoiser using unsupervised measurement domain loss
    
    Args:
        denoiser: Denoiser model
        dataloader: DataLoader with measurements y
        A_tensor: Steering matrix [2, M_rx, N_theta]
        num_epochs: Number of training epochs
        lr: Learning rate
        device: torch device
        model_name: Name for logging
    
    Returns:
        epoch_losses: List of losses per epoch
    """
    optimizer = optim.Adam(denoiser.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    epoch_losses = []
    
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {dataloader.batch_size}")
    print(f"  Device: {device}")
    print(f"{'='*70}\n")
    
    denoiser.train()
    A_batch_tensor = A_tensor.unsqueeze(0).to(device)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        batch_losses = []
        
        for batch_data in dataloader:
            # Handle both cases: dataset with/without ground truth
            if isinstance(batch_data, (list, tuple)):
                y_batch = batch_data[0]  # Get measurements only
            else:
                y_batch = batch_data
            y_batch = y_batch.to(device)
            
            # Create input: A^H * y
            x_input = complex_conj_transpose_matmul(A_batch_tensor, y_batch)
            
            # Forward pass through denoiser
            x_denoised = denoiser(x_input)
            
            # Compute measurements from denoised output
            A_expanded = A_batch_tensor.expand(x_denoised.shape[0], -1, -1, -1)
            y_pred = complex_matmul(A_expanded, x_denoised)
            
            # Unsupervised loss: measurement domain
            loss = criterion(y_pred, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        avg_loss = np.mean(batch_losses)
        epoch_losses.append(avg_loss)
        
        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"Epoch [{epoch+1:3d}/{num_epochs}] | Loss: {avg_loss:.6e} | Time: {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\nTraining complete! Total time: {total_time:.1f}s")
    print(f"Final loss: {epoch_losses[-1]:.6e}")
    
    return epoch_losses


# -----------------------------------------------------------------
# Evaluation Function
# -----------------------------------------------------------------

def evaluate_denoiser(denoiser, dataset, A_tensor, device, model_name):
    """
    Evaluate denoiser on entire dataset
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    denoiser.eval()
    A_batch_tensor = A_tensor.unsqueeze(0).to(device)
    
    dataloader_eval = DataLoader(dataset, batch_size=64, shuffle=False)
    
    measurement_errors = []
    real_magnitudes = []
    imag_magnitudes = []
    
    with torch.no_grad():
        for batch_data in dataloader_eval:
            # Handle both cases: dataset with/without ground truth
            if isinstance(batch_data, (list, tuple)):
                y_batch = batch_data[0]  # Get measurements only
            else:
                y_batch = batch_data
            y_batch = y_batch.to(device)
            
            # Input: A^H * y
            x_input = complex_conj_transpose_matmul(A_batch_tensor, y_batch)
            
            # Denoised output
            x_denoised = denoiser(x_input)
            
            # Predicted measurements
            A_expanded = A_batch_tensor.expand(x_denoised.shape[0], -1, -1, -1)
            y_pred = complex_matmul(A_expanded, x_denoised)
            
            # Measurement error
            mse_measurement = torch.mean((y_pred - y_batch) ** 2).item()
            measurement_errors.append(mse_measurement)
            
            # Analyze output statistics
            real_mag = torch.mean(torch.abs(x_denoised[:, 0, :])).item()
            imag_mag = torch.mean(torch.abs(x_denoised[:, 1, :])).item()
            real_magnitudes.append(real_mag)
            imag_magnitudes.append(imag_mag)
    
    metrics = {
        'measurement_mse': np.mean(measurement_errors),
        'real_magnitude': np.mean(real_magnitudes),
        'imag_magnitude': np.mean(imag_magnitudes),
        'imag_to_real_ratio': np.mean(imag_magnitudes) / (np.mean(real_magnitudes) + 1e-10)
    }
    
    print(f"\n{model_name} - Evaluation Metrics:")
    print(f"  Measurement MSE: {metrics['measurement_mse']:.6e}")
    print(f"  Real magnitude: {metrics['real_magnitude']:.6f}")
    print(f"  Imaginary magnitude: {metrics['imag_magnitude']:.6f}")
    print(f"  Imag/Real ratio: {metrics['imag_to_real_ratio']:.6f}")
    
    return metrics


# -----------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------

def plot_training_curves(losses_real, losses_complex, losses_original, save_path):
    """Plot training loss curves for all three models"""
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(losses_real) + 1)
    
    plt.plot(epochs, losses_real, 'b-', linewidth=2.5, label='Real-Only Output (Deep, 1 channel)', alpha=0.9)
    plt.plot(epochs, losses_complex, 'r-', linewidth=2.5, label='Complex Output (Deep, 2 channels)', alpha=0.9)
    plt.plot(epochs, losses_original, 'g-', linewidth=2.5, label='Original CNNDenoiser (Shallow + Residual, 2 channels)', alpha=0.9)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Measurement MSE Loss', fontsize=12)
    plt.title('Training Loss Comparison: Three Denoiser Architectures', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {save_path}")
    plt.close()


def plot_output_comparison(denoiser_real, denoiser_complex, denoiser_original, dataset, A_tensor, device, save_path):
    """Visualize outputs from all three denoisers on a sample"""
    denoiser_real.eval()
    denoiser_complex.eval()
    denoiser_original.eval()
    
    # Get one sample
    sample_data = dataset[25]
    if isinstance(sample_data, (list, tuple)):
        y_sample = sample_data[0].unsqueeze(0).to(device)  # Get measurements only
    else:
        y_sample = sample_data.unsqueeze(0).to(device)
    A_batch = A_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Input
        x_input = complex_conj_transpose_matmul(A_batch, y_sample)
        
        # Outputs
        x_real = denoiser_real(x_input)
        x_complex = denoiser_complex(x_input)
        x_original = denoiser_original(x_input)
    
    # Convert to numpy
    x_input_np = x_input.squeeze(0).cpu().numpy()
    x_real_np = x_real.squeeze(0).cpu().numpy()
    x_complex_np = x_complex.squeeze(0).cpu().numpy()
    x_original_np = x_original.squeeze(0).cpu().numpy()
    
    # Create figure (4 rows, 3 columns)
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    # Angle bins for x-axis
    N_theta = x_input_np.shape[1]
    theta = np.linspace(-25, 25, N_theta)
    
    # Row 1: Input (A^H * y)
    axes[0, 0].plot(theta, x_input_np[0, :], 'b-', linewidth=1.5)
    axes[0, 0].set_title('Input: Real Part', fontweight='bold')
    axes[0, 0].set_ylabel('Input\n(A^H * y)', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(theta, x_input_np[1, :], 'r-', linewidth=1.5)
    axes[0, 1].set_title('Input: Imaginary Part', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].plot(theta, np.abs(x_input_np[0, :] + 1j*x_input_np[1, :]), 'g-', linewidth=1.5)
    axes[0, 2].set_title('Input: Magnitude', fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Real-Only Output
    axes[1, 0].plot(theta, x_real_np[0, :], 'b-', linewidth=1.5)
    axes[1, 0].set_title('Real-Only: Real Part', fontweight='bold')
    axes[1, 0].set_ylabel('Real-Only\n(Deep, 1ch)', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(theta, x_real_np[1, :], 'r-', linewidth=1.5)
    axes[1, 1].set_title('Real-Only: Imaginary (=0)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    axes[1, 2].plot(theta, np.abs(x_real_np[0, :] + 1j*x_real_np[1, :]), 'g-', linewidth=1.5)
    axes[1, 2].set_title('Real-Only: Magnitude', fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Row 3: Complex Output
    axes[2, 0].plot(theta, x_complex_np[0, :], 'b-', linewidth=1.5)
    axes[2, 0].set_title('Complex: Real Part', fontweight='bold')
    axes[2, 0].set_ylabel('Complex\n(Deep, 2ch)', fontsize=11, fontweight='bold')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(theta, x_complex_np[1, :], 'r-', linewidth=1.5)
    axes[2, 1].set_title('Complex: Imaginary Part', fontweight='bold')
    axes[2, 1].grid(True, alpha=0.3)
    
    axes[2, 2].plot(theta, np.abs(x_complex_np[0, :] + 1j*x_complex_np[1, :]), 'g-', linewidth=1.5)
    axes[2, 2].set_title('Complex: Magnitude', fontweight='bold')
    axes[2, 2].grid(True, alpha=0.3)
    
    # Row 4: Original CNNDenoiser
    axes[3, 0].plot(theta, x_original_np[0, :], 'b-', linewidth=1.5)
    axes[3, 0].set_title('Original: Real Part', fontweight='bold')
    axes[3, 0].set_xlabel('Angle (degrees)')
    axes[3, 0].set_ylabel('Original\n(Shallow+Res, 2ch)', fontsize=11, fontweight='bold')
    axes[3, 0].grid(True, alpha=0.3)
    
    axes[3, 1].plot(theta, x_original_np[1, :], 'r-', linewidth=1.5)
    axes[3, 1].set_title('Original: Imaginary Part', fontweight='bold')
    axes[3, 1].set_xlabel('Angle (degrees)')
    axes[3, 1].grid(True, alpha=0.3)
    
    axes[3, 2].plot(theta, np.abs(x_original_np[0, :] + 1j*x_original_np[1, :]), 'g-', linewidth=1.5)
    axes[3, 2].set_title('Original: Magnitude', fontweight='bold')
    axes[3, 2].set_xlabel('Angle (degrees)')
    axes[3, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Denoiser Output Comparison: Three Architectures', fontsize=16, fontweight='bold', y=0.996)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Output comparison saved to: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_real, metrics_complex, metrics_original, save_path):
    """Create bar chart comparing metrics for all three models"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    models = ['Real-Only\n(Deep, 1ch)', 'Complex\n(Deep, 2ch)', 'Original\n(Shallow+Res)']
    colors = ['blue', 'red', 'green']
    
    # Measurement MSE
    mse_values = [metrics_real['measurement_mse'], 
                  metrics_complex['measurement_mse'],
                  metrics_original['measurement_mse']]
    bars = axes[0].bar(models, mse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Measurement Domain MSE\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(mse_values):
        axes[0].text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Real Magnitude
    real_values = [metrics_real['real_magnitude'], 
                   metrics_complex['real_magnitude'],
                   metrics_original['real_magnitude']]
    axes[1].bar(models, real_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Magnitude', fontsize=12)
    axes[1].set_title('Real Part Magnitude', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(real_values):
        axes[1].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Imaginary Magnitude
    imag_values = [metrics_real['imag_magnitude'], 
                   metrics_complex['imag_magnitude'],
                   metrics_original['imag_magnitude']]
    axes[2].bar(models, imag_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Magnitude', fontsize=12)
    axes[2].set_title('Imaginary Part Magnitude\n(Lower is Better for Real Targets)', fontsize=12, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(imag_values):
        if v > 1e-10:  # Only show if non-zero
            axes[2].text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            axes[2].text(i, 1e-10, f'{v:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Performance Metrics Comparison: Three Architectures', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to: {save_path}")
    plt.close()


# -----------------------------------------------------------------
# Main Comparison
# -----------------------------------------------------------------

def main():
    print("="*80)
    print("COMPARISON: THREE DENOISER ARCHITECTURES")
    print("="*80)
    print()
    print("Experiment Setup:")
    print("  - All denoisers: Input = A^H * y (2 channels: real, imag)")
    print("  - Denoiser 1: Real-Only (Deep encoder-decoder, 1 output channel)")
    print("  - Denoiser 2: Complex (Deep encoder-decoder, 2 output channels)")
    print("  - Denoiser 3: Original (Shallow residual, 2 output channels)")
    print("  - Training: Unsupervised (measurement domain loss)")
    print("="*80)
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load dataset
    print("\n[1] Loading dataset...")
    dataset = MIMOSAR_Dataset(MAT_FILE, return_ground_truth=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=2, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    A_tensor = dataset.A
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    print(f"  ✓ Steering matrix A: {list(A_tensor.shape)}")
    
    # Initialize models
    print("\n[2] Initializing models...")
    denoiser_real = CNNDenoiser_RealOutput(in_channels=2, out_channels=1, enforce_positivity=False).to(DEVICE)
    denoiser_complex = CNNDenoiser_ComplexOutput(in_channels=2, out_channels=2).to(DEVICE)
    denoiser_original = CNNDenoiser(in_channels=2, out_channels=2).to(DEVICE)
    
    print(f"  ✓ Real-Only Denoiser: {sum(p.numel() for p in denoiser_real.parameters()):,} parameters")
    print(f"  ✓ Complex Denoiser: {sum(p.numel() for p in denoiser_complex.parameters()):,} parameters")
    print(f"  ✓ Original Denoiser: {sum(p.numel() for p in denoiser_original.parameters()):,} parameters")
    
    # Train Real-Only Denoiser
    print("\n[3] Training Real-Only Output Denoiser...")
    losses_real = train_denoiser(denoiser_real, dataloader, A_tensor, 
                                 NUM_EPOCHS, LEARNING_RATE, DEVICE, 
                                 "Real-Only Denoiser")
    torch.save(denoiser_real.state_dict(), CHECKPOINT_REAL)
    print(f"  ✓ Model saved: {CHECKPOINT_REAL}")
    
    # Train Complex Denoiser
    print("\n[4] Training Complex Output Denoiser...")
    losses_complex = train_denoiser(denoiser_complex, dataloader, A_tensor, 
                                   NUM_EPOCHS, LEARNING_RATE, DEVICE, 
                                   "Complex Denoiser")
    torch.save(denoiser_complex.state_dict(), CHECKPOINT_COMPLEX)
    print(f"  ✓ Model saved: {CHECKPOINT_COMPLEX}")
    
    # Train Original Denoiser
    print("\n[5] Training Original CNNDenoiser...")
    losses_original = train_denoiser(denoiser_original, dataloader, A_tensor, 
                                    NUM_EPOCHS, LEARNING_RATE, DEVICE, 
                                    "Original CNNDenoiser")
    torch.save(denoiser_original.state_dict(), CHECKPOINT_ORIGINAL)
    print(f"  ✓ Model saved: {CHECKPOINT_ORIGINAL}")
    
    # Evaluate all models
    print("\n[6] Evaluating models...")
    metrics_real = evaluate_denoiser(denoiser_real, dataset, A_tensor, DEVICE, "Real-Only Denoiser")
    metrics_complex = evaluate_denoiser(denoiser_complex, dataset, A_tensor, DEVICE, "Complex Denoiser")
    metrics_original = evaluate_denoiser(denoiser_original, dataset, A_tensor, DEVICE, "Original CNNDenoiser")
    
    # Generate visualizations
    print("\n[7] Generating visualizations...")
    plot_training_curves(losses_real, losses_complex, losses_original,
                        os.path.join(RESULTS_DIR, 'training_curves.png'))
    
    plot_output_comparison(denoiser_real, denoiser_complex, denoiser_original, dataset, A_tensor, DEVICE,
                          os.path.join(RESULTS_DIR, 'output_comparison.png'))
    
    #plot_metrics_comparison(metrics_real, metrics_complex, metrics_original,
    #                       os.path.join(RESULTS_DIR, 'metrics_comparison.png'))
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print()
    print("1️⃣  Real-Only Output Denoiser (Deep, 1 channel):")
    print(f"    • Final training loss: {losses_real[-1]:.6e}")
    print(f"    • Measurement MSE: {metrics_real['measurement_mse']:.6e}")
    print(f"    • Imaginary magnitude: {metrics_real['imag_magnitude']:.6e} (should be ~0)")
    print(f"    • Parameters: {sum(p.numel() for p in denoiser_real.parameters()):,}")
    print()
    print("2️⃣  Complex Output Denoiser (Deep, 2 channels):")
    print(f"    • Final training loss: {losses_complex[-1]:.6e}")
    print(f"    • Measurement MSE: {metrics_complex['measurement_mse']:.6e}")
    print(f"    • Imaginary magnitude: {metrics_complex['imag_magnitude']:.6e}")
    print(f"    • Imag/Real ratio: {metrics_complex['imag_to_real_ratio']:.4f}")
    print(f"    • Parameters: {sum(p.numel() for p in denoiser_complex.parameters()):,}")
    print()
    print("3️⃣  Original CNNDenoiser (Shallow + Residual, 2 channels):")
    print(f"    • Final training loss: {losses_original[-1]:.6e}")
    print(f"    • Measurement MSE: {metrics_original['measurement_mse']:.6e}")
    print(f"    • Imaginary magnitude: {metrics_original['imag_magnitude']:.6e}")
    print(f"    • Imag/Real ratio: {metrics_original['imag_to_real_ratio']:.4f}")
    print(f"    • Parameters: {sum(p.numel() for p in denoiser_original.parameters()):,}")
    print()
    
    # Determine winner
    mse_values = {
        'Real-Only': metrics_real['measurement_mse'],
        'Complex': metrics_complex['measurement_mse'],
        'Original': metrics_original['measurement_mse']
    }
    winner = min(mse_values, key=mse_values.get)
    
    print("🏆 Winner (Lowest Measurement MSE):")
    print(f"    {winner} Denoiser with MSE = {mse_values[winner]:.6e}")
    print()
    
    print("📊 Key Observations:")
    print("    • Real-Only: Forces imaginary=0 at architecture level")
    print("    • Complex: Allows network to learn imaginary component")
    print("    • Original: Uses residual connection (output = input + CNN(input))")
    print("    • Original: Much fewer parameters (shallow architecture)")
    print()
    print("="*80)
    print(f"✅ COMPARISON COMPLETE! Results saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()

