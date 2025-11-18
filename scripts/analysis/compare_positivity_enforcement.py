"""
Compare Impact of Positivity Enforcement on CNNDenoiser_RealOutput

This script trains two versions of CNNDenoiser_RealOutput and compares their performance:
1. With Positivity Enforcement: ReLU at final layer (enforce_positivity=True)
2. Without Positivity Enforcement: No ReLU (enforce_positivity=False)

Both use the same architecture (deep encoder-decoder, 1 output channel for real-only).
All trained unsupervised using measurement domain loss.
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
from core.models import CNNDenoiser_RealOutput

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
MAT_FILE = 'data/FL_MIMO_SAR_data.mat'

# Training Hyperparameters
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Output
CHECKPOINT_WITH_RELU = 'checkpoints/denoiser_with_positivity.pth'
CHECKPOINT_NO_RELU = 'checkpoints/denoiser_no_positivity.pth'
RESULTS_DIR = 'results/comparison_positivity/'

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------
# Training Function
# -----------------------------------------------------------------

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
        all_outputs: All denoised outputs for visualization
    """
    denoiser.eval()
    A_batch_tensor = A_tensor.unsqueeze(0).to(device)
    
    dataloader_eval = DataLoader(dataset, batch_size=64, shuffle=False)
    
    measurement_errors = []
    all_outputs = []
    
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
            
            # Store outputs (real channel only)
            all_outputs.append(x_denoised[:, 0, :].cpu())
    
    # Concatenate all outputs
    all_outputs = torch.cat(all_outputs, dim=0)  # [N_samples, N_theta]
    
    # Compute statistics
    min_val = all_outputs.min().item()
    max_val = all_outputs.max().item()
    mean_val = all_outputs.mean().item()
    std_val = all_outputs.std().item()
    
    # Count negative values
    num_negative = (all_outputs < 0).sum().item()
    total_values = all_outputs.numel()
    percent_negative = 100.0 * num_negative / total_values
    
    metrics = {
        'measurement_mse': np.mean(measurement_errors),
        'min_value': min_val,
        'max_value': max_val,
        'mean_value': mean_val,
        'std_value': std_val,
        'num_negative': num_negative,
        'percent_negative': percent_negative,
    }
    
    print(f"\n{model_name} - Evaluation Metrics:")
    print(f"  Measurement MSE: {metrics['measurement_mse']:.6e}")
    print(f"  Value range: [{metrics['min_value']:.6f}, {metrics['max_value']:.6f}]")
    print(f"  Mean ± Std: {metrics['mean_value']:.6f} ± {metrics['std_value']:.6f}")
    print(f"  Negative values: {metrics['num_negative']:,} / {total_values:,} ({metrics['percent_negative']:.2f}%)")
    
    return metrics, all_outputs


# -----------------------------------------------------------------
# Visualization Functions
# -----------------------------------------------------------------

def plot_training_curves(losses_with_relu, losses_no_relu, save_path):
    """Plot training loss curves for both models"""
    plt.figure(figsize=(12, 6))
    
    epochs = range(1, len(losses_with_relu) + 1)
    
    plt.plot(epochs, losses_with_relu, 'b-', linewidth=2.5, 
             label='With Positivity (ReLU)', alpha=0.9)
    plt.plot(epochs, losses_no_relu, 'r-', linewidth=2.5, 
             label='Without Positivity (No ReLU)', alpha=0.9)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Measurement MSE Loss', fontsize=12)
    plt.title('Training Loss Comparison: Positivity Enforcement Impact', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {save_path}")
    plt.close()


def plot_output_histograms(outputs_with_relu, outputs_no_relu, save_path):
    """
    Plot histograms of output values to visualize positivity enforcement
    
    This is the KEY visualization showing the impact of ReLU!
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Flatten outputs for histogram
    vals_with_relu = outputs_with_relu.flatten().numpy()
    vals_no_relu = outputs_no_relu.flatten().numpy()
    
    # Determine common bins for fair comparison
    all_vals = np.concatenate([vals_with_relu, vals_no_relu])
    bins = np.linspace(all_vals.min(), all_vals.max(), 50)
    
    # Plot 1: With ReLU (should be all >= 0)
    ax = axes[0]
    ax.hist(vals_with_relu, bins=bins, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('Output Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('WITH Positivity Enforcement (ReLU)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate statistics
    min_val = vals_with_relu.min()
    max_val = vals_with_relu.max()
    num_neg = (vals_with_relu < 0).sum()
    ax.text(0.95, 0.95, f'Min: {min_val:.4f}\nMax: {max_val:.4f}\nNegative: {num_neg}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Without ReLU (may have negatives)
    ax = axes[1]
    ax.hist(vals_no_relu, bins=bins, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='blue', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('Output Value', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('WITHOUT Positivity Enforcement (No ReLU)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Annotate statistics
    min_val = vals_no_relu.min()
    max_val = vals_no_relu.max()
    num_neg = (vals_no_relu < 0).sum()
    percent_neg = 100.0 * num_neg / len(vals_no_relu)
    ax.text(0.95, 0.95, f'Min: {min_val:.4f}\nMax: {max_val:.4f}\nNegative: {num_neg}\n({percent_neg:.2f}%)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Output histograms saved to: {save_path}")
    plt.close()


def plot_sample_comparison(denoiser_with_relu, denoiser_no_relu, dataset, A_tensor, device, save_path):
    """
    Plot sample reconstructions comparing both models
    """
    denoiser_with_relu.eval()
    denoiser_no_relu.eval()
    
    A_batch_tensor = A_tensor.unsqueeze(0).to(device)
    
    # Get a few samples
    indices = [0, 10, 20, 30]
    num_samples = len(indices)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get measurement
            y_sample = dataset[idx]
            if isinstance(y_sample, (list, tuple)):
                y_sample = y_sample[0]
            y_sample = y_sample.unsqueeze(0).to(device)
            
            # Input
            x_input = complex_conj_transpose_matmul(A_batch_tensor, y_sample)
            
            # Reconstruct with both models
            x_with_relu = denoiser_with_relu(x_input)
            x_no_relu = denoiser_no_relu(x_input)
            
            # Extract real channel
            vals_with_relu = x_with_relu[0, 0, :].cpu().numpy()
            vals_no_relu = x_no_relu[0, 0, :].cpu().numpy()
            
            # Plot
            ax1 = axes[i, 0] if num_samples > 1 else axes[0]
            ax2 = axes[i, 1] if num_samples > 1 else axes[1]
            
            x_angles = np.linspace(-25, 25, len(vals_with_relu))
            ax1.plot(x_angles, vals_with_relu, 'b-', linewidth=1.5)
            ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax1.set_title(f'Sample {idx} - WITH ReLU', fontsize=10, fontweight='bold')
            ax1.set_ylabel('Reflectivity', fontsize=9)
            ax1.grid(True, alpha=0.3)
            ax1.text(0.02, 0.98, f'Min: {vals_with_relu.min():.4f}',
                    transform=ax1.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            
            x_angles = np.linspace(-25, 25, len(vals_no_relu))
            ax2.plot(x_angles, vals_no_relu, 'r-', linewidth=1.5)
            ax2.axhline(y=0, color='blue', linestyle='--', linewidth=1, alpha=0.5)
            ax2.set_title(f'Sample {idx} - WITHOUT ReLU', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Reflectivity', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # Highlight if negatives exist
            num_neg = (vals_no_relu < 0).sum()
            if num_neg > 0:
                ax2.text(0.02, 0.98, f'Min: {vals_no_relu.min():.4f}\n⚠️ {num_neg} negative values',
                        transform=ax2.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
            else:
                ax2.text(0.02, 0.98, f'Min: {vals_no_relu.min():.4f}',
                        transform=ax2.transAxes, fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
            
            if i == num_samples - 1:
                ax1.set_xlabel('Angle Index', fontsize=9)
                ax2.set_xlabel('Angle Index', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample comparison saved to: {save_path}")
    plt.close()


def plot_metrics_comparison(metrics_with_relu, metrics_no_relu, save_path):
    """
    Create bar chart comparing key metrics
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    labels = ['With ReLU', 'No ReLU']
    colors = ['blue', 'red']
    
    # Plot 1: Measurement MSE
    ax = axes[0]
    mse_values = [metrics_with_relu['measurement_mse'], metrics_no_relu['measurement_mse']]
    bars = ax.bar(labels, mse_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('MSE', fontsize=11)
    ax.set_title('Measurement MSE\n(Lower is Better)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate values
    for bar, val in zip(bars, mse_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Value Range
    ax = axes[1]
    min_values = [metrics_with_relu['min_value'], metrics_no_relu['min_value']]
    max_values = [metrics_with_relu['max_value'], metrics_no_relu['max_value']]
    
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, min_values, width, label='Min', color='lightblue', 
                   edgecolor='black', alpha=0.7)
    bars2 = ax.bar(x + width/2, max_values, width, label='Max', color='lightcoral', 
                   edgecolor='black', alpha=0.7)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Zero')
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Output Value Range', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Negative Values Percentage
    ax = axes[2]
    neg_percentages = [metrics_with_relu['percent_negative'], metrics_no_relu['percent_negative']]
    bars = ax.bar(labels, neg_percentages, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Negative Values\n(Should be 0% with ReLU)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotate values
    for bar, val in zip(bars, neg_percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to: {save_path}")
    plt.close()


# -----------------------------------------------------------------
# Main Comparison Script
# -----------------------------------------------------------------

def main():
    print("\n" + "="*80)
    print("POSITIVITY ENFORCEMENT COMPARISON")
    print("="*80)
    print("\nComparing CNNDenoiser_RealOutput with and without ReLU at final layer")
    print("Both use identical architecture (deep encoder-decoder, 1 output channel)")
    print("Training: Unsupervised (measurement domain loss)")
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
    denoiser_with_relu = CNNDenoiser_RealOutput(
        in_channels=2, 
        out_channels=1, 
        enforce_positivity=True
    ).to(DEVICE)
    
    denoiser_no_relu = CNNDenoiser_RealOutput(
        in_channels=2, 
        out_channels=1, 
        enforce_positivity=False
    ).to(DEVICE)
    
    params_with = sum(p.numel() for p in denoiser_with_relu.parameters())
    params_no = sum(p.numel() for p in denoiser_no_relu.parameters())
    
    print(f"  ✓ With ReLU: {params_with:,} parameters")
    print(f"  ✓ Without ReLU: {params_no:,} parameters")
    print(f"  ✓ Same architecture: {params_with == params_no}")
    
    # Train With ReLU
    print("\n[3] Training WITH Positivity Enforcement (ReLU)...")
    losses_with_relu = train_denoiser(denoiser_with_relu, dataloader, A_tensor, 
                                      NUM_EPOCHS, LEARNING_RATE, DEVICE, 
                                      "Denoiser WITH ReLU")
    torch.save(denoiser_with_relu.state_dict(), CHECKPOINT_WITH_RELU)
    print(f"  ✓ Model saved: {CHECKPOINT_WITH_RELU}")
    
    # Train Without ReLU
    print("\n[4] Training WITHOUT Positivity Enforcement (No ReLU)...")
    losses_no_relu = train_denoiser(denoiser_no_relu, dataloader, A_tensor, 
                                    NUM_EPOCHS, LEARNING_RATE, DEVICE, 
                                    "Denoiser WITHOUT ReLU")
    torch.save(denoiser_no_relu.state_dict(), CHECKPOINT_NO_RELU)
    print(f"  ✓ Model saved: {CHECKPOINT_NO_RELU}")
    
    # Evaluate all models
    print("\n[5] Evaluating models...")
    metrics_with_relu, outputs_with_relu = evaluate_denoiser(
        denoiser_with_relu, dataset, A_tensor, DEVICE, "WITH ReLU"
    )
    metrics_no_relu, outputs_no_relu = evaluate_denoiser(
        denoiser_no_relu, dataset, A_tensor, DEVICE, "WITHOUT ReLU"
    )
    
    # Generate visualizations
    print("\n[6] Generating visualizations...")
    plot_training_curves(losses_with_relu, losses_no_relu,
                        os.path.join(RESULTS_DIR, 'training_curves.png'))
    
    plot_output_histograms(outputs_with_relu, outputs_no_relu,
                          os.path.join(RESULTS_DIR, 'output_histograms.png'))
    
    plot_sample_comparison(denoiser_with_relu, denoiser_no_relu, dataset, A_tensor, DEVICE,
                          os.path.join(RESULTS_DIR, 'sample_comparison.png'))
    
    plot_metrics_comparison(metrics_with_relu, metrics_no_relu,
                           os.path.join(RESULTS_DIR, 'metrics_comparison.png'))
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY: POSITIVITY ENFORCEMENT")
    print("="*80)
    print()
    print("1️⃣  WITH POSITIVITY (ReLU at final layer):")
    print(f"    • Final training loss: {losses_with_relu[-1]:.6e}")
    print(f"    • Measurement MSE: {metrics_with_relu['measurement_mse']:.6e}")
    print(f"    • Value range: [{metrics_with_relu['min_value']:.6f}, {metrics_with_relu['max_value']:.6f}]")
    print(f"    • Mean ± Std: {metrics_with_relu['mean_value']:.6f} ± {metrics_with_relu['std_value']:.6f}")
    print(f"    • Negative values: {metrics_with_relu['num_negative']:,} ({metrics_with_relu['percent_negative']:.2f}%) ✓")
    print(f"    • Parameters: {params_with:,}")
    print()
    print("2️⃣  WITHOUT POSITIVITY (No ReLU):")
    print(f"    • Final training loss: {losses_no_relu[-1]:.6e}")
    print(f"    • Measurement MSE: {metrics_no_relu['measurement_mse']:.6e}")
    print(f"    • Value range: [{metrics_no_relu['min_value']:.6f}, {metrics_no_relu['max_value']:.6f}]")
    print(f"    • Mean ± Std: {metrics_no_relu['mean_value']:.6f} ± {metrics_no_relu['std_value']:.6f}")
    
    # Highlight negatives if they exist
    if metrics_no_relu['percent_negative'] > 0:
        print(f"    • Negative values: {metrics_no_relu['num_negative']:,} ({metrics_no_relu['percent_negative']:.2f}%) ⚠️")
    else:
        print(f"    • Negative values: {metrics_no_relu['num_negative']:,} ({metrics_no_relu['percent_negative']:.2f}%)")
    print(f"    • Parameters: {params_no:,}")
    print()
    
    # Determine winner
    mse_with = metrics_with_relu['measurement_mse']
    mse_no = metrics_no_relu['measurement_mse']
    
    if mse_with < mse_no:
        winner = "WITH ReLU"
        improvement = 100.0 * (mse_no - mse_with) / mse_no
        print(f"🏆 Winner (Lowest Measurement MSE): WITH ReLU")
        print(f"    MSE = {mse_with:.6e} ({improvement:.2f}% better)")
    elif mse_no < mse_with:
        winner = "WITHOUT ReLU"
        improvement = 100.0 * (mse_with - mse_no) / mse_with
        print(f"🏆 Winner (Lowest Measurement MSE): WITHOUT ReLU")
        print(f"    MSE = {mse_no:.6e} ({improvement:.2f}% better)")
    else:
        print(f"🏆 Result: TIED (identical MSE)")
        print(f"    MSE = {mse_with:.6e}")
    print()
    
    print("📊 Key Observations:")
    print("    • ReLU enforces positivity at ARCHITECTURE level (hard constraint)")
    print(f"    • ReLU version: Min = {metrics_with_relu['min_value']:.6f} (guaranteed ≥ 0)")
    print(f"    • No-ReLU version: Min = {metrics_no_relu['min_value']:.6f}")
    
    if metrics_no_relu['percent_negative'] > 0:
        print(f"    • No-ReLU produces {metrics_no_relu['percent_negative']:.2f}% negative values")
        print("    • Negative values violate physical prior (reflectivity ≥ 0)")
    else:
        print("    • No-ReLU version learned to avoid negatives (interesting!)")
    
    if mse_with < mse_no:
        print("    • Positivity constraint IMPROVES reconstruction quality")
    elif mse_no < mse_with:
        print("    • Unconstrained network has slightly lower MSE")
        print("    • But produces physically invalid outputs (negatives)")
    else:
        print("    • Both achieve similar reconstruction quality")
    
    print()
    print("💡 Recommendation:")
    if mse_with <= mse_no * 1.01:  # Within 1% or better
        print("    ✓ USE enforce_positivity=True (ReLU)")
        print("    ✓ Enforces physical constraint without sacrificing performance")
        print("    ✓ Guarantees valid outputs for downstream applications")
    else:
        print("    ⚠️  Trade-off exists: positivity vs. reconstruction error")
        print("    ⚠️  Consider application requirements")
    
    print()
    print("="*80)
    print(f"✅ COMPARISON COMPLETE! Results saved to: {RESULTS_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()

