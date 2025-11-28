"""
2D SAR Image Reconstruction Inference Script

This script loads a trained model and reconstructs a full 2D SAR image from
measurements at multiple range bins.

Usage:
    python scripts/inference/inference_2d_image.py

Configuration:
    - DATA_FILE: Path to .mat file containing 'A' and 'received_signals_fft'
    - MODEL_PATH: Path to trained model checkpoint
    - OUTPUT_DIR: Directory to save results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from core.models import DBPNet
from core.data_loader import load_mat_file, to_tensor
from core.inference_utils import reconstruct_2d_image_with_input, polar_to_cartesian

# -----------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------
DATA_FILE = 'data/data.mat'  # Input data file
MODEL_PATH = 'checkpoints/dbp_model.pth'  # Trained model checkpoint
OUTPUT_DIR = 'results/inference_2d/'  # Output directory

# Model configuration (must match training configuration)
NUM_UNROLLS = 5
NUM_ADMM_STEPS = 2
DENOISER_TYPE = 'real'
ENFORCE_POSITIVITY = True
ADMM_ENFORCE_REAL = True
ADMM_ENFORCE_POSITIVITY = True

# Inference configuration
BATCH_SIZE = 64  # Number of range bins to process at once
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONVERT_TO_CARTESIAN = True  # If True, also convert to Cartesian coordinates

# Visualization configuration
COLORMAP = 'hot'  # 'hot', 'jet', 'viridis', 'gray'
DPI = 150
FIGSIZE = (16, 6)


def main():
    print("="*80)
    print("2D SAR IMAGE RECONSTRUCTION - INFERENCE")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Data file: {DATA_FILE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -----------------------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------------------
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: Data file not found: {DATA_FILE}")
        print("Please ensure the data file exists and contains:")
        print("  - 'A': Steering matrix [N_v, N_theta]")
        print("  - 'received_signals_fft': Measurements [N_ranges, N_v]")
        return
    
    data = load_mat_file(DATA_FILE)
    
    # Extract steering matrix A
    if 'A' not in data:
        print("ERROR: 'A' (steering matrix) not found in data file!")
        return
    
    A_complex = data['A'].astype(np.complex64)
    A_tensor = to_tensor(A_complex)  # Shape: [2, N_v, N_theta]
    
    # Extract measurements
    if 'received_signals_fft' not in data:
        print("ERROR: 'received_signals_fft' not found in data file!")
        return
    
    measurements = data['received_signals_fft'].astype(np.complex64)
    
    N_ranges = measurements.shape[0]
    N_v = measurements.shape[1]
    N_theta = A_tensor.shape[2]
    
    print(f"Data loaded successfully:")
    print(f"  Measurements shape: {measurements.shape} [N_ranges, N_v]")
    print(f"  N_ranges (range bins): {N_ranges}")
    print(f"  N_v (virtual antennas): {N_v}")
    print(f"  N_theta (angle bins): {N_theta}")
    print(f"  Steering matrix shape: {A_tensor.shape} [2, N_v, N_theta]\n")
    
    # -----------------------------------------------------------------
    # 2. Load Trained Model
    # -----------------------------------------------------------------
    print("Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model checkpoint not found: {MODEL_PATH}")
        print("Please train the model first using: python scripts/training/train_configurable.py")
        return
    
    # Initialize model with same configuration as training
    model = DBPNet(
        A_tensor,
        num_iterations=NUM_UNROLLS,
        N_admm_steps=NUM_ADMM_STEPS,
        denoiser_type=DENOISER_TYPE,
        enforce_positivity=ENFORCE_POSITIVITY,
        admm_enforce_real=ADMM_ENFORCE_REAL,
        admm_enforce_positivity=ADMM_ENFORCE_POSITIVITY
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    print("Model loaded successfully!")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"  Unrolled iterations: {NUM_UNROLLS}")
    print(f"  ADMM steps per iteration: {NUM_ADMM_STEPS}\n")
    
    # -----------------------------------------------------------------
    # 3. Reconstruct 2D Image
    # -----------------------------------------------------------------
    print("="*80)
    print("RECONSTRUCTING 2D SAR IMAGE")
    print("="*80)
    print(f"Processing {N_ranges} range bins in batches of {BATCH_SIZE}...\n")
    
    # Use the reusable function
    reconstructed_image, measurement_magnitude = reconstruct_2d_image_with_input(
        model=model,
        measurements=measurements,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        verbose=True
    )
    
    print(f"\nReconstruction complete!")
    print(f"  Input (measurement domain): {measurement_magnitude.shape} [N_ranges, N_v]")
    print(f"  Output (angle-range image): {reconstructed_image.shape} [N_ranges, N_theta]")
    print(f"  Output range: [{reconstructed_image.min():.6f}, {reconstructed_image.max():.6f}]\n")
    
    # -----------------------------------------------------------------
    # 4. Convert to Cartesian Coordinates (Optional)
    # -----------------------------------------------------------------
    cartesian_image = None
    if CONVERT_TO_CARTESIAN:
        print("="*80)
        print("CONVERTING TO CARTESIAN COORDINATES")
        print("="*80)
        
        # Check if required data is available
        required_keys = ['ranges', 'angles', 'x_grid', 'y_grid', 'x_radar', 'y_radar']
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print(f"WARNING: Cannot convert to Cartesian. Missing data: {missing_keys}")
            print("Skipping Cartesian conversion...\n")
        else:
            try:
                ranges = data['ranges'].ravel()  # Ensure 1D
                angles = data['angles'].ravel()  # Ensure 1D
                x_grid = data['x_grid']
                y_grid = data['y_grid']
                x_radar = float(data['x_radar'])
                y_radar = float(data['y_radar'])
                
                print(f"Grid parameters:")
                print(f"  Ranges: {len(ranges)} bins from {ranges.min():.2f}m to {ranges.max():.2f}m")
                print(f"  Angles: {len(angles)} bins from {angles.min():.2f}° to {angles.max():.2f}°")
                print(f"  Cartesian grid: {x_grid.shape}")
                print(f"  X extent: [{x_grid.min():.2f}, {x_grid.max():.2f}] m")
                print(f"  Y extent: [{y_grid.min():.2f}, {y_grid.max():.2f}] m")
                print(f"  Radar position: ({x_radar:.2f}, {y_radar:.2f}) m\n")
                
                print("Converting polar to Cartesian (using cubic interpolation)...")
                cartesian_image = polar_to_cartesian(
                    reconstructed_image, ranges, angles,
                    x_grid, y_grid, x_radar, y_radar,
                    method='cubic'
                )
                
                print(f"Cartesian conversion complete!")
                print(f"  Cartesian image shape: {cartesian_image.shape}")
                print(f"  Value range: [{cartesian_image.min():.6f}, {cartesian_image.max():.6f}]\n")
                
            except Exception as e:
                print(f"ERROR during Cartesian conversion: {e}")
                print("Continuing without Cartesian image...\n")
                cartesian_image = None
    
    # -----------------------------------------------------------------
    # 5. Save Results
    # -----------------------------------------------------------------
    print("Saving results...")
    
    # Save as .mat file (MATLAB compatible)
    results_mat = {
        'reconstructed_image': reconstructed_image,
        'measurement_magnitude': measurement_magnitude,
        'N_ranges': N_ranges,
        'N_theta': N_theta,
        'N_v': N_v
    }
    if cartesian_image is not None:
        results_mat['cartesian_image'] = cartesian_image
    
    mat_file_path = os.path.join(OUTPUT_DIR, 'reconstructed_image.mat')
    scipy.io.savemat(mat_file_path, results_mat)
    print(f"  Results saved to: {mat_file_path}")
    
    # Save as numpy file (Python)
    npz_data = {
        'reconstructed_image': reconstructed_image,
        'measurement_magnitude': measurement_magnitude
    }
    if cartesian_image is not None:
        npz_data['cartesian_image'] = cartesian_image
    
    npz_file_path = os.path.join(OUTPUT_DIR, 'reconstructed_image.npz')
    np.savez(npz_file_path, **npz_data)
    print(f"  Results saved to: {npz_file_path}")
    
    # -----------------------------------------------------------------
    # 5. Visualization
    # -----------------------------------------------------------------
    print("\nGenerating visualizations...")
    
    # Create angle axis for labels
    angle_axis = np.linspace(-25, 25, N_theta)
    
    # --- Plot 1: Side-by-side comparison ---
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    
    # Left: Input measurements (magnitude)
    im1 = axes[0].imshow(measurement_magnitude, aspect='auto', cmap=COLORMAP, 
                         interpolation='nearest', origin='upper')
    axes[0].set_xlabel('Virtual Antenna Index', fontsize=12)
    axes[0].set_ylabel('Range Bin', fontsize=12)
    axes[0].set_title('Input: Measurement Domain\n|received_signals_fft|', 
                     fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Magnitude', fontsize=11)
    
    # Right: Reconstructed image
    im2 = axes[1].imshow(reconstructed_image, aspect='auto', cmap=COLORMAP,
                         interpolation='nearest', origin='upper',
                         extent=[angle_axis[0], angle_axis[-1], N_ranges-1, 0])
    axes[1].set_xlabel('Angle (degrees)', fontsize=12)
    axes[1].set_ylabel('Range Bin', fontsize=12)
    axes[1].set_title('Output: Reconstructed Angle-Range Image\n(Network Reconstruction)', 
                     fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Reflectivity', fontsize=11)
    
    plt.tight_layout()
    comparison_path = os.path.join(OUTPUT_DIR, 'inference_2d_comparison.png')
    plt.savefig(comparison_path, dpi=DPI, bbox_inches='tight')
    print(f"  Comparison plot saved to: {comparison_path}")
    plt.close()
    
    # --- Plot 2: Reconstructed image only (large) ---
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(reconstructed_image, aspect='auto', cmap=COLORMAP,
                   interpolation='nearest', origin='upper',
                   extent=[angle_axis[0], angle_axis[-1], N_ranges-1, 0])
    ax.set_xlabel('Angle (degrees)', fontsize=13)
    ax.set_ylabel('Range Bin', fontsize=13)
    ax.set_title('Reconstructed 2D SAR Image (Angle-Range)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Reflectivity', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    recon_path = os.path.join(OUTPUT_DIR, 'reconstructed_image_full.png')
    plt.savefig(recon_path, dpi=DPI, bbox_inches='tight')
    print(f"  Full reconstruction plot saved to: {recon_path}")
    plt.close()
    
    # --- Plot 3: Log scale visualization ---
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    
    # Left: Input (log scale)
    measurement_log = 20 * np.log10(measurement_magnitude + 1e-10)
    im1 = axes[0].imshow(measurement_log, aspect='auto', cmap=COLORMAP,
                         interpolation='nearest', origin='upper')
    axes[0].set_xlabel('Virtual Antenna Index', fontsize=12)
    axes[0].set_ylabel('Range Bin', fontsize=12)
    axes[0].set_title('Input (dB scale)', fontsize=13, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label('Magnitude (dB)', fontsize=11)
    
    # Right: Output (log scale)
    recon_log = 20 * np.log10(np.abs(reconstructed_image) + 1e-10)
    im2 = axes[1].imshow(recon_log, aspect='auto', cmap=COLORMAP,
                         interpolation='nearest', origin='upper',
                         extent=[angle_axis[0], angle_axis[-1], N_ranges-1, 0])
    axes[1].set_xlabel('Angle (degrees)', fontsize=12)
    axes[1].set_ylabel('Range Bin', fontsize=12)
    axes[1].set_title('Output (dB scale)', fontsize=13, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label('Reflectivity (dB)', fontsize=11)
    
    plt.tight_layout()
    log_path = os.path.join(OUTPUT_DIR, 'inference_2d_comparison_dB.png')
    plt.savefig(log_path, dpi=DPI, bbox_inches='tight')
    print(f"  Log-scale comparison saved to: {log_path}")
    plt.close()
    
    # --- Plot 4: Cartesian image (if available) ---
    if cartesian_image is not None:
        print("  Generating Cartesian visualization...")
        
        # Large Cartesian plot
        fig, ax = plt.subplots(figsize=(12, 12))
        im = ax.imshow(cartesian_image, 
                      extent=[x_grid.min(), x_grid.max(), y_grid.max(), y_grid.min()],
                      cmap=COLORMAP, interpolation='nearest', origin='upper')
        
        # Plot radar position
        ax.plot(x_radar, y_radar, 'r*', markersize=20, 
               markeredgecolor='white', markeredgewidth=2, label='Radar Position')
        
        ax.set_xlabel('X (meters)', fontsize=13)
        ax.set_ylabel('Y (meters)', fontsize=13)
        ax.set_title('SAR Image in Cartesian Coordinates', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Reflectivity', fontsize=12)
        
        plt.tight_layout()
        cartesian_path = os.path.join(OUTPUT_DIR, 'cartesian_image.png')
        plt.savefig(cartesian_path, dpi=DPI, bbox_inches='tight')
        print(f"  Cartesian plot saved to: {cartesian_path}")
        plt.close()
        
        # Side-by-side: Polar vs Cartesian
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left: Polar
        angle_axis = np.linspace(angles.min(), angles.max(), N_theta) if 'angles' in locals() else np.linspace(-25, 25, N_theta)
        im1 = axes[0].imshow(reconstructed_image, aspect='auto', cmap=COLORMAP,
                           interpolation='nearest', origin='upper',
                           extent=[angle_axis[0], angle_axis[-1], N_ranges-1, 0])
        axes[0].set_xlabel('Angle (degrees)', fontsize=12)
        axes[0].set_ylabel('Range Bin', fontsize=12)
        axes[0].set_title('Polar Domain (Range-Angle)', fontsize=13, fontweight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Reflectivity', fontsize=11)
        
        # Right: Cartesian
        im2 = axes[1].imshow(cartesian_image,
                           extent=[x_grid.min(), x_grid.max(), y_grid.max(), y_grid.min()],
                           cmap=COLORMAP, interpolation='nearest', origin='upper')
        axes[1].plot(x_radar, y_radar, 'r*', markersize=15, 
                    markeredgecolor='white', markeredgewidth=1.5, label='Radar')
        axes[1].set_xlabel('X (meters)', fontsize=12)
        axes[1].set_ylabel('Y (meters)', fontsize=12)
        axes[1].set_title('Cartesian Domain (X-Y)', fontsize=13, fontweight='bold')
        axes[1].set_aspect('equal')
        axes[1].legend(fontsize=10)
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Reflectivity', fontsize=11)
        
        plt.tight_layout()
        comparison_cart_path = os.path.join(OUTPUT_DIR, 'polar_vs_cartesian.png')
        plt.savefig(comparison_cart_path, dpi=DPI, bbox_inches='tight')
        print(f"  Polar vs Cartesian comparison saved to: {comparison_cart_path}")
        plt.close()
    
    # -----------------------------------------------------------------
    # 6. Summary Statistics
    # -----------------------------------------------------------------
    print("\n" + "="*80)
    print("RECONSTRUCTION STATISTICS")
    print("="*80)
    print(f"Input Measurements (|y|):")
    print(f"  Shape: {measurement_magnitude.shape}")
    print(f"  Min:   {measurement_magnitude.min():.6f}")
    print(f"  Max:   {measurement_magnitude.max():.6f}")
    print(f"  Mean:  {measurement_magnitude.mean():.6f}")
    print(f"  Std:   {measurement_magnitude.std():.6f}")
    
    print(f"\nReconstructed Image (x):")
    print(f"  Shape: {reconstructed_image.shape}")
    print(f"  Min:   {reconstructed_image.min():.6f}")
    print(f"  Max:   {reconstructed_image.max():.6f}")
    print(f"  Mean:  {reconstructed_image.mean():.6f}")
    print(f"  Std:   {reconstructed_image.std():.6f}")
    
    # Sparsity analysis
    threshold = 0.1 * reconstructed_image.max()
    nonzero_ratio = np.sum(reconstructed_image > threshold) / reconstructed_image.size
    print(f"\nSparsity Analysis:")
    print(f"  Pixels above {threshold:.4f} (10% of max): {nonzero_ratio*100:.2f}%")
    
    print("="*80)
    print("\nInference complete! All results saved to:", OUTPUT_DIR)
    print("="*80)


if __name__ == '__main__':
    main()

