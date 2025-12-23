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
COMPARE_WITH_GT = True  # If True and GT available, generate comparison plots

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
    # 3b. Load Ground Truth (if available)
    # -----------------------------------------------------------------
    x_gt_polar = None
    x_gt_cartesian = None
    has_ground_truth = False
    
    if 'x' in data:
        print("="*80)
        print("GROUND TRUTH DETECTED")
        print("="*80)
        x_gt_polar = data['x'].astype(np.float64)
        
        # Handle different possible shapes
        if x_gt_polar.ndim == 1:
            # Single sample, replicate
            x_gt_polar = np.tile(x_gt_polar, (N_ranges, 1))
        elif x_gt_polar.ndim == 2:
            # Check if transpose is needed
            if x_gt_polar.shape[0] == N_ranges and x_gt_polar.shape[1] == N_theta:
                pass  # Already correct
            elif x_gt_polar.shape[0] == N_theta and x_gt_polar.shape[1] == N_ranges:
                x_gt_polar = x_gt_polar.T  # Transpose
            else:
                print(f"WARNING: GT shape {x_gt_polar.shape} doesn't match expected [{N_ranges}, {N_theta}]")
        
        has_ground_truth = True
        print(f"Ground truth loaded: {x_gt_polar.shape}")
        print(f"  GT range: [{x_gt_polar.min():.6f}, {x_gt_polar.max():.6f}]\n")
    else:
        print("No ground truth ('x') found in data file.\n")
    
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
                
                # Convert ground truth to Cartesian if available
                if has_ground_truth:
                    print("Converting ground truth to Cartesian...")
                    x_gt_cartesian = polar_to_cartesian(
                        x_gt_polar, ranges, angles,
                        x_grid, y_grid, x_radar, y_radar,
                        method='cubic', verbose=False
                    )
                    print(f"  GT Cartesian shape: {x_gt_cartesian.shape}")
                    print(f"  GT Cartesian range: [{x_gt_cartesian.min():.6f}, {x_gt_cartesian.max():.6f}]\n")
                
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
    if has_ground_truth:
        results_mat['x_ground_truth_polar'] = x_gt_polar
        results_mat['error_polar'] = np.abs(x_gt_polar - reconstructed_image)
        if x_gt_cartesian is not None:
            results_mat['x_ground_truth_cartesian'] = x_gt_cartesian
            results_mat['error_cartesian'] = np.abs(x_gt_cartesian - cartesian_image)
    
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
    if has_ground_truth:
        npz_data['x_ground_truth_polar'] = x_gt_polar
        npz_data['error_polar'] = np.abs(x_gt_polar - reconstructed_image)
        if x_gt_cartesian is not None:
            npz_data['x_ground_truth_cartesian'] = x_gt_cartesian
            npz_data['error_cartesian'] = np.abs(x_gt_cartesian - cartesian_image)
    
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
    # 5b. Ground Truth Comparison (if available)
    # -----------------------------------------------------------------
    if COMPARE_WITH_GT and has_ground_truth:
        print("\n" + "="*80)
        print("GROUND TRUTH COMPARISON")
        print("="*80)
        
        # Compute network input (matched filter output: A^H @ y)
        print("Computing network input (matched filter output: A^H @ y)...")
        # We need to apply A^H to the measurements
        A_batch = A_tensor.unsqueeze(0).to(DEVICE)
        
        # Process measurements to get matched filter output
        from core.utils import complex_conj_transpose_matmul
        measurements_tensor = torch.from_numpy(
            np.stack((measurements.real, measurements.imag), axis=1)
        ).to(DEVICE)  # [N_ranges, 2, N_v]
        
        with torch.no_grad():
            network_input = complex_conj_transpose_matmul(A_batch, measurements_tensor)  # [N_ranges, 2, N_theta]
        
        network_input_polar = network_input[:, 0, :].cpu().numpy()  # Extract real part [N_ranges, N_theta]
        print(f"  Network input shape: {network_input_polar.shape}")
        print(f"  Network input range: [{network_input_polar.min():.6f}, {network_input_polar.max():.6f}]\n")
        
        # --- Figure 1: Polar Domain Comparison (1x3 grid) ---
        print("Generating polar domain comparison...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Network input (A^H @ y)
        im1 = axes[0].imshow(network_input_polar, aspect='auto', cmap=COLORMAP,
                            interpolation='nearest', origin='upper',
                            extent=[angle_axis[0], angle_axis[-1], N_ranges-1, 0])
        axes[0].set_xlabel('Angle (degrees)', fontsize=11)
        axes[0].set_ylabel('Range Bin', fontsize=11)
        axes[0].set_title('Network Input (A^H @ y)\nMatched Filter', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=axes[0])
        
        # Ground truth
        im2 = axes[1].imshow(x_gt_polar, aspect='auto', cmap=COLORMAP,
                            interpolation='nearest', origin='upper',
                            extent=[angle_axis[0], angle_axis[-1], N_ranges-1, 0])
        axes[1].set_xlabel('Angle (degrees)', fontsize=11)
        axes[1].set_ylabel('Range Bin', fontsize=11)
        axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
        plt.colorbar(im2, ax=axes[1])
        
        # Network output
        im3 = axes[2].imshow(reconstructed_image, aspect='auto', cmap=COLORMAP,
                            interpolation='nearest', origin='upper',
                            extent=[angle_axis[0], angle_axis[-1], N_ranges-1, 0])
        axes[2].set_xlabel('Angle (degrees)', fontsize=11)
        axes[2].set_ylabel('Range Bin', fontsize=11)
        axes[2].set_title('Network Output', fontsize=12, fontweight='bold')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        polar_comp_path = os.path.join(OUTPUT_DIR, 'comparison_gt_polar.png')
        plt.savefig(polar_comp_path, dpi=DPI, bbox_inches='tight')
        print(f"  Polar comparison saved to: {polar_comp_path}")
        plt.close()
        
        # Compute polar domain metrics (for reference)
        mse_polar = np.mean((x_gt_polar - reconstructed_image)**2)
        nmse_polar = mse_polar / (np.mean(x_gt_polar**2) + 1e-10)
        psnr_polar = 10 * np.log10(np.max(x_gt_polar)**2 / (mse_polar + 1e-10))
        mae_polar = np.mean(np.abs(x_gt_polar - reconstructed_image))
        
        print(f"\n  Polar Domain Metrics:")
        print(f"    MSE:  {mse_polar:.8f}")
        print(f"    NMSE: {nmse_polar:.8f}")
        print(f"    PSNR: {psnr_polar:.2f} dB")
        print(f"    MAE:  {mae_polar:.8f}")
        
        # --- Figure 2: Cartesian Domain Comparison (if available) ---
        if x_gt_cartesian is not None and cartesian_image is not None:
            print("\nGenerating Cartesian domain comparison...")
            
            # Convert network input to Cartesian
            print("Converting network input to Cartesian...")
            network_input_cartesian = polar_to_cartesian(
                network_input_polar, ranges, angles,
                x_grid, y_grid, x_radar, y_radar,
                method='cubic', verbose=False
            )
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Network input
            im1 = axes[0].imshow(network_input_cartesian, cmap=COLORMAP,
                                extent=[x_grid.min(), x_grid.max(), y_grid.max(), y_grid.min()],
                                interpolation='nearest', origin='upper')
            axes[0].set_xlabel('X (meters)', fontsize=11)
            axes[0].set_ylabel('Y (meters)', fontsize=11)
            axes[0].set_title('Network Input (A^H @ y)\nMatched Filter', fontsize=12, fontweight='bold')
            axes[0].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0])
            
            # Ground truth
            im2 = axes[1].imshow(x_gt_cartesian, cmap=COLORMAP,
                                extent=[x_grid.min(), x_grid.max(), y_grid.max(), y_grid.min()],
                                interpolation='nearest', origin='upper')
            axes[1].set_xlabel('X (meters)', fontsize=11)
            axes[1].set_ylabel('Y (meters)', fontsize=11)
            axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[1].set_aspect('equal')
            plt.colorbar(im2, ax=axes[1])
            
            # Network output
            im3 = axes[2].imshow(cartesian_image, cmap=COLORMAP,
                                extent=[x_grid.min(), x_grid.max(), y_grid.max(), y_grid.min()],
                                interpolation='nearest', origin='upper')
            axes[2].set_xlabel('X (meters)', fontsize=11)
            axes[2].set_ylabel('Y (meters)', fontsize=11)
            axes[2].set_title('Network Output', fontsize=12, fontweight='bold')
            axes[2].set_aspect('equal')
            plt.colorbar(im3, ax=axes[2])
            
            plt.tight_layout()
            cart_comp_path = os.path.join(OUTPUT_DIR, 'comparison_gt_cartesian.png')
            plt.savefig(cart_comp_path, dpi=DPI, bbox_inches='tight')
            print(f"  Cartesian comparison saved to: {cart_comp_path}")
            plt.close()
            
            # Compute Cartesian domain metrics (for reference)
            mse_cart = np.mean((x_gt_cartesian - cartesian_image)**2)
            nmse_cart = mse_cart / (np.mean(x_gt_cartesian**2) + 1e-10)
            psnr_cart = 10 * np.log10(np.max(x_gt_cartesian)**2 / (mse_cart + 1e-10))
            mae_cart = np.mean(np.abs(x_gt_cartesian - cartesian_image))
            
            print(f"\n  Cartesian Domain Metrics:")
            print(f"    MSE:  {mse_cart:.8f}")
            print(f"    NMSE: {nmse_cart:.8f}")
            print(f"    PSNR: {psnr_cart:.2f} dB")
            print(f"    MAE:  {mae_cart:.8f}")
        
        print("\nGround truth comparison complete!")
    
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

