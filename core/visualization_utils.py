"""
Visualization utilities for showing intermediate network outputs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .utils import complex_matmul


def plot_unrolled_iterations(intermediates, x_gt=None, sample_idx=0, save_path='unrolled_iterations.png'):
    """
    Plot the progressive refinement through unrolled iterations.
    
    Args:
        intermediates: Dictionary from model.forward(return_intermediates=True)
        x_gt: Ground truth x (optional) [batch, 2, N_theta]
        sample_idx: Which sample from batch to plot (default: 0)
        save_path: Where to save the plot
    """
    
    # Extract data for the selected sample
    x_init = intermediates['x_init'][sample_idx]  # [2, N_theta]
    x_after_denoiser = [x[sample_idx] for x in intermediates['x_after_denoiser']]
    x_after_admm = [x[sample_idx] for x in intermediates['x_after_admm']]
    x_final = intermediates['x_final'][sample_idx]  # [2, N_theta]
    
    num_iterations = len(x_after_admm)
    N_theta = x_init.shape[1]
    
    # Convert to numpy and get magnitude
    x_init_np = torch.real(torch.view_as_complex(x_init.permute(1, 0).contiguous())).cpu().numpy()
    x_denoiser_np = [torch.real(torch.view_as_complex(x.permute(1, 0).contiguous())).cpu().numpy() 
                     for x in x_after_denoiser]
    x_admm_np = [torch.real(torch.view_as_complex(x.permute(1, 0).contiguous())).cpu().numpy() 
                 for x in x_after_admm]
    x_final_np = torch.real(torch.view_as_complex(x_final.permute(1, 0).contiguous())).cpu().numpy()
    
    if x_gt is not None:
        x_gt_sample = x_gt[sample_idx]
        x_gt_np = torch.real(torch.view_as_complex(x_gt_sample.permute(1, 0).contiguous())).cpu().numpy()
    else:
        x_gt_np = None
    
    # Create angle axis
    theta = np.linspace(-25.0, 25.0, N_theta)
    
    # Create figure with subplots
    # Layout: 1 row for initialization, then num_iterations rows (each with denoiser and ADMM)
    fig = plt.figure(figsize=(16, 3 + 2.5 * num_iterations))
    gs = GridSpec(num_iterations + 1, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Normalize for better visualization
    if x_gt_np is not None:
        vmax = np.max(x_gt_np)
    else:
        vmax = max(np.max(x_init_np), np.max(x_final_np))
    
    # Row 0: Initialization
    ax0 = fig.add_subplot(gs[0, :])
    if x_gt_np is not None:
        ax0.plot(theta, x_gt_np / vmax, 'r-', label='Ground Truth', linewidth=2, alpha=0.7)
    ax0.plot(theta, x_init_np / vmax, 'b-', label='Initial (A^H @ y)', linewidth=2)
    ax0.set_title('Initialization: Matched Filter Output', fontsize=12, fontweight='bold')
    ax0.set_ylabel('Normalized Real Part')
    ax0.legend()
    ax0.grid(True, alpha=0.3)
    ax0.set_xlim([-25, 25])
    
    # Rows 1 to num_iterations: Show denoiser and ADMM outputs
    for i in range(num_iterations):
        # Left column: After denoiser
        ax_den = fig.add_subplot(gs[i+1, 0])
        if x_gt_np is not None:
            ax_den.plot(theta, x_gt_np / vmax, 'r-', label='Ground Truth', 
                       linewidth=1.5, alpha=0.5)
        ax_den.plot(theta, x_denoiser_np[i] / vmax, 'g-', 
                   label=f'After Denoiser {i+1}', linewidth=2)
        ax_den.set_title(f'Iteration {i+1}: After CNN Denoiser', fontsize=11, fontweight='bold')
        ax_den.set_ylabel('Normalized Real Part')
        ax_den.legend(fontsize=9)
        ax_den.grid(True, alpha=0.3)
        ax_den.set_xlim([-25, 25])
        if i == num_iterations - 1:  # Add xlabel to last row
            ax_den.set_xlabel('Angle (degrees)')
        
        # Right column: After ADMM
        ax_admm = fig.add_subplot(gs[i+1, 1])
        if x_gt_np is not None:
            ax_admm.plot(theta, x_gt_np / vmax, 'r-', label='Ground Truth', 
                        linewidth=1.5, alpha=0.5)
        ax_admm.plot(theta, x_admm_np[i] / vmax, 'b-', 
                    label=f'After ADMM {i+1}', linewidth=2)
        ax_admm.set_title(f'Iteration {i+1}: After ADMM (Data Consistency)', fontsize=11, fontweight='bold')
        ax_admm.set_ylabel('Normalized Real Part')
        ax_admm.legend(fontsize=9)
        ax_admm.grid(True, alpha=0.3)
        ax_admm.set_xlim([-25, 25])
        if i == num_iterations - 1:  # Add xlabel to last row
            ax_admm.set_xlabel('Angle (degrees)')
    
    #plt.suptitle(f'Progressive Refinement Through Unrolled Iterations (Sample {sample_idx})', 
    #             fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Unrolled iterations plot saved to {save_path}")
    plt.close()


def plot_iteration_comparison(intermediates, x_gt=None, sample_idx=0, save_path='iteration_comparison.png'):
    """
    Plot all iterations in a single figure for easy comparison.
    
    Args:
        intermediates: Dictionary from model.forward(return_intermediates=True)
        x_gt: Ground truth x (optional) [batch, 2, N_theta]
        sample_idx: Which sample from batch to plot (default: 0)
        save_path: Where to save the plot
    """
    
    # Extract data for the selected sample
    x_init = intermediates['x_init'][sample_idx]
    x_after_admm = [x[sample_idx] for x in intermediates['x_after_admm']]
    
    num_iterations = len(x_after_admm)
    N_theta = x_init.shape[1]
    
    # Convert to numpy and get magnitude
    x_init_np = torch.real(torch.view_as_complex(x_init.permute(1, 0).contiguous())).cpu().numpy()
    x_admm_np = [torch.real(torch.view_as_complex(x.permute(1, 0).contiguous())).cpu().numpy() 
                 for x in x_after_admm]
    
    if x_gt is not None:
        x_gt_sample = x_gt[sample_idx]
        x_gt_np = torch.real(torch.view_as_complex(x_gt_sample.permute(1, 0).contiguous())).cpu().numpy()
    else:
        x_gt_np = None
    
    # Create angle axis
    theta = np.linspace(-25.0, 25.0, N_theta)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize
    if x_gt_np is not None:
        vmax = np.max(x_gt_np)
    else:
        vmax = max(np.max(x_init_np), np.max(x_admm_np[-1]))
    
    # Plot ground truth
    if x_gt_np is not None:
        ax.plot(theta, x_gt_np / vmax, 'r-', label='Ground Truth', linewidth=3, alpha=0.8)
    
    # Plot initialization
    ax.plot(theta, x_init_np / vmax, 'gray', label='Initialization (A^H @ y)', 
            linewidth=2, linestyle='--', alpha=0.6)
    
    # Plot each iteration with different colors
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, num_iterations))
    for i in range(num_iterations):
        ax.plot(theta, x_admm_np[i] / vmax, color=colors[i], 
               label=f'After Iteration {i+1}', linewidth=2, alpha=0.8)
    
    ax.set_title(f'Comparison of All Iterations (Sample {sample_idx})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Angle (degrees)', fontsize=12)
    ax.set_ylabel('Normalized Real Part', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-25, 25])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Iteration comparison plot saved to {save_path}")
    plt.close()


def plot_measurement_domain_progression(intermediates, y_gt, A_tensor, sample_idx=0, 
                                        save_path='measurement_domain_progression.png'):
    """
    Plot measurement domain (y = A*x) for each iteration to show physical consistency.
    
    Args:
        intermediates: Dictionary from model.forward(return_intermediates=True)
        y_gt: Ground truth measurements [batch, 2, N_v]
        A_tensor: Steering matrix [2, N_v, N_theta]
        sample_idx: Which sample from batch to plot (default: 0)
        save_path: Where to save the plot
    """
    
    # Extract data for the selected sample
    x_init = intermediates['x_init'][sample_idx:sample_idx+1]  # Keep batch dim
    x_after_admm = [x[sample_idx:sample_idx+1] for x in intermediates['x_after_admm']]
    y_gt_sample = y_gt[sample_idx]  # [2, N_v]
    
    num_iterations = len(x_after_admm)
    N_v = y_gt_sample.shape[1]
    
    # Prepare A for matrix multiplication
    A_batch = A_tensor.unsqueeze(0)
    
    # Compute y_hat for initialization and each iteration
    y_init = complex_matmul(A_batch, x_init)[0]  # [2, N_v]
    y_after_admm = [complex_matmul(A_batch, x)[0] for x in x_after_admm]
    
    # Convert to numpy and get magnitude
    y_gt_np = torch.real(torch.view_as_complex(y_gt_sample.permute(1, 0).contiguous())).cpu().numpy()
    y_init_np = torch.real(torch.view_as_complex(y_init.permute(1, 0).contiguous())).cpu().numpy()
    y_admm_np = [torch.real(torch.view_as_complex(y.permute(1, 0).contiguous())).cpu().numpy() 
                 for y in y_after_admm]
    
    # Also get real and imaginary parts for detailed view
    y_gt_real = y_gt_sample[0].cpu().numpy()
    y_gt_imag = y_gt_sample[1].cpu().numpy()
    y_init_real = y_init[0].cpu().numpy()
    y_init_imag = y_init[1].cpu().numpy()
    y_admm_real = [y[0].cpu().numpy() for y in y_after_admm]
    y_admm_imag = [y[1].cpu().numpy() for y in y_after_admm]
    
    # Create antenna index
    antenna_idx = np.arange(N_v)
    
    # Create figure with subplots: 3 columns (magnitude, real, imag)
    fig = plt.figure(figsize=(18, 3 + 2 * num_iterations))
    gs = GridSpec(num_iterations + 1, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 0: Initialization
    # Real Part
    ax0_mag = fig.add_subplot(gs[0, 0])
    ax0_mag.plot(antenna_idx, y_gt_np, 'r-o', label='Measured y', linewidth=2, markersize=8)
    ax0_mag.plot(antenna_idx, y_init_np, 'b-s', label='y_init = A*x_init', linewidth=2, markersize=6, alpha=0.7)
    ax0_mag.set_title('Initialization: Real Part', fontsize=11, fontweight='bold')
    ax0_mag.set_xlabel('Virtual Antenna Index')
    ax0_mag.set_ylabel('Real Part')
    ax0_mag.legend(fontsize=9)
    ax0_mag.grid(True, alpha=0.3)
    
    # Real part
    ax0_real = fig.add_subplot(gs[0, 1])
    ax0_real.plot(antenna_idx, y_gt_real, 'r-o', label='Measured y', linewidth=2, markersize=8)
    ax0_real.plot(antenna_idx, y_init_real, 'b-s', label='y_init = A*x_init', linewidth=2, markersize=6, alpha=0.7)
    ax0_real.set_title('Initialization: Real Part', fontsize=11, fontweight='bold')
    ax0_real.set_xlabel('Virtual Antenna Index')
    ax0_real.set_ylabel('Real Amplitude')
    ax0_real.legend(fontsize=9)
    ax0_real.grid(True, alpha=0.3)
    
    # Imaginary part
    ax0_imag = fig.add_subplot(gs[0, 2])
    ax0_imag.plot(antenna_idx, y_gt_imag, 'r-o', label='Measured y', linewidth=2, markersize=8)
    ax0_imag.plot(antenna_idx, y_init_imag, 'b-s', label='y_init = A*x_init', linewidth=2, markersize=6, alpha=0.7)
    ax0_imag.set_title('Initialization: Imaginary Part', fontsize=11, fontweight='bold')
    ax0_imag.set_xlabel('Virtual Antenna Index')
    ax0_imag.set_ylabel('Imaginary Amplitude')
    ax0_imag.legend(fontsize=9)
    ax0_imag.grid(True, alpha=0.3)
    
    # Rows 1 to num_iterations: Show measurement domain after each ADMM
    for i in range(num_iterations):
        # Real Part
        ax_mag = fig.add_subplot(gs[i+1, 0])
        ax_mag.plot(antenna_idx, y_gt_np, 'r-o', label='Measured y', linewidth=2, markersize=8)
        ax_mag.plot(antenna_idx, y_admm_np[i], 'b-s', label=f'A*x_{i+1}', linewidth=2, markersize=6, alpha=0.7)
        ax_mag.set_title(f'Iteration {i+1}: Real Part', fontsize=11, fontweight='bold')
        ax_mag.set_xlabel('Virtual Antenna Index')
        ax_mag.set_ylabel('Real Part')
        ax_mag.legend(fontsize=9)
        ax_mag.grid(True, alpha=0.3)
        
        # Real part
        ax_real = fig.add_subplot(gs[i+1, 1])
        ax_real.plot(antenna_idx, y_gt_real, 'r-o', label='Measured y', linewidth=2, markersize=8)
        ax_real.plot(antenna_idx, y_admm_real[i], 'b-s', label=f'A*x_{i+1}', linewidth=2, markersize=6, alpha=0.7)
        ax_real.set_title(f'Iteration {i+1}: Real Part', fontsize=11, fontweight='bold')
        ax_real.set_xlabel('Virtual Antenna Index')
        ax_real.set_ylabel('Real Amplitude')
        ax_real.legend(fontsize=9)
        ax_real.grid(True, alpha=0.3)
        
        # Imaginary part
        ax_imag = fig.add_subplot(gs[i+1, 2])
        ax_imag.plot(antenna_idx, y_gt_imag, 'r-o', label='Measured y', linewidth=2, markersize=8)
        ax_imag.plot(antenna_idx, y_admm_imag[i], 'b-s', label=f'A*x_{i+1}', linewidth=2, markersize=6, alpha=0.7)
        ax_imag.set_title(f'Iteration {i+1}: Imaginary Part', fontsize=11, fontweight='bold')
        ax_imag.set_xlabel('Virtual Antenna Index')
        ax_imag.set_ylabel('Imaginary Amplitude')
        ax_imag.legend(fontsize=9)
        ax_imag.grid(True, alpha=0.3)
    
    #plt.suptitle(f'Measurement Domain Progression (y = A*x) - Sample {sample_idx}', 
    #             fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Measurement domain progression plot saved to {save_path}")
    plt.close()


def plot_mse_progression(intermediates, x_gt, sample_idx=0, save_path='mse_progression.png'):
    """
    Plot MSE after each iteration to show quantitative improvement.
    
    Args:
        intermediates: Dictionary from model.forward(return_intermediates=True)
        x_gt: Ground truth x [batch, 2, N_theta]
        sample_idx: Which sample from batch to plot (default: 0)
        save_path: Where to save the plot
    """
    
    if x_gt is None:
        print("Cannot plot MSE progression without ground truth")
        return
    
    # Extract data
    x_init = intermediates['x_init'][sample_idx]
    x_after_admm = [x[sample_idx] for x in intermediates['x_after_admm']]
    x_gt_sample = x_gt[sample_idx]
    
    # Compute MSE for each iteration
    mse_init = torch.mean((x_init - x_gt_sample) ** 2).item()
    mse_iterations = [torch.mean((x - x_gt_sample) ** 2).item() for x in x_after_admm]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = [0] + list(range(1, len(mse_iterations) + 1))
    mse_values = [mse_init] + mse_iterations
    
    ax.plot(iterations, mse_values, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MSE (Image Domain)', fontsize=12)
    ax.set_title(f'MSE Progression Through Iterations (Sample {sample_idx})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(iterations)
    
    # Add percentage improvement annotation
    improvement = (mse_init - mse_iterations[-1]) / mse_init * 100
    ax.text(0.5, 0.95, f'Total Improvement: {improvement:.1f}%', 
           transform=ax.transAxes, fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"MSE progression plot saved to {save_path}")
    plt.close()

