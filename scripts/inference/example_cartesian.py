"""
Example: Polar to Cartesian Conversion

Demonstrates how to use the polar_to_cartesian function to convert
reconstructed SAR images from polar to Cartesian coordinates.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.models import DBPNet
from core.data_loader import load_mat_file, to_tensor
from core.inference_utils import reconstruct_2d_image, polar_to_cartesian


# Example 1: Basic Cartesian conversion
def example_basic_cartesian():
    """Basic usage of polar to Cartesian conversion"""
    
    # Load trained model
    data = load_mat_file('data/data.mat')
    A_tensor = to_tensor(data['A'])
    
    model = DBPNet(A_tensor, num_iterations=5, N_admm_steps=2)
    model.load_state_dict(torch.load('checkpoints/dbp_model.pth'))
    
    # Reconstruct polar image
    measurements = data['received_signals_fft']
    polar_image = reconstruct_2d_image(model, measurements)  # [N_ranges, N_angles]
    
    # Load grid parameters
    ranges = data['ranges'].ravel()     # [N_ranges]
    angles = data['angles'].ravel()     # [N_angles]
    x_grid = data['x_grid']             # [N_x, N_y]
    y_grid = data['y_grid']             # [N_x, N_y]
    x_radar = float(data['x_radar'])
    y_radar = float(data['y_radar'])
    
    # Convert to Cartesian
    cartesian_image = polar_to_cartesian(
        polar_image, ranges, angles,
        x_grid, y_grid, x_radar, y_radar,
        method='cubic'
    )
    
    print(f"Polar image shape: {polar_image.shape}")
    print(f"Cartesian image shape: {cartesian_image.shape}")
    
    return polar_image, cartesian_image


# Example 2: Visualization comparison
def example_visualization():
    """Compare polar and Cartesian visualizations"""
    
    # Get both images
    polar_img, cart_img = example_basic_cartesian()
    
    # Load grid info
    data = load_mat_file('data/data.mat')
    ranges = data['ranges'].ravel()
    angles = data['angles'].ravel()
    x_grid = data['x_grid']
    y_grid = data['y_grid']
    x_radar = float(data['x_radar'])
    y_radar = float(data['y_radar'])
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Polar domain
    im1 = ax1.imshow(polar_img, aspect='auto', cmap='hot',
                    extent=[angles.min(), angles.max(), 
                           len(ranges)-1, 0])
    ax1.set_xlabel('Angle (degrees)')
    ax1.set_ylabel('Range Bin')
    ax1.set_title('Polar Domain')
    plt.colorbar(im1, ax=ax1)
    
    # Cartesian domain
    im2 = ax2.imshow(cart_img, cmap='hot',
                    extent=[x_grid.min(), x_grid.max(),
                           y_grid.max(), y_grid.min()])
    ax2.plot(x_radar, y_radar, 'r*', markersize=15, label='Radar')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.set_title('Cartesian Domain')
    ax2.set_aspect('equal')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('polar_vs_cartesian_comparison.png', dpi=150)
    print("Comparison saved to: polar_vs_cartesian_comparison.png")


# Example 3: Different interpolation methods
def example_interpolation_methods():
    """Compare different interpolation methods"""
    
    # Setup
    data = load_mat_file('data/data.mat')
    A_tensor = to_tensor(data['A'])
    model = DBPNet(A_tensor, num_iterations=5, N_admm_steps=2)
    model.load_state_dict(torch.load('checkpoints/dbp_model.pth'))
    
    measurements = data['received_signals_fft']
    polar_image = reconstruct_2d_image(model, measurements)
    
    ranges = data['ranges'].ravel()
    angles = data['angles'].ravel()
    x_grid = data['x_grid']
    y_grid = data['y_grid']
    x_radar = float(data['x_radar'])
    y_radar = float(data['y_radar'])
    
    # Try different methods
    methods = ['nearest', 'linear', 'cubic']
    results = {}
    
    for method in methods:
        print(f"Testing {method} interpolation...")
        cart_img = polar_to_cartesian(
            polar_image, ranges, angles,
            x_grid, y_grid, x_radar, y_radar,
            method=method
        )
        results[method] = cart_img
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, method in zip(axes, methods):
        im = ax.imshow(results[method], cmap='hot')
        ax.set_title(f'{method.capitalize()} Interpolation')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('interpolation_comparison.png', dpi=150)
    print("Interpolation comparison saved!")
    
    return results


# Example 4: Custom post-processing in Cartesian domain
def example_custom_processing():
    """Apply custom processing in Cartesian domain"""
    
    # Get Cartesian image
    data = load_mat_file('data/data.mat')
    A_tensor = to_tensor(data['A'])
    model = DBPNet(A_tensor, num_iterations=5, N_admm_steps=2)
    model.load_state_dict(torch.load('checkpoints/dbp_model.pth'))
    
    measurements = data['received_signals_fft']
    polar_image = reconstruct_2d_image(model, measurements)
    
    cartesian_image = polar_to_cartesian(
        polar_image,
        data['ranges'].ravel(),
        data['angles'].ravel(),
        data['x_grid'],
        data['y_grid'],
        float(data['x_radar']),
        float(data['y_radar'])
    )
    
    # Apply custom processing in Cartesian domain
    # 1. Normalize
    cart_norm = cartesian_image / cartesian_image.max()
    
    # 2. Apply threshold
    threshold = 0.1
    cart_thresh = np.where(cart_norm > threshold, cart_norm, 0)
    
    # 3. Apply Gaussian smoothing (optional)
    from scipy.ndimage import gaussian_filter
    cart_smooth = gaussian_filter(cart_thresh, sigma=2.0)
    
    # 4. Convert to dB scale
    cart_db = 20 * np.log10(cart_smooth + 1e-10)
    
    # Visualize processing steps
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.ravel()
    
    titles = ['Original', 'Normalized', 'Thresholded', 'Smoothed (dB)']
    images = [cartesian_image, cart_norm, cart_thresh, cart_db]
    
    for ax, title, img in zip(axes, titles, images):
        im = ax.imshow(img, cmap='hot')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('cartesian_processing_steps.png', dpi=150)
    print("Processing steps saved!")
    
    return cart_db


if __name__ == '__main__':
    print("Polar to Cartesian Conversion Examples")
    print("="*60)
    
    # Run examples (uncomment as needed)
    # polar, cartesian = example_basic_cartesian()
    # example_visualization()
    # results = example_interpolation_methods()
    # processed = example_custom_processing()
    
    print("\nExamples complete! See function docstrings for details.")

