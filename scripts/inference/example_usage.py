"""
Example: Using the reconstruct_2d_image function

This demonstrates how to use the reusable inference function
in your own scripts.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from core.models import DBPNet
from core.data_loader import load_mat_file, to_tensor
from core.inference_utils import reconstruct_2d_image, reconstruct_2d_image_with_input


# Example 1: Basic usage
def example_basic():
    """Basic usage of the reconstruction function"""
    
    # Load your trained model
    A_tensor = torch.randn(2, 8, 51)  # Example steering matrix
    model = DBPNet(A_tensor, num_iterations=5, N_admm_steps=2)
    model.load_state_dict(torch.load('checkpoints/dbp_model.pth'))
    
    # Load your measurements
    data = load_mat_file('data/data.mat')
    measurements = data['received_signals_fft']  # [N_ranges, N_v]
    
    # Reconstruct the image
    reconstructed_image = reconstruct_2d_image(
        model=model,
        measurements=measurements,
        device='cuda',
        batch_size=64
    )
    
    print(f"Reconstructed image shape: {reconstructed_image.shape}")
    return reconstructed_image


# Example 2: With input for comparison
def example_with_comparison():
    """Get both reconstructed image and input magnitude"""
    
    # Load model
    A_tensor = torch.randn(2, 8, 51)
    model = DBPNet(A_tensor, num_iterations=5, N_admm_steps=2)
    model.load_state_dict(torch.load('checkpoints/dbp_model.pth'))
    
    # Load measurements
    data = load_mat_file('data/data.mat')
    measurements = data['received_signals_fft']
    
    # Reconstruct with input
    recon_image, input_mag = reconstruct_2d_image_with_input(
        model=model,
        measurements=measurements,
        device='cuda'
    )
    
    # Quick visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(input_mag, aspect='auto', cmap='hot')
    ax1.set_title('Input Measurements')
    ax2.imshow(recon_image, aspect='auto', cmap='hot')
    ax2.set_title('Reconstructed Image')
    plt.savefig('quick_comparison.png')
    
    return recon_image, input_mag


# Example 3: Custom processing
def example_custom_processing():
    """Use the function with custom post-processing"""
    
    # Load model
    A_tensor = torch.randn(2, 8, 51)
    model = DBPNet(A_tensor, num_iterations=5, N_admm_steps=2)
    model.load_state_dict(torch.load('checkpoints/dbp_model.pth'))
    
    # Load measurements
    data = load_mat_file('data/data.mat')
    measurements = data['received_signals_fft']
    
    # Reconstruct
    recon_image = reconstruct_2d_image(model, measurements)
    
    # Apply custom post-processing
    # 1. Normalize
    recon_normalized = recon_image / recon_image.max()
    
    # 2. Apply threshold
    threshold = 0.1
    recon_thresholded = np.where(recon_normalized > threshold, recon_normalized, 0)
    
    # 3. Convert to dB scale
    recon_db = 20 * np.log10(recon_thresholded + 1e-10)
    
    return recon_db


# Example 4: Processing multiple datasets
def example_batch_datasets():
    """Process multiple data files"""
    
    # Load model once
    A_tensor = torch.randn(2, 8, 51)
    model = DBPNet(A_tensor, num_iterations=5, N_admm_steps=2)
    model.load_state_dict(torch.load('checkpoints/dbp_model.pth'))
    model = model.to('cuda')
    model.eval()
    
    # Process multiple files
    data_files = ['data1.mat', 'data2.mat', 'data3.mat']
    results = []
    
    for data_file in data_files:
        data = load_mat_file(data_file)
        measurements = data['received_signals_fft']
        
        # Reconstruct
        recon_image = reconstruct_2d_image(
            model=model,
            measurements=measurements,
            device='cuda',
            verbose=False  # Disable progress bar for batch processing
        )
        
        results.append(recon_image)
    
    return results


if __name__ == '__main__':
    # Run examples (uncomment as needed)
    
    # image = example_basic()
    # recon, input_mag = example_with_comparison()
    # recon_db = example_custom_processing()
    # all_results = example_batch_datasets()
    
    print("Example usage demonstrated!")
    print("See function docstrings for more details.")

