"""
Inference utilities for 2D SAR image reconstruction
"""

import torch
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator


def reconstruct_2d_image(model, measurements, device='cuda', batch_size=64, verbose=True):
    """
    Reconstruct 2D SAR image by passing each range bin through the trained network.
    
    This function takes measurements for multiple range bins and processes each one
    through the network to create a full 2D angle-range image.
    
    Args:
        model: Trained DBPNet model
        measurements: Complex measurements array of shape [N_ranges, N_v] 
                     Can be numpy array (complex) or torch tensor [N_ranges, 2, N_v]
        device: Device to run inference on ('cuda' or 'cpu')
        batch_size: Number of range bins to process simultaneously (default: 64)
        verbose: If True, show progress bar (default: True)
    
    Returns:
        reconstructed_image: Numpy array of shape [N_ranges, N_theta] containing
                           the reconstructed real-valued reflectivity image
    
    Example:
        >>> model = DBPNet(...)
        >>> model.load_state_dict(torch.load('checkpoints/dbp_model.pth'))
        >>> measurements = data['received_signals_fft']  # [256, 8]
        >>> image = reconstruct_2d_image(model, measurements, device='cuda')
        >>> print(image.shape)  # (256, 51)
    """
    
    model.eval()
    model = model.to(device)
    
    # Convert measurements to tensor if needed
    if isinstance(measurements, np.ndarray):
        if np.iscomplexobj(measurements):
            # Complex numpy array: [N_ranges, N_v] -> [N_ranges, 2, N_v]
            measurements_real = measurements.real.astype(np.float32)
            measurements_imag = measurements.imag.astype(np.float32)
            measurements_tensor = torch.from_numpy(
                np.stack((measurements_real, measurements_imag), axis=1)
            )
        else:
            # Assume already in [N_ranges, 2, N_v] format
            measurements_tensor = torch.from_numpy(measurements.astype(np.float32))
    else:
        measurements_tensor = measurements
    
    N_ranges = measurements_tensor.shape[0]
    
    # Determine output size by doing a single forward pass
    with torch.no_grad():
        sample_input = measurements_tensor[0:1].to(device)
        sample_output = model(sample_input)
        N_theta = sample_output.shape[2]  # [batch, 2, N_theta]
    
    # Initialize output array
    reconstructed_image = np.zeros((N_ranges, N_theta), dtype=np.float32)
    
    # Process in batches for efficiency
    num_batches = (N_ranges + batch_size - 1) // batch_size
    
    with torch.no_grad():
        if verbose:
            range_iterator = tqdm(range(0, N_ranges, batch_size), 
                                desc="Reconstructing image", 
                                unit="batch",
                                total=num_batches)
        else:
            range_iterator = range(0, N_ranges, batch_size)
        
        for start_idx in range_iterator:
            end_idx = min(start_idx + batch_size, N_ranges)
            
            # Extract batch of measurements
            y_batch = measurements_tensor[start_idx:end_idx].to(device)
            
            # Forward pass through network
            x_hat_batch = model(y_batch)
            
            # Extract real part (channel 0) and convert to numpy
            # Shape: [batch, 2, N_theta] -> [batch, N_theta]
            x_hat_real = x_hat_batch[:, 0, :].cpu().numpy()
            
            # Store in output array
            reconstructed_image[start_idx:end_idx, :] = x_hat_real
    
    return reconstructed_image


def reconstruct_2d_image_with_input(model, measurements, device='cuda', batch_size=64, verbose=True):
    """
    Reconstruct 2D SAR image and also return the magnitude of input measurements.
    
    Convenience function that returns both the reconstructed image and the input
    measurement magnitude for easy comparison visualization.
    
    Args:
        model: Trained DBPNet model
        measurements: Complex measurements array of shape [N_ranges, N_v]
        device: Device to run inference on ('cuda' or 'cpu')
        batch_size: Number of range bins to process simultaneously (default: 64)
        verbose: If True, show progress bar (default: True)
    
    Returns:
        tuple: (reconstructed_image, measurement_magnitude)
            - reconstructed_image: [N_ranges, N_theta] reconstructed reflectivity
            - measurement_magnitude: [N_ranges, N_v] magnitude of input measurements
    
    Example:
        >>> image, input_mag = reconstruct_2d_image_with_input(model, measurements)
        >>> plt.subplot(121); plt.imshow(input_mag)
        >>> plt.subplot(122); plt.imshow(image)
    """
    
    # Reconstruct the image
    reconstructed_image = reconstruct_2d_image(model, measurements, device, batch_size, verbose)
    
    # Compute input magnitude
    if isinstance(measurements, np.ndarray):
        if np.iscomplexobj(measurements):
            measurement_magnitude = np.abs(measurements)
        else:
            # Assume [N_ranges, 2, N_v] format - compute magnitude
            measurement_magnitude = np.sqrt(measurements[:, 0]**2 + measurements[:, 1]**2)
    else:
        # Torch tensor [N_ranges, 2, N_v]
        measurements_np = measurements.cpu().numpy()
        measurement_magnitude = np.sqrt(measurements_np[:, 0]**2 + measurements_np[:, 1]**2)
    
    return reconstructed_image, measurement_magnitude


def polar_to_cartesian(polar_image, ranges, angles, x_grid, y_grid, 
                      x_radar, y_radar, method='cubic', verbose=True):
    """
    Convert polar domain SAR image to Cartesian coordinates with radar offset.
    
    This function transforms a reconstructed SAR image from polar coordinates
    (range-angle) to Cartesian coordinates (x-y), accounting for the radar
    position offset.
    
    Args:
        polar_image: [N_ranges, N_angles] - Reconstructed polar image
        ranges: [N_ranges] - Range bins in meters (1D array)
        angles: [N_angles] - Angle bins in degrees (1D array, e.g., -25 to +25)
        x_grid: [N_x, N_y] - X-coordinate meshgrid in meters
        y_grid: [N_x, N_y] - Y-coordinate meshgrid in meters
        x_radar: scalar - Radar X position in meters
        y_radar: scalar - Radar Y position in meters
        method: Interpolation method: 'cubic' (default), 'linear', or 'nearest'
        verbose: If True, print diagnostic information about coverage
    
    Returns:
        cartesian_image: [N_x, N_y] - Image in Cartesian coordinates
    
    Coordinate System:
        The radar is located at (x_radar, y_radar).
        For a point at range r and angle θ from the radar:
            x = x_radar + r × sin(θ)
            y = y_radar + r × cos(θ)
        
        Convention:
            - θ = 0° is along +Y axis (down-range direction)
            - Positive θ is toward +X axis (cross-range right)
    
    Example:
        >>> # After reconstructing polar image
        >>> polar_img = reconstruct_2d_image(model, measurements)  # [256, 51]
        >>> 
        >>> # Load grid parameters from data.mat
        >>> data = load_mat_file('data/data.mat')
        >>> ranges = data['ranges']      # [256]
        >>> angles = data['angles']      # [51]
        >>> x_grid = data['x_grid']      # [512, 512]
        >>> y_grid = data['y_grid']      # [512, 512]
        >>> x_radar = data['x_radar']    # scalar
        >>> y_radar = data['y_radar']    # scalar
        >>> 
        >>> # Convert to Cartesian
        >>> cart_img = polar_to_cartesian(polar_img, ranges, angles,
        ...                                x_grid, y_grid, x_radar, y_radar)
        >>> print(cart_img.shape)  # (512, 512)
    """
    
    # Input validation
    if polar_image.ndim != 2:
        raise ValueError(f"polar_image must be 2D, got shape {polar_image.shape}")
    
    if len(ranges) != polar_image.shape[0]:
        raise ValueError(f"Length of ranges ({len(ranges)}) must match polar_image.shape[0] ({polar_image.shape[0]})")
    
    if len(angles) != polar_image.shape[1]:
        raise ValueError(f"Length of angles ({len(angles)}) must match polar_image.shape[1] ({polar_image.shape[1]})")
    
    if x_grid.shape != y_grid.shape:
        raise ValueError(f"x_grid and y_grid must have same shape, got {x_grid.shape} and {y_grid.shape}")
    
    # Convert to numpy arrays if needed
    ranges = np.asarray(ranges, dtype=np.float64)
    angles = np.asarray(angles, dtype=np.float64)
    polar_image = np.asarray(polar_image, dtype=np.float64)
    x_grid = np.asarray(x_grid, dtype=np.float64)
    y_grid = np.asarray(y_grid, dtype=np.float64)
    
    # Ensure ranges and angles are sorted
    if not np.all(np.diff(ranges) > 0):
        raise ValueError("ranges must be sorted in ascending order")
    if not np.all(np.diff(angles) > 0):
        raise ValueError("angles must be sorted in ascending order")
    
    # Create polar grid interpolator
    # The interpolator expects (ranges, angles) as grid coordinates
    interpolator = RegularGridInterpolator(
        (ranges, angles),           # Polar grid coordinates
        polar_image,                # Values on polar grid
        method=method,              # Interpolation method
        bounds_error=False,         # Don't raise error on out-of-bounds
        fill_value=0.0              # Fill out-of-bounds regions with 0
    )
    
    # Transform Cartesian grid to polar coordinates (relative to radar)
    dx = x_grid - x_radar  # X relative to radar [N_x, N_y]
    dy = y_grid - y_radar  # Y relative to radar [N_x, N_y]
    
    # Convert to polar coordinates
    r = np.sqrt(dx**2 + dy**2)                    # Range [N_x, N_y]
    theta_rad = np.arctan2(dy, dx)                # Angle in radians [N_x, N_y] - CORRECTED
    theta_deg = np.degrees(theta_rad)             # Angle in degrees [N_x, N_y]
    
    if verbose:
        # Diagnostic information
        print(f"\n  Polar to Cartesian Conversion Diagnostics:")
        print(f"  Polar coverage:")
        print(f"    Range: [{ranges.min():.2f}, {ranges.max():.2f}] m")
        print(f"    Angle: [{angles.min():.2f}, {angles.max():.2f}]°")
        print(f"  Cartesian grid coverage:")
        print(f"    Computed range: [{r.min():.2f}, {r.max():.2f}] m")
        print(f"    Computed angle: [{theta_deg.min():.2f}, {theta_deg.max():.2f}]°")
        
        # Check coverage
        in_range = (r >= ranges.min()) & (r <= ranges.max())
        in_angle = (theta_deg >= angles.min()) & (theta_deg <= angles.max())
        in_both = in_range & in_angle
        coverage = 100 * np.sum(in_both) / in_both.size
        print(f"  Coverage: {coverage:.1f}% of Cartesian pixels fall within polar grid")
        
        if coverage < 10:
            print(f"  ⚠️  WARNING: Less than 10% coverage! Most of the image will be zeros.")
            print(f"     Check that your Cartesian grid matches the polar coverage.")
    
    # Create points array for interpolation
    # Shape: [N_x*N_y, 2] where each row is [range, angle]
    points = np.stack([r.ravel(), theta_deg.ravel()], axis=-1)
    
    # Interpolate from polar grid to Cartesian points
    cartesian_values = interpolator(points)  # [N_x*N_y]
    
    # Reshape back to 2D Cartesian grid
    cartesian_image = cartesian_values.reshape(x_grid.shape)  # [N_x, N_y]
    
    return cartesian_image

