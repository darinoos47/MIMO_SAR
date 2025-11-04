import torch
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import h5py

from models import DBPNet
from data_loader import to_tensor

# -----------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------
MAT_FILE = 'FL_MIMO_SAR_data.mat'
MODEL_PATH = 'dbp_model.pth'
RESULT_IMAGE_PATH = 'dbp_reconstructed_image.png'
GT_IMAGE_PATH = 'ground_truth_image.png'

# DBP network iterations (must match trained model)
NUM_DBP_ITERATIONS = 5

# Parameters from script.m for Cartesian grid
X_DOMAIN_LENGTH = 15.0
Y_DOMAIN_LENGTH = 4.125
BANDWIDTH = 2e9
RANGE_RESOLUTION = 3e8 / (2 * BANDWIDTH) # 0.075 m

# Parameters from script.m for Polar grid
SAMPLING_RATE = 10e6
PULSE_DURATION = 200e-6
CHIRP_RATE = BANDWIDTH / PULSE_DURATION
NUM_SAMPLES = int(round(PULSE_DURATION * SAMPLING_RATE)) # 2000
# NOTE: NUM_ANGLE_BINS will be determined from actual data (A matrix shape)
START_ANGLE_DEG = 25.0
END_ANGLE_DEG = -25.0

# -----------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------
def load_mat_file(filepath):
    """Loads .mat file, handling v7.3 files."""
    try:
        data = scipy.io.loadmat(filepath)
    except NotImplementedError:
        print("Reading v7.3 MAT file with h5py...")
        data = {}
        with h5py.File(filepath, 'r') as f:
            for k in f.keys():
                data[k] = f[k][()]
        
        # Handle complex data loaded by h5py
        def fix_complex(d):
            if d.dtype.names and ('real' in d.dtype.names) and ('imag' in d.dtype.names):
                return d['real'] + 1j * d['imag']
            return d
            
        data = {k: fix_complex(v) for k, v in data.items()}
    return data

def tensor_to_complex(tensor):
    """Converts a [batch, 2, ...] tensor back to a complex numpy array."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.numpy()
    return data[:, 0, ...] + 1j * data[:, 1, ...]

def plot_image(image_data, x_coords, y_coords, title, save_path):
    """Plots and saves the reconstructed image."""
    print(f"Plotting '{title}'...")
    plt.figure(figsize=(8, 6))
    
    # Use 20*log10 for dB scale, handle zeros
    image_db = 20 * np.log10(np.abs(image_data) + 1e-10)
    
    # Normalize
    image_db = image_db - np.max(image_db)
    
    plt.imshow(
        image_db,
        extent=[x_coords[0], x_coords[-1], y_coords[-1], y_coords[0]],
        aspect='auto',
        cmap='jet',
        vmin=-30, # Clip at -30 dB
        vmax=0
    )
    plt.colorbar(label='Amplitude (dB)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(title)
    plt.savefig(save_path)
    print(f"Image saved to {save_path}")
    plt.close()

# -----------------------------------------------------------------
# 3. Main Inference and Reconstruction
# -----------------------------------------------------------------
def main():
    # --- Setup Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    # --- Load Data ---
    print(f"Loading data from {MAT_FILE}...")
    data = load_mat_file(MAT_FILE)
    
    # A: [N_v, N_theta]
    A_complex = data['A'].astype(np.complex64)
    A_tensor = to_tensor(A_complex).to(device)
    
    # Get NUM_ANGLE_BINS from the steering matrix shape
    # A_tensor shape: [2, N_v, N_theta]
    NUM_ANGLE_BINS = A_tensor.shape[2]
    print(f"Number of angle bins from data: {NUM_ANGLE_BINS}")
    
    # y: [N_r, N_v, N_l] -> transpose to [N_l, N_r, N_v]
    y_measurements = np.transpose(data['received_signals_fft'], (2, 0, 1)).astype(np.complex64)
    
    # x_scan: [1, N_l] -> flatten to [N_l]
    x_scan_positions = data['x_scan'].flatten()
    
    # Ground Truth: [N_y_cart, N_x_cart]
    object_gt = data['object']
    
    num_aperture_steps = y_measurements.shape[0]
    num_range_bins = y_measurements.shape[1]
    
    # --- Load Trained Model ---
    print(f"Loading trained model from {MODEL_PATH}...")
    model = DBPNet(A_tensor, num_iterations=NUM_DBP_ITERATIONS).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode
    
    # --- Run Inference ---
    print("Running inference on all measurements...")
    # This will hold all enhanced polar images: [N_l, N_r, N_theta]
    x_hat_polar_all = np.zeros(
        (num_aperture_steps, num_range_bins, NUM_ANGLE_BINS), 
        dtype=np.complex64
    )
    
    with torch.no_grad():
        for i in range(num_aperture_steps):
            # Get all range bins for this aperture step: [N_r, N_v]
            y_aperture_i = y_measurements[i, :, :]
            
            # Convert to 2-channel tensor: [N_r, 2, N_v]
            y_tensor_i = to_tensor(y_aperture_i).to(device)
            
            # Pass through the DBP network
            # Output x_hat_i is shape [N_r, 2, N_theta]
            x_hat_tensor_i = model(y_tensor_i)
            
            # Convert back to complex numpy array: [N_r, N_theta]
            x_hat_polar_all[i, :, :] = tensor_to_complex(x_hat_tensor_i)
            
            print(f"  Processed aperture step {i+1}/{num_aperture_steps}")
            
    print("Inference complete.")

    # --- Modified Backprojection (MBP) ---
    print("Starting Modified Backprojection...")
    
    # 1. Define Polar Coordinate Grid (from script.m)
    #    [cite: 732]
    ranges_polar = np.linspace(
        -SAMPLING_RATE / 2, SAMPLING_RATE / 2, num_range_bins
    ) * 3e8 / (2 * CHIRP_RATE)
    #    [cite: 733]
    angles_polar_deg = np.linspace(START_ANGLE_DEG, END_ANGLE_DEG, NUM_ANGLE_BINS)
    
    # 2. Define Cartesian Output Grid (from script.m)
    dx = dy = RANGE_RESOLUTION
    x_cart = np.arange(dx / 2, X_DOMAIN_LENGTH, dx)
    y_cart = np.arange(Y_DOMAIN_LENGTH / 2 - dy / 2, -Y_DOMAIN_LENGTH / 2, -dy)
    X_cart_grid, Y_cart_grid = np.meshgrid(x_cart, y_cart)
    
    final_image = np.zeros_like(X_cart_grid, dtype=np.complex64)
    
    # 3. Create interpolators for each aperture step's polar image
    #    This is the "upsampling" part of MBP [cite: 595, 599]
    interpolators = []
    for i in range(num_aperture_steps):
        # We need (angle, range) for RegularGridInterpolator
        interpolator = RegularGridInterpolator(
            (angles_polar_deg, ranges_polar),
            x_hat_polar_all[i, :, :].T, # Transpose to [N_theta, N_r]
            method='linear',
            bounds_error=False,
            fill_value=0
        )
        interpolators.append(interpolator)
        
    # 4. Loop over every pixel in the final Cartesian grid
    for m in range(X_cart_grid.shape[0]): # y-axis
        for n in range(X_cart_grid.shape[1]): # x-axis
            x_m = X_cart_grid[m, n]
            y_n = Y_cart_grid[m, n]
            
            pixel_sum = 0.0 + 0.0j
            
            # 5. Sum contributions from each aperture step (Eq. 31) [cite: 730]
            for i in range(num_aperture_steps):
                x_aperture = x_scan_positions[i]
                
                # Calculate relative range and angle for this pixel
                # [cite: 732-733]
                rel_x = x_m - x_aperture
                rel_y = y_n
                
                # $r_{mn}=\sqrt{(x_{m}-i)^{2}+y_{n}^{2}}$ [cite: 732]
                r_mn = np.sqrt(rel_x**2 + rel_y**2)
                
                # $\theta_{mn}=$ arctan $((x_{m}-i)/y_{n})$ [cite: 733]
                # We use arctan2 for a stable 4-quadrant angle
                theta_mn_rad = np.arctan2(rel_x, rel_y)
                theta_mn_deg = np.degrees(theta_mn_rad)
                
                # 6. Interpolate the value from the polar image
                # This finds the value at [C_theta_mn, C_r_mn] [cite: 730-733]
                point = (theta_mn_deg, r_mn)
                pixel_sum += interpolators[i](point)
                
            final_image[m, n] = pixel_sum
            
    print("Modified Backprojection complete.")

    # --- Plot and Save Results ---
    plot_image(
        final_image, 
        x_cart, 
        y_cart, 
        'DBP Reconstructed Image', 
        RESULT_IMAGE_PATH
    )
    
    # Plot ground truth for comparison
    plot_image(
        object_gt, 
        x_cart, 
        y_cart, 
        'Ground Truth', 
        GT_IMAGE_PATH
    )

if __name__ == '__main__':
    main()
