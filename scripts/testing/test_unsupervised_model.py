import torch
import torch.nn as nn # <-- Import nn for the loss function
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import h5py

# Add project root to path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


from core.models import DBPNet
from core.data_loader import to_tensor
# We need complex_matmul to create the measurement domain plot
from core.utils import complex_matmul 

# -----------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------
MAT_FILE = '../../data/FL_MIMO_SAR_data.mat'
MODEL_PATH = '../../checkpoints/dbp_model.pth' # The unsupervised model
PLOT_SAVE_PATH_IMAGE = 'unsupervised_model_output_image.png'
PLOT_SAVE_PATH_MEAS = 'unsupervised_model_output_measurement.png'

# --- Model Hyperparameters ---
# !!! CRITICAL: These MUST MATCH the values from your 'train.py' script !!!
NUM_UNROLLS = 1      # Currently train.py uses 1
NUM_ADMM_STEPS = 1   # Currently train.py uses 1   

# --- Plotting Parameters (from MATLAB script) ---
# NOTE: NUM_ANGLE_BINS will be determined from actual data, not hardcoded
START_ANGLE_DEG = 25.0
END_ANGLE_DEG = -25.0

def load_mat_file(filepath):
    """Loads .mat file, handling v7.3 files with correct transposes."""
    try:
        data = scipy.io.loadmat(filepath)
    except NotImplementedError:
        print(f"Reading v7.3 MAT file {filepath} with h5py...")
        data = {}
        with h5py.File(filepath, 'r') as f:
            for k in f.keys():
                v = f[k]
                # Handle transpose: h5py loads as [dim1, dim0]
                # We need [dim0, dim1] to match MATLAB
                if v.ndim == 2:
                    v = v[()] # Load data
                    v = v.T    # Transpose back to MATLAB's shape
                else:
                    v = v[()] # Load data
                
                # Handle complex data saved as a struct
                if v.dtype.names and ('real' in v.dtype.names) and ('imag' in v.dtype.names):
                    v = v['real'] + 1j * v['imag']
                
                data[k] = v
    return data

def tensor_to_complex_np(tensor):
    """
    Converts a [batch, 2, ...] tensor to a complex numpy array.
    Squeezes the batch dimension.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.numpy()
    # Squeeze out the batch dim if it's 1, then combine channels
    data_squeezed = data.squeeze(axis=0)
    return data_squeezed[0] + 1j * data_squeezed[1]

# -----------------------------------------------------------------
# 2. Main Test Function
# -----------------------------------------------------------------
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    # -----------------------------------------------------------------
    # 1. Load Data (CORRECTED - Same logic as data_loader.py)
    # -----------------------------------------------------------------
    print(f"Loading data from {MAT_FILE}...")
    data = load_mat_file(MAT_FILE)
    
    # Normalized steering matrix 'A'
    A_np = data['A'].astype(np.complex64)
    A_tensor = to_tensor(A_np).to(device)
    A_batch_tensor = A_tensor.unsqueeze(0) # For complex_matmul
    
    # Load measurements the SAME way as data_loader.py does
    measurements = data['received_signals_fft']
    print(f"Raw measurements shape: {measurements.shape}")
    
    # Handle 2D or 3D data (same as MIMOSAR_Dataset)
    if measurements.ndim == 3:
        # Transpose from (N_r, N_v, N_l) to (N_l, N_r, N_v)
        measurements_transposed = np.transpose(measurements, (2, 0, 1))
    else:
        # It's 2D (N_samples, N_v), so just add an N_l dimension
        measurements_transposed = measurements.reshape(1, measurements.shape[0], measurements.shape[1])
    
    # Reshape to flat list of samples: [N_l * N_samples, N_v]
    measurements_flat = measurements_transposed.reshape(-1, measurements_transposed.shape[-1])
    print(f"Total samples available: {measurements_flat.shape[0]}")
    print(f"Virtual antennas per sample: {measurements_flat.shape[1]}")
    
    # *** SELECT A SPECIFIC SAMPLE TO TEST ***
    # The training DataLoader returns samples one-by-one from this flattened array
    # With shuffle=True and batch_size=64, the last batch in epoch 500 would contain
    # random samples. To test on the FIRST sample (index 0):
    SAMPLE_IDX = 25  # Change this to test different samples
    
    y_gt_np = measurements_flat[SAMPLE_IDX].astype(np.complex64)  # Shape [8]
    print(f"\nTesting on sample index: {SAMPLE_IDX}")
    print(f"y_gt_np shape: {y_gt_np.shape}")
    
    # Ground truth reflectivity 'x' (optional, for visualization only)
    # NOTE: Unsupervised training never uses this!
    if 'x' in data:
        x_gt_all = data['x']
        print(f"Ground truth 'x' loaded with shape: {x_gt_all.shape}")
        
        # If x is 2D [N_samples, N_theta], select the same sample we're testing
        if x_gt_all.ndim == 2 and x_gt_all.shape[0] > SAMPLE_IDX:
            x_gt_np = x_gt_all[SAMPLE_IDX].astype(np.complex64)  # Select sample SAMPLE_IDX
            print(f"  Selected sample {SAMPLE_IDX}: x_gt shape = {x_gt_np.shape}")
        # If x is 1D [N_theta], use it directly (single sample dataset)
        elif x_gt_all.ndim == 1:
            x_gt_np = x_gt_all.astype(np.complex64)
            print(f"  Using single-sample ground truth: x_gt shape = {x_gt_np.shape}")
        # Otherwise flatten as fallback
        else:
            x_gt_np = x_gt_all.flatten().astype(np.complex64)
            print(f"  Using flattened ground truth: x_gt shape = {x_gt_np.shape}")
    else:
        x_gt_np = None
        print("No ground truth 'x' available (not needed for unsupervised testing)")

    # -----------------------------------------------------------------
    # 2. Load Trained Model
    # -----------------------------------------------------------------
    print(f"Loading trained model from {MODEL_PATH}...")
    
    model = DBPNet(
        A_tensor, 
        num_iterations=NUM_UNROLLS, 
        N_admm_steps=NUM_ADMM_STEPS
    ).to(device)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    # *** IMPORTANT: BatchNorm behavior ***
    # If trained on very few samples, BatchNorm statistics may not be reliable.
    # For debugging/testing on the same data used for training, use train() mode.
    # For deployment on new data, use eval() mode.
    # NOTE: When testing single samples (batch_size=1) with very small N_theta (e.g., 1),
    # we MUST use eval() mode otherwise BatchNorm will fail.
    USE_TRAIN_MODE = False  # Changed to False due to small dimensions
    
    if USE_TRAIN_MODE:
        model.train()
        print("Model loaded successfully (using train mode for BatchNorm consistency).")
        print("WARNING: Using train mode - suitable for debugging with training data only!")
    else:
        model.eval()
        print("Model loaded successfully (using eval mode).")

    # -----------------------------------------------------------------
    # 3. Run Inference
    # -----------------------------------------------------------------
    print("Running inference on the single range sample...")
    
    # Prepare the 'y' tensor: add a batch dimension [1, 2, 8]
    y_tensor = to_tensor(y_gt_np).unsqueeze(0).to(device)
    
    # <<< NEW: Instantiate the loss function >>>
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        # Get the network's reconstruction, x_hat
        x_hat_tensor = model(y_tensor) # Output shape: [1, 2, 1001]
        
        # Create the measurement domain output
        # y_hat = A * x_hat
        y_hat_tensor = complex_matmul(A_batch_tensor, x_hat_tensor) # Shape: [1, 2, 8]
        
        # <<< NEW: Calculate the final measurement loss >>>
        loss = criterion(y_hat_tensor, y_tensor)
        
    # Convert outputs to numpy
    x_hat_np = tensor_to_complex_np(x_hat_tensor) # Shape [1001]
    y_hat_np = tensor_to_complex_np(y_hat_tensor) # Shape [8]
    
    print("Inference complete.")
    print(f"\n>>> Final Measurement Loss (MSE): {loss.item():.10f} <<<\n")

    # -----------------------------------------------------------------
    # 4. Plot the Results
    # -----------------------------------------------------------------
    
    # --- PLOT 1: Image Domain (x vs x_hat) ---
    print(f"Plotting Image Domain results to {PLOT_SAVE_PATH_IMAGE}...")
    
    # Get NUM_ANGLE_BINS from actual data
    NUM_ANGLE_BINS = x_hat_np.shape[0]
    theta = np.linspace(START_ANGLE_DEG, END_ANGLE_DEG, NUM_ANGLE_BINS)
    x_hat_abs = np.abs(x_hat_np)
    
    plt.figure(figsize=(12, 7))
    
    # Plot ground truth only if available AND dimensions match
    if x_gt_np is not None and x_gt_np.shape[0] == NUM_ANGLE_BINS:
        x_gt_abs = np.abs(x_gt_np)
        plt.plot(theta, x_gt_abs / np.max(x_gt_abs), 'r', label='Ground Truth (x)', linewidth=2)
    elif x_gt_np is not None:
        print(f"  Note: Ground truth 'x' has shape {x_gt_np.shape} but output has {NUM_ANGLE_BINS} angle bins - skipping ground truth plot")
    
    plt.plot(theta, x_hat_abs / np.max(x_hat_abs), 'b--', label='Network Output (x_hat)', linewidth=2)
    plt.title(f'Unsupervised Model Output (Image Domain) - Sample {SAMPLE_IDX}')
    plt.xlabel('Angle (degree)')
    plt.ylabel('Normalized Amplitude')
    plt.xlim([START_ANGLE_DEG, END_ANGLE_DEG])
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_SAVE_PATH_IMAGE)
    print("Image plot saved.")
    plt.close()

    # --- PLOT 2: Measurement Domain (y vs y_hat) ---
    print(f"Plotting Measurement Domain results to {PLOT_SAVE_PATH_MEAS}...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot Real Part
    ax1.plot(np.real(y_gt_np), 'r-o', label='Ground Truth (y)', linewidth=2, markersize=6)
    ax1.plot(np.real(y_hat_np), 'b--s', label='Network Output (A*x_hat)', linewidth=2, markersize=6, alpha=0.7)
    ax1.set_title(f'Measurement Domain (Real Part) - Sample {SAMPLE_IDX}')
    ax1.set_xlabel('Virtual Antenna Index')
    ax1.set_ylabel('Real Amplitude')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Imaginary Part
    ax2.plot(np.imag(y_gt_np), 'r-o', label='Ground Truth (y)', linewidth=2, markersize=6)
    ax2.plot(np.imag(y_hat_np), 'b--s', label='Network Output (A*x_hat)', linewidth=2, markersize=6, alpha=0.7)
    ax2.set_title(f'Measurement Domain (Imaginary Part) - Sample {SAMPLE_IDX}')
    ax2.set_xlabel('Virtual Antenna Index')
    ax2.set_ylabel('Imaginary Amplitude')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH_MEAS)
    print("Measurement plot saved.")
    plt.close()
    
    # --- PLOT 3: Ground Truth x (if dimensions don't match network output) ---
    if x_gt_np is not None and x_gt_np.shape[0] != NUM_ANGLE_BINS:
        print(f"Plotting separate Ground Truth 'x' (has different dimensions than network output)...")
        
        plt.figure(figsize=(12, 7))
        x_indices = np.arange(x_gt_np.shape[0])
        plt.plot(x_indices, np.abs(x_gt_np), 'r-o', label=f'Ground Truth |x| (shape: {x_gt_np.shape})', linewidth=2, markersize=4)
        plt.title(f'Ground Truth Reflectivity - Sample {SAMPLE_IDX}')
        plt.xlabel('Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True)
        plt.savefig('unsupervised_model_ground_truth_x.png')
        print("Ground truth x plot saved to unsupervised_model_ground_truth_x.png")
        plt.close()

if __name__ == '__main__':
    main()
