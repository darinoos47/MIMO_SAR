import torch
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os
import h5py

# Add project root to path
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)


from core.models import DBPNet
from core.data_loader import MIMOSAR_Dataset, to_tensor # Need to_tensor
from torch.utils.data import DataLoader
from core.utils import complex_matmul # Need complex_matmul

# -----------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------
MAT_FILE = '../../data/FL_MIMO_SAR_data.mat'
MODEL_PATH = '../../checkpoints/dbp_model.pth' # The unsupervised model
OUTPUT_DIR = 'test_results'
NUM_PLOTS_TO_SAVE = 5 # Save plots for the first 5 samples

# --- Model Hyperparameters ---
# !!! CRITICAL: These MUST MATCH the values from your 'train.py' script !!!
NUM_UNROLLS = 5      
NUM_ADMM_STEPS = 3   

# --- Plotting Parameters (from MATLAB script) ---
# NOTE: NUM_ANGLE_BINS will be determined from actual data (A matrix shape)
START_ANGLE_DEG = 25.0
END_ANGLE_DEG = -25.0

# -----------------------------------------------------------------
# 2. Helper Functions (copied here to avoid changing utils.py)
# -----------------------------------------------------------------
def load_mat_file(filepath):
    """
    Loads .mat file, handling v7.3 files with h5py,
    correcting for transposes and complex structs.
    """
    try:
        data = scipy.io.loadmat(filepath)
    except NotImplementedError:
        print(f"Reading v7.3 MAT file {filepath} with h5py...")
        data = {}
        with h5py.File(filepath, 'r') as f:
            for k in f.keys():
                v = f[k]
                # Handle transpose: h5py loads as [dim1, dim0]
                # We need [dim0, dim1]
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
# 3. Main Test Function
# -----------------------------------------------------------------
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------------------
    print("Loading test dataset...")
    # Use the UNCHANGED data loader (only provides 'y')
    dataset = MIMOSAR_Dataset(MAT_FILE)
    
    # Use batch_size=1 and shuffle=False to iterate one by one
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # --- Manually Load Ground Truth 'x' ---
    print("Manually loading ground truth 'x' dataset...")
    data = load_mat_file(MAT_FILE)
    x_gt_all_np = data['x'].astype(np.complex64) # Shape [2000, 1001]
    # Convert all 2000 samples to a tensor and move to device
    x_gt_all_tensor = to_tensor(x_gt_all_np).to(device) # Shape [2, 2000, 1001]
    # Permute to [2000, 2, 1001] to be batch-like
    x_gt_all_tensor = x_gt_all_tensor.permute(1, 0, 2)
    print(f"Loaded {x_gt_all_tensor.shape[0]} ground truth 'x' samples.")
    
    # Load A
    A_tensor = dataset.A.to(device)
    A_batch_tensor = A_tensor.unsqueeze(0) # For complex_matmul
    
    # Get NUM_ANGLE_BINS from the steering matrix shape
    # A_tensor shape: [2, N_v, N_theta]
    NUM_ANGLE_BINS = A_tensor.shape[2]
    print(f"Number of angle bins from data: {NUM_ANGLE_BINS}")

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
    model.eval()
    print("Model loaded successfully.")

    # -----------------------------------------------------------------
    # 3. Run Full Dataset Evaluation
    # -----------------------------------------------------------------
    print(f"Running evaluation on {len(dataset)} test samples...")
    
    image_criterion = nn.MSELoss()
    meas_criterion = nn.MSELoss()
    
    image_losses = []
    meas_losses = []
    
    with torch.no_grad():
        # Enumerate() gives us the index 'i' (0, 1, 2, ...)
        # Since shuffle=False, 'i' corresponds to the sample index
        for i, y_batch in enumerate(dataloader):
            y_batch = y_batch.to(device)
            
            # --- Manually get the corresponding x_batch ---
            # x_gt_all_tensor is [2000, 2, 1001]
            # We select index 'i' to get [1, 2, 1001]
            x_batch = x_gt_all_tensor[i:i+1] 

            # --- Run Inference ---
            x_hat_batch = model(y_batch) # [1, 2, 1001]
            y_hat_batch = complex_matmul(A_batch_tensor, x_hat_batch) # [1, 2, 8]
            
            # --- Calculate Losses ---
            img_loss = image_criterion(x_hat_batch, x_batch)
            mes_loss = meas_criterion(y_hat_batch, y_batch)
            
            image_losses.append(img_loss.item())
            meas_losses.append(mes_loss.item())
            
            # --- Save Example Plots ---
            if i < NUM_PLOTS_TO_SAVE:
                print(f"  Saving plots for sample {i}...")
                
                # Convert to numpy
                x_true_np = tensor_to_complex_np(x_batch)
                x_hat_np = tensor_to_complex_np(x_hat_batch)
                y_true_np = tensor_to_complex_np(y_batch)
                y_hat_np = tensor_to_complex_np(y_hat_batch)
                
                # --- PLOT 1: Image Domain ---
                theta = np.linspace(START_ANGLE_DEG, END_ANGLE_DEG, NUM_ANGLE_BINS)
                plt.figure(figsize=(12, 7))
                plt.plot(theta, np.abs(x_true_np), 'r', label='Ground Truth (x)', linewidth=2)
                plt.plot(theta, np.abs(x_hat_np), 'b--', label='Network Output (x_hat)', linewidth=2)
                plt.title(f'Sample {i} - Image Domain')
                plt.xlabel('Angle (degree)')
                plt.ylabel('Amplitude (Abs)')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(OUTPUT_DIR, f'sample_{i}_image_plot.png'))
                plt.close()

                # --- PLOT 2: Measurement Domain ---
                plt.figure(figsize=(12, 7))
                plt.plot(np.real(y_true_np), 'r', label='Ground Truth (y)', linewidth=2)
                plt.plot(np.real(y_hat_np), 'bo', label='Network Output (A*x_hat)', markersize=6)
                plt.title(f'Sample {i} - Measurement Domain')
                plt.xlabel('Virtual Antenna Index')
                plt.ylabel('Amplitude (Real Part)')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(OUTPUT_DIR, f'sample_{i}_meas_plot.png'))
                plt.close()

    # -----------------------------------------------------------------
    # 4. Print Final Results
    # -----------------------------------------------------------------
    avg_img_loss = np.mean(image_losses)
    avg_mes_loss = np.mean(meas_losses)

    print("\n--- Full Dataset Test Complete ---")
    print(f"  Total Samples Evaluated: {len(dataset)}")
    print(f"  Average Image Domain Loss (x_hat vs x):   {avg_img_loss:.10f}")
    print(f"  Average Meas. Domain Loss (A*x_hat vs y): {avg_mes_loss:.10f}")
    print(f"  Saved {NUM_PLOTS_TO_SAVE} example plots to '{OUTPUT_DIR}/' directory.")

if __name__ == '__main__':
    main()
