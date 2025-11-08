import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import h5py

from models import DCLayer_ADMM
from data_loader import to_tensor
from utils import complex_matmul, tensor_to_complex_np, complex_conj_transpose_matmul

# -----------------------------------------------------------------
# 1. Configuration (from your MATLAB script)
# -----------------------------------------------------------------
MAT_FILE = 'data_test_ADMM.mat' # Assumes this has A, x, and y
PLOT_SAVE_PATH_X = 'admm_test_plot_x.png'
PLOT_SAVE_PATH_Y = 'admm_test_plot_y.png'

# Parameters from your MATLAB script
RHO_VALUE = 1e3
EPSILON_VALUE = 1e-3
NUM_ITER = 1 # We will set the layer's N_admm_steps to 1

def load_mat_file(filepath):
    """Loads .mat file, handling v7.3 files."""
    try:
        data = scipy.io.loadmat(filepath)
    except NotImplementedError:
        print(f"Reading v7.3 MAT file {filepath} with h5py...")
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

# -----------------------------------------------------------------
# 2. Main Test Function
# -----------------------------------------------------------------
def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Using device: {device} ---")

    # -----------------------------------------------------------------
    # 1. Load Data (A, x, and y)
    # -----------------------------------------------------------------
    print(f"Loading data from {MAT_FILE}...")
    data = load_mat_file(MAT_FILE)
    
    # Ground truth reflectivity 'x' (this will be our 'r')
    x_gt_np = data['x'].flatten().astype(np.complex64)
    print(f"x_gt_np.shape is: {x_gt_np.shape}")
    
    # Ground truth measurement 'y'
    y_gt_np = data['y'].flatten().astype(np.complex64)
    
    # Normalized steering matrix 'A'
    A_np = data['A'].astype(np.complex64)
    A_tensor = to_tensor(A_np).to(device)
    A_batch_tensor = A_tensor.unsqueeze(0) # Shape [1, 2, 8, 1001]

    # -----------------------------------------------------------------
    # 2. Instantiate and Configure the ADMM Layer
    # -----------------------------------------------------------------
    print(f"Configuring DCLayer_ADMM with N_admm_steps = {NUM_ITER}...")
    
    admm_layer = DCLayer_ADMM(
        A_tensor, 
        N_admm_steps=NUM_ITER
    ).to(device)
    
    # Manually set the layer's parameters to match MATLAB
    admm_layer.log_rho.data = torch.log(torch.tensor(RHO_VALUE, device=device))
    admm_layer.log_epsilon.data = torch.log(torch.tensor(EPSILON_VALUE, device=device))
    
    print(f"  Set rho = {RHO_VALUE}")
    print(f"  Set epsilon = {EPSILON_VALUE}")
    
    # Set to eval mode
    admm_layer.eval()

    # -----------------------------------------------------------------
    # 3. Prepare Inputs (as per your script)
    # -----------------------------------------------------------------
    # r = x (the ground truth)
    r_n_tensor = to_tensor(x_gt_np).unsqueeze(0).to(device) # Shape [1, 2, 1001]
    print(f"r_n_tensor shape is: {r_n_tensor.shape}")
    
    # y = y (the measurement)
    y_tensor = to_tensor(y_gt_np).unsqueeze(0).to(device) # Shape [1, 2, 8]
    
    # r = Ah*y (matched filter result as initial guess)
    r_n_tensor = complex_conj_transpose_matmul(A_batch_tensor, y_tensor)
    
    # u0 = zeros
    u_in_tensor = torch.zeros_like(y_tensor) # Shape [1, 2, 8]

    # -----------------------------------------------------------------
    # 4. Run One Step of the ADMM Layer
    # -----------------------------------------------------------------
    print("Running one forward pass of DCLayer_ADMM...")
    
    with torch.no_grad():
        # Get the output x_estimated
        x_estimated_tensor, u_out_tensor = admm_layer(r_n_tensor, y_tensor, u_in_tensor)
        
    # Convert for plotting
    x_estimated_np = tensor_to_complex_np(x_estimated_tensor) # No .squeeze() needed
    
    print("Inference complete.")

    # -----------------------------------------------------------------
    # 5. Plot the Results (same as MATLAB)
    # -----------------------------------------------------------------
    print(f"Plotting results to {PLOT_SAVE_PATH_X} and {PLOT_SAVE_PATH_Y}...")

    # --- Figure 1: Plot x ---
    plt.figure(figsize=(12, 7))
    plt.hold_on = True # Keep this for compatibility, though not needed in modern matplotlib
    # plt.plot(np.abs(x_gt_np), linewidth=1.5, label='x prior (r = x_true)')
    plt.plot(np.abs(np.abs(r_n_tensor[0,0,:].cpu() + 1j *r_n_tensor[0, 1, :].cpu())), linewidth=1.5, label='x prior (r = x_true)')
    plt.plot(np.abs(x_estimated_np), 'o', linewidth=1.5, markersize=4, label='x estimated ADMM')
    plt.legend(["x prior (r = x_true)", "x estimated ADMM"])
    plt.grid(True)
    plt.title('Reflectivity Comparison (Python)')
    plt.savefig(PLOT_SAVE_PATH_X)
    
    # --- Figure 2: Plot y ---
    # We need A*r and A*x_estimated in Python
    with torch.no_grad():
        A_r_tensor = complex_matmul(A_batch_tensor, r_n_tensor)
        A_x_est_tensor = complex_matmul(A_batch_tensor, x_estimated_tensor)
    
    # Convert to numpy
    y_true_np = y_gt_np
    A_r_np = tensor_to_complex_np(A_r_tensor).squeeze()
    A_x_est_np = tensor_to_complex_np(A_x_est_tensor).squeeze()
    
    plt.figure(figsize=(12, 7))
    plt.hold_on = True
    plt.plot(np.real(y_true_np), linewidth=1.5, label='A*x_true')
    plt.plot(np.real(A_r_np), linewidth=1.5, label='A*x_initial')
    plt.plot(np.real(A_x_est_np), 'o', linewidth=1.5, markersize=4, label='A*x_estimated')
    plt.legend(["A*x_true", "A*x_initial", "A*x_estimated"])
    plt.grid(True)
    plt.title('Measurement Domain Comparison (Python)')
    plt.savefig(PLOT_SAVE_PATH_Y)

    print("Plots saved.")

if __name__ == '__main__':
    main()
