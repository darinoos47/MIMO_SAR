import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np
import h5py

def to_tensor(data):
    """
    Converts a complex numpy array to a 2-channel torch tensor 
    (channel 0: real, channel 1: imag).
    """
    if np.iscomplexobj(data):
        data_real = data.real.astype(np.float32)
        data_imag = data.imag.astype(np.float32)
        return torch.from_numpy(np.stack((data_real, data_imag), axis=0))
    else:
        return torch.from_numpy(data.astype(np.float32))

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

class MIMOSAR_Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading the FL-MIMO-SAR data.
    
    Supports both supervised and unsupervised training modes.
    
    Each item is a single measurement vector 'y' from one range bin 
    at one aperture step. Optionally returns ground truth 'x' if available.
    """
    def __init__(self, mat_file_path, return_ground_truth=False):
        print(f"Loading data from {mat_file_path}...")
        print(f"mat_file_path is: {mat_file_path}") # Added for clarity
        data = load_mat_file(mat_file_path)

        # 1. Load Measurements 'y'
        #    MATLAB shape is [N_samples, N_v] -> [2000, 8]
        measurements = data['received_signals_fft']
        
        if measurements.ndim == 3:
            # Transpose from (N_r, N_v, N_l) to (N_l, N_r, N_v)
            measurements_transposed = np.transpose(measurements, (2, 0, 1))
        else:
            # It's 2D (N_samples, N_v), so just add an N_l dimension
            measurements_transposed = measurements.reshape(1, measurements.shape[0], measurements.shape[1])

        # 2. Reshape to a flat list of samples
        #    Shape becomes [N_l * N_samples, N_v]
        self.data = measurements_transposed.reshape(-1, measurements_transposed.shape[-1])
        
        self.num_samples = self.data.shape[0]
        self.num_virtual_ant = self.data.shape[1]
        
        # 3. Load Steering Matrix 'A'
        #    MATLAB shape is [N_v, N_theta] -> [8, 1001]
        self.A_complex = data['A'].astype(np.complex64)
        self.A = to_tensor(self.A_complex) # Shape [2, 8, 1001]
        self.num_angle_bins = self.A.shape[2]
        
        # 4. Load Ground Truth 'x' if requested and available
        self.return_ground_truth = return_ground_truth
        self.x_data = None
        self.has_ground_truth = False
        
        if return_ground_truth:
            if 'x' in data:
                x_complex = data['x'].astype(np.complex64)
                print(f"  Ground truth 'x' loaded with shape: {x_complex.shape}")
                
                # Handle different possible shapes
                if x_complex.ndim == 1:
                    # Single sample: [N_theta]
                    # Replicate for all samples if needed
                    if self.num_samples == 1:
                        self.x_data = x_complex.reshape(1, -1)
                    else:
                        print(f"  WARNING: Single ground truth for {self.num_samples} samples - replicating")
                        self.x_data = np.tile(x_complex, (self.num_samples, 1))
                elif x_complex.ndim == 2:
                    # Multiple samples: [N_samples, N_theta]
                    # Check if shapes match
                    if x_complex.shape[0] == self.num_samples:
                        self.x_data = x_complex
                    else:
                        print(f"  WARNING: Ground truth samples ({x_complex.shape[0]}) != measurement samples ({self.num_samples})")
                        # Use as many as we have
                        self.x_data = x_complex[:self.num_samples]
                
                self.has_ground_truth = True
                print(f"  Stored ground truth 'x' with shape: {self.x_data.shape}")
            else:
                print(f"  WARNING: Ground truth requested but 'x' not found in data file!")
                print(f"  Will use unsupervised training mode.")
        
        print(f"Data loaded successfully.")
        print(f"  Total training samples: {self.num_samples}")
        print(f"  Virtual antennas (N_v): {self.num_virtual_ant}")
        print(f"  Angle bins (N_theta): {self.num_angle_bins}")
        print(f"  Steering matrix 'A' shape: {list(self.A.shape)}")
        print(f"  Ground truth available: {self.has_ground_truth}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the complex measurement vector y at index idx
        y_complex = self.data[idx] # Shape [N_v]
        
        # Convert to 2-channel tensor [2, N_v]
        y_tensor = to_tensor(y_complex)
        
        # Optionally return ground truth x
        if self.return_ground_truth and self.has_ground_truth:
            x_complex = self.x_data[idx] # Shape [N_theta]
            x_tensor = to_tensor(x_complex) # Shape [2, N_theta]
            return y_tensor, x_tensor
        else:
            return y_tensor
