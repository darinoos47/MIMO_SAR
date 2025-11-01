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
    
    FOR FULL UNSUPERVISED TRAINING
    
    Each item is a single measurement vector 'y' from one range bin 
    at one aperture step.
    """
    def __init__(self, mat_file_path):
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
        
        print(f"Data loaded successfully.")
        print(f"  Total training samples: {self.num_samples}")
        print(f"  Virtual antennas (N_v): {self.num_virtual_ant}")
        print(f"  Angle bins (N_theta): {self.num_angle_bins}")
        print(f"  Steering matrix 'A' shape: {list(self.A.shape)}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get the complex measurement vector y at index idx
        y_complex = self.data[idx] # Shape [N_v]
        
        # Convert to 2-channel tensor [2, N_v]
        y_tensor = to_tensor(y_complex)
        
        return y_tensor
