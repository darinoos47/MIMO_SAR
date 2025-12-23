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
                data[k] = _read_h5py_item(f[k])
    return data


def _read_h5py_item(item):
    """
    Recursively read an h5py item (Dataset or Group).
    
    Args:
        item: h5py Dataset or Group
    
    Returns:
        Loaded data (numpy array, dict for Groups, or complex array)
    """
    if isinstance(item, h5py.Group):
        # This is a struct in MATLAB - load it as a dictionary
        result = {}
        for key in item.keys():
            result[key] = _read_h5py_item(item[key])
        return result
    elif isinstance(item, h5py.Dataset):
        # This is a regular array
        v = item[()]  # Load data
        
        # Handle transpose: h5py loads as [dim1, dim0]
        # We need [dim0, dim1] to match MATLAB
        if v.ndim == 2:
            v = v.T
        
        # Handle complex data saved as a struct with 'real' and 'imag' fields
        if v.dtype.names and ('real' in v.dtype.names) and ('imag' in v.dtype.names):
            v = v['real'] + 1j * v['imag']
        
        return v
    else:
        # Fallback for unknown types
        return item[()]

class MIMOSAR_Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading the FL-MIMO-SAR data.
    
    Supports both supervised and unsupervised training modes.
    
    Each item is a single measurement vector 'y' from one range bin 
    at one aperture step. Optionally returns ground truth 'x' if available.
    """
    def __init__(self, mat_file_path, return_ground_truth=False):
        return_ground_truth = True
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


class MIMOSAR_MultiPosition_Dataset(Dataset):
    """
    Custom PyTorch Dataset for multi-position SAR data.
    
    Flattens measurements from all radar positions into a single training dataset.
    Each position contributes its range measurements as independent samples.
    
    Expected data structure in MAT file:
        - N_positions: Number of radar positions
        - position_1, position_2, ..., position_N: Each contains:
            - received_signals_fft: [N_ranges, N_v] measurements
            - A: [N_theta, N_v] steering matrix (transposed)
            - x_polar: [N_ranges, N_theta] ground truth (optional)
    
    The dataset treats each range bin from each position as an independent sample.
    Total samples = N_positions × N_ranges
    """
    def __init__(self, mat_file_path, return_ground_truth=False):
        return_ground_truth = True  # Always load GT if available
        print(f"Loading multi-position SAR data from {mat_file_path}...")
        data = load_mat_file(mat_file_path)
        
        # Check if this is multi-position data
        if 'N_positions' not in data:
            raise ValueError(
                f"File {mat_file_path} does not contain 'N_positions' field.\n"
                f"This is not multi-position SAR data. Use MIMOSAR_Dataset instead."
            )
        
        N_positions = int(data['N_positions'])
        print(f"  Multi-position data detected: {N_positions} positions")
        
        # Initialize storage for all positions
        all_measurements = []
        all_ground_truths = []
        
        # Extract data from each position
        for pos_idx in range(1, N_positions + 1):
            field_name = f'position_{pos_idx}'
            
            if field_name not in data:
                print(f"  WARNING: {field_name} not found, skipping...")
                continue
            
            pos_data = data[field_name]
            
            # Extract measurements [N_ranges, N_v]
            measurements = pos_data['received_signals_fft']
            all_measurements.append(measurements)
            
            # Extract ground truth if available [N_ranges, N_theta]
            if return_ground_truth and 'x_polar' in pos_data:
                x_polar = pos_data['x_polar']
                all_ground_truths.append(x_polar)
        
        # Flatten all positions into single dataset
        # Shape: [total_samples, N_v] where total_samples = N_positions × N_ranges
        self.data = np.vstack(all_measurements)
        self.num_samples = self.data.shape[0]
        self.num_virtual_ant = self.data.shape[1]
        
        # Load steering matrix A from first position
        # All positions should have the same A (same angle grid)
        # MATLAB saves as A' (transposed), so we get [N_theta, N_v] after h5py load
        # We need to transpose to [N_v, N_theta] for our convention
        first_pos_data = data['position_1']
        A_from_matlab = first_pos_data['A']  # [N_theta, N_v] from MATLAB
        A = A_from_matlab.T  # Transpose to [N_v, N_theta]
        
        self.A_complex = A.astype(np.complex64)
        self.A = to_tensor(self.A_complex)  # Shape [2, N_v, N_theta]
        self.num_angle_bins = self.A.shape[2]
        
        # Handle ground truth
        self.return_ground_truth = return_ground_truth
        self.x_data = None
        self.has_ground_truth = False
        
        if return_ground_truth and len(all_ground_truths) > 0:
            # Flatten ground truth from all positions
            self.x_data = np.vstack(all_ground_truths)
            
            if self.x_data.shape[0] == self.num_samples:
                self.has_ground_truth = True
                print(f"  Ground truth loaded for all positions")
            else:
                print(f"  WARNING: GT samples ({self.x_data.shape[0]}) != total samples ({self.num_samples})")
        
        print(f"Multi-position data loaded successfully.")
        print(f"  Total training samples: {self.num_samples} (from {N_positions} positions)")
        print(f"  Virtual antennas (N_v): {self.num_virtual_ant}")
        print(f"  Angle bins (N_theta): {self.num_angle_bins}")
        print(f"  Steering matrix 'A' shape: {list(self.A.shape)}")
        print(f"  Ground truth available: {self.has_ground_truth}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get the complex measurement vector y at index idx
        y_complex = self.data[idx]  # Shape [N_v]
        
        # Convert to 2-channel tensor [2, N_v]
        y_tensor = to_tensor(y_complex)
        
        # Optionally return ground truth x
        if self.return_ground_truth and self.has_ground_truth:
            x_complex = self.x_data[idx]  # Shape [N_theta]
            x_tensor = to_tensor(x_complex)  # Shape [2, N_theta]
            return y_tensor, x_tensor
        else:
            return y_tensor


def load_dataset_auto(mat_file_path, return_ground_truth=False):
    """
    Automatically detect and load the appropriate dataset type.
    
    Detects whether the file contains single-position or multi-position data
    and returns the appropriate Dataset object.
    
    Args:
        mat_file_path: Path to .mat file
        return_ground_truth: Whether to load ground truth if available
    
    Returns:
        Dataset object (MIMOSAR_Dataset or MIMOSAR_MultiPosition_Dataset)
    
    Example:
        >>> dataset = load_dataset_auto('data/data_training_sar.mat')
        Auto-detected: Multi-position SAR data
        Multi-position data detected: 10 positions
        >>> print(len(dataset))  # 2560 (10 positions × 256 ranges)
    """
    # Peek at file to detect type
    data = load_mat_file(mat_file_path)
    
    if 'N_positions' in data:
        print(f"Auto-detected: Multi-position SAR data")
        return MIMOSAR_MultiPosition_Dataset(mat_file_path, return_ground_truth)
    else:
        print(f"Auto-detected: Single-position data")
        return MIMOSAR_Dataset(mat_file_path, return_ground_truth)
