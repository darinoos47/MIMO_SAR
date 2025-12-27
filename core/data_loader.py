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


class MultiRangeMIMOSAR_Dataset(Dataset):
    """
    Custom PyTorch Dataset for multi-range processing (without flattening ranges).
    
    Returns all range bins together as 3D tensors for joint processing by 2D CNNs.
    This enables the denoiser to capture dependencies between different ranges.
    
    Expected data structure in MAT file (single or multi-position):
        Single position:
            - received_signals_fft: [N_ranges, N_v] measurements
            - A: [N_v, N_theta] or [N_theta, N_v] steering matrix
            - x: [N_ranges, N_theta] ground truth (optional)
        Multi-position:
            - N_positions: Number of radar positions
            - position_1, position_2, ...: Each contains above fields
    
    Each sample is the full measurement matrix for one position: [N_ranges, 2, N_v]
    Total samples = N_positions (not N_positions × N_ranges)
    """
    def __init__(self, mat_file_path, return_ground_truth=False):
        return_ground_truth = True  # Always load GT if available
        print(f"Loading multi-range data from {mat_file_path}...")
        data = load_mat_file(mat_file_path)
        
        # Check if this is multi-position data
        is_multi_position = 'N_positions' in data
        
        if is_multi_position:
            N_positions = int(data['N_positions'])
            print(f"  Multi-position data detected: {N_positions} positions")
            
            # Extract data from each position (keep ranges together)
            self.measurements_list = []
            self.ground_truth_list = []
            
            for pos_idx in range(1, N_positions + 1):
                field_name = f'position_{pos_idx}'
                
                if field_name not in data:
                    print(f"  WARNING: {field_name} not found, skipping...")
                    continue
                
                pos_data = data[field_name]
                
                # Extract measurements [N_ranges, N_v]
                measurements = pos_data['received_signals_fft']
                self.measurements_list.append(measurements)
                
                # Extract ground truth if available [N_ranges, N_theta]
                if return_ground_truth and 'x_polar' in pos_data:
                    x_polar = pos_data['x_polar']
                    self.ground_truth_list.append(x_polar)
            
            # Load steering matrix A from first position
            first_pos_data = data['position_1']
            A_from_matlab = first_pos_data['A']  # [N_theta, N_v] from MATLAB
            A = A_from_matlab.T  # Transpose to [N_v, N_theta]
            
            self.num_samples = len(self.measurements_list)
            
        else:
            # Single position: treat as one sample with all ranges
            print(f"  Single position data detected")
            
            measurements = data['received_signals_fft']
            
            # Handle different shapes
            if measurements.ndim == 2:
                # [N_ranges, N_v]
                self.measurements_list = [measurements]
            elif measurements.ndim == 3:
                # [N_ranges, N_v, N_samples] - split along sample dimension
                N_samples = measurements.shape[2]
                self.measurements_list = [measurements[:, :, i] for i in range(N_samples)]
            else:
                raise ValueError(f"Unexpected measurements shape: {measurements.shape}")
            
            # Load steering matrix
            A_complex = data['A'].astype(np.complex64)
            if A_complex.shape[0] < A_complex.shape[1]:
                # [N_v, N_theta]
                A = A_complex
            else:
                # [N_theta, N_v] - transpose
                A = A_complex.T
            
            # Ground truth
            self.ground_truth_list = []
            if return_ground_truth and 'x' in data:
                x = data['x']
                if x.ndim == 2:
                    # [N_ranges, N_theta]
                    self.ground_truth_list = [x]
                elif x.ndim == 3:
                    # [N_ranges, N_theta, N_samples]
                    N_samples = x.shape[2]
                    self.ground_truth_list = [x[:, :, i] for i in range(N_samples)]
            
            self.num_samples = len(self.measurements_list)
        
        # Store metadata
        self.A_complex = A.astype(np.complex64)
        self.A = to_tensor(self.A_complex)  # Shape [2, N_v, N_theta]
        self.num_virtual_ant = self.A.shape[1]
        self.num_angle_bins = self.A.shape[2]
        
        # *** NEW: Find max range bins and pad all samples ***
        max_ranges = max(meas.shape[0] for meas in self.measurements_list)
        self.num_ranges = max_ranges
        
        # Store original range counts for masking
        self.valid_range_counts = []
        
        # Pad measurements to max_ranges
        padded_measurements = []
        for meas in self.measurements_list:
            N_ranges = meas.shape[0]
            self.valid_range_counts.append(N_ranges)
            if N_ranges < max_ranges:
                # Pad with zeros: [N_ranges, N_v] -> [max_ranges, N_v]
                pad_size = max_ranges - N_ranges
                meas_padded = np.pad(meas, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
                padded_measurements.append(meas_padded)
            else:
                padded_measurements.append(meas)
        self.measurements_list = padded_measurements
        
        # Pad ground truth if available
        if self.ground_truth_list:
            padded_gt = []
            for gt in self.ground_truth_list:
                N_ranges = gt.shape[0]
                if N_ranges < max_ranges:
                    # Pad with zeros: [N_ranges, N_theta] -> [max_ranges, N_theta]
                    pad_size = max_ranges - N_ranges
                    gt_padded = np.pad(gt, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
                    padded_gt.append(gt_padded)
                else:
                    padded_gt.append(gt)
            self.ground_truth_list = padded_gt
        
        # Handle ground truth
        self.return_ground_truth = return_ground_truth
        self.has_ground_truth = len(self.ground_truth_list) == self.num_samples
        
        print(f"Multi-range data loaded successfully.")
        print(f"  Total samples: {self.num_samples}")
        print(f"  Range bins per sample: {self.num_ranges} (padded to max)")
        print(f"  Virtual antennas (N_v): {self.num_virtual_ant}")
        print(f"  Angle bins (N_theta): {self.num_angle_bins}")
        print(f"  Steering matrix 'A' shape: {list(self.A.shape)}")
        print(f"  Ground truth available: {self.has_ground_truth}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get all range measurements for this sample
        y_complex = self.measurements_list[idx]  # Shape [N_ranges, N_v]
        
        # Convert to tensor: [N_ranges, 2, N_v]
        y_tensor_list = []
        for r_idx in range(y_complex.shape[0]):
            y_r = to_tensor(y_complex[r_idx])  # [2, N_v]
            y_tensor_list.append(y_r)
        y_tensor = torch.stack(y_tensor_list, dim=0)  # [N_ranges, 2, N_v]
        
        # Create mask for valid ranges: [N_ranges]
        # 1.0 for valid ranges, 0.0 for padded ranges
        valid_count = self.valid_range_counts[idx]
        mask = torch.zeros(y_tensor.shape[0], dtype=torch.float32)
        mask[:valid_count] = 1.0
        
        # Optionally return ground truth x
        if self.return_ground_truth and self.has_ground_truth:
            x_complex = self.ground_truth_list[idx]  # Shape [N_ranges, N_theta]
            
            # Convert to tensor: [N_ranges, 2, N_theta]
            x_tensor_list = []
            for r_idx in range(x_complex.shape[0]):
                x_r = to_tensor(x_complex[r_idx])  # [2, N_theta]
                x_tensor_list.append(x_r)
            x_tensor = torch.stack(x_tensor_list, dim=0)  # [N_ranges, 2, N_theta]
            
            return y_tensor, x_tensor, mask
        else:
            return y_tensor, mask


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


def load_dataset_multirange(mat_file_path, return_ground_truth=False):
    """
    Load dataset for multi-range processing (all ranges together as 3D tensors).
    
    Use this when training with use_multi_range_denoiser=True.
    Returns MultiRangeMIMOSAR_Dataset which provides samples of shape:
        - Measurements: [N_ranges, 2, N_v]
        - Ground truth: [N_ranges, 2, N_theta]
    
    Args:
        mat_file_path: Path to .mat file
        return_ground_truth: Whether to load ground truth if available
    
    Returns:
        MultiRangeMIMOSAR_Dataset object
    
    Example:
        >>> dataset = load_dataset_multirange('data/data_training_sar.mat')
        Multi-position data detected: 10 positions
        >>> y, x = dataset[0]
        >>> print(y.shape)  # [57, 2, 8] - all ranges together
    """
    return MultiRangeMIMOSAR_Dataset(mat_file_path, return_ground_truth)
