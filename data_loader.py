import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import numpy as np

def to_tensor(data):
    """
    Converts a complex numpy array to a 2-channel torch tensor 
    (channel 0: real, channel 1: imag).
    """
    if np.iscomplexobj(data):
        # Stack real and imag parts as channels
        data_real = data.real.astype(np.float32)
        data_imag = data.imag.astype(np.float32)
        return torch.from_numpy(np.stack((data_real, data_imag), axis=0))
    else:
        # If already real, just convert
        return torch.from_numpy(data.astype(np.float32))

class MIMOSAR_Dataset(Dataset):
    """
    Custom PyTorch Dataset for loading the FL-MIMO-SAR data.
    
    Each item is a single measurement vector 'y' from one range bin 
    at one aperture step.
    
    The complex vector y of shape [N_v] is returned as a
    real tensor of shape [2, N_v].
    """
    def __init__(self, mat_file_path):
        print(f"Loading data from {mat_file_path}...")
        try:
            data = scipy.io.loadmat(mat_file_path)
        except NotImplementedError:
            print("Failed to read .mat file. Trying with h5py...")
            import h5py
            data = {}
            with h5py.File(mat_file_path, 'r') as f:
                data['A'] = f['A'][()]
                data['received_signals_fft'] = f['received_signals_fft'][()]
                # object is optional for unsupervised training
                if 'object' in f:
                    data['object'] = f['object'][()]
            
            # h5py might load as (imag, real), fix if necessary
            if data['A'].dtype.names:
                 data['A'] = data['A']['real'] + 1j * data['A']['imag']
            if data['received_signals_fft'].dtype.names:
                data['received_signals_fft'] = data['received_signals_fft']['real'] + 1j * data['received_signals_fft']['imag']


        # measurements shape from MATLAB: (N_r, N_v, N_l)
        # N_r = num_samples (range), N_v = virtual_ant_num, N_l = scan_steps
        measurements = data['received_signals_fft']
        
        # We want our dataset to be a flat list of all (N_r * N_l) measurements.
        # Each measurement is a vector of length N_v.
        #
        # 1. Transpose from (N_r, N_v, N_l) to (N_l, N_r, N_v)
        #    This groups by aperture step, then range bin.
        measurements_transposed = np.transpose(measurements, (2, 0, 1))
        
        # 2. Reshape to (N_l * N_r, N_v)
        self.num_samples = measurements.shape[0] * measurements.shape[2]
        self.num_virtual_ant = measurements.shape[1]
        self.data = measurements_transposed.reshape(self.num_samples, self.num_virtual_ant)
        
        # Store steering matrix 'A'
        self.A_complex = data['A'].astype(np.complex64)
        self.A = to_tensor(self.A_complex) # Shape [2, N_v, N_theta]

        self.num_angle_bins = self.A.shape[2]
        
        print(f"Data loaded successfully.")
        print(f"  Total training samples (N_l * N_r): {self.num_samples}")
        print(f"  Virtual antennas (N_v): {self.num_virtual_ant}")
        print(f"  Angle bins (N_theta): {self.num_angle_bins}")
        print(f"  Steering matrix 'A' shape: {list(self.A.shape)}")


    def __len__(self):
        # Total number of samples (N_l * N_r)
        return self.num_samples

    def __getitem__(self, idx):
        # Get the complex measurement vector y at index idx
        y_complex = self.data[idx] # Shape [N_v]
        
        # Convert to 2-channel tensor [2, N_v]
        y_tensor = to_tensor(y_complex)
        
        return y_tensor

# --- Example Usage (main block) ---
if __name__ == '__main__':
    
    # 1. Create the Dataset
    dataset = MIMOSAR_Dataset('FL_MIMO_SAR_data.mat')
    
    # 2. Create the DataLoader
    # This will batch the samples for training
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    # 3. Test by fetching one batch
    y_batch = next(iter(dataloader))
    
    print("\n--- DataLoader Test ---")
    print(f"Batch shape: {list(y_batch.shape)}")
    print(f"(Batch_size, Channels, N_v)")
