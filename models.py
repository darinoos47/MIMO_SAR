import torch
import torch.nn as nn
from utils import complex_matmul, complex_conj_transpose_matmul

# -----------------------------------------------------------------
# PHASE 2 (From previous step)
# -----------------------------------------------------------------
class CNNDenoiser(nn.Module):
    """
    Implements the CNN-based denoiser with a residual architecture
    as shown in Fig. 8 of the paper. [cite: 379-385, 395]
    
    The network operates on the 1D angular reflectivity profile 'x',
    which is represented as a 2-channel tensor (real, imag).
    
    Input shape: [batch_size, 2, N_theta]
    Output shape: [batch_size, 2, N_theta]
    """
    def __init__(self, in_channels=2, out_channels=2, num_filters=32, kernel_size=3):
        super(CNNDenoiser, self).__init__()
        
        # Padding is (kernel_size - 1) / 2 = (3-1)/2 = 1 to keep size same
        padding = (kernel_size - 1) // 2
        
        # First convolutional block [cite: 379-381]
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, 
                out_channels=num_filters, 
                kernel_size=kernel_size, 
                padding=padding,
                bias=False 
            ),
            nn.BatchNorm1d(num_features=num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Second convolutional block [cite: 382-384]
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=kernel_size,
                padding=padding,
                bias=False
            ),
            nn.BatchNorm1d(num_features=num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Final convolutional layer [cite: 385]
        self.conv_final = nn.Conv1d(
            in_channels=num_filters,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True 
        )

    def forward(self, x):
        """
        Forward pass with the residual connection.
        The output of the blocks is added back to the input. [cite: 395]
        """
        identity = x
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_final(out)
        return identity + out

# -----------------------------------------------------------------
# PHASE 3 (New code)
# -----------------------------------------------------------------
class DCLayer(nn.Module):
    """
    Implements the Data Consistency (DC) layer.
    
    This layer performs one step of gradient descent on the data fidelity
    term:  L(x) = ||y - Ax||^2
    
    The update is: x_out = x_in - mu * grad(L(x_in))
    grad(L(x_in)) = A.H @ (A @ x_in - y)
    
    So: x_out = x_in - mu * A.H @ (A @ x_in - y)
    
    where 'mu' is a learnable step-size parameter.
    """
    def __init__(self, A_tensor):
        super(DCLayer, self).__init__()
        
        # Register 'A' as a non-trainable buffer
        self.register_buffer('A', A_tensor)
        
        # 'mu' is a learnable parameter, initialized to a small value
        self.mu = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_in, y):
        """
        x_in: [batch, 2, N_theta] (current estimate of x from denoiser)
        y:    [batch, 2, N_v]     (original measurement vector)
        """
        
        # 1. Project to measurement domain: A @ x_in
        Ax = complex_matmul(self.A, x_in)
        
        # 2. Compute residual in measurement domain: (A @ x_in - y)
        residual = Ax - y
        
        # 3. Project residual back to image domain: A.H @ (A @ x_in - y)
        correction = complex_conj_transpose_matmul(self.A, residual)
        
        # 4. Perform the gradient descent update
        # We constrain mu to be positive using relu
        x_out = x_in - torch.relu(self.mu) * correction
        
        return x_out

class DBPNet(nn.Module):
    """
    Implements the full unrolled Deep Basis Pursuit (DBP) network
    as shown in Fig. 7. [cite: 363]
    
    It alternates between a shared CNNDenoiser (28a) and a DCLayer (28b)
    for a fixed number of iterations.
    """
    def __init__(self, A_tensor, num_iterations=5):
        super(DBPNet, self).__init__()
        self.num_iterations = num_iterations
        
        # Register A and A.H as non-trainable buffers
        self.register_buffer('A', A_tensor)
        A_H_tensor = torch.stack((A_tensor[0].T, -A_tensor[1].T), dim=0)
        self.register_buffer('A_H', A_H_tensor)
        
        # 1. The single, shared CNN-based denoiser (28a) [cite: 562]
        #    Weights are shared across all iterations. [cite: 573]
        self.denoiser = CNNDenoiser()
        
        # 2. A list of Data Consistency (DC) layers (28b) [cite: 564]
        #    Each layer has its own learnable step-size 'mu'.
        self.dc_layers = nn.ModuleList(
            [DCLayer(A_tensor) for _ in range(num_iterations)]
        )

    def forward(self, y):
        """
        Forward pass through the unrolled network.
        
        y: [batch, 2, N_v] (original measurement vectors)
        """
        
        # 1. Get initial estimate x_0 = A.H @ y (Matched Filter) [cite: 769]
        x = complex_conj_transpose_matmul(self.A, y)
        
        # 2. Unroll the iterations
        for i in range(self.num_iterations):
            # Denoising step (r_n = D_w(x_{n-1})) [cite: 562]
            r = self.denoiser(x)
            
            # Data Consistency step (x_n = ...) [cite: 564, 570]
            x = self.dc_layers[i](r, y)
            
        return x

# --- Example Usage (main block) ---
if __name__ == '__main__':
    from data_loader import MIMOSAR_Dataset, DataLoader
    
    # 1. Load the dataset to get 'A'
    dataset = MIMOSAR_Dataset('../FL_MIMO_SAR_data.mat')
    A_tensor = dataset.A
    
    # 2. Get one batch of data
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    y_batch = next(iter(dataloader))

    # 3. Initialize the DBPNet model
    model = DBPNet(A_tensor, num_iterations=5)
    
    # 4. Pass the batch through the model
    x_hat = model(y_batch)
    
    print("\n--- DBPNet Test ---")
    print(f"Steering Matrix 'A' shape: {list(A_tensor.shape)}")
    print(f"Input 'y' batch shape:  {list(y_batch.shape)}")
    print(f"Output 'x' batch shape: {list(x_hat.shape)}")
    
    # Check that output shape is correct
    assert list(x_hat.shape) == [
        y_batch.shape[0], 2, A_tensor.shape[2]
    ]
    print("Test passed: Input and output shapes are correct.")
