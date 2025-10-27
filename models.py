import torch
import torch.nn as nn
from utils import (
    complex_matmul, complex_conj_transpose_matmul,
    complex_matmul_tensor, complex_batch_matmul_vec,
    complex_batch_inverse, complex_project_l2_ball
)

# -----------------------------------------------------------------
# 1. CNN Denoiser (Unchanged)
# -----------------------------------------------------------------
class CNNDenoiser(nn.Module):
    """
    Implements the CNN-based denoiser with a residual architecture
    [cite_start]as shown in Fig. 8 of the paper. [cite: 379-385, 395]
    """
    def __init__(self, in_channels=2, out_channels=2, num_filters=32, kernel_size=3):
        super(CNNDenoiser, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.conv_final = nn.Conv1d(num_filters, out_channels, kernel_size, padding=padding, bias=True)

    def forward(self, x):
        identity = x
        out = self.conv_block_1(x)
        out = self.conv_block_2(out)
        out = self.conv_final(out)
        return identity + out

# -----------------------------------------------------------------
# 2. New ADMM-based Data Consistency Layer
# -----------------------------------------------------------------
class DCLayer_ADMM(nn.Module):
    """
    Implements the Data Consistency (DC) layer using N steps of ADMM
    to solve:
    arg min_x (1/2)||x - r_n||^2  s.t. ||y - Ax||_2 <= epsilon
    
    Where r_n is the output of the denoiser.
    """
    def __init__(self, A_tensor, N_admm_steps=3):
        super(DCLayer_ADMM, self).__init__()
        self.N_admm_steps = N_admm_steps
        
        # Learnable parameters (in log-space for stability)
        # rho: ADMM penalty parameter
        # epsilon: l2-ball radius
        self.log_rho = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.01)))
        
        # Store A (shape [2, N_v, N_theta])
        self.register_buffer('A', A_tensor)
        
        # Precompute A_H (shape [2, N_theta, N_v])
        A_H = torch.stack((A_tensor[0].T, -A_tensor[1].T), dim=0)
        self.register_buffer('A_H', A_H)
        
        # Precompute A @ A.H (shape [2, N_v, N_v])
        A_A_H = complex_matmul_tensor(A_tensor, A_H)
        self.register_buffer('A_A_H', A_A_H)

    def forward(self, r_n, y, u_in):
        """
        r_n:   [batch, 2, N_theta] (denoiser output)
        y:     [batch, 2, N_v]     (original measurement)
        u_in:  [batch, 2, N_v]     (dual variable from prev iter)
        
        Returns:
        x_out: [batch, 2, N_theta]
        u_out: [batch, 2, N_v]
        """
        # Get scalar parameters
        rho = torch.exp(self.log_rho)
        epsilon = torch.exp(self.log_epsilon)
        
        # --- Precompute fixed part of x-update (SMW identity) ---
        # M = (A @ A.H + rho * I)
        I = torch.eye(self.A.shape[1], device=r_n.device) # N_v x N_v
        I_tensor = torch.stack((I, torch.zeros_like(I)), dim=0).unsqueeze(0)
        M = self.A_A_H.unsqueeze(0) + rho * I_tensor
        
        # M_inv = (A @ A.H + rho * I)^-1
        M_inv = complex_batch_inverse(M)
        
        # --- Initialize ADMM variables ---
        u = u_in
        z = y # Start z close to y
        
        # --- Run N_admm_steps ---
        for _ in range(self.N_admm_steps):
            
            # 1. x-update (using SMW)
            # x = (I + rho*A.H*A)^-1 @ (r_n + rho*A.H*(z-u))
            # x = 1/rho * (rhs - A.H @ (A.H*A+rho*I)^-1 @ A @ rhs)
            
            # Let rhs = (r_n + rho * A.H @ (z - u))
            z_minus_u = z - u
            A_H_z_u = complex_conj_transpose_matmul(self.A.unsqueeze(0), z_minus_u)
            rhs = r_n + rho * A_H_z_u
            
            # temp_x = rhs / rho
            temp_x = rhs / (rho + 1e-8)
            
            # temp_M_inv_Ax = M_inv @ (A @ temp_x)
            A_temp_x = complex_matmul(self.A.unsqueeze(0), temp_x)
            temp_M_inv_Ax = complex_batch_matmul_vec(M_inv, A_temp_x)
            
            # x = temp_x - A.H @ temp_M_inv_Ax
            x = temp_x - complex_conj_transpose_matmul(self.A.unsqueeze(0), temp_M_inv_Ax)

            # 2. z-update
            # z = project(A @ x + u) onto l2-ball(y, epsilon)
            Ax = complex_matmul(self.A.unsqueeze(0), x)
            Ax_plus_u = Ax + u
            z = complex_project_l2_ball(Ax_plus_u, y, epsilon)

            # 3. u-update
            # u = u + A @ x - z
            u = u + Ax - z
            
        return x, u

# -----------------------------------------------------------------
# 3. Updated DBP Network
# -----------------------------------------------------------------
class DBPNet(nn.Module):
    """
    Implements the full unrolled Deep Basis Pursuit (DBP) network
    as shown in Fig. 7.
    
    This is now a STATEFUL network that passes the ADMM dual
    variable 'u' between iterations.
    """
    def __init__(self, A_tensor, num_iterations=5, N_admm_steps=3):
        super(DBPNet, self).__init__()
        self.num_iterations = num_iterations
        
        # Register A as a non-trainable buffer for the initial step
        self.register_buffer('A', A_tensor)
        
        # 1. The single, shared CNN-based denoiser (28a)
        self.denoiser = CNNDenoiser()
        
        # 2. A list of stateful ADMM-based Data Consistency layers
        self.dc_layers = nn.ModuleList(
            [DCLayer_ADMM(A_tensor, N_admm_steps) for _ in range(num_iterations)]
        )

    def forward(self, y):
        """
        Forward pass through the unrolled, stateful network.
        
        y: [batch, 2, N_v] (original measurement vectors)
        """
        
        # 1. Get initial estimate x_0 = A.H @ y (Matched Filter)
        x = complex_conj_transpose_matmul(self.A.unsqueeze(0), y)
        
        # 2. Initialize the ADMM dual variable 'u'
        u = torch.zeros_like(y)
        
        # 3. Unroll the iterations
        for i in range(self.num_iterations):
            # Denoising step (r_n = D_w(x_{n-1}))
            r = self.denoiser(x)
            
            # Stateful Data Consistency step (x_n, u_n = ...)
            x, u = self.dc_layers[i](r, y, u)
            
        return x

# --- Example Usage (main block) ---
if __name__ == '__main__':
    from data_loader import MIMOSAR_Dataset, DataLoader
    
    print("--- Testing new ADMM-based DBPNet ---")
    
    # 1. Load the dataset to get 'A'
    dataset = MIMOSAR_Dataset('../FL_MIMO_SAR_data.mat')
    A_tensor = dataset.A
    
    # 2. Get one batch of data
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    y_batch = next(iter(dataloader))

    # 3. Initialize the DBPNet model with N_admm_steps=3
    model = DBPNet(A_tensor, num_iterations=5, N_admm_steps=3)
    
    # 4. Pass the batch through the model
    x_hat = model(y_batch)
    
    print(f"Steering Matrix 'A' shape: {list(A_tensor.shape)}")
    print(f"Input 'y' batch shape:  {list(y_batch.shape)}")
    print(f"Output 'x' batch shape: {list(x_hat.shape)}")
    
    assert list(x_hat.shape) == [
        y_batch.shape[0], 2, A_tensor.shape[2]
    ]
    print("Test passed: Input and output shapes are correct.")
