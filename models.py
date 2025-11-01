import torch
import torch.nn as nn
from utils import (
    complex_matmul, complex_conj_transpose_matmul,
    complex_matmul_tensor, complex_batch_matmul_vec,
    complex_batch_inverse, complex_project_l2_ball
)

# -----------------------------------------------------------------
# 1. CNN Denoiser (Unchanged - keep as is)
# -----------------------------------------------------------------
class CNNDenoiser(nn.Module):
    # ... (no changes here) ...
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
# 2. ADMM Layer (Modified to match new MATLAB code - NO SMW)
# -----------------------------------------------------------------
class DCLayer_ADMM(nn.Module):
    """
    Implements the Data Consistency (DC) layer using N steps of ADMM
    by *directly solving* the (I + rho*A.H*A) system, as per
    the user's new MATLAB file.
    
    This version does NOT use the SMW identity.
    """
    def __init__(self, A_tensor, N_admm_steps=3):
        super(DCLayer_ADMM, self).__init__()
        self.N_admm_steps = N_admm_steps
        
        # Learnable parameters
        self.log_rho = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.01)))
        
        # Store A (shape [2, N_v, N_theta])
        self.register_buffer('A', A_tensor)
        
        # Precompute A_H (shape [2, N_theta, N_v])
        A_H = torch.stack((A_tensor[0].T, -A_tensor[1].T), dim=0)
        self.register_buffer('A_H', A_H)
        
        # Precompute A.H @ A (shape [2, N_theta, N_theta])
        # This is the large N x N matrix
        A_H_A = complex_matmul_tensor(A_H, A_tensor)
        self.register_buffer('A_H_A', A_H_A)

    def forward(self, r_n, y, u_in):
        """
        r_n:   [batch, 2, N_theta] (denoiser output)
        y:     [batch, 2, N_v]     (original measurement)
        u_in:  [batch, 2, N_v]     (dual variable from prev iter)
        """
        # Get scalar parameters
        rho = torch.exp(self.log_rho)
        epsilon = torch.exp(self.log_epsilon)
        
        # Get batch size and N_theta
        batch_size = r_n.shape[0]
        N_theta = self.A.shape[2]
        
        # --- Precompute M_mat = (I + rho * A.H @ A) ---
        # This is (1, 2, N_theta, N_theta)
        I = torch.eye(N_theta, device=r_n.device)
        I_tensor = torch.stack((I, torch.zeros_like(I)), dim=0).unsqueeze(0)
        M_mat = I_tensor + rho * self.A_H_A.unsqueeze(0)
        
        # --- Initialize ADMM variables ---
        u = u_in
        z = y 
        x = r_n # Start x at the denoiser output
        
        A_batch = self.A.unsqueeze(0)
        
        # --- Run N_admm_steps ---
        for _ in range(self.N_admm_steps):
            
            # -----------------------------------------------
            # 1. x-update (Direct Solve, matching MATLAB)
            # Solves (I + rho*A.H*A) * x = (r_n + rho*A.H*(z-u))
            # -----------------------------------------------
            
            # --- Build rhs = (r_n + rho*A.H*(z-u)) ---
            z_minus_u = z - u
            A_H_z_u = complex_conj_transpose_matmul(A_batch, z_minus_u)
            rhs = r_n + rho * A_H_z_u # Shape: [B, 2, N_theta]
            
            # --- Solve M_mat * x = rhs ---
            # We build the [B, 2N, 2N] real block matrix system
            
            # M_mat: [1, 2, N, N] -> M_r, M_i: [B, N, N]
            M_r = M_mat[:, 0].expand(batch_size, -1, -1)
            M_i = M_mat[:, 1].expand(batch_size, -1, -1)
            
            # rhs: [B, 2, N] -> R_r, R_i: [B, N]
            R_r = rhs[:, 0]
            R_i = rhs[:, 1]
            
            # Build M_block [B, 2N, 2N]
            top_row = torch.cat([M_r, -M_i], dim=2)
            bot_row = torch.cat([M_i, M_r], dim=2)
            M_block = torch.cat([top_row, bot_row], dim=1)
            
            # Build R_block [B, 2N, 1]
            R_block = torch.cat([R_r, R_i], dim=1).unsqueeze(-1)
            
            # Solve the large real system
            x_block = torch.linalg.solve(M_block, R_block) # Shape [B, 2N, 1]
            
            # Unpack x
            x_r = x_block[:, :N_theta, 0].unsqueeze(1)
            x_i = x_block[:, N_theta:, 0].unsqueeze(1)
            x = torch.cat([x_r, x_i], dim=1) # Shape [B, 2, N_theta]
            
            # -----------------------------------------------
            # 2. z-update (Matches MATLAB)
            # z = y + L2Proj(A*x+u-y, epsilon)
            # -----------------------------------------------
            Ax = complex_matmul(A_batch, x)
            Ax_plus_u_minus_y = Ax + u - y
            
            # Project the *residual* vector
            z_residual = complex_project_l2_ball(Ax_plus_u_minus_y, torch.zeros_like(y), epsilon)
            
            # Add back to y
            z = y + z_residual

            # -----------------------------------------------
            # 3. u-update (Matches MATLAB)
            # u = u + A*x - z
            # -----------------------------------------------
            u = u + Ax - z
            
        return x, u

# -----------------------------------------------------------------
# 3. Updated DBP Network (Unchanged - keep as is)
# -----------------------------------------------------------------
class DBPNet(nn.Module):
    # ... (no changes here) ...
    def __init__(self, A_tensor, num_iterations=5, N_admm_steps=3):
        super(DBPNet, self).__init__()
        self.num_iterations = num_iterations
        
        self.register_buffer('A', A_tensor)
        self.denoiser = CNNDenoiser()
        
        self.dc_layers = nn.ModuleList(
            [DCLayer_ADMM(A_tensor, N_admm_steps) for _ in range(num_iterations)]
        )

    def forward(self, y):
        x = complex_conj_transpose_matmul(self.A.unsqueeze(0), y)
        u = torch.zeros_like(y)
        
        for i in range(self.num_iterations):
            r = self.denoiser(x)
            x, u = self.dc_layers[i](r, y, u)
            
        return x
