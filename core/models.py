import torch
import torch.nn as nn
from .utils import (
    complex_matmul, complex_conj_transpose_matmul,
    complex_matmul_tensor, complex_batch_matmul_vec,
    complex_batch_inverse, complex_project_l2_ball
)

# -----------------------------------------------------------------
# 1. CNN Denoiser (Unchanged - keep as is)
# -----------------------------------------------------------------
class CNNDenoiser(nn.Module):
    """
    Original CNN Denoiser (Shallow Residual Architecture)
    - 2 convolutional blocks with 32 filters each
    - Residual connection: output = input + CNN(input)
    - 2 output channels (real + imaginary)
    - ~3,586 parameters
    
    Note: Kept for backward compatibility with existing checkpoints
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
# 1b. CNN Denoiser - Real Output (Architectural Constraint)
# -----------------------------------------------------------------
class CNNDenoiser_RealOutput(nn.Module):
    """
    CNN Denoiser with Real-Only Output (Deep Encoder-Decoder)
    - Deep architecture: 6 convolutional layers
    - 1 output channel (REAL ONLY) - enforces physical constraint at architecture level
    - Imaginary part is forced to zero by design
    - Optional positivity enforcement (ReLU) for non-negative reflectivity
    - ~62,369 parameters
    
    Performance: Best results for real-valued targets (2.46× better than complex output)
    Use case: When target reflectivity is known to be real-valued (MIMO SAR)
    """
    def __init__(self, in_channels=2, out_channels=1, num_filters=32, kernel_size=3,
                 enforce_positivity=True):
        super(CNNDenoiser_RealOutput, self).__init__()
        
        self.enforce_positivity = enforce_positivity
        padding = kernel_size // 2
        
        # Encoder: progressively increase channels
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters * 4),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: progressively decrease channels to 1 (real only)
        decoder_layers = [
            nn.Conv1d(num_filters * 4, num_filters * 2, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_filters * 2, num_filters, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_filters, out_channels, kernel_size, padding=padding, bias=True),  # out_channels=1
        ]
        
        # Optionally add ReLU to enforce positivity (reflectivity ≥ 0)
        if self.enforce_positivity:
            decoder_layers.append(nn.ReLU())
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """
        Args:
            x: [batch, 2, N] - complex input (real, imag)
        Returns:
            x_out: [batch, 2, N] - output with real channel filled, imag=0
        """
        # Encode and decode to get real part only
        z = self.encoder(x)
        real_part = self.decoder(z)  # [batch, 1, N]
        
        # Create 2-channel output with imaginary part = 0
        batch_size, _, N = x.shape
        x_out = torch.zeros(batch_size, 2, N, device=x.device, dtype=x.dtype)
        x_out[:, 0:1, :] = real_part  # Real channel
        # x_out[:, 1, :] remains zero (imaginary)
        
        return x_out


# -----------------------------------------------------------------
# 1c. CNN Denoiser - Complex Output (No Constraint)
# -----------------------------------------------------------------
class CNNDenoiser_ComplexOutput(nn.Module):
    """
    CNN Denoiser with Complex Output (Deep Encoder-Decoder)
    - Deep architecture: 6 convolutional layers
    - 2 output channels (real + imaginary) - both learnable
    - No architectural constraint on imaginary part
    - ~62,466 parameters
    
    Performance: Good, but 2.46× worse than Real-Only for real targets
    Use case: When target can have imaginary components, or as baseline comparison
    """
    def __init__(self, in_channels=2, out_channels=2, num_filters=32, kernel_size=3):
        super(CNNDenoiser_ComplexOutput, self).__init__()
        
        padding = kernel_size // 2
        
        # Encoder: progressively increase channels
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters * 4),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: progressively decrease channels to 2 (real + imag)
        self.decoder = nn.Sequential(
            nn.Conv1d(num_filters * 4, num_filters * 2, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_filters * 2, num_filters, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(num_filters, out_channels, kernel_size, padding=padding, bias=True),  # out_channels=2
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, 2, N] - complex input (real, imag)
        Returns:
            x_out: [batch, 2, N] - complex output (real, imag)
        """
        z = self.encoder(x)
        x_out = self.decoder(z)
        return x_out


# -----------------------------------------------------------------
# 2. ADMM Layer (Using Sherman-Morrison-Woodbury Identity)
# -----------------------------------------------------------------
class DCLayer_ADMM(nn.Module):
    """
    Implements the Data Consistency (DC) layer using N steps of ADMM
    with the Sherman-Morrison-Woodbury (SMW) identity for efficient computation.
    
    The SMW identity transforms the x-update from solving a large N×N system
    (I + rho*A^H*A) to solving a smaller M×M system (1/rho*I_M + A*A^H),
    where M = N_v (measurements) and N = N_theta (unknowns).
    
    Since M << N typically, this provides significant computational speedup.
    
    Args:
        A_tensor: Steering matrix [2, N_v, N_theta]
        N_admm_steps: Number of ADMM iterations
        enforce_real: If True, project x to real values (discard imaginary part)
        enforce_positivity: If True, clamp x to non-negative values
    """
    def __init__(self, A_tensor, N_admm_steps=3, enforce_real=True, enforce_positivity=True):
        super(DCLayer_ADMM, self).__init__()
        self.N_admm_steps = N_admm_steps
        self.enforce_real = enforce_real
        self.enforce_positivity = enforce_positivity
        
        # Learnable parameters
        self.log_rho = nn.Parameter(torch.log(torch.tensor(1.0)))
        self.log_epsilon = nn.Parameter(torch.log(torch.tensor(0.01)))
        
        # Store A (shape [2, N_v, N_theta])
        self.register_buffer('A', A_tensor)

    def forward(self, r_n, y, u_in):
        """
        Forward pass using ADMM with SMW identity for efficient x-update.
        
        Args:
            r_n:   [batch, 2, N_theta] (denoiser output)
            y:     [batch, 2, N_v]     (original measurement)
            u_in:  [batch, 2, N_v]     (dual variable from prev iter)
        
        Returns:
            x: [batch, 2, N_theta] (updated reflectivity estimate)
            u: [batch, 2, N_v]     (updated dual variable)
        """
        # Get scalar parameters
        rho = torch.exp(self.log_rho)
        epsilon = torch.exp(self.log_epsilon)
        
        # --- Initialize ADMM variables ---
        u = u_in
        z = y 
        x = r_n  # Start x at the denoiser output
        
        A_batch = self.A.unsqueeze(0)
        
        # --- Run N_admm_steps ---
        for _ in range(self.N_admm_steps):
            
            # -----------------------------------------------
            # 1) x-update using SMW identity
            #    We want: x = (I + rho*A^H*A)^{-1} * b
            #    where:   b = r_n + rho*A^H*(z - u)
            #    
            #    Using SMW: (I + rho*A^H*A)^{-1}*b = b - A^H*((1/rho)*I_M + A*A^H)^{-1}*(A*b)
            #    
            #    This reduces the inversion from N×N to M×M (much faster!)
            # -----------------------------------------------

            # --- Build b = r_n + rho * A^H * (z - u) ---
            z_minus_u = z - u                                        # [B, 2, M]
            A_H_z_u   = complex_conj_transpose_matmul(A_batch, z_minus_u)  # [B, 2, N]
            b         = r_n + rho * A_H_z_u                         # [B, 2, N]

            # --- Compute A*b (size M, much smaller than N) ---
            Ab = complex_matmul(A_batch, b)                         # [B, 2, M]

            # --- Build S = (1/rho)*I_M + A*A^H  (complex M×M matrix) ---
            B, _, M_meas, N_theta = A_batch.shape                   # A_batch: [B, 2, M, N]
            I_M = torch.eye(M_meas, device=b.device).expand(B, M_meas, M_meas)  # [B, M, M]

            # Split real/imag parts of A
            A_r = A_batch[:, 0]                                      # [B, M, N]
            A_i = A_batch[:, 1]                                      # [B, M, N]

            # Compute A*A^H (real/imag parts) using batched matrix multiplication
            # Real(A*A^H) = A_r @ A_r^T + A_i @ A_i^T
            # Imag(A*A^H) = A_i @ A_r^T - A_r @ A_i^T
            ArArT = torch.bmm(A_r, A_r.transpose(1, 2))             # [B, M, M]
            AiAiT = torch.bmm(A_i, A_i.transpose(1, 2))             # [B, M, M]
            ArAiT = torch.bmm(A_r, A_i.transpose(1, 2))             # [B, M, M]
            AiArT = torch.bmm(A_i, A_r.transpose(1, 2))             # [B, M, M]

            S_r = (1.0 / (rho + 1e-12)) * I_M + (ArArT + AiAiT)     # [B, M, M]
            S_i = (AiArT - ArAiT)                                   # [B, M, M]

            # --- Convert complex system to real block form and solve ---
            #     [ S_r  -S_i ][T_r] = [Ab_r]
            #     [ S_i   S_r ][T_i]   [Ab_i]
            Ab_r = Ab[:, 0]                                         # [B, M]
            Ab_i = Ab[:, 1]                                         # [B, M]

            top_row = torch.cat([S_r, -S_i], dim=2)                 # [B, M, 2M]
            bot_row = torch.cat([S_i,  S_r], dim=2)                 # [B, M, 2M]
            S_block = torch.cat([top_row, bot_row], dim=1)          # [B, 2M, 2M]

            R_block = torch.cat([Ab_r, Ab_i], dim=1).unsqueeze(-1)  # [B, 2M, 1]

            # Solve the small M×M system (much faster than N×N!)
            T_block = torch.linalg.solve(S_block, R_block)          # [B, 2M, 1]

            # Unpack solution
            T_r = T_block[:, :M_meas, 0].unsqueeze(1)               # [B, 1, M]
            T_i = T_block[:,  M_meas:, 0].unsqueeze(1)              # [B, 1, M]
            T   = torch.cat([T_r, T_i], dim=1)                      # [B, 2, M]

            # --- Complete SMW formula: x = b - A^H @ T ---
            A_H_T = complex_conj_transpose_matmul(A_batch, T)       # [B, 2, N]
            x = b - A_H_T                                           # [B, 2, N]

            # -----------------------------------------------
            # ENFORCE PHYSICAL CONSTRAINTS
            # -----------------------------------------------
            # 1. Enforce realness: discard imaginary part
            if self.enforce_real:
                x[:, 1, :] = 0  # Set imaginary channel to zero
            
            # 2. Enforce positivity: clamp to non-negative
            if self.enforce_positivity:
                x[:, 0, :] = torch.clamp(x[:, 0, :], min=0)

            
            # -----------------------------------------------
            # 2) z-update: z = y + L2Proj(A*x + u - y, epsilon)
            # -----------------------------------------------
            Ax = complex_matmul(A_batch, x)
            Ax_plus_u_minus_y = Ax + u - y
            
            # Project the residual onto L2 ball
            z_residual = complex_project_l2_ball(Ax_plus_u_minus_y, torch.zeros_like(y), epsilon)
            
            # Add back to y
            z = y + z_residual

            # -----------------------------------------------
            # 3) u-update: u = u + A*x - z
            # -----------------------------------------------
            u = u + Ax - z
            
        return x, u

# -----------------------------------------------------------------
# 3. Updated DBP Network
# -----------------------------------------------------------------
class DBPNet(nn.Module):
    """
    Deep Back Projection Network with configurable denoiser architecture.
    
    Args:
        A_tensor: Steering matrix [2, N_v, N_theta]
        num_iterations: Number of unrolled iterations
        N_admm_steps: Number of ADMM steps per iteration
        denoiser_type: Type of denoiser architecture
            - 'original': Original shallow residual (3.6K params, backward compatible)
            - 'real': Deep encoder-decoder with real-only output (62K params, best for real targets)
            - 'complex': Deep encoder-decoder with complex output (62K params)
        num_filters: Number of filters in first layer (default: 32)
        enforce_positivity: If True, enforce non-negative output (only for 'real' denoiser)
        admm_enforce_real: If True, ADMM projects x to real values (discard imaginary)
        admm_enforce_positivity: If True, ADMM clamps x to non-negative values
    """
    def __init__(self, A_tensor, num_iterations=5, N_admm_steps=3, 
                 denoiser_type='original', num_filters=32, enforce_positivity=True,
                 admm_enforce_real=True, admm_enforce_positivity=True):
        super(DBPNet, self).__init__()
        self.num_iterations = num_iterations
        self.denoiser_type = denoiser_type
        self.enforce_positivity = enforce_positivity
        self.admm_enforce_real = admm_enforce_real
        self.admm_enforce_positivity = admm_enforce_positivity
        
        self.register_buffer('A', A_tensor)
        
        # Select denoiser architecture based on type
        if denoiser_type == 'real':
            positivity_str = " + Positivity" if enforce_positivity else ""
            print(f"Using CNNDenoiser_RealOutput (Deep, 1 channel, ~62K params{positivity_str})")
            self.denoiser = CNNDenoiser_RealOutput(in_channels=2, out_channels=1, 
                                                   num_filters=num_filters,
                                                   enforce_positivity=enforce_positivity)
        elif denoiser_type == 'complex':
            print(f"Using CNNDenoiser_ComplexOutput (Deep, 2 channels, ~62K params)")
            self.denoiser = CNNDenoiser_ComplexOutput(in_channels=2, out_channels=2, 
                                                      num_filters=num_filters)
        elif denoiser_type == 'original':
            print(f"Using CNNDenoiser (Shallow residual, 2 channels, ~3.6K params)")
            self.denoiser = CNNDenoiser(in_channels=2, out_channels=2, 
                                       num_filters=num_filters)
        else:
            raise ValueError(f"Unknown denoiser_type: {denoiser_type}. "
                           f"Choose from: 'original', 'real', 'complex'")
        
        # Print ADMM constraint info
        constraints = []
        if admm_enforce_real:
            constraints.append("Real")
        if admm_enforce_positivity:
            constraints.append("Positive")
        constraint_str = " + ".join(constraints) if constraints else "None"
        print(f"ADMM constraints: {constraint_str}")
        
        self.dc_layers = nn.ModuleList(
            [DCLayer_ADMM(A_tensor, N_admm_steps, 
                         enforce_real=admm_enforce_real,
                         enforce_positivity=admm_enforce_positivity) 
             for _ in range(num_iterations)]
        )
    
    def load_pretrained_denoiser(self, denoiser_path):
        """
        Load pre-trained denoiser weights.
        
        Args:
            denoiser_path: Path to saved denoiser state dict
        """
        print(f"Loading pre-trained denoiser from {denoiser_path}")
        denoiser_state = torch.load(denoiser_path)
        self.denoiser.load_state_dict(denoiser_state)
        print("Pre-trained denoiser loaded successfully!")
    
    def freeze_denoiser(self):
        """Freeze denoiser parameters (no gradient updates)."""
        for param in self.denoiser.parameters():
            param.requires_grad = False
        print("Denoiser parameters frozen (no gradient updates)")
    
    def unfreeze_denoiser(self):
        """Unfreeze denoiser parameters (allow gradient updates)."""
        for param in self.denoiser.parameters():
            param.requires_grad = True
        print("Denoiser parameters unfrozen (gradient updates enabled)")
    
    def get_trainable_params(self):
        """Return only trainable parameters (useful when denoiser is frozen)."""
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, y, return_intermediates=False):
        """
        Forward pass through the network.
        
        Args:
            y: Measurements [batch, 2, N_v]
            return_intermediates: If True, returns intermediate outputs from each iteration
            
        Returns:
            If return_intermediates=False:
                x: Final reconstructed reflectivity [batch, 2, N_theta]
            If return_intermediates=True:
                Dictionary with:
                    'x_final': Final output
                    'x_init': Initial estimate (A^H @ y)
                    'x_after_denoiser': List of outputs after denoiser at each iteration
                    'x_after_admm': List of outputs after ADMM at each iteration
        """
        x = complex_conj_transpose_matmul(self.A.unsqueeze(0), y)
        u = torch.zeros_like(y)
        
        if return_intermediates:
            intermediates = {
                'x_init': x.detach().clone(),
                'x_after_denoiser': [],
                'x_after_admm': []
            }
        
        for i in range(self.num_iterations):
            r = self.denoiser(x)
            # r = x # debug: Bypass denoiser to check ADMM
            
            if return_intermediates:
                intermediates['x_after_denoiser'].append(r.detach().clone())
            
            x, u = self.dc_layers[i](r, y, u)
            
            if return_intermediates:
                intermediates['x_after_admm'].append(x.detach().clone())
            
        if return_intermediates:
            intermediates['x_final'] = x
            return intermediates
        else:
            return x
