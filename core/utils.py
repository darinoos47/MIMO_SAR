import torch
import numpy as np

# --- Existing Functions (from previous step) ---

def complex_to_tensor(complex_array):
    """Converts a complex numpy array to a 2-channel torch tensor."""
    data_real = complex_array.real.astype(np.float32)
    data_imag = complex_array.imag.astype(np.float32)
    return torch.from_numpy(np.stack((data_real, data_imag), axis=0))

def complex_matmul(A_tensor, x_tensor):
    """
    Performs complex matrix-vector multiplication: y = A @ x (batch-wise)
    
    A_tensor: [1, 2, N_v, N_theta] (complex matrix A, batched)
    x_tensor: [batch, 2, N_theta] (complex vector x)
    
    Returns:
    y_tensor: [batch, 2, N_v] (complex vector y)
    """
    # <<< FIX HERE: Get batch_size from x_tensor >>>
    batch_size = x_tensor.shape[0]

    # <<< FIX HERE: Expand A to match the batch size of x >>>
    A_r = A_tensor[:, 0, :, :].expand(batch_size, -1, -1)  # [batch, N_v, N_theta]
    A_i = A_tensor[:, 1, :, :].expand(batch_size, -1, -1)  # [batch, N_v, N_theta]
    
    x_r = x_tensor[:, 0, :].unsqueeze(1)  # [batch, 1, N_theta]
    x_i = x_tensor[:, 1, :].unsqueeze(1)  # [batch, 1, N_theta]

    # This bmm is now valid: [batch, N_v, N_theta] @ [batch, N_theta, 1]
    y_r = torch.bmm(A_r, x_r.transpose(1, 2)) - torch.bmm(A_i, x_i.transpose(1, 2))
    y_i = torch.bmm(A_r, x_i.transpose(1, 2)) + torch.bmm(A_i, x_r.transpose(1, 2))
    
    return torch.cat((y_r.squeeze(-1).unsqueeze(1), y_i.squeeze(-1).unsqueeze(1)), dim=1)

def complex_conj_transpose_matmul(A_tensor, y_tensor):
    """
    Performs complex conj-transpose matrix-vector multiplication: x = A.H @ y
    
    A_tensor: [1, 2, N_v, N_theta] OR [batch, 2, N_v, N_theta] (complex matrix A, batched)
    y_tensor: [batch, 2, N_v] (complex vector y)
    
    Returns:
    x_tensor: [batch, 2, N_theta] (complex vector x)
    """
    # <<< FIX HERE: Get batch_size from y_tensor >>>
    batch_size = y_tensor.shape[0]
    A_batch_size = A_tensor.shape[0]

    # <<< FIX HERE: Handle both single A and batch A >>>
    if A_batch_size == 1:
        # Original behavior: expand single A to match batch
        A_r_T = A_tensor[:, 0, :, :].transpose(1, 2).expand(batch_size, -1, -1)  # [batch, N_theta, N_v]
        A_i_T = A_tensor[:, 1, :, :].transpose(1, 2).expand(batch_size, -1, -1)  # [batch, N_theta, N_v]
    else:
        # New behavior: A is already batched, just transpose
        A_r_T = A_tensor[:, 0, :, :].transpose(1, 2).contiguous()  # [batch, N_theta, N_v]
        A_i_T = A_tensor[:, 1, :, :].transpose(1, 2).contiguous()  # [batch, N_theta, N_v]

    y_r = y_tensor[:, 0, :].unsqueeze(1)  # [batch, 1, N_v]
    y_i = y_tensor[:, 1, :].unsqueeze(1)  # [batch, 1, N_v]
    
    # x = (A_r.T - j A_i.T) @ (y_r + j y_i)
    # x_r = A_r.T @ y_r + A_i.T @ y_i
    x_r = torch.bmm(A_r_T, y_r.transpose(1, 2)) + torch.bmm(A_i_T, y_i.transpose(1, 2))
    
    # x_i = A_r.T @ y_i - A_i.T @ y_r
    x_i = torch.bmm(A_r_T, y_i.transpose(1, 2)) - torch.bmm(A_i_T, y_r.transpose(1, 2))
    
    return torch.cat((x_r.squeeze(-1).unsqueeze(1), x_i.squeeze(-1).unsqueeze(1)), dim=1)


# --- New Functions for ADMM Layer (Unchanged) ---

def complex_matmul_tensor(A, B):
    """
    Performs complex matrix multiplication: C = A @ B
    A, B are non-batched 2-channel tensors.
    
    A: [2, N, K]
    B: [2, K, M]
    Returns:
    C: [2, N, M]
    """
    A_r, A_i = A[0], A[1]
    B_r, B_i = B[0], B[1]
    
    C_r = A_r @ B_r - A_i @ B_i
    C_i = A_r @ B_i + A_i @ B_r
    
    return torch.stack((C_r, C_i), dim=0)

def complex_batch_matmul_vec(M, v):
    """
    Performs complex batch matrix-vector multiplication: out = M @ v
    
    M: [batch, 2, N, N]
    v: [batch, 2, N]
    Returns:
    out: [batch, 2, N]
    """
    M_r, M_i = M[:, 0], M[:, 1] # [batch, N, N]
    v_r, v_i = v[:, 0].unsqueeze(-1), v[:, 1].unsqueeze(-1) # [batch, N, 1]
    
    out_r = M_r @ v_r - M_i @ v_i
    out_i = M_r @ v_i + M_i @ v_r
    
    return torch.cat((out_r.squeeze(-1).unsqueeze(1), out_i.squeeze(-1).unsqueeze(1)), dim=1)

def complex_batch_inverse(M):
    """
    Computes the inverse of a complex batch of matrices.
    
    M: [batch, 2, N, N]
    Returns:
    M_inv: [batch, 2, N, N]
    """
    # Convert [B, 2, N, N] -> [B, N, N, 2]
    M_permuted = M.permute(0, 2, 3, 1).contiguous()
    # View as [B, N, N] complex
    M_complex = torch.view_as_complex(M_permuted)
    
    # Compute inverse
    M_inv_complex = torch.linalg.inv(M_complex)
    
    # Convert back to [B, N, N, 2] real
    M_inv_real = torch.view_as_real(M_inv_complex)
    # Convert back to [B, 2, N, N]
    M_inv_tensor = M_inv_real.permute(0, 3, 1, 2)
    
    return M_inv_tensor

def complex_project_l2_ball(v, center, radius):
    """
    Projects a batch of complex vectors 'v' onto the l2-ball 
    centered at 'center' with 'radius'.
    
    v, center: [batch, 2, N]
    radius: scalar tensor
    Returns:
    z: [batch, 2, N]
    """
    d_tensor = v - center # [batch, 2, N]
    
    # Calculate l2 norm: ||d||_2 = sqrt(sum(d_r^2 + d_i^2))
    # d_tensor**2 sums d_r^2 and d_i^2
    d_norm = torch.sqrt(torch.sum(d_tensor**2, dim=(1, 2))) # [batch]
    
    # Get scale factor: min(1, radius / ||d||)
    # Add epsilon for numerical stability
    scale = torch.min(torch.tensor(1.0, device=v.device), radius / (d_norm + 1e-8))
    
    # Reshape scale for broadcasting: [batch] -> [batch, 1, 1]
    scale = scale.unsqueeze(-1).unsqueeze(-1)
    
    # z = center + d * scale
    return center + d_tensor * scale

# --- New Function for Adjoint Test (from previous step) ---

def complex_inner_product(a, b):
    """
    Computes the complex inner product <a, b> = sum(a_conj * b)
    for a batch of vectors.
    
    a, b: [batch, 2, N]
    Returns:
    product: [batch, 2] (a batch of complex scalars)
    """
    
    # a_conj: [batch, 2, N]
    a_conj = torch.stack((a[:, 0], -a[:, 1]), dim=1)
    
    # (a_r - j*a_i) * (b_r + j*b_i) = (a_r*b_r + a_i*b_i) + j*(a_r*b_i - a_i*b_r)
    
    # prod_r = a_conj_r * b_r - a_conj_i * b_i
    prod_r = a_conj[:, 0, :] * b[:, 0, :] - a_conj[:, 1, :] * b[:, 1, :]
    
    # prod_i = a_conj_r * b_i + a_conj_i * b_r
    prod_i = a_conj[:, 0, :] * b[:, 1, :] + a_conj[:, 1, :] * b[:, 0, :]
    
    # Sum over the N dimension
    sum_r = torch.sum(prod_r, dim=1)
    sum_i = torch.sum(prod_i, dim=1)
    
    # Return as a [batch, 2] complex scalar
    return torch.stack((sum_r, sum_i), dim=1)

def tensor_to_complex_np(tensor):
    """Converts a [batch, 2, ...] tensor to a complex numpy array."""
    if tensor.is_cuda:
        tensor = tensor.cpu()
    data = tensor.numpy()
    # Squeeze out the batch dim if it's 1, then combine channels
    data_squeezed = data.squeeze(axis=0)
    return data_squeezed[0] + 1j * data_squeezed[1]
