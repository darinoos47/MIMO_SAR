import torch

def complex_to_tensor(complex_array):
    """Converts a complex numpy array to a 2-channel torch tensor."""
    import numpy as np
    data_real = complex_array.real.astype(np.float32)
    data_imag = complex_array.imag.astype(np.float32)
    return torch.from_numpy(np.stack((data_real, data_imag), axis=0))

def complex_matmul(A_tensor, x_tensor):
    """
    Performs complex matrix multiplication: y = A @ x
    
    A_tensor: [2, N_v, N_theta] (complex matrix A)
    x_tensor: [batch, 2, N_theta] (complex vector x)
    
    Returns:
    y_tensor: [batch, 2, N_v] (complex vector y)
    """
    
    # A = A_r + j * A_i
    A_r = A_tensor[0, :, :].unsqueeze(0)  # [1, N_v, N_theta]
    A_i = A_tensor[1, :, :].unsqueeze(0)  # [1, N_v, N_theta]
    
    # x = x_r + j * x_i
    x_r = x_tensor[:, 0, :].unsqueeze(1)  # [batch, 1, N_theta]
    x_i = x_tensor[:, 1, :].unsqueeze(1)  # [batch, 1, N_theta]

    # Use torch.bmm for batch matrix multiplication
    # y_r = A_r @ x_r - A_i @ x_i
    y_r = torch.bmm(A_r, x_r.transpose(1, 2)) - torch.bmm(A_i, x_i.transpose(1, 2))
    
    # y_i = A_r @ x_i + A_i @ x_r
    y_i = torch.bmm(A_r, x_i.transpose(1, 2)) + torch.bmm(A_i, x_r.transpose(1, 2))
    
    # Result shape is [batch, N_v, 1], squeeze to [batch, N_v]
    # Then stack to [batch, 2, N_v]
    return torch.cat((y_r.squeeze(-1).unsqueeze(1), y_i.squeeze(-1).unsqueeze(1)), dim=1)

def complex_conj_transpose_matmul(A_tensor, y_tensor):
    """
    Performs complex conjugate transpose matrix multiplication: x = A.H @ y
    
    A_tensor: [2, N_v, N_theta] (complex matrix A)
    y_tensor: [batch, 2, N_v] (complex vector y)
    
    Returns:
    x_tensor: [batch, 2, N_theta] (complex vector x)
    """
    
    # A.H = A_r.T - j * A_i.T
    # We get A_r and A_i from A_tensor
    # A_r.T has shape [N_theta, N_v]
    # A_i.T has shape [N_theta, N_v]
    A_r_T = A_tensor[0, :, :].T.unsqueeze(0)  # [1, N_theta, N_v]
    A_i_T = A_tensor[1, :, :].T.unsqueeze(0)  # [1, N_theta, N_v]

    # y = y_r + j * y_i
    y_r = y_tensor[:, 0, :].unsqueeze(1)  # [batch, 1, N_v]
    y_i = y_tensor[:, 1, :].unsqueeze(1)  # [batch, 1, N_v]
    
    # x_r = A_r.T @ y_r + A_i.T @ y_i
    x_r = torch.bmm(A_r_T, y_r.transpose(1, 2)) + torch.bmm(A_i_T, y_i.transpose(1, 2))
    
    # x_i = A_r.T @ y_i - A_i.T @ y_r
    x_i = torch.bmm(A_r_T, y_i.transpose(1, 2)) - torch.bmm(A_i_T, y_r.transpose(1, 2))
    
    # Stack to [batch, 2, N_theta]
    return torch.cat((x_r.squeeze(-1).unsqueeze(1), x_i.squeeze(-1).unsqueeze(1)), dim=1)
