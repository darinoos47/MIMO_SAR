"""
MNIST Polar Domain Inversion: Single-Range vs Multi-Range Comparison

This script demonstrates the benefit of multi-range (2D) processing vs single-range (1D)
processing for image reconstruction in polar (angle-range) domain.

Setup:
- Uses MNIST digits as ground truth images in polar domain
- Radar is static (no SAR movement)
- Forward model: y = A @ x for each range bin
- Two reconstruction methods compared:
  1. Single-Range: 1D CNN denoiser, processes each range independently
  2. Multi-Range: 2D CNN denoiser, processes all ranges simultaneously

Both methods use unsupervised training (measurement consistency loss).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from core.models import CNNDenoiser_RealOutput, MultiRangeDenoiser
from core.utils import complex_matmul, complex_conj_transpose_matmul

# =============================================================================
# Configuration
# =============================================================================

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Polar domain dimensions
N_THETA = 51      # Number of angle bins (columns in polar image)
N_RANGES = 57     # Number of range bins (rows in polar image)
N_V = 8           # Number of virtual antennas (measurements per range)

# Training configuration
NUM_TRAIN_SAMPLES = 10000    # Number of MNIST samples for training
NUM_TEST_SAMPLES = 100      # Number of MNIST samples for testing
BATCH_SIZE = 32
EPOCHS_SINGLE_RANGE = 1   # Epochs for single-range denoiser
EPOCHS_MULTI_RANGE = 1    # Epochs for multi-range denoiser
LEARNING_RATE = 1e-3

# Denoiser configuration
NUM_FILTERS = 32            # Number of filters in first layer
ENFORCE_POSITIVITY = True   # ReLU at output

# Output paths
OUTPUT_DIR = 'results/mnist_polar_comparison'

# Checkpoint configuration
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs

def get_checkpoint_path(model_type):
    """Generate checkpoint path based on dataset and network type."""
    return f'checkpoints/{DATASET}_{NETWORK_TYPE}_{model_type}.pth'

# Dataset selection: 'mnist', 'fashion_mnist', or 'emnist'
DATASET = 'fashion_mnist'

# Network architecture: 'cnn' or 'unet'
NETWORK_TYPE = 'cnn'  # Options: 'cnn', 'unet'

# Random seed for reproducibility
SEED = 42

# =============================================================================
# UNet Wrapper Classes
# =============================================================================

class UNetDenoiser2D(nn.Module):
    """
    UNet-based denoiser for 2D (multi-range) processing.
    
    Uses pre-trained UNet architecture from torch.hub with ReLU for positivity.
    Handles padding to ensure dimensions are divisible by 16.
    """
    def __init__(self, in_channels=2, out_channels=1, init_features=32, enforce_positivity=True):
        super().__init__()
        self.enforce_positivity = enforce_positivity
        self.relu = nn.ReLU()
        
        # Load UNet from torch hub
        self.unet = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            pretrained=False
        )
    
    def _pad_to_multiple(self, x, multiple=16):
        """Pad input to be divisible by multiple."""
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, h, w
    
    def forward(self, x):
        """
        Args:
            x: [batch, N_ranges, 2, N_theta] - adjoint image
            
        Returns:
            x_out: [batch, N_ranges, 2, N_theta] - denoised image
        """
        batch_size, N_ranges, _, N_theta = x.shape
        
        # Reshape: [batch, N_ranges, 2, N_theta] -> [batch, 2, N_ranges, N_theta]
        x = x.permute(0, 2, 1, 3)
        
        # Pad to multiple of 16
        x_padded, orig_h, orig_w = self._pad_to_multiple(x, 16)
        
        # Apply UNet
        out = self.unet(x_padded)  # [batch, 1, H_padded, W_padded]
        
        # Crop back to original size
        out = out[:, :, :orig_h, :orig_w]  # [batch, 1, N_ranges, N_theta]
        
        # Apply positivity
        if self.enforce_positivity:
            out = self.relu(out)
        
        # Create 2-channel output (real part only, imag=0)
        x_out = torch.zeros(batch_size, 2, N_ranges, N_theta, device=x.device)
        x_out[:, 0, :, :] = out.squeeze(1)
        
        # Reshape back: [batch, 2, N_ranges, N_theta] -> [batch, N_ranges, 2, N_theta]
        x_out = x_out.permute(0, 2, 1, 3)
        
        return x_out


class UNetDenoiser1D(nn.Module):
    """
    UNet-based denoiser for 1D (single-range) processing.
    
    Replicates 1D signal to 16-pixel height for UNet compatibility (minimum for pooling).
    """
    def __init__(self, in_channels=2, out_channels=1, init_features=32, enforce_positivity=True):
        super().__init__()
        self.enforce_positivity = enforce_positivity
        self.relu = nn.ReLU()
        self.min_height = 16  # Minimum height for UNet pooling
        
        # Load UNet from torch hub
        self.unet = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            pretrained=False
        )
    
    def _pad_to_multiple(self, x, multiple=16):
        """Pad input to be divisible by multiple."""
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return x, h, w
    
    def forward(self, x):
        """
        Args:
            x: [batch, 2, N_theta] - single range adjoint
            
        Returns:
            x_out: [batch, 2, N_theta] - denoised range
        """
        batch_size, _, N_theta = x.shape
        
        # Replicate to create minimum height: [batch, 2, N_theta] -> [batch, 2, min_height, N_theta]
        x = x.unsqueeze(2).repeat(1, 1, self.min_height, 1)
        
        # Pad width to multiple of 16
        x_padded, orig_h, orig_w = self._pad_to_multiple(x, 16)
        
        # Apply UNet
        out = self.unet(x_padded)  # [batch, 1, H_padded, W_padded]
        
        # Crop back to original size
        out = out[:, :, :orig_h, :orig_w]  # [batch, 1, min_height, N_theta]
        
        # Apply positivity
        if self.enforce_positivity:
            out = self.relu(out)
        
        # Average over height dimension and create 2-channel output
        out = out.mean(dim=2)  # [batch, 1, N_theta]
        x_out = torch.zeros(batch_size, 2, N_theta, device=x.device)
        x_out[:, 0, :] = out.squeeze(1)
        
        return x_out


# =============================================================================
# Steering Matrix Generation
# =============================================================================

def generate_steering_matrix(N_v, N_theta, device):
    """
    Generate a synthetic steering matrix A for the forward model.
    
    The steering matrix models the relationship between angle bins and
    virtual antenna measurements. Uses a simplified uniform linear array model.
    
    Args:
        N_v: Number of virtual antennas (measurements)
        N_theta: Number of angle bins
        device: torch device
        
    Returns:
        A_tensor: Steering matrix [2, N_v, N_theta] (complex as 2 channels)
    """
    # Create angle bins from -60 to 60 degrees
    theta = np.linspace(-25, 25, N_theta) * np.pi / 180  # in radians
    
    # Virtual antenna positions (normalized half-wavelength spacing)
    d = 0.5  # spacing in wavelengths
    antenna_positions = np.arange(N_v) * d
    
    # Create steering matrix: A[v, theta] = exp(j * 2 * pi * d_v * sin(theta))
    A_complex = np.zeros((N_v, N_theta), dtype=np.complex64)
    for v in range(N_v):
        for t in range(N_theta):
            phase = 2 * np.pi * antenna_positions[v] * np.sin(theta[t])
            A_complex[v, t] = np.exp(1j * phase)
    
    # Convert to 2-channel tensor [2, N_v, N_theta]
    A_real = A_complex.real.astype(np.float32)
    A_imag = A_complex.imag.astype(np.float32)
    A_tensor = torch.from_numpy(np.stack([A_real, A_imag], axis=0)).to(device)
    
    return A_tensor


# =============================================================================
# MNIST Data Pipeline
# =============================================================================

def load_dataset_polar(dataset_name, num_train, num_test, n_theta, n_ranges, device):
    """
    Load dataset (MNIST, Fashion-MNIST, or EMNIST) and resize to polar domain dimensions.
    
    Args:
        dataset_name: 'mnist', 'fashion_mnist', or 'emnist'
        num_train: Number of training samples
        num_test: Number of test samples
        n_theta: Number of angle bins (width)
        n_ranges: Number of range bins (height)
        device: torch device
        
    Returns:
        train_images: [num_train, n_ranges, n_theta] tensor (real, positive)
        test_images: [num_test, n_ranges, n_theta] tensor (real, positive)
        dataset_display_name: Human-readable name of the dataset
    """
    # Define transform to resize to polar dimensions
    transform = transforms.Compose([
        transforms.Resize((n_ranges, n_theta)),
        transforms.ToTensor(),  # Converts to [0, 1] range
    ])
    
    # Select dataset
    if dataset_name.lower() == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root='./data/mnist', 
            train=True, 
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data/mnist', 
            train=False, 
            download=True,
            transform=transform
        )
        display_name = "MNIST"
        
    elif dataset_name.lower() == 'fashion_mnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            root='./data/fashion_mnist', 
            train=True, 
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root='./data/fashion_mnist', 
            train=False, 
            download=True,
            transform=transform
        )
        display_name = "Fashion-MNIST"
        
    elif dataset_name.lower() == 'emnist':
        # EMNIST has multiple splits; using 'letters' for variety
        train_dataset = torchvision.datasets.EMNIST(
            root='./data/emnist', 
            split='letters',
            train=True, 
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.EMNIST(
            root='./data/emnist', 
            split='letters',
            train=False, 
            download=True,
            transform=transform
        )
        display_name = "EMNIST (Letters)"
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                        f"Choose from: 'mnist', 'fashion_mnist', 'emnist'")
    
    # Extract subset of images
    train_images = []
    for i in range(min(num_train, len(train_dataset))):
        img, _ = train_dataset[i]
        train_images.append(img.squeeze(0))  # Remove channel dim -> [n_ranges, n_theta]
    train_images = torch.stack(train_images).to(device)  # [num_train, n_ranges, n_theta]
    
    test_images = []
    for i in range(min(num_test, len(test_dataset))):
        img, _ = test_dataset[i]
        test_images.append(img.squeeze(0))
    test_images = torch.stack(test_images).to(device)  # [num_test, n_ranges, n_theta]
    
    return train_images, test_images, display_name


def create_measurements(x_true, A_tensor):
    """
    Generate measurements using forward model: y = A @ x for each range.
    
    Args:
        x_true: Ground truth polar image [batch, N_ranges, N_theta] (real)
        A_tensor: Steering matrix [2, N_v, N_theta]
        
    Returns:
        y: Measurements [batch, N_ranges, 2, N_v] (complex as 2 channels)
    """
    batch_size, N_ranges, N_theta = x_true.shape
    device = x_true.device
    
    # Convert real image to complex format [batch, N_ranges, 2, N_theta]
    # Real part = x_true, Imaginary part = 0
    x_complex = torch.zeros(batch_size, N_ranges, 2, N_theta, device=device)
    x_complex[:, :, 0, :] = x_true  # Real channel
    
    # Apply forward model to each range
    y_list = []
    A_batch = A_tensor.unsqueeze(0)  # [1, 2, N_v, N_theta]
    
    for r in range(N_ranges):
        x_r = x_complex[:, r, :, :]  # [batch, 2, N_theta]
        y_r = complex_matmul(A_batch, x_r)  # [batch, 2, N_v]
        y_list.append(y_r)
    
    y = torch.stack(y_list, dim=1)  # [batch, N_ranges, 2, N_v]
    return y


def apply_adjoint(y, A_tensor):
    """
    Apply adjoint operator: x_adj = A^H @ y for each range.
    
    Args:
        y: Measurements [batch, N_ranges, 2, N_v]
        A_tensor: Steering matrix [2, N_v, N_theta]
        
    Returns:
        x_adj: Adjoint image [batch, N_ranges, 2, N_theta]
    """
    batch_size, N_ranges = y.shape[:2]
    device = y.device
    
    A_batch = A_tensor.unsqueeze(0)  # [1, 2, N_v, N_theta]
    
    x_adj_list = []
    for r in range(N_ranges):
        y_r = y[:, r, :, :]  # [batch, 2, N_v]
        x_r = complex_conj_transpose_matmul(A_batch, y_r)  # [batch, 2, N_theta]
        x_adj_list.append(x_r)
    
    x_adj = torch.stack(x_adj_list, dim=1)  # [batch, N_ranges, 2, N_theta]
    return x_adj


# =============================================================================
# Single-Range Denoiser (Method 1)
# =============================================================================

class SingleRangeWrapper(nn.Module):
    """
    Wrapper that applies a 1D denoiser to each range independently.
    
    Takes a multi-range input, processes each range with the same 1D denoiser,
    and stacks the results back together.
    """
    def __init__(self, denoiser_1d):
        super().__init__()
        self.denoiser = denoiser_1d
    
    def forward(self, x):
        """
        Args:
            x: [batch, N_ranges, 2, N_theta] - adjoint image
            
        Returns:
            x_out: [batch, N_ranges, 2, N_theta] - denoised image
        """
        batch_size, N_ranges = x.shape[:2]
        
        # Process each range independently
        x_out_list = []
        for r in range(N_ranges):
            x_r = x[:, r, :, :]  # [batch, 2, N_theta]
            x_r_denoised = self.denoiser(x_r)  # [batch, 2, N_theta]
            x_out_list.append(x_r_denoised)
        
        x_out = torch.stack(x_out_list, dim=1)  # [batch, N_ranges, 2, N_theta]
        return x_out


def train_single_range_denoiser(train_images, A_tensor, epochs, device):
    """
    Train the single-range (1D) denoiser.
    
    Uses unsupervised measurement consistency loss: ||A @ x_out - y||^2
    
    Args:
        train_images: [num_samples, N_ranges, N_theta] ground truth images
        A_tensor: Steering matrix [2, N_v, N_theta]
        epochs: Number of training epochs
        device: torch device
        
    Returns:
        model: Trained SingleRangeWrapper
        losses: List of training losses
    """
    print("\n" + "="*70)
    arch_name = "UNet" if NETWORK_TYPE == 'unet' else "1D CNN"
    print(f"TRAINING SINGLE-RANGE DENOISER (Method 1: {arch_name})")
    print("="*70)
    
    # Create denoiser based on network type
    if NETWORK_TYPE == 'unet':
        denoiser_1d = UNetDenoiser1D(
            in_channels=2,
            out_channels=1,
            init_features=NUM_FILTERS,
            enforce_positivity=ENFORCE_POSITIVITY
        ).to(device)
    else:
        denoiser_1d = CNNDenoiser_RealOutput(
            in_channels=2,
            out_channels=1,
            num_filters=NUM_FILTERS,
            enforce_positivity=ENFORCE_POSITIVITY
        ).to(device)
    
    model = SingleRangeWrapper(denoiser_1d).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Denoiser parameters: {num_params:,}")
    print(f"  Processing: Each range independently")
    print(f"  Epochs: {epochs}")
    
    # Create measurements
    print("\n  Creating measurements from training images...")
    y_train = create_measurements(train_images, A_tensor)  # [N, N_ranges, 2, N_v]
    x_adj_train = apply_adjoint(y_train, A_tensor)  # [N, N_ranges, 2, N_theta]
    
    # Create dataloader
    dataset = TensorDataset(x_adj_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Check for existing checkpoint and resume if available
    start_epoch = 0
    losses = []
    checkpoint_path = get_checkpoint_path('single_range')
    
    if os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle both old format (just state dict) and new format (full checkpoint)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint['losses']
            print(f"  Resuming from epoch {start_epoch}")
        else:
            # Old format: checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            print(f"  Loaded old-format checkpoint (starting fresh from epoch 0)")
    
    # Skip training if already completed
    if start_epoch >= epochs:
        print(f"  Training already completed ({start_epoch} epochs)")
        return model, losses
    
    # Training loop
    model.train()
    A_batch = A_tensor.unsqueeze(0)
    
    pbar = tqdm(range(start_epoch, epochs), desc="  Training")
    for epoch in pbar:
        epoch_loss = 0.0
        for x_adj_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            x_out = model(x_adj_batch)  # [batch, N_ranges, 2, N_theta]
            
            # Compute measurement consistency loss for each range
            loss = 0.0
            N_ranges = x_out.shape[1]
            for r in range(N_ranges):
                x_r = x_out[:, r, :, :]  # [batch, 2, N_theta]
                y_r = y_batch[:, r, :, :]  # [batch, 2, N_v]
                y_pred_r = complex_matmul(A_batch, x_r)  # [batch, 2, N_v]
                loss += torch.mean((y_pred_r - y_r) ** 2)
            loss = loss / N_ranges
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Save checkpoint every CHECKPOINT_INTERVAL epochs
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
            }, checkpoint_path)
            pbar.write(f"  Checkpoint saved at epoch {epoch + 1}")
    
    print(f"\n  Final loss: {losses[-1]:.6f}")
    
    return model, losses


# =============================================================================
# Multi-Range Denoiser (Method 2)
# =============================================================================

def train_multi_range_denoiser(train_images, A_tensor, epochs, device):
    """
    Train the multi-range (2D) denoiser.
    
    Uses unsupervised measurement consistency loss: ||A @ x_out - y||^2
    
    Args:
        train_images: [num_samples, N_ranges, N_theta] ground truth images
        A_tensor: Steering matrix [2, N_v, N_theta]
        epochs: Number of training epochs
        device: torch device
        
    Returns:
        model: Trained MultiRangeDenoiser
        losses: List of training losses
    """
    print("\n" + "="*70)
    arch_name = "UNet" if NETWORK_TYPE == 'unet' else "2D CNN"
    print(f"TRAINING MULTI-RANGE DENOISER (Method 2: {arch_name})")
    print("="*70)
    
    # Create denoiser based on network type
    if NETWORK_TYPE == 'unet':
        model = UNetDenoiser2D(
            in_channels=2,
            out_channels=1,
            init_features=NUM_FILTERS,
            enforce_positivity=ENFORCE_POSITIVITY
        ).to(device)
    else:
        model = MultiRangeDenoiser(
            num_filters=NUM_FILTERS,
            enforce_positivity=ENFORCE_POSITIVITY,
            output_real_only=True
        ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Denoiser parameters: {num_params:,}")
    print(f"  Processing: All ranges simultaneously (2D)")
    print(f"  Epochs: {epochs}")
    
    # Create measurements
    print("\n  Creating measurements from training images...")
    y_train = create_measurements(train_images, A_tensor)  # [N, N_ranges, 2, N_v]
    x_adj_train = apply_adjoint(y_train, A_tensor)  # [N, N_ranges, 2, N_theta]
    
    # Create dataloader
    dataset = TensorDataset(x_adj_train, y_train)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Check for existing checkpoint and resume if available
    start_epoch = 0
    losses = []
    checkpoint_path = get_checkpoint_path('multi_range')
    
    if os.path.exists(checkpoint_path):
        print(f"  Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle both old format (just state dict) and new format (full checkpoint)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            losses = checkpoint['losses']
            print(f"  Resuming from epoch {start_epoch}")
        else:
            # Old format: checkpoint is just the state dict
            model.load_state_dict(checkpoint)
            print(f"  Loaded old-format checkpoint (starting fresh from epoch 0)")
    
    # Skip training if already completed
    if start_epoch >= epochs:
        print(f"  Training already completed ({start_epoch} epochs)")
        return model, losses
    
    # Training loop
    model.train()
    A_batch = A_tensor.unsqueeze(0)
    
    pbar = tqdm(range(start_epoch, epochs), desc="  Training")
    for epoch in pbar:
        epoch_loss = 0.0
        for x_adj_batch, y_batch in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            x_out = model(x_adj_batch)  # [batch, N_ranges, 2, N_theta]
            
            # Compute measurement consistency loss for each range
            loss = 0.0
            N_ranges = x_out.shape[1]
            for r in range(N_ranges):
                x_r = x_out[:, r, :, :]  # [batch, 2, N_theta]
                y_r = y_batch[:, r, :, :]  # [batch, 2, N_v]
                y_pred_r = complex_matmul(A_batch, x_r)  # [batch, 2, N_v]
                loss += torch.mean((y_pred_r - y_r) ** 2)
            loss = loss / N_ranges
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
        
        # Save checkpoint every CHECKPOINT_INTERVAL epochs
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
            }, checkpoint_path)
            pbar.write(f"  Checkpoint saved at epoch {epoch + 1}")
    
    print(f"\n  Final loss: {losses[-1]:.6f}")
    
    return model, losses


# =============================================================================
# Visualization
# =============================================================================

def compute_metrics(x_true, x_recon):
    """
    Compute reconstruction quality metrics.
    
    Args:
        x_true: Ground truth [N_ranges, N_theta] (real)
        x_recon: Reconstructed [N_ranges, N_theta] (real)
        
    Returns:
        mse: Mean squared error
        psnr: Peak signal-to-noise ratio
        ssim: Structural similarity (simplified)
    """
    # MSE
    mse = torch.mean((x_true - x_recon) ** 2).item()
    
    # PSNR (assuming max value is 1.0)
    if mse > 0:
        psnr = 10 * np.log10(1.0 / mse)
    else:
        psnr = float('inf')
    
    # Simplified SSIM (correlation-based)
    x_true_flat = x_true.flatten()
    x_recon_flat = x_recon.flatten()
    
    # Normalize
    x_true_norm = x_true_flat - x_true_flat.mean()
    x_recon_norm = x_recon_flat - x_recon_flat.mean()
    
    # Correlation
    corr = torch.sum(x_true_norm * x_recon_norm) / (
        torch.sqrt(torch.sum(x_true_norm**2)) * torch.sqrt(torch.sum(x_recon_norm**2)) + 1e-8
    )
    ssim = corr.item()
    
    return mse, psnr, ssim


def visualize_comparison(test_image, x_adj, x_single, x_multi, output_path, sample_idx=0, dataset_name="MNIST"):
    """
    Create comparison visualization of reconstruction methods.
    
    Args:
        test_image: Ground truth [N_ranges, N_theta] (real)
        x_adj: Adjoint reconstruction [N_ranges, 2, N_theta]
        x_single: Single-range denoiser output [N_ranges, 2, N_theta]
        x_multi: Multi-range denoiser output [N_ranges, 2, N_theta]
        output_path: Path to save figure
        sample_idx: Sample index for title
    """
    # Extract real parts (channel 0)
    x_adj_real = x_adj[:, 0, :].cpu().numpy()
    x_single_real = x_single[:, 0, :].cpu().numpy()
    x_multi_real = x_multi[:, 0, :].cpu().numpy()
    x_true = test_image.cpu().numpy()
    
    # Compute metrics
    mse_adj, psnr_adj, ssim_adj = compute_metrics(test_image, x_adj[:, 0, :])
    mse_single, psnr_single, ssim_single = compute_metrics(test_image, x_single[:, 0, :])
    mse_multi, psnr_multi, ssim_multi = compute_metrics(test_image, x_multi[:, 0, :])
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Color scale for ground truth and reconstructions (0 to 1)
    vmin_recon = 0
    vmax_recon = max(x_true.max(), x_single_real.max(), x_multi_real.max())
    
    # Separate color scale for adjoint (can be much larger)
    vmin_adj = 0
    vmax_adj = x_adj_real.max()
    
    # Ground Truth
    im0 = axes[0].imshow(x_true, aspect='auto', cmap='viridis', vmin=vmin_recon, vmax=vmax_recon)
    axes[0].set_title(f'Ground Truth\n({dataset_name} in Polar Domain)', fontsize=11)
    axes[0].set_xlabel('Angle Bin')
    axes[0].set_ylabel('Range Bin')
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    
    # Adjoint (separate color scale - note the different range!)
    im1 = axes[1].imshow(x_adj_real, aspect='auto', cmap='viridis', vmin=vmin_adj, vmax=vmax_adj)
    axes[1].set_title(f'Adjoint (A^H @ y)\n(Note: different scale!)', fontsize=11)
    axes[1].set_xlabel('Angle Bin')
    axes[1].set_ylabel('Range Bin')
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    
    # Single-Range Denoiser
    im2 = axes[2].imshow(x_single_real, aspect='auto', cmap='viridis', vmin=vmin_recon, vmax=vmax_recon)
    axes[2].set_title(f'Single-Range (1D CNN)\nMSE={mse_single:.4f}, PSNR={psnr_single:.1f}dB', fontsize=11)
    axes[2].set_xlabel('Angle Bin')
    axes[2].set_ylabel('Range Bin')
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
    
    # Multi-Range Denoiser
    im3 = axes[3].imshow(x_multi_real, aspect='auto', cmap='viridis', vmin=vmin_recon, vmax=vmax_recon)
    axes[3].set_title(f'Multi-Range (2D CNN)\nMSE={mse_multi:.4f}, PSNR={psnr_multi:.1f}dB', fontsize=11)
    axes[3].set_xlabel('Angle Bin')
    axes[3].set_ylabel('Range Bin')
    plt.colorbar(im3, ax=axes[3], shrink=0.8)
    
    plt.suptitle(f'Polar Domain Reconstruction Comparison (Sample {sample_idx})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n  Saved comparison figure to: {output_path}")
    
    # Print metrics summary
    print("\n  Reconstruction Quality Metrics:")
    print(f"  {'Method':<25} {'MSE':>10} {'PSNR (dB)':>12} {'SSIM':>10}")
    print(f"  {'-'*57}")
    print(f"  {'Adjoint (A^H @ y)':<25} {mse_adj:>10.4f} {psnr_adj:>12.2f} {ssim_adj:>10.4f}")
    print(f"  {'Single-Range (1D CNN)':<25} {mse_single:>10.4f} {psnr_single:>12.2f} {ssim_single:>10.4f}")
    print(f"  {'Multi-Range (2D CNN)':<25} {mse_multi:>10.4f} {psnr_multi:>12.2f} {ssim_multi:>10.4f}")
    
    return {
        'adjoint': {'mse': mse_adj, 'psnr': psnr_adj, 'ssim': ssim_adj},
        'single_range': {'mse': mse_single, 'psnr': psnr_single, 'ssim': ssim_single},
        'multi_range': {'mse': mse_multi, 'psnr': psnr_multi, 'ssim': ssim_multi}
    }


def visualize_training_curves(losses_single, losses_multi, output_path):
    """
    Plot training loss curves for both methods.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs_single = range(1, len(losses_single) + 1)
    epochs_multi = range(1, len(losses_multi) + 1)
    
    ax.semilogy(epochs_single, losses_single, 'b-', linewidth=2, label='Single-Range (1D CNN)')
    ax.semilogy(epochs_multi, losses_multi, 'r-', linewidth=2, label='Multi-Range (2D CNN)')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Measurement Consistency Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved training curves to: {output_path}")


def visualize_multiple_samples(test_images, single_model, multi_model, A_tensor, output_path, num_samples=5):
    """
    Visualize reconstruction for multiple test samples.
    """
    single_model.eval()
    multi_model.eval()
    
    # Select samples
    num_samples = min(num_samples, len(test_images))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(14, 3*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for i in range(num_samples):
            test_image = test_images[i:i+1]  # [1, N_ranges, N_theta]
            
            # Generate measurements and adjoint
            y = create_measurements(test_image, A_tensor)
            x_adj = apply_adjoint(y, A_tensor)
            
            # Apply denoisers
            x_single = single_model(x_adj)
            x_multi = multi_model(x_adj)
            
            # Extract real parts
            x_true = test_image[0].cpu().numpy()
            x_adj_real = x_adj[0, :, 0, :].cpu().numpy()
            x_single_real = x_single[0, :, 0, :].cpu().numpy()
            x_multi_real = x_multi[0, :, 0, :].cpu().numpy()
            
            # Color scale for reconstructions (ground truth range)
            vmax_recon = max(x_true.max(), x_single_real.max(), x_multi_real.max())
            # Separate scale for adjoint
            vmax_adj = x_adj_real.max()
            
            # Plot
            axes[i, 0].imshow(x_true, aspect='auto', cmap='turbo', vmin=0, vmax=vmax_recon)
            axes[i, 0].set_title('Ground Truth' if i == 0 else '', fontsize=16, fontweight='bold')
            axes[i, 0].set_ylabel(f'Sample {i+1}', fontsize=16, fontweight='bold')
            
            axes[i, 1].imshow(x_adj_real, aspect='auto', cmap='turbo', vmin=0, vmax=vmax_adj)
            axes[i, 1].set_title('Input' if i == 0 else '', fontsize=16, fontweight='bold')
            
            axes[i, 2].imshow(x_single_real, aspect='auto', cmap='turbo', vmin=0, vmax=vmax_recon)
            axes[i, 2].set_title('Single-range Output' if i == 0 else '', fontsize=16, fontweight='bold')
            
            axes[i, 3].imshow(x_multi_real, aspect='auto', cmap='turbo', vmin=0, vmax=vmax_recon)
            axes[i, 3].set_title('Multi-range Output' if i == 0 else '', fontsize=16, fontweight='bold')
    
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    #plt.suptitle('Multi-Sample Reconstruction Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved multi-sample comparison to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("POLAR DOMAIN INVERSION COMPARISON")
    print("Single-Range (1D) vs Multi-Range (2D) Denoiser")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Dataset: {DATASET}")
    print(f"  Polar dimensions: {N_RANGES} ranges × {N_THETA} angles")
    print(f"  Measurements per range: {N_V}")
    print(f"  Training samples: {NUM_TRAIN_SAMPLES}")
    print(f"  Test samples: {NUM_TEST_SAMPLES}")
    
    # Set random seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Generate steering matrix
    print("\n[1] Generating steering matrix...")
    A_tensor = generate_steering_matrix(N_V, N_THETA, DEVICE)
    print(f"A_tensor[0, 3, 23] is {A_tensor[0, 3, 23]}")
    print(f"A_tensor[1, 3, 23] is {A_tensor[1, 3, 23]}")
    print(f"  Steering matrix shape: {list(A_tensor.shape)}")
    
    # Load dataset
    print(f"\n[2] Loading {DATASET} data and resizing to polar domain...")
    train_images, test_images, dataset_display_name = load_dataset_polar(
        DATASET, NUM_TRAIN_SAMPLES, NUM_TEST_SAMPLES, N_THETA, N_RANGES, DEVICE
    )
    print(f"  Dataset: {dataset_display_name}")
    print(f"  Training images: {list(train_images.shape)}")
    print(f"  Test images: {list(test_images.shape)}")
    
    # Train single-range denoiser
    single_model, losses_single = train_single_range_denoiser(
        train_images, A_tensor, EPOCHS_SINGLE_RANGE, DEVICE
    )
    
    # Save single-range model
    single_checkpoint_path = get_checkpoint_path('single_range')
    torch.save(single_model.state_dict(), single_checkpoint_path)
    print(f"  Saved model to: {single_checkpoint_path}")
    
    # Train multi-range denoiser
    multi_model, losses_multi = train_multi_range_denoiser(
        train_images, A_tensor, EPOCHS_MULTI_RANGE, DEVICE
    )
    
    # Save multi-range model
    multi_checkpoint_path = get_checkpoint_path('multi_range')
    torch.save(multi_model.state_dict(), multi_checkpoint_path)
    print(f"  Saved model to: {multi_checkpoint_path}")
    
    # Visualization
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Plot training curves
    visualize_training_curves(
        losses_single, losses_multi,
        os.path.join(OUTPUT_DIR, 'training_curves.png')
    )
    
    # Evaluate on test samples
    print("\n[3] Evaluating on test samples...")
    single_model.eval()
    multi_model.eval()
    
    with torch.no_grad():
        # Pick a specific test sample
        test_idx = 0
        test_image = test_images[test_idx:test_idx+1]  # [1, N_ranges, N_theta]
        
        # Generate measurements and adjoint
        y_test = create_measurements(test_image, A_tensor)
        x_adj = apply_adjoint(y_test, A_tensor)
        
        # Apply denoisers
        x_single = single_model(x_adj)
        x_multi = multi_model(x_adj)
        
        # Visualize comparison
        metrics = visualize_comparison(
            test_image[0], x_adj[0], x_single[0], x_multi[0],
            os.path.join(OUTPUT_DIR, 'reconstruction_comparison.png'),
            sample_idx=test_idx,
            dataset_name=dataset_display_name
        )
    
    # Multiple sample visualization
    print("\n[4] Visualizing multiple test samples...")
    visualize_multiple_samples(
        test_images, single_model, multi_model, A_tensor,
        os.path.join(OUTPUT_DIR, 'multi_sample_comparison.png'),
        num_samples=5
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nOutput files saved to: {OUTPUT_DIR}/")
    print("  - reconstruction_comparison.png")
    print("  - multi_sample_comparison.png")
    print("  - training_curves.png")
    print(f"\nCheckpoints saved to:")
    print(f"  - {get_checkpoint_path('single_range')}")
    print(f"  - {get_checkpoint_path('multi_range')}")
    
    # Print final comparison
    improvement_mse = (metrics['single_range']['mse'] - metrics['multi_range']['mse']) / metrics['single_range']['mse'] * 100
    improvement_psnr = metrics['multi_range']['psnr'] - metrics['single_range']['psnr']
    
    print(f"\n  Multi-Range vs Single-Range Improvement:")
    print(f"    MSE reduction: {improvement_mse:.1f}%")
    print(f"    PSNR increase: {improvement_psnr:.2f} dB")
    
    if improvement_mse > 0:
        print("\n  ✓ Multi-Range (2D) denoiser outperforms Single-Range (1D)!")
        print("    This demonstrates the benefit of jointly processing all ranges.")
    else:
        print("\n  Note: Results may vary. Try adjusting hyperparameters or training longer.")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

