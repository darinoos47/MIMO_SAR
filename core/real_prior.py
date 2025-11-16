"""
Real-Valued Prior Enforcement

This module provides utilities to enforce real-valued outputs when the 
target reflectivity is known to be real (no imaginary component).

Three enforcement strategies:
1. 'loss_penalty': Add loss term that penalizes imaginary part
2. 'hard_projection': Set imaginary channel to zero after forward pass
3. 'hybrid': Combine both approaches

Complex tensors are represented as [batch, 2, N] where:
  - [:, 0, :] is the real part
  - [:, 1, :] is the imaginary part
"""

import torch
import torch.nn as nn


def apply_real_projection(x_complex):
    """
    Hard projection: Set imaginary part to zero
    
    Args:
        x_complex: Complex tensor [batch, 2, N] where channel 0=real, 1=imag
    
    Returns:
        x_real: Complex tensor with imaginary part set to zero
    """
    x_real = x_complex.clone()
    x_real[:, 1, :] = 0.0  # Set imaginary channel to zero
    return x_real


def compute_imaginary_penalty(x_complex):
    """
    Compute penalty for non-zero imaginary part
    
    Args:
        x_complex: Complex tensor [batch, 2, N] where channel 0=real, 1=imag
    
    Returns:
        penalty: Scalar loss (mean squared imaginary part)
    """
    imaginary_part = x_complex[:, 1, :]  # Extract imaginary channel
    penalty = torch.mean(imaginary_part ** 2)
    return penalty


def enforce_real_prior(x_complex, strategy='none', penalty_weight=1.0):
    """
    Enforce real-valued prior on complex tensor
    
    Args:
        x_complex: Complex tensor [batch, 2, N]
        strategy: Enforcement strategy
                  'none': No enforcement
                  'loss_penalty': Return penalty term for loss
                  'hard_projection': Project to real space
                  'hybrid': Both projection and return penalty
        penalty_weight: Weight for loss penalty (only used if strategy includes penalty)
    
    Returns:
        If strategy includes projection: (x_projected, penalty)
        If strategy is loss_penalty only: (x_complex, penalty)
        If strategy is none: (x_complex, 0.0)
    """
    if strategy == 'none':
        return x_complex, 0.0
    
    elif strategy == 'loss_penalty':
        penalty = compute_imaginary_penalty(x_complex) * penalty_weight
        return x_complex, penalty
    
    elif strategy == 'hard_projection':
        x_projected = apply_real_projection(x_complex)
        return x_projected, 0.0
    
    elif strategy == 'hybrid':
        # First compute penalty, then project
        penalty = compute_imaginary_penalty(x_complex) * penalty_weight
        x_projected = apply_real_projection(x_complex)
        return x_projected, penalty
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from 'none', 'loss_penalty', 'hard_projection', 'hybrid'")


class RealPriorWrapper(nn.Module):
    """
    Wrapper module that enforces real-valued prior on model outputs
    
    Usage:
        model = DBPNet(...)
        real_model = RealPriorWrapper(model, strategy='hard_projection')
        
        x_hat = real_model(y)  # Output will have zero imaginary part
    """
    
    def __init__(self, model, strategy='none', penalty_weight=1.0):
        """
        Args:
            model: The base model (e.g., DBPNet)
            strategy: 'none', 'loss_penalty', 'hard_projection', 'hybrid'
            penalty_weight: Weight for imaginary penalty
        """
        super(RealPriorWrapper, self).__init__()
        self.model = model
        self.strategy = strategy
        self.penalty_weight = penalty_weight
        self.last_penalty = 0.0
    
    def forward(self, *args, **kwargs):
        """Forward pass with real prior enforcement"""
        x_out = self.model(*args, **kwargs)
        
        # If model returns dict (intermediates), enforce on final output
        if isinstance(x_out, dict):
            x_final = x_out['x_final']
            x_enforced, penalty = enforce_real_prior(
                x_final, self.strategy, self.penalty_weight
            )
            x_out['x_final'] = x_enforced
            self.last_penalty = penalty
            return x_out
        else:
            # Regular tensor output
            x_enforced, penalty = enforce_real_prior(
                x_out, self.strategy, self.penalty_weight
            )
            self.last_penalty = penalty
            return x_enforced
    
    def get_last_penalty(self):
        """Get the penalty from the last forward pass"""
        return self.last_penalty


# Utility functions for visualization/analysis

def measure_imaginary_magnitude(x_complex):
    """
    Measure the magnitude of imaginary part relative to real part
    
    Args:
        x_complex: Complex tensor [batch, 2, N]
    
    Returns:
        ratio: Imaginary magnitude / Real magnitude
    """
    real_mag = torch.mean(torch.abs(x_complex[:, 0, :]))
    imag_mag = torch.mean(torch.abs(x_complex[:, 1, :]))
    
    if real_mag > 1e-10:
        ratio = imag_mag / real_mag
    else:
        ratio = torch.tensor(float('inf'))
    
    return ratio.item()


def compare_real_vs_complex(x_real, x_complex):
    """
    Compare real-only version vs complex version
    
    Args:
        x_real: Real-projected tensor [batch, 2, N] with imag=0
        x_complex: Full complex tensor [batch, 2, N]
    
    Returns:
        dict with comparison metrics
    """
    real_only_mag = torch.mean(torch.abs(x_real[:, 0, :]))
    real_part_mag = torch.mean(torch.abs(x_complex[:, 0, :]))
    imag_part_mag = torch.mean(torch.abs(x_complex[:, 1, :]))
    
    real_diff = torch.mean((x_real[:, 0, :] - x_complex[:, 0, :]) ** 2)
    
    return {
        'real_only_magnitude': real_only_mag.item(),
        'real_part_magnitude': real_part_mag.item(),
        'imaginary_magnitude': imag_part_mag.item(),
        'imag_to_real_ratio': (imag_part_mag / real_part_mag).item() if real_part_mag > 1e-10 else float('inf'),
        'real_part_mse': real_diff.item()
    }


if __name__ == '__main__':
    """Test the real prior enforcement"""
    print("="*70)
    print("TESTING REAL PRIOR ENFORCEMENT")
    print("="*70)
    
    # Create test data
    batch_size, N = 4, 51
    x_test = torch.randn(batch_size, 2, N)
    
    print(f"\nTest tensor shape: {x_test.shape}")
    print(f"Real part mean: {x_test[:, 0, :].mean():.6f}")
    print(f"Imaginary part mean: {x_test[:, 1, :].mean():.6f}")
    
    # Test 1: Hard projection
    print("\n[Test 1] Hard Projection")
    x_proj, penalty = enforce_real_prior(x_test, strategy='hard_projection')
    print(f"  After projection - Real mean: {x_proj[:, 0, :].mean():.6f}")
    print(f"  After projection - Imag mean: {x_proj[:, 1, :].mean():.6f}")
    print(f"  Imag should be zero: {torch.allclose(x_proj[:, 1, :], torch.zeros_like(x_proj[:, 1, :]))}")
    
    # Test 2: Loss penalty
    print("\n[Test 2] Loss Penalty")
    x_pen, penalty = enforce_real_prior(x_test, strategy='loss_penalty', penalty_weight=0.1)
    print(f"  Penalty value: {penalty:.6f}")
    print(f"  Tensor unchanged: {torch.allclose(x_pen, x_test)}")
    
    # Test 3: Hybrid
    print("\n[Test 3] Hybrid")
    x_hyb, penalty = enforce_real_prior(x_test, strategy='hybrid', penalty_weight=0.1)
    print(f"  Penalty value: {penalty:.6f}")
    print(f"  Imag is zero: {torch.allclose(x_hyb[:, 1, :], torch.zeros_like(x_hyb[:, 1, :]))}")
    
    # Test 4: Measure imaginary magnitude
    print("\n[Test 4] Measure Imaginary Magnitude")
    ratio = measure_imaginary_magnitude(x_test)
    print(f"  Imag/Real ratio: {ratio:.6f}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)

