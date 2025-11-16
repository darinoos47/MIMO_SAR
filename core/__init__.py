"""
MIMO SAR Core Modules

This package contains the core functionality for MIMO SAR deep learning:
- models: DBPNet architecture (CNN Denoiser + ADMM)
- data_loader: Dataset loader for .mat files
- utils: Complex-valued operations and utilities
- visualization_utils: Plotting and visualization functions
- real_prior: Real-valued prior enforcement utilities
"""

from .models import CNNDenoiser, DCLayer_ADMM, DBPNet
from .data_loader import MIMOSAR_Dataset
from .utils import complex_matmul, complex_conj_transpose_matmul, complex_to_tensor
from .visualization_utils import (
    plot_unrolled_iterations,
    plot_iteration_comparison,
    plot_mse_progression,
    plot_measurement_domain_progression
)
from .real_prior import enforce_real_prior, measure_imaginary_magnitude

__all__ = [
    # Models
    'CNNDenoiser',
    'DCLayer_ADMM',
    'DBPNet',
    # Data
    'MIMOSAR_Dataset',
    # Utils
    'complex_matmul',
    'complex_conj_transpose_matmul',
    'complex_to_tensor',
    # Visualization
    'plot_unrolled_iterations',
    'plot_iteration_comparison',
    'plot_mse_progression',
    'plot_measurement_domain_progression',
    # Real Prior
    'enforce_real_prior',
    'measure_imaginary_magnitude',
]

