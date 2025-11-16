# Supervised Training Guide

This guide explains how to use the new supervised training capabilities in the MIMO SAR framework.

## Overview

The framework now supports three training modes:
1. **Supervised**: Uses ground truth reflectivity x (image domain loss)
2. **Unsupervised**: Uses only measurements y (measurement domain loss) - original mode
3. **Hybrid**: Combines both supervised and unsupervised losses

## Quick Start

### 1. Using the New Training Script

```bash
python train_configurable.py
```

### 2. Configure Training Mode

Edit `train_configurable.py` (lines 27-30):

```python
TRAINING_MODE = 'supervised'    # Options: 'supervised', 'unsupervised', 'hybrid'
SUPERVISED_WEIGHT = 1.0         # Weight for image domain loss
UNSUPERVISED_WEIGHT = 1.0       # Weight for measurement domain loss
```

## Training Modes Explained

### Supervised Mode

**When to use**: When you have ground truth reflectivity x for all training samples

**Loss function**:
```
Loss = ||x_hat - x_gt||²
```

**Configuration**:
```python
TRAINING_MODE = 'supervised'
SUPERVISED_WEIGHT = 1.0
```

**Advantages**:
- Directly optimizes image quality
- Faster convergence
- Better reconstruction quality when ground truth is accurate

**Requirements**:
- Dataset must contain 'x' field with shape [N_samples, N_theta] or [N_theta]

### Unsupervised Mode

**When to use**: When ground truth x is not available (original mode)

**Loss function**:
```
Loss = ||A*x_hat - y||²
```

**Configuration**:
```python
TRAINING_MODE = 'unsupervised'
UNSUPERVISED_WEIGHT = 1.0
```

**Advantages**:
- No ground truth needed
- Works with real measured data
- Physically consistent outputs

**Requirements**:
- Only needs measurements y

### Hybrid Mode

**When to use**: When you have ground truth and want to combine both objectives

**Loss function**:
```
Loss = α*||x_hat - x_gt||² + β*||A*x_hat - y||²
```

**Configuration**:
```python
TRAINING_MODE = 'hybrid'
SUPERVISED_WEIGHT = 1.0      # α
UNSUPERVISED_WEIGHT = 0.1    # β
```

**Advantages**:
- Balances image quality and measurement consistency
- More robust to imperfect ground truth
- Better generalization

**Typical weight settings**:
- Equal weights: `α = β = 1.0` (balanced)
- Favor supervised: `α = 1.0, β = 0.1` (when ground truth is reliable)
- Favor unsupervised: `α = 0.1, β = 1.0` (when ground truth is noisy)

## Dataset Requirements

### Ground Truth Format

Your MATLAB file should contain 'x' with one of these shapes:

1. **Single sample**: `[N_theta]` 
   - Example: `[51]` or `[1001]`
   - Will be replicated for all training samples

2. **Multiple samples**: `[N_samples, N_theta]`
   - Example: `[1000, 51]` - 1000 samples, 51 angle bins each
   - Must match number of measurement samples

### Fallback Behavior

If ground truth is requested but not available:
```
*** WARNING: supervised training requested but ground truth not available!
*** Falling back to UNSUPERVISED training mode ***
```

The training will automatically continue in unsupervised mode.

## Example Workflows

### Example 1: Train with full supervision

```python
# In train_configurable.py
TRAINING_MODE = 'supervised'
SUPERVISED_WEIGHT = 1.0
NUM_EPOCHS = 100

# Run
python train_configurable.py
```

**Expected output**:
```
--- Training Mode: SUPERVISED ---
Ground truth 'x' loaded with shape: (1000, 51)
Stored ground truth 'x' with shape: (1000, 51)
Ground truth available: True
```

### Example 2: Hybrid training with emphasis on supervision

```python
# In train_configurable.py
TRAINING_MODE = 'hybrid'
SUPERVISED_WEIGHT = 1.0      # Image domain
UNSUPERVISED_WEIGHT = 0.1    # Measurement domain
NUM_EPOCHS = 150

# Run
python train_configurable.py
```

**Expected output**:
```
--- Training Mode: HYBRID ---
Training Configuration:
  Mode: hybrid
  Supervised weight: 1.0
  Unsupervised weight: 0.1
  
Epoch [1/150], Step [1/16], Total Loss: 0.123456
  Supervised (image): 0.120000
  Unsupervised (measurement): 0.034560
```

### Example 3: Automatic fallback

```python
# Dataset doesn't have 'x', but request supervised training
TRAINING_MODE = 'supervised'

# Run
python train_configurable.py
```

**Expected output**:
```
WARNING: Ground truth requested but 'x' not found in data file!
Will use unsupervised training mode.

*** WARNING: supervised training requested but ground truth not available!
*** Falling back to UNSUPERVISED training mode ***
```

## Output Files

The training script generates:

1. **Model checkpoint**: `dbp_model.pth`
   - Trained model weights

2. **Loss curves**: `training_loss_{mode}.png`
   - Training loss vs epochs
   - Separate curves for supervised/unsupervised components in hybrid mode
   - Log scale for better visualization

3. **Debug plots**:
   - `train_debug_plot_y.png` - Measurement domain (y vs y_hat)
   - `train_debug_plot_x.png` - Image domain (x_hat vs x_gt if available)

## Comparison with Original train.py

| Feature | train.py (original) | train_configurable.py (new) |
|---------|-------------------|----------------------------|
| Unsupervised training | ✓ | ✓ |
| Supervised training | ✗ | ✓ |
| Hybrid training | ✗ | ✓ |
| Configurable weights | ✗ | ✓ |
| Loss tracking | Basic | Detailed |
| Automatic fallback | ✗ | ✓ |

## Tips and Best Practices

### 1. Start with Supervised if Ground Truth Available
- Supervised training typically converges faster
- Provides better initial results

### 2. Use Hybrid for Robustness
- Balances image quality and physical consistency
- More robust to imperfect ground truth
- Recommended settings: `α=1.0, β=0.1`

### 3. Monitor Both Loss Components
- In hybrid mode, watch both supervised and unsupervised losses
- If one dominates, adjust weights accordingly

### 4. Experiment with Weights
- Try different weight ratios to find optimal balance
- Higher supervised weight → sharper images
- Higher unsupervised weight → better measurement fit

### 5. Ground Truth Quality Matters
- If ground truth is noisy, use hybrid mode with lower supervised weight
- If ground truth is perfect (simulation), use pure supervised mode

## Troubleshooting

### Issue: "Ground truth samples != measurement samples"

**Cause**: Mismatch between number of x and y samples

**Solution**: 
```python
# Check your MATLAB data
# x should be [1000, 51] if y is [1000, 8]
```

### Issue: Training unstable in hybrid mode

**Cause**: Loss components have very different magnitudes

**Solution**: Adjust weights to balance losses
```python
# If supervised loss >> unsupervised loss:
SUPERVISED_WEIGHT = 0.1
UNSUPERVISED_WEIGHT = 1.0

# Or vice versa
```

### Issue: Supervised training not improving

**Possible causes**:
1. Ground truth quality issues
2. Network too small for the problem
3. Learning rate too high/low

**Solutions**:
- Switch to hybrid mode
- Increase NUM_UNROLLS or NUM_ADMM_STEPS
- Adjust LEARNING_RATE

## Advanced: Modifying Data Loader

If you need custom ground truth handling, modify `data_loader.py`:

```python
# Custom ground truth loading
class MIMOSAR_Dataset(Dataset):
    def __init__(self, mat_file_path, return_ground_truth=False,
                 custom_x_key='my_ground_truth'):  # Add custom parameter
        ...
        if return_ground_truth:
            if custom_x_key in data:
                x_complex = data[custom_x_key].astype(np.complex64)
                ...
```

## Next Steps

1. Try all three modes with your dataset
2. Compare results visually using the debug plots
3. Experiment with different weight combinations in hybrid mode
4. Use `test_unsupervised_model.py` to evaluate trained models

For questions or issues, refer to the main README or codebase documentation.

