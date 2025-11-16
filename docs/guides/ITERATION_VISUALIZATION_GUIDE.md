# Iteration Visualization Guide

## Overview

The framework now includes powerful visualization tools to see how the network progressively refines the image reconstruction through each unrolled iteration. This helps you understand:

1. How much each iteration improves the result
2. The role of the denoiser vs ADMM at each step
3. Whether more iterations would help

## What Gets Visualized

After training completes, you'll get **4 types of plots**:

### 1. Detailed Iteration-by-Iteration Plot (`train_unrolled_iterations.png`)

Shows the complete progression through all iterations:
- **Top row**: Initial estimate (A^H @ y - matched filter)
- **Following rows**: For each iteration:
  - **Left column**: After CNN denoiser
  - **Right column**: After ADMM (data consistency)

**What to look for**:
- Does the denoiser remove noise effectively?
- Does ADMM enforce physical constraints?
- Are later iterations still improving?

### 2. Iteration Comparison Plot (`train_iteration_comparison.png`)

All iterations overlaid on one plot for easy comparison:
- Different colors for each iteration
- Shows progressive refinement at a glance
- Ground truth overlaid (if available)

**What to look for**:
- How quickly does it converge?
- Are later iterations making meaningful changes?
- Is final output close to ground truth?

### 3. Measurement Domain Progression Plot (`train_measurement_domain_progression.png`)

**NEW! Shows physical consistency through iterations**

Displays y = A*x for each iteration:
- **3 columns**: Magnitude, Real Part, Imaginary Part
- **Rows**: Initialization + Each iteration
- **Red circles**: Measured y (ground truth)
- **Blue squares**: A*x at each iteration

**What to look for**:
- How quickly does A*x match measured y?
- Is ADMM effectively enforcing y ≈ A*x?
- Are all antenna measurements converging?
- Real and imaginary parts both fitting well?

**Why this matters**:
- Shows physical consistency (key for unsupervised training!)
- Reveals if network "cheats" by ignoring measurements
- Diagnostic for ADMM hyperparameters (ρ, ε)

### 4. MSE Progression Plot (`train_mse_progression.png`)

**Only generated if ground truth is available**

Quantitative measure of improvement:
- X-axis: Iteration number
- Y-axis: Mean Squared Error vs ground truth
- Shows percentage improvement

**What to look for**:
- Is MSE steadily decreasing?
- When does it plateau?
- How much improvement per iteration?

## How to Use

### Automatic Generation (During Training)

The visualizations are **automatically generated** after training completes:

```bash
python train.py                 # For unsupervised training
python train_configurable.py    # For supervised/hybrid training
```

At the end of training, you'll see:
```
Generating progressive refinement visualizations...
  Creating detailed iteration-by-iteration plot...
  Unrolled iterations plot saved to train_unrolled_iterations.png
  Creating iteration comparison plot...
  Iteration comparison plot saved to train_iteration_comparison.png
  Creating MSE progression plot...
  MSE progression plot saved to train_mse_progression.png
  Creating measurement domain progression plot...
  Measurement domain progression plot saved to train_measurement_domain_progression.png
Progressive refinement visualizations complete!
```

### Manual Generation (For Testing)

You can also generate these plots manually for testing:

```python
from models import DBPNet
from visualization_utils import (plot_unrolled_iterations, 
                                 plot_iteration_comparison,
                                 plot_mse_progression)

# Load model and data
model = DBPNet(A_tensor, num_iterations=5).cuda()
model.load_state_dict(torch.load('dbp_model.pth'))
model.eval()

# Get sample
y_sample = ...  # [1, 2, N_v]
x_gt = ...      # [1, 2, N_theta] (optional)

# Run with intermediates
with torch.no_grad():
    intermediates = model(y_sample, return_intermediates=True)

# Generate plots
plot_unrolled_iterations(intermediates, x_gt=x_gt, sample_idx=0)
plot_iteration_comparison(intermediates, x_gt=x_gt, sample_idx=0)
plot_mse_progression(intermediates, x_gt=x_gt, sample_idx=0)  # Needs ground truth
```

## Understanding the Model Architecture

The visualization helps you see how the **unrolled architecture** works:

```
Input: y [measurements]
   ↓
x₀ = A^H @ y [initialization]
   ↓
┌─────────────────────────────┐
│ Iteration 1:                │
│  r₁ = Denoiser(x₀)          │ ← Removes noise, artifacts
│  x₁ = ADMM(r₁, y)           │ ← Enforces y ≈ A*x₁
└─────────────────────────────┘
   ↓
┌─────────────────────────────┐
│ Iteration 2:                │
│  r₂ = Denoiser(x₁)          │ ← Further refinement
│  x₂ = ADMM(r₂, y)           │ ← Maintains consistency
└─────────────────────────────┘
   ↓
... (repeat N times)
   ↓
Output: x_N [final estimate]
```

## Examples of What You Might See

### Good Training (Converging Well)

**Iteration Comparison Plot**:
- Clear separation between iteration curves
- Each iteration closer to ground truth
- Final iteration very close to ground truth

**MSE Progression**:
- Steady decrease
- Large drop in first few iterations
- Smaller improvements in later iterations

**Interpretation**: Network is working well, appropriate number of iterations.

### Too Few Iterations

**Iteration Comparison Plot**:
- Final iteration still far from ground truth
- Curves still changing significantly in last iteration

**MSE Progression**:
- Still decreasing steeply at final iteration
- No plateau

**Solution**: Increase `NUM_UNROLLS` in training configuration.

### Too Many Iterations

**Iteration Comparison Plot**:
- Last few iterations almost identical
- Minimal visual difference between late iterations

**MSE Progression**:
- Plateaus after iteration 2-3
- Flat line for remaining iterations

**Solution**: Can reduce `NUM_UNROLLS` for faster training without loss of quality.

### Denoiser vs ADMM Balance

**Detailed Iteration Plot**:
- Large changes after denoiser → Denoiser is doing most of the work
- Large changes after ADMM → ADMM making major corrections

**Interpretation**:
- If denoiser changes are large but ADMM changes are small: Good balance, denoiser learned good prior
- If ADMM changes are large: Denoiser output not physically consistent, may need more ADMM steps

## Configuration

### Number of Iterations Visualized

By default, shows all `NUM_UNROLLS` iterations from your training configuration.

To train with different numbers of iterations:

```python
# In train.py or train_configurable.py
NUM_UNROLLS = 5      # Number of iterations to visualize
NUM_ADMM_STEPS = 3   # ADMM steps per iteration
```

### Changing Sample to Visualize

By default, visualizes sample index 0 from the last batch. To change:

```python
# In visualization function calls
plot_unrolled_iterations(intermediates, x_gt=x_gt, sample_idx=5)  # Use sample 5
```

### Custom Visualization

The `visualization_utils.py` module provides building blocks. You can create custom plots:

```python
# Access intermediate outputs
intermediates = model(y, return_intermediates=True)

# Available data:
x_init = intermediates['x_init']                    # [batch, 2, N_theta]
x_after_denoiser = intermediates['x_after_denoiser']  # List of [batch, 2, N_theta]
x_after_admm = intermediates['x_after_admm']          # List of [batch, 2, N_theta]
x_final = intermediates['x_final']                    # [batch, 2, N_theta]

# Create your own analysis...
```

## Tips for Interpretation

### 1. Check Initialization Quality
- Is x_init (A^H @ y) a reasonable starting point?
- If initialization is very poor, network has to work harder

### 2. Monitor Denoiser Effect
- Does denoiser smooth/enhance appropriately?
- Or is it over-smoothing and losing detail?

### 3. Verify ADMM Consistency
- After ADMM, solution should be closer to satisfying y = A*x
- Large ADMM corrections may indicate denoiser is "hallucinating"

### 4. Convergence Check
- Does solution stabilize after a few iterations?
- Or still changing significantly at the end?

### 5. Supervised vs Unsupervised
- **Supervised**: MSE plot directly shows image domain error
- **Unsupervised**: Look at measurement domain consistency instead

## Troubleshooting

### "No ground truth available" (for MSE plot)

**Cause**: Running in unsupervised mode or ground truth not in dataset

**Solution**: 
- Use supervised/hybrid mode: `TRAINING_MODE = 'hybrid'` in `train_configurable.py`
- Or accept that MSE plot won't be generated (other plots still work)

### Plots look identical across iterations

**Possible causes**:
1. Network not trained yet (random weights)
2. Learning rate too low
3. Already converged (good!)

**Solution**: Check training loss curves to confirm convergence

### Memory errors when visualizing

**Cause**: Too many iterations or large batch size

**Solution**: Visualization uses `.detach().clone()` to avoid memory issues, but if still problems:
- Reduce batch size
- Reduce number of iterations
- Visualize on CPU: move data to CPU before plotting

## File Outputs

After training completes, you'll have these files:

```
train_unrolled_iterations.png               # Image domain: detailed iteration-by-iteration
train_iteration_comparison.png              # Image domain: all iterations overlaid
train_measurement_domain_progression.png    # Measurement domain: y = A*x per iteration (NEW!)
train_mse_progression.png                   # MSE vs iteration (if ground truth available)
```

For testing runs (manual generation):
```
test_unrolled_iterations.png
test_iteration_comparison.png
test_measurement_domain_progression.png
test_mse_progression.png
```

## Next Steps

After viewing the visualizations:

1. **Adjust NUM_UNROLLS** based on when convergence plateaus
2. **Modify architecture** if denoiser/ADMM balance seems off
3. **Compare training modes** (supervised vs unsupervised vs hybrid)
4. **Tune hyperparameters** based on observed behavior

For questions or issues, refer to the main documentation or check `visualization_utils.py` source code.

