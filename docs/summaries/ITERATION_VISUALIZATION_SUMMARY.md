# Iteration Visualization Feature - Summary

## What Was Added

Your framework now automatically visualizes how the network progressively refines the reconstruction through each unrolled iteration.

## Key Features

### 1. Modified Model (`models.py`)
Added `return_intermediates` parameter to `DBPNet.forward()`:

```python
# Normal forward pass
x = model(y)  # Returns final output

# With intermediates
intermediates = model(y, return_intermediates=True)
# Returns dictionary with:
#   - x_init: Initial estimate
#   - x_after_denoiser: List of outputs after each denoiser
#   - x_after_admm: List of outputs after each ADMM
#   - x_final: Final output
```

### 2. New Visualization Module (`visualization_utils.py`)

Four visualization functions:

1. **`plot_unrolled_iterations()`**: Detailed view of each iteration (image domain)
   - Shows denoiser and ADMM outputs side-by-side
   - Ground truth comparison (if available)

2. **`plot_iteration_comparison()`**: All iterations overlaid (image domain)
   - Quick comparison of progression
   - Color-coded by iteration

3. **`plot_measurement_domain_progression()`**: Physical consistency check (NEW!)
   - Shows y = A*x for each iteration
   - Displays magnitude, real, and imaginary parts
   - Verifies measurement domain fit

4. **`plot_mse_progression()`**: Quantitative improvement
   - MSE after each iteration
   - Requires ground truth

### 3. Automatic Integration

Both `train.py` and `train_configurable.py` now automatically generate these plots after training completes.

## Generated Files

After training, you'll see:
```
train_unrolled_iterations.png               # Image domain: detailed iteration-by-iteration
train_iteration_comparison.png              # Image domain: all iterations overlaid  
train_measurement_domain_progression.png    # Measurement domain: y = A*x per iteration (NEW!)
train_mse_progression.png                   # MSE progression (if ground truth available)
```

## Usage

### Automatic (During Training)

Just run training as usual:
```bash
python train.py
# or
python train_configurable.py
```

Visualizations are generated automatically at the end.

### Manual (For Analysis)

```python
from visualization_utils import plot_unrolled_iterations

# Load model and data...
with torch.no_grad():
    intermediates = model(y, return_intermediates=True)
    
plot_unrolled_iterations(intermediates, x_gt=x_gt, 
                         save_path='my_analysis.png')
```

## What You Can Learn

1. **Does each iteration help?**
   - See if output improves with each iteration
   - Identify when convergence occurs

2. **Denoiser vs ADMM roles**
   - See what denoiser contributes (noise removal, enhancement)
   - See what ADMM contributes (physical consistency)

3. **Optimal number of iterations**
   - If plateau early → can reduce NUM_UNROLLS
   - If still improving → can increase NUM_UNROLLS

4. **Training quality**
   - Compare to ground truth visually
   - Quantify improvement via MSE

5. **Physical consistency (NEW!)**
   - See how well A*x matches measured y
   - Verify network respects physical forward model
   - Diagnose if unsupervised training is working correctly
   - Check if ADMM hyperparameters (ρ, ε) are appropriate

## Example Interpretation

If you train with 5 iterations and see:
- **Iteration 1**: Large improvement from initialization
- **Iteration 2-3**: Continued refinement
- **Iteration 4-5**: Almost identical to iteration 3

**Conclusion**: 3 iterations sufficient, can reduce from 5 to 3 for faster training.

## Backward Compatible

- Old code still works (default `return_intermediates=False`)
- No changes needed to test scripts
- Optional feature, can disable if not needed

## Files Modified

- `models.py`: Added `return_intermediates` parameter
- `train.py`: Added visualization calls
- `train_configurable.py`: Added visualization calls
- `visualization_utils.py`: New file with plotting functions
- `ITERATION_VISUALIZATION_GUIDE.md`: Detailed documentation

## Testing

Tested with:
- Unsupervised training (no ground truth)
- Supervised training (with ground truth)
- 1-5 iterations
- All visualizations generating correctly

## Quick Reference

**To adjust number of iterations visualized:**
```python
# In train.py or train_configurable.py
NUM_UNROLLS = 3  # Will show 3 iterations
```

**To change which sample is plotted:**
```python
# In visualization calls
plot_unrolled_iterations(intermediates, sample_idx=5)  # Plot sample 5
```

**To disable (if desired):**
```python
# Comment out or remove the visualization section in train scripts
```

## Documentation

- Full guide: `ITERATION_VISUALIZATION_GUIDE.md`
- This summary: `ITERATION_VISUALIZATION_SUMMARY.md`

