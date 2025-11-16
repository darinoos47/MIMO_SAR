# Denoiser Measurement Domain Visualization

## Overview

The `train_denoiser_only.py` script now includes **measurement domain visualization** to help you verify that the denoiser preserves consistency with the forward model.

---

## What Was Added

### Measurement Domain Analysis (Section 7b)

After training the denoiser, the script now:

1. **Applies forward operator** to denoiser output: `y_hat = A @ x_denoised`
2. **Compares** with actual measurements: `y_gt`
3. **Visualizes** the difference in magnitude, real, and imaginary parts
4. **Calculates** measurement domain MSE

---

## Generated Outputs

When you run `train_denoiser_only.py`, you'll get:

### 1. Image Domain Visualization

**File**: `denoiser_output_<mode>.png`

Shows the denoiser output in the image domain (reflectivity vs. angle):
- Noisy input (A^H @ y)
- Denoiser output
- Ground truth (if available)

### 2. Measurement Domain Visualization (NEW!)

**File**: `denoiser_measurement_domain_<mode>.png`

Shows how well the denoiser preserves measurement consistency:

**Three subplots:**
- **Magnitude**: `|y_gt|` vs. `|y_hat|`
- **Real Part**: `Re(y_gt)` vs. `Re(y_hat)`
- **Imaginary Part**: `Im(y_gt)` vs. `Im(y_hat)`

Where:
- `y_gt`: Actual measurements
- `y_hat = A @ x_denoised`: Predicted measurements from denoiser output

### 3. Measurement Domain MSE (Terminal Output)

The script prints:
```
Measurement Domain MSE: 7.016613e-02
```

This quantifies how well the denoiser output satisfies the forward model.

---

## What This Tells You

### For Unsupervised Training

Since unsupervised loss is `MSE(A @ x_denoised, y)`, the measurement domain plot shows:
- **How well the denoiser learned** to preserve measurement consistency
- **If the loss is working correctly** (y_hat should match y_gt closely)
- **Where the denoiser struggles** (which measurements have larger errors)

### For Supervised Training

Since supervised loss is `MSE(x_denoised, x_gt)`, the measurement domain plot shows:
- **If the denoiser is overfitting** to ground truth at the expense of measurement consistency
- **The trade-off** between image domain accuracy and measurement domain fidelity
- **If the forward model is accurate** (good x_gt should give y_hat ≈ y_gt)

---

## Interpretation Guide

### Good Denoiser

✅ **Unsupervised mode**: y_hat closely matches y_gt (curves overlap)
✅ **Supervised mode**: y_hat reasonably matches y_gt (some deviation is OK)
✅ **Measurement Domain MSE**: Low value (e.g., < 1e-3)

### Problematic Denoiser

❌ **Large gaps** between y_hat and y_gt
❌ **Very high Measurement Domain MSE** (e.g., > 1e-1)
❌ **Unsupervised mode**: Indicates training failure
❌ **Supervised mode**: Indicates model mismatch or overfitting

---

## Example Usage

```bash
# Train denoiser (supervised)
python train_denoiser_only.py
# DENOISER_TRAINING_MODE = 'supervised'

# Check outputs:
# 1. denoiser_output_supervised.png (image domain)
# 2. denoiser_measurement_domain_supervised.png (NEW! measurement domain)
# 3. Terminal: Measurement Domain MSE
```

---

## Technical Details

### Forward Pass Implementation

```python
# Apply forward model to denoiser output
y_hat = complex_matmul(A_batch_tensor, x_denoised)

# Extract first sample
y_gt_sample = y_batch[0]      # Shape: [2, M_rx]
y_hat_sample = y_hat[0]       # Shape: [2, M_rx]

# Convert to complex numpy for plotting
y_gt_complex = torch.view_as_complex(y_gt_sample.permute(1, 0).contiguous())
y_hat_complex = torch.view_as_complex(y_hat_sample.permute(1, 0).contiguous())

# Extract magnitude, real, imaginary
y_gt_mag = np.abs(y_gt_complex)
y_gt_real = np.real(y_gt_complex)
y_gt_imag = np.imag(y_gt_complex)
```

### MSE Calculation

```python
# Measurement domain MSE (averaged over all samples in batch)
mse_measurement = torch.mean((y_hat - y_batch) ** 2).item()
```

---

## Benefits

1. **Verification**: Confirm denoiser is working as expected
2. **Debugging**: Identify if denoiser is violating forward model
3. **Comparison**: Compare supervised vs. unsupervised training
4. **Understanding**: Visualize the impact of denoiser on measurements
5. **Trust**: Ensure denoiser output is physically consistent

---

## Relation to Full Network

This measurement domain visualization is similar to what's available in the full network training (`train_configurable.py`), but specifically for the standalone denoiser:

| Feature | `train_denoiser_only.py` | `train_configurable.py` |
|---------|-------------------------|-------------------------|
| **Image Domain Plot** | ✅ Denoiser output | ✅ Unrolled iterations |
| **Measurement Domain Plot** | ✅ **NEW!** Denoiser output | ✅ Unrolled iterations |
| **Training Loss Curve** | ✅ | ✅ |
| **Intermediate Steps** | N/A (single step) | ✅ (multiple unrolled steps) |

---

## Summary

The denoiser training script now provides comprehensive visualization in **both domains**:

1. **Image Domain** (`denoiser_output_<mode>.png`): Shows reflectivity reconstruction
2. **Measurement Domain** (`denoiser_measurement_domain_<mode>.png`): Shows measurement consistency

This helps you understand and verify the denoiser's behavior before using it in the two-stage training pipeline!

---

## Next Steps

After training the denoiser and reviewing both visualizations:

1. **If measurement consistency is good**: Proceed to Stage 2
   ```bash
   python train_configurable.py  # TRAINING_STRATEGY = 'two_stage'
   ```

2. **If measurement consistency is poor**:
   - Adjust `NUM_EPOCHS` or `LEARNING_RATE`
   - Try different training mode (supervised ↔ unsupervised)
   - Check data quality

3. **Compare training modes**:
   - Run with `DENOISER_TRAINING_MODE = 'supervised'`
   - Run with `DENOISER_TRAINING_MODE = 'unsupervised'`
   - Compare the measurement domain plots

Enjoy the enhanced visualization! 🎉

