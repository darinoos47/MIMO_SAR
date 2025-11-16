# Measurement Domain Visualization - Quick Guide

## What's New

In addition to showing how the **image domain** (x - reflectivity) evolves through iterations, you can now see how the **measurement domain** (y = A*x) progresses!

## Why This Matters

### For Unsupervised Training
- Unsupervised loss is `||A*x - y||²` (measurement domain)
- This plot shows **exactly what the network is being trained on**
- Verifies the network is fitting the measurements correctly

### For Supervised/Hybrid Training
- Even with ground truth, physical consistency matters
- Shows if network is "cheating" by ignoring measurements
- Diagnostic for balancing supervised vs unsupervised weights

### For All Training Modes
- Validates ADMM is working correctly
- Shows if ρ and ε hyperparameters are appropriate
- Reveals convergence in the physical constraint

## What You See

### File Generated
`train_measurement_domain_progression.png`

### Layout
- **3 columns**: Magnitude | Real Part | Imaginary Part
- **Multiple rows**: One per iteration (init + all iterations)
- **Red circles** (●): Measured y (ground truth)
- **Blue squares** (■): A*x from network output

### Perfect Fit
When training works well, blue squares should **overlap with red circles** by the final iteration!

## Interpretation Guide

### Good Training
```
Iteration 1: Blue squares close to red circles
Iteration 2: Even closer  
Iteration 3: Almost perfectly overlapped
...
Final: Perfect overlap (measurement domain loss ≈ 0)
```
✓ ADMM is working
✓ Physical consistency enforced
✓ Training successful

### Problem: Not Converging
```
Iteration 1: Blue squares far from red
Iteration 2: Still far
Iteration 3: Still far
...
Final: Still significant gap
```
✗ Possible issues:
- ADMM steps too few (increase `NUM_ADMM_STEPS`)
- ρ too small (not enforcing constraint strongly enough)
- ε too large (allowing too much slack)
- Network not trained enough

### Problem: Too Aggressive
```
Initialization: Blue squares reasonably close
Iteration 1: Fits perfectly to measurements
Iteration 2-5: No further improvement in image domain
```
⚠️ Overfitting to measurements:
- ρ too large (over-emphasizing measurements)
- ε too small (no slack allowed)
- May sacrifice image quality for measurement fit

## Real vs Imaginary Parts

**Important**: Both real and imaginary parts should fit well!

- If only real part fits: Complex arithmetic issue
- If only imaginary part fits: Check conjugate operations  
- If neither fits: ADMM not working properly
- If both fit: ✓ Correct implementation

## Comparison with Other Plots

### Image Domain vs Measurement Domain

| Plot Type | Shows | Optimizes |
|-----------|-------|-----------|
| `train_unrolled_iterations.png` | x (reflectivity) | Image quality |
| `train_measurement_domain_progression.png` | y = A*x (measurements) | Physical consistency |

**Both should improve together** for good training!

### When to Worry

**Red flag 1**: Image domain looks great, measurement domain doesn't fit
→ Network ignoring physical constraints (supervised weight too high)

**Red flag 2**: Measurement domain fits perfectly, image domain looks noisy
→ Too much emphasis on measurements (unsupervised weight too high)

**Good balance**: Both improve together smoothly

## Practical Examples

### Example 1: Unsupervised Training Working Well

**Measurement Domain Plot**:
- Initialization: Some mismatch
- Iteration 1: Much better fit
- Iteration 2-3: Progressive improvement
- Final: Near-perfect overlap

**What this means**: Unsupervised training successfully learning from measurements alone!

### Example 2: Hybrid Training - Good Balance

**Measurement Domain Plot**:
- Converges in 2-3 iterations
- Small residual acceptable (ε parameter working)

**Image Domain Plot**:
- Also converges in 2-3 iterations
- Close to ground truth

**What this means**: Good balance between supervised and unsupervised losses!

### Example 3: Need to Adjust Hyperparameters

**Measurement Domain Plot**:
- Fits measurements too aggressively at iteration 1
- No room for image quality refinement

**Solution**:
```python
# In train_configurable.py
SUPERVISED_WEIGHT = 1.0     # Keep
UNSUPERVISED_WEIGHT = 0.1   # Reduce from 1.0 → 0.1
```

## How to Use

### Automatic (Default)
Just run training - plot is generated automatically:
```bash
python train.py
# or
python train_configurable.py
```

### Manual
```python
from visualization_utils import plot_measurement_domain_progression

# After getting intermediates from model
plot_measurement_domain_progression(intermediates, y_gt=y_batch, 
                                   A_tensor=A_tensor, sample_idx=0)
```

## Key Insights

1. **Magnitude alone is not enough** - check real/imag separately
2. **All 8 antennas matter** - each point should fit
3. **Initialization quality matters** - if A^H*y is already close, easier to converge
4. **Compare with image domain plots** - both should improve together

## Troubleshooting

### Gap remains large after many iterations

**Check**:
1. Are you using enough ADMM steps? Try increasing `NUM_ADMM_STEPS`
2. Is ρ too small? (Check model's learned `log_rho` parameter)
3. Is the model trained? (Random weights won't fit)

### Fits too quickly (iteration 1)

**Check**:
1. Is ρ too large? Might be over-constraining
2. Is supervised weight too low in hybrid mode?
3. Consider reducing `UNSUPERVISED_WEIGHT`

### Real and imaginary don't match symmetrically

**Check**:
1. Data loading - are complex numbers handled correctly?
2. Forward model - is A matrix correct?
3. Complex arithmetic in utils.py

## Next Steps

After viewing measurement domain progression:

1. **Compare with image domain plots** - do they tell the same story?
2. **Check training loss curves** - does measurement loss match visual fit?
3. **Adjust weights if needed** - balance supervised vs unsupervised
4. **Tune ADMM parameters** - if convergence is too slow/fast

For more details, see `ITERATION_VISUALIZATION_GUIDE.md`.

