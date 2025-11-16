# Two-Stage Training - Quick Reference

## Overview

The framework now supports **two training strategies**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **End-to-End** | Train denoiser + ADMM jointly | Simple workflow, optimal joint optimization |
| **Two-Stage** | Train denoiser first, then use with ADMM | Modular training, faster experimentation |

---

## Quick Start

### Option 1: End-to-End Training (Default)

```bash
# Edit train_configurable.py
# TRAINING_STRATEGY = 'end_to_end'
# TRAINING_MODE = 'supervised'  # or 'unsupervised', 'hybrid'

python train_configurable.py
```

---

### Option 2: Two-Stage Training

**Step 1: Train Denoiser**

```bash
# Edit train_denoiser_only.py
# DENOISER_TRAINING_MODE = 'unsupervised'  # or 'supervised'

python train_denoiser_only.py
```

**Output**: `denoiser_pretrained.pth`

**Step 2: Train Full Network**

```bash
# Edit train_configurable.py
# TRAINING_STRATEGY = 'two_stage'
# PRETRAINED_DENOISER_PATH = 'denoiser_pretrained.pth'
# FREEZE_DENOISER = True  # or False for fine-tuning
# TRAINING_MODE = 'supervised'  # or 'unsupervised', 'hybrid'

python train_configurable.py
```

---

## Configuration Options

### `train_denoiser_only.py`

```python
# Denoiser Training Mode
DENOISER_TRAINING_MODE = 'unsupervised'  # or 'supervised'

# For supervised mode
NOISE_LEVEL = 0.1  # Synthetic noise std dev

# Training params
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
```

### `train_configurable.py`

```python
# Training Strategy
TRAINING_STRATEGY = 'end_to_end'  # or 'two_stage'

# Two-Stage Options
PRETRAINED_DENOISER_PATH = 'denoiser_pretrained.pth'
FREEZE_DENOISER = True  # True: freeze denoiser, False: fine-tune

# Training Mode (applies to full network)
TRAINING_MODE = 'supervised'  # or 'unsupervised', 'hybrid'

# Loss weights (for hybrid mode)
SUPERVISED_WEIGHT = 1.0
UNSUPERVISED_WEIGHT = 1.0
```

---

## Training Modes Summary

### Denoiser Training Modes (Stage 1)

| Mode | Input | Target | Loss |
|------|-------|--------|------|
| **Unsupervised** | `A^H @ y` | - | `MSE(A @ x_denoised, y)` |
| **Supervised** | `A^H @ y + noise` | `x_gt` | `MSE(x_denoised, x_gt)` |

### Full Network Training Modes (Stage 2 or End-to-End)

| Mode | Loss | Requires Ground Truth |
|------|------|----------------------|
| **Supervised** | `MSE(x_hat, x_gt)` | ✅ Yes |
| **Unsupervised** | `MSE(A @ x_hat, y)` | ❌ No |
| **Hybrid** | `α*supervised + β*unsupervised` | ✅ Yes |

---

## Denoiser Handling in Two-Stage

| Option | `FREEZE_DENOISER` | Behavior | When to Use |
|--------|-------------------|----------|-------------|
| **Frozen** | `True` | Denoiser fixed, train ADMM only | Denoiser well-trained, faster training |
| **Fine-tune** | `False` | Update both denoiser + ADMM | Adapt denoiser to full network |

---

## Workflow Examples

### Example 1: Fully Unsupervised

```bash
# Stage 1: Unsupervised denoiser
python train_denoiser_only.py  # DENOISER_TRAINING_MODE = 'unsupervised'

# Stage 2: Unsupervised full network, frozen denoiser
python train_configurable.py    # TRAINING_STRATEGY = 'two_stage'
                                # TRAINING_MODE = 'unsupervised'
                                # FREEZE_DENOISER = True
```

### Example 2: Fully Supervised

```bash
# Stage 1: Supervised denoiser
python train_denoiser_only.py  # DENOISER_TRAINING_MODE = 'supervised'

# Stage 2: Supervised full network, frozen denoiser
python train_configurable.py    # TRAINING_STRATEGY = 'two_stage'
                                # TRAINING_MODE = 'supervised'
                                # FREEZE_DENOISER = True
```

### Example 3: Mixed (Unsupervised → Supervised)

```bash
# Stage 1: Unsupervised denoiser (no ground truth needed)
python train_denoiser_only.py  # DENOISER_TRAINING_MODE = 'unsupervised'

# Stage 2: Supervised full network (ground truth available)
python train_configurable.py    # TRAINING_STRATEGY = 'two_stage'
                                # TRAINING_MODE = 'supervised'
                                # FREEZE_DENOISER = False  # Fine-tune
```

---

## Files Generated

### After Stage 1 (`train_denoiser_only.py`)

- `denoiser_pretrained.pth` - Denoiser weights
- `denoiser_training_loss_<mode>.png` - Loss curve
- `denoiser_output_<mode>.png` - Visualization

### After Stage 2 (`train_configurable.py`)

- `dbp_model.pth` - Full network weights
- Visualization plots (unrolled iterations, measurement domain, etc.)

---

## Model Methods (Added to `DBPNet`)

```python
# Load pre-trained denoiser
model.load_pretrained_denoiser('denoiser_pretrained.pth')

# Freeze denoiser (no gradient updates)
model.freeze_denoiser()

# Unfreeze denoiser (allow gradient updates)
model.unfreeze_denoiser()

# Get only trainable parameters (for optimizer)
trainable_params = model.get_trainable_params()
optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
```

---

## Decision Tree: Which Strategy to Use?

```
Do you want to train denoiser separately?
│
├─ NO  → Use END-TO-END training
│         ✓ Simpler workflow
│         ✓ Joint optimization
│
└─ YES → Use TWO-STAGE training
          │
          ├─ Is denoiser well-trained?
          │  │
          │  ├─ YES → FREEZE_DENOISER = True
          │  │         ✓ Faster training
          │  │         ✓ Avoid overfitting
          │  │
          │  └─ NO  → FREEZE_DENOISER = False
          │            ✓ Fine-tune for better fit
          │
          └─ Do you have ground truth?
             │
             ├─ YES → Choose TRAINING_MODE
             │         • 'supervised' (image domain loss)
             │         • 'hybrid' (both domains)
             │
             └─ NO  → TRAINING_MODE = 'unsupervised'
                       (measurement domain loss)
```

---

## Performance Comparison

| Metric | End-to-End | Two-Stage (Frozen) | Two-Stage (Fine-tune) |
|--------|------------|--------------------|----------------------|
| Training Steps | 1 | 2 | 2 |
| Stage 2 Speed | - | Fast | Medium |
| Memory (Stage 2) | High | Low | High |
| Flexibility | Low | High | Medium |
| Likely Best Performance | ✓ | | ✓ (with tuning) |

---

## Key Benefits of Two-Stage Training

✅ **Modularity**: Debug denoiser and ADMM separately  
✅ **Flexibility**: Swap different denoisers without full retraining  
✅ **Efficiency**: Freeze denoiser for faster ADMM optimization  
✅ **Mixed Training**: Use unsupervised denoiser + supervised full network  
✅ **Transfer Learning**: Pre-train denoiser on different dataset

---

## Testing After Training

```bash
# Test on single sample
python test_unsupervised_model.py

# Test on full dataset
python test_full_dataset.py

# Generate final SAR image
python inference.py
```

---

## Related Documentation

- `TWO_STAGE_TRAINING_GUIDE.md` - Detailed guide with examples
- `SUPERVISED_TRAINING_GUIDE.md` - Supervised training modes
- `ITERATION_VISUALIZATION_GUIDE.md` - Visualizing network outputs
- `README.md` - Main project documentation
- `QUICK_REFERENCE.md` - Complete framework reference

---

## Common Issues

**"Pre-trained denoiser not found"**
→ Run `python train_denoiser_only.py` first

**Two-stage worse than end-to-end**
→ Try fine-tuning (`FREEZE_DENOISER = False`)

**Out of memory**
→ Set `FREEZE_DENOISER = True`

**Denoiser overfitting**
→ Reduce `NUM_EPOCHS` or add regularization

---

## Summary

The two-stage training framework provides maximum flexibility for MIMO SAR imaging:

- **Stage 1**: Focus on learning a good denoiser (supervised or unsupervised)
- **Stage 2**: Optimize ADMM parameters with pre-trained denoiser (frozen or fine-tuned)
- **Alternative**: Use end-to-end training for simplicity and joint optimization

Choose the strategy that best fits your data and research goals!

