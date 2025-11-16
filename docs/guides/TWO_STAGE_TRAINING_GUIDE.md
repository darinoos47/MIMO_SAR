# Two-Stage Training Guide

## Overview

This framework now supports **two training strategies**:

1. **End-to-End Training**: Train the entire network (denoiser + ADMM layers) jointly from scratch
2. **Two-Stage Training**: First train the denoiser separately, then use it with ADMM layers

The two-stage approach can be beneficial when:
- You want to focus on learning a good denoiser first
- You want to train ADMM parameters with a frozen, pre-trained denoiser
- You want to experiment with different denoiser architectures independently

---

## Training Strategies

### 1. End-to-End Training (Default)

Train all network parameters jointly from scratch.

```python
# In train_configurable.py
TRAINING_STRATEGY = 'end_to_end'
```

**Advantages:**
- Simpler workflow (single training script)
- Joint optimization of all parameters
- Network learns optimal interaction between denoiser and ADMM

**Disadvantages:**
- Requires careful hyperparameter tuning
- Longer training time per epoch

---

### 2. Two-Stage Training

**Stage 1**: Train the denoiser separately using `train_denoiser_only.py`

**Stage 2**: Load the pre-trained denoiser and train the full network (with denoiser frozen or fine-tuned)

```python
# In train_configurable.py
TRAINING_STRATEGY = 'two_stage'
PRETRAINED_DENOISER_PATH = 'denoiser_pretrained.pth'
FREEZE_DENOISER = True  # or False for fine-tuning
```

**Advantages:**
- Modular training: debug denoiser and ADMM separately
- Can use different loss functions for denoiser vs. full network
- Faster training in Stage 2 if denoiser is frozen

**Disadvantages:**
- Requires two separate training runs
- May not find the globally optimal solution (compared to end-to-end)

---

## Stage 1: Training the Denoiser

### Configuration

Edit `train_denoiser_only.py`:

```python
# Training Mode
DENOISER_TRAINING_MODE = 'unsupervised'  # or 'supervised'

# Training Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# For supervised training
NOISE_LEVEL = 0.1  # Std dev of synthetic noise
```

### Denoiser Training Modes

#### A. Unsupervised Denoiser Training

The denoiser is trained to preserve measurement domain consistency:

```
Input:  x_noisy = A^H @ y
Output: x_denoised = Denoiser(x_noisy)
Loss:   MSE(A @ x_denoised, y)
```

**When to use:**
- No ground truth reflectivity available
- Want denoiser to learn from raw measurements only

#### B. Supervised Denoiser Training

The denoiser is trained with synthetic noise on ground truth data:

```
Input:  x_noisy = A^H @ y + noise
Target: x_gt (ground truth reflectivity)
Output: x_denoised = Denoiser(x_noisy)
Loss:   MSE(x_denoised, x_gt)
```

**When to use:**
- Ground truth reflectivity available
- Want direct image domain supervision

### Running Stage 1

```bash
# Edit train_denoiser_only.py to set DENOISER_TRAINING_MODE
python train_denoiser_only.py
```

**Output:**
- `denoiser_pretrained.pth`: Saved denoiser weights
- `denoiser_training_loss_<mode>.png`: Training loss curve
- `denoiser_output_<mode>.png`: Visualization of denoiser output

---

## Stage 2: Training Full Network with Pre-trained Denoiser

### Configuration

Edit `train_configurable.py`:

```python
# Training Strategy
TRAINING_STRATEGY = 'two_stage'
PRETRAINED_DENOISER_PATH = 'denoiser_pretrained.pth'
FREEZE_DENOISER = True  # Frozen: train ADMM only; False: fine-tune

# Training Mode (applies to Stage 2)
TRAINING_MODE = 'supervised'  # or 'unsupervised', 'hybrid'
```

### Denoiser Handling Options

#### Option A: Frozen Denoiser (Recommended)

```python
FREEZE_DENOISER = True
```

- Denoiser weights are **fixed** (no gradient updates)
- Only ADMM parameters are trained
- Faster training, lower memory usage
- Good when denoiser is already well-trained

#### Option B: Fine-tuning

```python
FREEZE_DENOISER = False
```

- Denoiser weights are **updated** during training
- Both denoiser and ADMM parameters are trained
- Allows denoiser to adapt to full network context
- May improve performance but risk overfitting

### Running Stage 2

```bash
# Edit train_configurable.py to set TRAINING_STRATEGY = 'two_stage'
python train_configurable.py
```

**The script will:**
1. Load the pre-trained denoiser from `PRETRAINED_DENOISER_PATH`
2. Freeze/unfreeze denoiser based on `FREEZE_DENOISER`
3. Train the full network
4. Save the final model to `dbp_model.pth`

---

## Complete Workflow Examples

### Example 1: Unsupervised Two-Stage Training

**Stage 1: Train Denoiser (Unsupervised)**

```python
# In train_denoiser_only.py
DENOISER_TRAINING_MODE = 'unsupervised'
NUM_EPOCHS = 200
```

```bash
python train_denoiser_only.py
```

**Stage 2: Train Full Network (Unsupervised, Frozen Denoiser)**

```python
# In train_configurable.py
TRAINING_STRATEGY = 'two_stage'
TRAINING_MODE = 'unsupervised'
FREEZE_DENOISER = True
```

```bash
python train_configurable.py
```

---

### Example 2: Supervised Two-Stage Training

**Stage 1: Train Denoiser (Supervised)**

```python
# In train_denoiser_only.py
DENOISER_TRAINING_MODE = 'supervised'
NOISE_LEVEL = 0.1
NUM_EPOCHS = 200
```

```bash
python train_denoiser_only.py
```

**Stage 2: Train Full Network (Supervised, Frozen Denoiser)**

```python
# In train_configurable.py
TRAINING_STRATEGY = 'two_stage'
TRAINING_MODE = 'supervised'
FREEZE_DENOISER = True
```

```bash
python train_configurable.py
```

---

### Example 3: Mixed Training (Unsupervised Denoiser + Supervised Full Network)

**Stage 1: Train Denoiser (Unsupervised)**

```python
# In train_denoiser_only.py
DENOISER_TRAINING_MODE = 'unsupervised'
NUM_EPOCHS = 200
```

```bash
python train_denoiser_only.py
```

**Stage 2: Train Full Network (Supervised, Fine-tune Denoiser)**

```python
# In train_configurable.py
TRAINING_STRATEGY = 'two_stage'
TRAINING_MODE = 'supervised'
FREEZE_DENOISER = False  # Fine-tune the denoiser
```

```bash
python train_configurable.py
```

---

## Comparison: End-to-End vs. Two-Stage

| Aspect | End-to-End | Two-Stage (Frozen) | Two-Stage (Fine-tune) |
|--------|------------|-------------------|----------------------|
| **Training Steps** | 1 | 2 | 2 |
| **Denoiser Training** | Joint with ADMM | Separate | Separate + Fine-tune |
| **Training Time** | Medium | Stage 1 + Fast Stage 2 | Stage 1 + Medium Stage 2 |
| **Flexibility** | Low | High (swap denoisers) | Medium |
| **Memory Usage** | High | Low (Stage 2) | High (Stage 2) |
| **Debugging** | Harder | Easier (modular) | Easier (modular) |
| **Optimal Solution** | Likely best | May be suboptimal | Better than frozen |

---

## Model Architecture Details

### CNNDenoiser

The denoiser is a 1D CNN operating on complex-valued reflectivity:

- **Input**: Complex reflectivity (2 channels: real + imag), shape `[B, 2, N_theta]`
- **Output**: Denoised complex reflectivity, same shape
- **Architecture**:
  - Conv1D layers with ReLU and BatchNorm
  - Residual connection: `output = input + CNN(input)`

### DBPNet

The full network combines denoiser + ADMM:

```
x = A^H @ y                    # Matched filter initialization

For i = 1 to N1:
    r = Denoiser(x)            # CNN-based denoising
    x, u = ADMM_Layer(r, y, u) # Data consistency (ADMM)

return x
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `train_configurable.py` | Main training script (end-to-end or two-stage) |
| `train_denoiser_only.py` | Stage 1: Train denoiser separately |
| `models.py` | Model definitions (CNNDenoiser, DBPNet, DCLayer_ADMM) |
| `data_loader.py` | Dataset loading (with/without ground truth) |
| `utils.py` | Complex operations, L2-ball projection |
| `visualization_utils.py` | Visualization functions |
| `inference.py` | Final SAR image reconstruction |
| `test_unsupervised_model.py` | Test trained model on single sample |
| `test_full_dataset.py` | Evaluate model on full dataset |

---

## Tips and Best Practices

### 1. Hyperparameter Tuning

- **Denoiser Learning Rate**: Start with `1e-3` for standalone training
- **Full Network Learning Rate**: Use `1e-4` for two-stage (especially if fine-tuning)
- **ADMM Penalty Parameter (ρ)**: Tunable per-iteration in `DCLayer_ADMM`

### 2. When to Use Two-Stage Training

✅ **Use Two-Stage When:**
- Debugging: isolate issues in denoiser vs. ADMM
- Experimentation: try different denoisers with same ADMM
- Limited compute: freeze denoiser to speed up Stage 2
- Transfer learning: use denoiser trained on different data

❌ **Use End-to-End When:**
- Simplicity: want single training workflow
- Optimal performance: joint optimization likely better
- No separate denoiser use case

### 3. Choosing Frozen vs. Fine-tune

- **Frozen** (default): safer, faster, good for well-trained denoiser
- **Fine-tune**: use if denoiser was trained on different task or needs adaptation

### 4. Monitoring Training

- Check `denoiser_output_<mode>.png` after Stage 1
- Visualize unrolled iterations after Stage 2
- Compare end-to-end vs. two-stage performance

---

## Troubleshooting

### Issue: "Pre-trained denoiser not found"

**Solution**: Run `python train_denoiser_only.py` first, or set `TRAINING_STRATEGY = 'end_to_end'`

### Issue: "Expected more than 1 value per channel when training"

**Cause**: BatchNorm issue with very small feature dimensions

**Solution**: Use `model.eval()` mode for testing, or increase data dimensionality

### Issue: Two-stage performance worse than end-to-end

**Possible reasons:**
- Denoiser overfitted in Stage 1
- Denoiser trained with different loss than Stage 2
- Try fine-tuning instead of freezing

### Issue: Out of memory in Stage 2

**Solution**: Set `FREEZE_DENOISER = True` to reduce memory usage

---

## Advanced: Custom Denoiser Architectures

The two-stage framework makes it easy to experiment with different denoisers:

1. **Train custom denoiser** (replace `CNNDenoiser` in `models.py`)
2. **Save weights**: `torch.save(denoiser.state_dict(), 'custom_denoiser.pth')`
3. **Load in Stage 2**: Set `PRETRAINED_DENOISER_PATH = 'custom_denoiser.pth'`

**Note**: Custom denoiser must have same input/output interface as `CNNDenoiser`

---

## Summary

The two-stage training framework provides:

✅ **Flexibility**: Train denoiser and ADMM separately or jointly
✅ **Modularity**: Debug and optimize components independently  
✅ **Experimentation**: Try different denoisers without retraining full network
✅ **Efficiency**: Freeze denoiser for faster ADMM training
✅ **Mixed Training**: Combine supervised/unsupervised modes across stages

Choose the strategy that best fits your research goals and computational constraints!

