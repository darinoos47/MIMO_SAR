# Quick Start: Two-Stage Training

## ⚡ 5-Minute Quick Start

### Option 1: End-to-End Training (Simplest)

```bash
# Edit train_configurable.py (if needed):
# TRAINING_STRATEGY = 'end_to_end'
# TRAINING_MODE = 'unsupervised'  # or 'supervised' if you have ground truth

python train_configurable.py
```

**Done!** Model saved to `dbp_model.pth`

---

### Option 2: Two-Stage Training (Most Flexible)

**Step 1: Train Denoiser (2 minutes to configure)**

```bash
# Edit train_denoiser_only.py:
# DENOISER_TRAINING_MODE = 'unsupervised'  # or 'supervised'
# NUM_EPOCHS = 200

python train_denoiser_only.py
```

**Output:** `denoiser_pretrained.pth`

**Step 2: Train Full Network (2 minutes to configure)**

```bash
# Edit train_configurable.py:
# TRAINING_STRATEGY = 'two_stage'
# FREEZE_DENOISER = True  # or False to fine-tune
# TRAINING_MODE = 'unsupervised'  # or 'supervised'

python train_configurable.py
```

**Done!** Model saved to `dbp_model.pth`

---

## 🎯 Configuration Cheat Sheet

### `train_denoiser_only.py`

```python
DENOISER_TRAINING_MODE = 'unsupervised'  # Options:
                                          # - 'unsupervised': no ground truth needed
                                          # - 'supervised': requires ground truth

NUM_EPOCHS = 200          # Number of training epochs
BATCH_SIZE = 64           # Batch size
LEARNING_RATE = 1e-3      # Learning rate
NOISE_LEVEL = 0.1         # Synthetic noise (for supervised mode)
```

### `train_configurable.py`

```python
# Training Strategy
TRAINING_STRATEGY = 'end_to_end'  # Options:
                                   # - 'end_to_end': train all from scratch
                                   # - 'two_stage': use pre-trained denoiser

# Two-Stage Options (only if TRAINING_STRATEGY = 'two_stage')
PRETRAINED_DENOISER_PATH = 'denoiser_pretrained.pth'
FREEZE_DENOISER = True  # Options:
                        # - True: freeze denoiser (train ADMM only)
                        # - False: fine-tune denoiser + train ADMM

# Training Mode
TRAINING_MODE = 'supervised'  # Options:
                              # - 'supervised': requires ground truth
                              # - 'unsupervised': no ground truth needed
                              # - 'hybrid': both losses combined

# Model
NUM_UNROLLS = 5      # Number of unrolled iterations
NUM_ADMM_STEPS = 2   # ADMM steps per iteration

# Training
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
```

---

## 🔀 Common Workflows

### Workflow 1: No Ground Truth (Fully Unsupervised)

```bash
# Stage 1: Unsupervised denoiser
# train_denoiser_only.py: DENOISER_TRAINING_MODE = 'unsupervised'
python train_denoiser_only.py

# Stage 2: Unsupervised full network, frozen denoiser
# train_configurable.py:
#   TRAINING_STRATEGY = 'two_stage'
#   TRAINING_MODE = 'unsupervised'
#   FREEZE_DENOISER = True
python train_configurable.py
```

---

### Workflow 2: Have Ground Truth (Fully Supervised)

```bash
# Stage 1: Supervised denoiser
# train_denoiser_only.py: DENOISER_TRAINING_MODE = 'supervised'
python train_denoiser_only.py

# Stage 2: Supervised full network, frozen denoiser
# train_configurable.py:
#   TRAINING_STRATEGY = 'two_stage'
#   TRAINING_MODE = 'supervised'
#   FREEZE_DENOISER = True
python train_configurable.py
```

---

### Workflow 3: Mixed (Best of Both Worlds)

```bash
# Stage 1: Unsupervised denoiser (no ground truth needed)
# train_denoiser_only.py: DENOISER_TRAINING_MODE = 'unsupervised'
python train_denoiser_only.py

# Stage 2: Supervised full network (use ground truth if available)
# train_configurable.py:
#   TRAINING_STRATEGY = 'two_stage'
#   TRAINING_MODE = 'supervised'
#   FREEZE_DENOISER = False  # Fine-tune denoiser
python train_configurable.py
```

---

## 🧪 Testing Your Model

After training, test the model:

```bash
# Test on single sample
python test_unsupervised_model.py

# Evaluate on full dataset
python test_full_dataset.py

# Generate final SAR image
python inference.py
```

---

## 📊 Output Files

### After Stage 1 (`train_denoiser_only.py`)

- `denoiser_pretrained.pth` - Pre-trained denoiser weights
- `denoiser_training_loss_<mode>.png` - Loss curve
- `denoiser_output_<mode>.png` - Denoiser visualization

### After Stage 2 or End-to-End (`train_configurable.py`)

- `dbp_model.pth` - Full network weights
- Various visualization plots (unrolled iterations, measurement domain, etc.)

---

## ❓ Which Strategy Should I Use?

### Use End-to-End if:
- ✅ You want the simplest workflow
- ✅ You want optimal joint optimization
- ✅ You don't need to debug components separately

### Use Two-Stage if:
- ✅ You want to debug denoiser and ADMM separately
- ✅ You want to experiment with different denoisers
- ✅ You want faster training in Stage 2 (frozen denoiser)
- ✅ You want to mix supervised/unsupervised modes

---

## 🚨 Troubleshooting

### "Pre-trained denoiser not found"
**Solution:** Run `train_denoiser_only.py` first

### "Ground truth not available"
**Solution:** Set `TRAINING_MODE = 'unsupervised'` or `DENOISER_TRAINING_MODE = 'unsupervised'`

### Out of memory
**Solution:** 
- Reduce `BATCH_SIZE`
- Set `FREEZE_DENOISER = True` (for two-stage)

### Poor performance
**Solution:**
- Try fine-tuning: `FREEZE_DENOISER = False`
- Try end-to-end training
- Increase `NUM_EPOCHS`
- Adjust `LEARNING_RATE`

---

## 📚 More Information

- **Detailed Guide:** `TWO_STAGE_TRAINING_GUIDE.md`
- **Quick Reference:** `TWO_STAGE_TRAINING_SUMMARY.md`
- **Full Documentation:** `README.md`
- **Implementation Details:** `TWO_STAGE_IMPLEMENTATION_SUMMARY.md`

---

## 🎓 Key Concepts (1-Minute Read)

### End-to-End Training
Train denoiser + ADMM together from scratch.
- **Pros:** Simple, optimal
- **Cons:** Less flexible

### Two-Stage Training
1. Train denoiser separately
2. Use pre-trained denoiser with ADMM

- **Pros:** Flexible, modular, faster Stage 2
- **Cons:** Requires two training runs

### Frozen vs Fine-tune
- **Frozen:** Denoiser weights fixed (faster, safer)
- **Fine-tune:** Denoiser weights updated (better fit, slower)

### Supervised vs Unsupervised
- **Supervised:** Use ground truth (requires `x_gt`)
- **Unsupervised:** Use forward model (no `x_gt` needed)

---

## ✅ Validation

All functionality has been tested and verified:
- ✅ Denoiser training (supervised & unsupervised)
- ✅ Two-stage training (frozen & fine-tune)
- ✅ Forward pass with intermediates
- ✅ Parameter freezing/unfreezing
- ✅ Integration with existing code

**You're ready to go!** 🚀

