# Curriculum Training Quick Start

## 🎯 Problem

Standard denoiser training fails in later iterations due to **domain shift**:
- Trained on: `A^H @ y`
- Iteration 1 sees: `A^H @ y` ✅
- Iteration 2+ sees: `x_after_ADMM` ❌ (different distribution!)

## 💡 Solution

**Progressive Curriculum Training:** Train on accumulated synthetic data from deeper iterations.

---

## 🚀 Usage (3 Steps)

### Step 1: Configure

Edit `train_denoiser_curriculum.py`:

```python
NUM_CURRICULUM_STAGES = 3          # Number of iteration depths
CURRICULUM_TRAINING_MODE = 'unsupervised'  # or 'supervised'
CURRICULUM_RETRAINING_STRATEGY = 'from_scratch'  # or 'fine_tune'
EPOCHS_PER_STAGE = 200             # Epochs per stage
```

### Step 2: Train

```bash
python train_denoiser_curriculum.py
```

**Output:**
- `denoiser_curriculum.pth` - Trained denoiser
- `curriculum_stage_losses.png` - Loss per stage
- `curriculum_final_output.png` - Image domain results
- `curriculum_measurement_domain.png` - Measurement consistency

### Step 3: Use in Two-Stage Training

Edit `train_configurable.py`:

```python
TRAINING_STRATEGY = 'two_stage'
PRETRAINED_DENOISER_PATH = 'denoiser_curriculum.pth'
FREEZE_DENOISER = True  # or False to fine-tune
```

Run:
```bash
python train_configurable.py
```

---

## 📊 What Happens

```
Stage 0: Train on 51 samples (A^H @ y)
         ↓ Generate synthetic data
         
Stage 1: Train on 102 samples (original + synthetic_0)
         ↓ Generate synthetic data
         
Stage 2: Train on 153 samples (original + synthetic_0 + synthetic_1)
         ↓
         
Result: Denoiser that works well at ANY iteration depth!
```

---

## ⚙️ Configuration Guide

### Training Mode

| Mode | Loss | Use When |
|------|------|----------|
| `'supervised'` | `MSE(output, x_gt)` | You have ground truth |
| `'unsupervised'` | `MSE(A@output, y)` | You want measurement consistency |

### Retraining Strategy

| Strategy | Behavior | Speed | Performance |
|----------|----------|-------|-------------|
| `'from_scratch'` | Reinitialize each stage | Slower ⏱️⏱️ | Better 🎯🎯🎯 |
| `'fine_tune'` | Continue training | Faster ⏱️ | Good 🎯🎯 |

### Number of Stages

| Stages | Dataset Growth | Use Case |
|--------|---------------|----------|
| 2 | 51 → 102 | Quick test |
| 3 | 51 → 153 | Recommended default |
| 5 | 51 → 306 | Deep unrolled network |

---

## 📈 Expected Results

### Loss Curves

- Stage 0: Baseline performance
- Stage 1+: Lower loss (more diverse data)
- Each stage should converge

### Measurement Domain MSE

- Good: < 1e-3
- Acceptable: 1e-3 to 1e-2
- Poor: > 1e-2

---

## 🆚 Quick Comparison

|  | Standard | Curriculum |
|--|----------|-----------|
| **Training time** | ~30 sec | ~90 sec (3 stages) |
| **Dataset size** | 51 | 153 (3× larger) |
| **Iteration 1** | ✅ Excellent | ✅ Excellent |
| **Iteration 2+** | ❌ Poor (domain shift) | ✅ Good (trained on similar data) |

---

## 🔧 Troubleshooting

**Loss increases in later stages?**
→ Reduce `LEARNING_RATE` or try `'fine_tune'`

**Out of memory?**
→ Reduce `BATCH_SIZE` or `NUM_CURRICULUM_STAGES`

**Too slow?**
→ Reduce `EPOCHS_PER_STAGE` or use `'fine_tune'`

**No improvement?**
→ Increase `NUM_CURRICULUM_STAGES` or check visualizations

---

## 📚 Full Documentation

See `CURRICULUM_TRAINING_GUIDE.md` for:
- Detailed algorithm explanation
- Technical details
- Advanced configuration
- Theory and references

---

## ✅ Checklist

- [ ] Configure `NUM_CURRICULUM_STAGES` (recommend 3)
- [ ] Choose `CURRICULUM_TRAINING_MODE` (supervised/unsupervised)
- [ ] Choose `CURRICULUM_RETRAINING_STRATEGY` (from_scratch/fine_tune)
- [ ] Run `python train_denoiser_curriculum.py`
- [ ] Check output plots for quality
- [ ] Use `denoiser_curriculum.pth` in `train_configurable.py`
- [ ] Compare with standard training

Happy training! 🎓🚀

