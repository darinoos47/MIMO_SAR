# Progressive Curriculum Training for Denoiser

## 🎯 The Problem

### Domain Shift in Unrolled Networks

When training a standard denoiser for use in an unrolled network:

**❌ Issue:**
- The denoiser is trained on input distribution: `A^H @ y` (matched filter output)
- In **iteration 1**, input is indeed `A^H @ y` → **Works well** ✅
- In **iterations 2+**, input is `x_after_ADMM` → **Different distribution** → **Poor performance** ❌

**Root Cause:** The denoiser has never seen the kind of data it receives in later iterations!

---

## 💡 The Solution: Progressive Curriculum Training

Train a single shared denoiser that can handle inputs from **any iteration depth** by progressively expanding the training dataset with synthetic samples from deeper iterations.

### Key Idea

Instead of training only on `A^H @ y`, progressively add synthetic training samples that represent what the denoiser will see at each iteration depth:

```
Stage 0: Train on [A^H @ y]                                    → 51 samples
Stage 1: Train on [A^H @ y, x_after_ADMM_0]                    → 102 samples
Stage 2: Train on [A^H @ y, x_after_ADMM_0, x_after_ADMM_1]    → 153 samples
...
```

This is a form of **curriculum learning** where the model progressively learns harder cases.

---

## 🔧 How It Works

### Algorithm Overview

```
For each curriculum stage k = 0, 1, 2, ..., K-1:
    
    1. Prepare dataset:
       - Accumulate all previous synthetic samples
       - Dataset size = (k+1) × N_original
    
    2. Train denoiser:
       - Option A: Train from scratch on full accumulated dataset
       - Option B: Fine-tune existing denoiser
    
    3. Generate synthetic data for next stage:
       - For each sample: x_synthetic = ADMM(Denoiser(x_current))
       - Add to accumulated dataset
    
    4. Repeat until K stages completed
```

### Detailed Steps

#### **Stage 0: Train on Original Data**

**Dataset:**
- Inputs: `A^H @ y` for all N samples
- Targets: `x_gt` (supervised) or `y` (unsupervised)

**Training:**
- Train denoiser for `EPOCHS_PER_STAGE` epochs

**Synthetic Data Generation:**
- For each sample:
  ```python
  x_input = A^H @ y
  x_denoised = Denoiser(x_input)
  y_temp = A @ x_input
  x_after_admm_0 = ADMM(x_denoised, y_temp)
  ```
- Store `x_after_admm_0` for all N samples

---

#### **Stage 1: Train on Original + Stage-0 Synthetic**

**Dataset:**
- Original: `A^H @ y` → target `x_gt` or `y` (N samples)
- Synthetic: `x_after_admm_0` → target **same** `x_gt` or `y` (N samples)
- **Total: 2N samples**

**Training:**
- Optionally reinitialize denoiser from scratch
- Train for `EPOCHS_PER_STAGE` epochs

**Synthetic Data Generation:**
- For each sample:
  ```python
  x_input = x_after_admm_0  # From previous stage
  x_denoised = Denoiser(x_input)
  y_temp = A @ x_input
  x_after_admm_1 = ADMM(x_denoised, y_temp)
  ```
- Store `x_after_admm_1` for all N samples

---

#### **Stage k: Continue Progressively**

**Dataset:**
- Accumulated: `[A^H @ y, x_after_admm_0, ..., x_after_admm_{k-1}]`
- **Total: (k+1) × N samples**

**Pattern:** Each stage adds one more iteration depth of synthetic data.

---

## 🎛️ Configuration

### Key Parameters in `train_denoiser_curriculum.py`

```python
# Curriculum Configuration
NUM_CURRICULUM_STAGES = 3  # Number of iteration depths (e.g., 0, 1, 2)
CURRICULUM_TRAINING_MODE = 'unsupervised'  # 'supervised' or 'unsupervised'
CURRICULUM_RETRAINING_STRATEGY = 'from_scratch'  # 'from_scratch' or 'fine_tune'

# Training
EPOCHS_PER_STAGE = 200  # Epochs for each curriculum stage
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# ADMM for synthetic data generation
NUM_ADMM_STEPS = 1  # Fixed ADMM steps (frozen parameters)

# Output
MODEL_SAVE_PATH = 'denoiser_curriculum.pth'
```

### Parameter Guide

| Parameter | Description | Options |
|-----------|-------------|---------|
| `NUM_CURRICULUM_STAGES` | Number of iteration depths to train for | Typically 2-5 |
| `CURRICULUM_TRAINING_MODE` | Loss function | `'supervised'` or `'unsupervised'` |
| `CURRICULUM_RETRAINING_STRATEGY` | How to retrain at each stage | `'from_scratch'` or `'fine_tune'` |
| `EPOCHS_PER_STAGE` | Training epochs per stage | 100-300 recommended |
| `NUM_ADMM_STEPS` | ADMM steps for synthetic data | Usually 1 |

---

## 🚀 Usage

### Basic Usage

```bash
# 1. Configure parameters in train_denoiser_curriculum.py
# 2. Run training
python train_denoiser_curriculum.py

# 3. Use the curriculum-trained denoiser in two-stage training
python train_configurable.py
```

Make sure to set in `train_configurable.py`:
```python
TRAINING_STRATEGY = 'two_stage'
PRETRAINED_DENOISER_PATH = 'denoiser_curriculum.pth'
```

---

## 📊 Outputs

### Generated Files

1. **`denoiser_curriculum.pth`**
   - Trained denoiser weights
   - Ready to use in two-stage training

2. **`curriculum_stage_losses.png`**
   - Loss curves for each curriculum stage
   - Shows convergence at each stage

3. **`curriculum_final_output.png`**
   - Image domain visualization
   - Compares: noisy input, denoiser output, ground truth

4. **`curriculum_measurement_domain.png`**
   - Measurement domain consistency
   - Plots: `y_gt` vs. `y_hat = A @ x_denoised`

### Terminal Output

```
Stage 0: Dataset size = 51 samples
Stage 1: Dataset size = 102 samples
Stage 2: Dataset size = 153 samples
...
Final Measurement Domain MSE: X.XXe-XX
```

---

## 🔬 Training Modes

### Supervised Mode

**Loss:** `MSE(denoiser_output, x_gt)`

**When to use:**
- You have ground truth images
- You want the denoiser to directly learn the image reconstruction

**Pros:**
- Direct supervision on target images
- Often faster convergence

**Cons:**
- Requires ground truth data
- May not perfectly preserve measurement consistency

---

### Unsupervised Mode

**Loss:** `MSE(A @ denoiser_output, y_gt)`

**When to use:**
- You want measurement consistency
- You don't have ground truth (or want to ignore it)

**Pros:**
- Enforces forward model consistency
- Works without ground truth

**Cons:**
- May converge slower
- Indirect learning of image structure

---

## 📈 Retraining Strategies

### Option A: Train from Scratch

**`CURRICULUM_RETRAINING_STRATEGY = 'from_scratch'`**

- At each stage: Reinitialize denoiser with random weights
- Train on the full accumulated dataset from scratch

**Pros:**
- Fresh learning without bias from previous stage
- May find better global minima
- More exploration

**Cons:**
- Longer training time
- May "forget" earlier learning initially

**Recommended when:**
- You have enough time/compute
- Dataset size grows significantly
- You want maximum performance

---

### Option B: Fine-Tune

**`CURRICULUM_RETRAINING_STRATEGY = 'fine_tune'`**

- At each stage: Keep existing weights
- Continue training on the expanded dataset

**Pros:**
- Faster (builds on previous knowledge)
- Smoother learning curve
- Preserves earlier learning

**Cons:**
- May get stuck in local minima
- Less exploration of weight space

**Recommended when:**
- Time/compute is limited
- Dataset growth is moderate
- Incremental improvement is sufficient

---

## 🎓 Example: 3-Stage Curriculum

### Configuration

```python
NUM_CURRICULUM_STAGES = 3
EPOCHS_PER_STAGE = 200
CURRICULUM_RETRAINING_STRATEGY = 'from_scratch'
```

### Execution

**Stage 0:**
- Dataset: 51 original samples (A^H @ y)
- Train from scratch: 200 epochs
- Final loss: ~0.001
- Generate: x_after_admm_0

**Stage 1:**
- Dataset: 102 samples (original + synthetic_0)
- Reinitialize and train from scratch: 200 epochs
- Final loss: ~0.0008
- Generate: x_after_admm_1

**Stage 2:**
- Dataset: 153 samples (original + synthetic_0 + synthetic_1)
- Reinitialize and train from scratch: 200 epochs
- Final loss: ~0.0006

**Result:**
- Denoiser trained on distributions from iterations 0, 1, 2
- Can generalize to all iteration depths in unrolled network

---

## 🆚 Comparison: Standard vs. Curriculum Training

| Aspect | Standard Training | Curriculum Training |
|--------|------------------|---------------------|
| **Training data** | Only A^H @ y | A^H @ y + synthetic samples from deeper iterations |
| **Dataset size** | N samples | (K+1) × N samples |
| **Training time** | Faster (single stage) | Slower (K stages) |
| **Iteration 1 performance** | Excellent ✅ | Excellent ✅ |
| **Iteration 2+ performance** | Poor ❌ (domain shift) | Good ✅ (trained on similar data) |
| **Generalization** | Limited to initial distribution | Robust across iteration depths |
| **Use case** | Single-iteration or end-to-end training | Multi-iteration unrolled networks |

---

## 💡 When to Use Curriculum Training

### ✅ Use Curriculum Training When:

1. You're using a **shared denoiser** across multiple unrolled iterations
2. You observe **degradation in performance** in later iterations
3. You have a **small dataset** (curriculum augments it)
4. You want the denoiser to be **robust** to different input distributions
5. You're using **two-stage training** (denoiser → full network)

### ❌ Consider Standard Training When:

1. You're doing **end-to-end training** (denoiser + ADMM jointly)
2. You have a **very large dataset** already
3. You use **per-iteration denoisers** (different denoiser at each iteration)
4. **Training time** is critical
5. You have **only 1 unrolled iteration** (no domain shift)

---

## 🔍 Understanding the Loss Curves

### What to Expect

**Stage 0:**
- Starts high, decreases steadily
- This is standard training on original data

**Stage 1:**
- May start higher (if from_scratch) or lower (if fine_tune)
- Should converge to lower loss than Stage 0
- Dataset is 2× larger → more diverse

**Stage K:**
- Typically lowest final loss
- Dataset is (K+1)× larger
- Denoiser has seen most diverse data

### Interpreting the Plot

`curriculum_stage_losses.png` shows:
- Different color for each stage
- Overlapping curves → easy to compare convergence
- Later stages typically have lower final loss

---

## 🛠️ Troubleshooting

### Issue 1: Loss Increases in Later Stages

**Possible causes:**
- Learning rate too high
- From-scratch retraining causing unstable optimization

**Solutions:**
- Reduce `LEARNING_RATE`
- Try `CURRICULUM_RETRAINING_STRATEGY = 'fine_tune'`
- Increase `EPOCHS_PER_STAGE`

---

### Issue 2: No Improvement Over Standard Training

**Possible causes:**
- Not enough curriculum stages
- Synthetic data too similar to original

**Solutions:**
- Increase `NUM_CURRICULUM_STAGES`
- Check that ADMM parameters are reasonable
- Visualize synthetic data to ensure diversity

---

### Issue 3: Out of Memory

**Possible causes:**
- Accumulated dataset becomes very large
- Batch size too large

**Solutions:**
- Reduce `BATCH_SIZE`
- Reduce `NUM_CURRICULUM_STAGES`
- Use gradient accumulation

---

### Issue 4: Training Takes Too Long

**Possible causes:**
- Too many stages × epochs

**Solutions:**
- Reduce `EPOCHS_PER_STAGE` (e.g., 100 instead of 200)
- Reduce `NUM_CURRICULUM_STAGES`
- Use `CURRICULUM_RETRAINING_STRATEGY = 'fine_tune'`

---

## 📝 Technical Details

### Ground Truth Handling

For synthetic samples `x_after_admm_k`, the target is **always the original**:

**Supervised:**
```python
loss = MSE(Denoiser(x_after_admm_k), x_gt_original)
```

**Unsupervised:**
```python
loss = MSE(A @ Denoiser(x_after_admm_k), y_original)
```

This is correct because:
- `x_gt_original` is the true target for all intermediate iterations
- We want the denoiser to drive toward the true solution, regardless of input quality

---

### ADMM Parameters

During synthetic data generation, ADMM parameters are **frozen** (not learned). This is because:

1. We only want to train the denoiser (not ADMM)
2. ADMM will be learned later in Stage 2 (two-stage) or jointly (end-to-end)
3. Fixed ADMM gives consistent synthetic data across stages

---

### Dataset Accumulation

At stage k, the dataset contains:
```
[A^H @ y, x_after_admm_0, x_after_admm_1, ..., x_after_admm_{k-1}]
```

Each sample is paired with the same original `x_gt` and `y`:
```
Sample 0: (A^H @ y,           x_gt_0, y_0)
Sample 1: (x_after_admm_0[0], x_gt_0, y_0)
Sample 2: (x_after_admm_1[0], x_gt_0, y_0)
...
```

This grows linearly: Stage k has `(k+1) × N` samples.

---

## 🎉 Benefits

1. **Robustness:** Denoiser generalizes across iteration depths
2. **Data Augmentation:** Expands small datasets with synthetic samples
3. **Better Performance:** Addresses domain shift in later iterations
4. **Principled:** Based on curriculum learning theory
5. **Flexible:** Works with supervised/unsupervised, from-scratch/fine-tune

---

## 🚀 Next Steps

After curriculum training:

1. **Visualize Results:**
   - Check `curriculum_stage_losses.png` for convergence
   - Check `curriculum_final_output.png` for image quality
   - Check `curriculum_measurement_domain.png` for consistency

2. **Use in Two-Stage Training:**
   ```bash
   python train_configurable.py
   ```
   With:
   ```python
   TRAINING_STRATEGY = 'two_stage'
   PRETRAINED_DENOISER_PATH = 'denoiser_curriculum.pth'
   FREEZE_DENOISER = True  # or False for fine-tuning
   ```

3. **Compare Performance:**
   - Train with standard denoiser: `python train_denoiser_only.py`
   - Train with curriculum denoiser: Use `denoiser_curriculum.pth`
   - Evaluate on test set: `python test_unsupervised_model.py`

4. **Experiment:**
   - Try different `NUM_CURRICULUM_STAGES` (2, 3, 5)
   - Compare `from_scratch` vs. `fine_tune`
   - Try supervised vs. unsupervised

---

## 📚 References

This implements ideas from:
- **Curriculum Learning:** Bengio et al., 2009
- **Self-Training:** Scudder, 1965; Yarowsky, 1995
- **Unrolled Optimization:** Gregor & LeCun, 2010; Monga et al., 2021

Enjoy curriculum training! 🎓✨

