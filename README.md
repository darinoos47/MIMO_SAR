# MIMO SAR Deep Learning Framework

Deep learning-based image reconstruction for FMCW MIMO SAR using unrolled networks with CNN-based denoising and ADMM data consistency layers.

## 🌟 Features

- **Unrolled Network Architecture**: Combines CNN denoiser with ADMM optimization
- **Flexible Training Modes**: Supervised, unsupervised, and hybrid training
- **Two Training Strategies**: End-to-end or two-stage (modular) training
- **Curriculum Training**: Progressive learning to address domain shift in unrolled networks
- **Real-Valued Prior Enforcement**: Leverage prior knowledge that reflectivity is real
- **Progressive Visualization**: View intermediate outputs at each unrolled iteration
- **Measurement Domain Analysis**: Visualize data consistency through the network
- **Dynamic Dimensioning**: Automatically adapts to variable dataset sizes
- **Complex-Valued Operations**: Efficient handling of complex radar data

---

## 📁 Project Structure

```
MIMO_SAR/
├── data_loader.py                      # Dataset loading with optional ground truth
├── models.py                           # CNNDenoiser, DCLayer_ADMM, DBPNet
├── utils.py                            # Complex operations, L2-ball projection
├── visualization_utils.py              # Plotting functions for analysis
│
├── train.py                            # Original unsupervised training script
├── train_configurable.py               # Configurable training (supervised/unsupervised/hybrid)
├── train_denoiser_only.py              # Standalone denoiser training (Stage 1)
├── train_denoiser_curriculum.py        # Curriculum training for denoiser (addresses domain shift)
│
├── test_unsupervised_model.py          # Test model on single sample
├── test_full_dataset.py                # Evaluate on full dataset
├── inference.py                        # Generate final SAR image (Cartesian)
│
├── compare_training_strategies.py      # Compare end-to-end vs. two-stage
│
├── FL_MIMO_SAR_data.mat                # Dataset (y, A, x)
├── dbp_model.pth                       # Saved model weights
├── denoiser_pretrained.pth             # Pre-trained denoiser (standard training)
├── denoiser_curriculum.pth             # Curriculum-trained denoiser (robust to domain shift)
│
└── Documentation/
    ├── README.md                       # This file
    ├── QUICK_REFERENCE.md              # Quick reference card
    ├── SUPERVISED_TRAINING_GUIDE.md    # Supervised training guide
    ├── TWO_STAGE_TRAINING_GUIDE.md     # Two-stage training guide
    ├── TWO_STAGE_TRAINING_SUMMARY.md   # Two-stage quick reference
    ├── CURRICULUM_TRAINING_GUIDE.md    # Curriculum training detailed guide
    ├── CURRICULUM_TRAINING_QUICK_START.md # Curriculum training quick reference
    ├── ITERATION_VISUALIZATION_GUIDE.md # Visualization guide
    └── MEASUREMENT_DOMAIN_VISUALIZATION.md
```

---

## 🚀 Quick Start

### 1. Basic Unsupervised Training

```bash
python train.py
```

Trains the network using measurement domain loss (no ground truth required).

### 2. Configurable Training

```bash
# Edit train_configurable.py to set:
# TRAINING_MODE = 'supervised'  # or 'unsupervised', 'hybrid'
# TRAINING_STRATEGY = 'end_to_end'  # or 'two_stage'

python train_configurable.py
```

### 3. Two-Stage Training

**Step 1: Train Denoiser**

```bash
# Edit train_denoiser_only.py to set:
# DENOISER_TRAINING_MODE = 'unsupervised'  # or 'supervised'

python train_denoiser_only.py
```

**Step 2: Train Full Network**

```bash
# Edit train_configurable.py to set:
# TRAINING_STRATEGY = 'two_stage'
# FREEZE_DENOISER = True  # or False for fine-tuning

python train_configurable.py
```

### 4. Curriculum Training (Advanced)

Train a denoiser that's robust to domain shift across unrolled iterations:

```bash
# Edit train_denoiser_curriculum.py to set:
# NUM_CURRICULUM_STAGES = 3
# CURRICULUM_TRAINING_MODE = 'unsupervised'  # or 'supervised'
# CURRICULUM_RETRAINING_STRATEGY = 'from_scratch'  # or 'fine_tune'

python train_denoiser_curriculum.py

# Then use in two-stage training:
python train_configurable.py
# (with PRETRAINED_DENOISER_PATH = 'denoiser_curriculum.pth')
```

See `CURRICULUM_TRAINING_GUIDE.md` for details.

### 5. Testing

```bash
# Test on single sample
python test_unsupervised_model.py

# Evaluate on full dataset
python test_full_dataset.py

# Generate final SAR image
python inference.py
```

---

## 🎯 Training Modes

### Supervised Training

Requires ground truth reflectivity `x` in the dataset.

```python
# In train_configurable.py
TRAINING_MODE = 'supervised'
```

**Loss**: `MSE(x_hat, x_gt)` (image domain)

### Unsupervised Training

Does not require ground truth (uses forward model).

```python
# In train_configurable.py
TRAINING_MODE = 'unsupervised'
```

**Loss**: `MSE(A @ x_hat, y)` (measurement domain)

### Hybrid Training

Combines both supervised and unsupervised losses.

```python
# In train_configurable.py
TRAINING_MODE = 'hybrid'
SUPERVISED_WEIGHT = 1.0
UNSUPERVISED_WEIGHT = 1.0
```

**Loss**: `α * MSE(x_hat, x_gt) + β * MSE(A @ x_hat, y)`

---

## 🏗️ Training Strategies

### End-to-End Training (Default)

Train the entire network (denoiser + ADMM) jointly from scratch.

```python
# In train_configurable.py
TRAINING_STRATEGY = 'end_to_end'
```

**Advantages:**
- Simple workflow
- Joint optimization
- Likely best performance

### Two-Stage Training

**Stage 1**: Train denoiser separately  
**Stage 2**: Train full network with pre-trained denoiser (frozen or fine-tuned)

```python
# Stage 1: train_denoiser_only.py
DENOISER_TRAINING_MODE = 'unsupervised'  # or 'supervised'

# Stage 2: train_configurable.py
TRAINING_STRATEGY = 'two_stage'
PRETRAINED_DENOISER_PATH = 'denoiser_pretrained.pth'
FREEZE_DENOISER = True  # or False for fine-tuning
```

**Advantages:**
- Modular training (debug components separately)
- Flexibility (swap different denoisers)
- Faster Stage 2 if denoiser frozen
- Can mix supervised/unsupervised across stages

See `TWO_STAGE_TRAINING_GUIDE.md` for detailed examples.

---

## 🎓 Curriculum Training (Advanced)

### The Problem: Domain Shift

When using a **shared denoiser** across multiple unrolled iterations, standard training creates a domain shift problem:

- **Trained on**: `A^H @ y` (matched filter output)
- **Iteration 1 sees**: `A^H @ y` ✅ Works well
- **Iteration 2+ sees**: `x_after_ADMM` ❌ Different distribution → Poor performance

### The Solution: Progressive Curriculum Training

Train the denoiser on progressively accumulated synthetic data from deeper iterations:

```
Stage 0: Train on [A^H @ y]                                    → 51 samples
Stage 1: Train on [A^H @ y, x_after_ADMM_0]                    → 102 samples  
Stage 2: Train on [A^H @ y, x_after_ADMM_0, x_after_ADMM_1]    → 153 samples
...
```

### Configuration

```python
# In train_denoiser_curriculum.py
NUM_CURRICULUM_STAGES = 3          # Number of iteration depths
CURRICULUM_TRAINING_MODE = 'unsupervised'  # or 'supervised'
CURRICULUM_RETRAINING_STRATEGY = 'from_scratch'  # or 'fine_tune'
EPOCHS_PER_STAGE = 200             # Epochs per curriculum stage
```

### Benefits

- **Robustness**: Denoiser works well at all iteration depths
- **Data Augmentation**: Expands small datasets with synthetic samples  
- **Better Performance**: Addresses domain shift in later iterations
- **Principled**: Based on curriculum learning theory

### When to Use

**✅ Use curriculum training when:**
- Using a shared denoiser across multiple iterations
- Observing performance degradation in later iterations
- Have a small dataset (curriculum augments it)
- Using two-stage training

**❌ Use standard training when:**
- Doing end-to-end training (denoiser + ADMM jointly)
- Have a very large dataset already
- Using per-iteration denoisers
- Training time is critical

See `CURRICULUM_TRAINING_GUIDE.md` for complete details and `CURRICULUM_TRAINING_QUICK_START.md` for quick reference.

---

## 🧠 Network Architecture

### Problem Formulation

**Forward Model:**
```
y = A * x + noise
```

- `y`: Measurements on MIMO virtual array (complex)
- `A`: Steering matrix (forward operator)
- `x`: Reflectivity at different angles (complex)

**Objective:** Reconstruct `x` from `y` given `A`

### Network: DBPNet

Unrolls ADMM iterations into a trainable network:

```
Initialization:
    x = A^H @ y  (matched filter)
    u = 0        (dual variable)

For i = 1 to N1 (num_iterations):
    r = CNNDenoiser(x)              # CNN-based denoising
    x, u = DCLayer_ADMM(r, y, u)    # ADMM data consistency

Output: x  (reconstructed reflectivity)
```

**Layers:**
- **CNNDenoiser**: 1D CNN with residual connection (replaces L1-norm prior)
- **DCLayer_ADMM**: ADMM optimization layer (x-update, z-update, u-update)
  - Uses Sherman-Morrison-Woodbury identity for efficient matrix inversion

**Hyperparameters:**
- `NUM_UNROLLS` (N1): Number of unrolled iterations (default: 5)
- `NUM_ADMM_STEPS` (N2): Internal ADMM steps per iteration (default: 2)

---

## 📊 Visualization Features

The framework provides comprehensive visualization of the network's behavior:

### 1. Unrolled Iterations (Image Domain)

Shows denoiser and ADMM outputs at each unrolled step:

```python
plot_unrolled_iterations(intermediates, x_gt=None, sample_idx=0)
```

### 2. Iteration Comparison

Overlays all iteration outputs for easy comparison:

```python
plot_iteration_comparison(intermediates, x_gt=None, sample_idx=0)
```

### 3. MSE Progression

Plots reconstruction error vs. iteration (requires ground truth):

```python
plot_mse_progression(intermediates, x_gt, sample_idx=0)
```

### 4. Measurement Domain Progression

Shows `y = A*x` at each iteration (magnitude, real, imaginary):

```python
plot_measurement_domain_progression(intermediates, y_gt, A_tensor, sample_idx=0)
```

See `ITERATION_VISUALIZATION_GUIDE.md` for details.

---

## 📦 Dataset Format

The dataset is stored in `FL_MIMO_SAR_data.mat` (MATLAB format):

**Required Fields:**
- `received_signals_fft`: Measurements `y` - shape `[N_samples, M_rx]` (complex)
- `A`: Steering matrix - shape `[M_rx, N_theta]` (complex)

**Optional Field:**
- `x`: Ground truth reflectivity - shape `[N_samples, N_theta]` or `[N_theta]` (complex)

**Notes:**
- Complex arrays are loaded as `dtype=np.complex64`
- The framework automatically handles 1D vs. 2D `x` data
- If using h5py (MATLAB v7.3), arrays are automatically transposed

---

## 🔧 Configuration Reference

### Key Parameters in `train_configurable.py`

```python
# Data
MAT_FILE = 'FL_MIMO_SAR_data.mat'
MODEL_SAVE_PATH = 'dbp_model.pth'

# Training Hyperparameters
NUM_EPOCHS = 500
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# Model Hyperparameters
NUM_UNROLLS = 5       # N1: unrolled iterations
NUM_ADMM_STEPS = 2    # N2: internal ADMM steps

# Training Mode
TRAINING_MODE = 'supervised'  # 'supervised', 'unsupervised', 'hybrid'
SUPERVISED_WEIGHT = 1.0
UNSUPERVISED_WEIGHT = 1.0

# Training Strategy
TRAINING_STRATEGY = 'end_to_end'  # 'end_to_end', 'two_stage'
PRETRAINED_DENOISER_PATH = 'denoiser_pretrained.pth'
FREEZE_DENOISER = True  # For two_stage only
```

### Key Parameters in `train_denoiser_only.py`

```python
# Data
MAT_FILE = 'FL_MIMO_SAR_data.mat'
MODEL_SAVE_PATH = 'denoiser_pretrained.pth'

# Training Hyperparameters
NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Denoiser Training Mode
DENOISER_TRAINING_MODE = 'unsupervised'  # 'supervised', 'unsupervised'
NOISE_LEVEL = 0.1  # For supervised mode
```

---

## 🎯 Real-Valued Prior Enforcement (NEW!)

### The Problem

If your target reflectivity is **real-valued** (no imaginary component), the network may still learn complex outputs because there's no explicit constraint in standard training.

### The Solution

Enforce real-valued outputs using three strategies:

**1. Loss Penalty** (`loss_penalty`)
- Adds penalty: `loss += λ * ||imaginary||²`
- Soft constraint, smooth gradients
- Use for gentle enforcement

**2. Hard Projection** (`hard_projection`)
- Sets imaginary channel to zero immediately
- Exact real outputs guaranteed
- Use when real-only is required

**3. Hybrid** (`hybrid`) ⭐ **Recommended**
- Combines penalty + projection
- Best of both worlds
- Default choice for most cases

### Configuration

Add to any training script:

```python
# Real-Valued Prior Enforcement
ENFORCE_REAL_PRIOR = True  # Enable real prior
REAL_PRIOR_STRATEGY = 'hybrid'  # 'loss_penalty', 'hard_projection', 'hybrid'
REAL_PRIOR_WEIGHT = 0.1  # Penalty weight λ
```

### Quick Example

```bash
# Enable in your training script
nano train_configurable.py
# Set: ENFORCE_REAL_PRIOR = True
#      REAL_PRIOR_STRATEGY = 'hybrid'
#      REAL_PRIOR_WEIGHT = 0.1

# Run training
python train_configurable.py

# Output will show:
# --- Real Prior: HYBRID (weight=0.1) ---
```

### Benefits

✅ Leverages prior knowledge that reflectivity is real  
✅ Better reconstruction quality  
✅ Simpler, more interpretable results  
✅ Works with all training modes and strategies  
✅ Easy to configure and use

See `REAL_PRIOR_GUIDE.md` for complete details and examples.

---

## 📚 Documentation

- **QUICK_REFERENCE.md**: Complete quick reference card
- **SUPERVISED_TRAINING_GUIDE.md**: Detailed supervised training guide
- **TWO_STAGE_TRAINING_GUIDE.md**: Comprehensive two-stage training guide
- **TWO_STAGE_TRAINING_SUMMARY.md**: Two-stage quick reference
- **CURRICULUM_TRAINING_GUIDE.md**: Progressive curriculum training detailed guide
- **CURRICULUM_TRAINING_QUICK_START.md**: Curriculum training quick reference
- **REAL_PRIOR_GUIDE.md**: Real-valued prior enforcement guide
- **ITERATION_VISUALIZATION_GUIDE.md**: Visualization features guide
- **MEASUREMENT_DOMAIN_VISUALIZATION.md**: Measurement domain analysis

---

## 🧪 Testing and Evaluation

### Single Sample Testing

```bash
# Edit test_unsupervised_model.py to set SAMPLE_IDX
python test_unsupervised_model.py
```

**Outputs:**
- `unsupervised_model_image_domain.png`: Reconstructed reflectivity
- `unsupervised_model_measurement_domain.png`: Measurement domain comparison

### Full Dataset Evaluation

```bash
python test_full_dataset.py
```

**Outputs:**
- Per-sample plots in `test_results/`
- Average MSE across dataset

### Final SAR Image Generation

```bash
python inference.py
```

**Outputs:**
- `dbp_reconstructed_image.png`: Reconstructed SAR image (Cartesian)
- `ground_truth_image.png`: Ground truth SAR image (if available)

---

## 🎓 Key Concepts

### ADMM (Alternating Direction Method of Multipliers)

Optimization algorithm for constrained problems:

```
Minimize: f(x) + g(z)
Subject to: x = z
```

**ADMM Steps:**
1. **x-update**: Minimize with respect to x (data consistency)
2. **z-update**: Minimize with respect to z (L2-ball projection)
3. **u-update**: Update dual variable (Lagrange multiplier)

### Unrolled Networks

- Fixed number of iterations (e.g., 5)
- Each iteration = trainable layer
- Gradient flows through entire network
- Learns optimal parameters for specific problem

### CNN Denoiser as Prior

- Replaces traditional regularization (e.g., L1-norm)
- Learns task-specific prior from data
- More expressive than hand-crafted priors

---

## 🔬 Research Features

This framework supports various research directions:

1. **Architecture Search**: Modify CNN denoiser architecture
2. **Loss Functions**: Experiment with different loss formulations
3. **Transfer Learning**: Use two-stage training with denoisers from other domains
4. **Interpretability**: Visualize intermediate outputs to understand network behavior
5. **Robustness**: Test supervised vs. unsupervised under noise/model mismatch

---

## 🛠️ Requirements

- Python 3.7+
- PyTorch 1.8+
- NumPy
- SciPy
- Matplotlib
- h5py (for MATLAB v7.3 files)

---

## 📝 Citation

If you use this framework in your research, please cite:

```
[Your paper citation here]
```

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional denoiser architectures (e.g., ResNet, U-Net)
- Support for multi-channel data
- Advanced visualization tools
- Performance optimizations

---

## 📄 License

[Your license here]

---

## 🙏 Acknowledgments

This framework implements unrolled optimization networks for MIMO SAR imaging, inspired by:

- **Deep ADMM-Net**: Learning-based approach for compressive sensing
- **Plug-and-Play Priors**: Using denoisers as regularizers
- **Model-Based Deep Learning**: Combining domain knowledge with neural networks

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your email].
