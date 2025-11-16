# Real-Valued Prior Enforcement

## 🎯 Overview

When the target reflectivity `x` is known to be **real-valued** (no imaginary component), this prior knowledge can be leveraged to improve reconstruction quality. This guide explains how to enforce real-valued outputs in your training.

---

## 📊 The Problem

In many SAR imaging scenarios, the reflectivity is inherently real. However, the network may learn to output complex values (with non-zero imaginary parts) during training because:

1. **Complex representation**: Data is represented as 2-channel tensors `[real, imag]`
2. **No explicit constraint**: Standard loss functions don't penalize imaginary components
3. **Measurement noise**: Noise in measurements can lead to spurious imaginary components
4. **Optimization artifacts**: Network may use imaginary parts as "degrees of freedom"

**Result**: Unnecessary complexity in the solution, potentially worse reconstruction.

---

## 💡 The Solution: Three Enforcement Strategies

### **1. Loss Penalty** (`loss_penalty`)

Add a penalty term to the loss function that discourages non-zero imaginary parts:

```python
loss_total = loss_primary + λ * ||x_imaginary||²
```

**How it works:**
- Network output can be complex during forward pass
- Loss penalizes large imaginary components
- Network learns to minimize imaginary part
- Gradual enforcement through learning

**Pros:**
- Soft constraint (allows small violations)
- Smooth gradients
- Network learns naturally

**Cons:**
- Doesn't guarantee exactly zero imaginary part
- Requires tuning penalty weight λ

**When to use:**
- You want gradual learning
- Small imaginary components are acceptable
- You're doing unsupervised training (measurement consistency more important)

---

### **2. Hard Projection** (`hard_projection`)

Force imaginary part to zero immediately after forward pass:

```python
x_output[: 1, :] = 0  # Set imaginary channel to zero
```

**How it works:**
- Network output is modified before loss computation
- Imaginary part is explicitly set to zero
- Only real part affects the loss and gradients

**Pros:**
- Guarantees exactly zero imaginary part
- Simple and deterministic
- No hyperparameter tuning

**Cons:**
- Hard constraint (no flexibility)
- May interfere with gradient flow
- Could slow convergence initially

**When to use:**
- You MUST have real-only outputs
- You want simplicity
- You're doing supervised training (direct image comparison)

---

### **3. Hybrid** (`hybrid`)

Combine both approaches:

```python
# 1. Compute penalty before projection
penalty = λ * ||x_imaginary||²

# 2. Project to real space
x_output[:, 1, :] = 0

# 3. Add penalty to loss
loss_total = loss_primary + penalty
```

**How it works:**
- Penalty encourages network to learn real outputs
- Projection ensures outputs are exactly real
- Best of both worlds

**Pros:**
- Exact real outputs (via projection)
- Smooth learning (via penalty)
- Robust to any situation

**Cons:**
- Slightly more computation
- One extra hyperparameter

**When to use:**
- **RECOMMENDED DEFAULT** - works well in most cases
- You want guaranteed real outputs
- You want smooth training dynamics

---

## ⚙️ Configuration

### In `train_configurable.py`, `train.py`, or `train_denoiser_curriculum.py`:

```python
# Real-Valued Prior Enforcement
ENFORCE_REAL_PRIOR = True  # Enable real prior
REAL_PRIOR_STRATEGY = 'hybrid'  # Strategy: 'loss_penalty', 'hard_projection', 'hybrid'
REAL_PRIOR_WEIGHT = 0.1  # Penalty weight λ (for loss_penalty and hybrid)
```

### Parameters

| Parameter | Options | Description |
|-----------|---------|-------------|
| `ENFORCE_REAL_PRIOR` | `True` / `False` | Enable or disable real prior enforcement |
| `REAL_PRIOR_STRATEGY` | `'loss_penalty'` | Add penalty term to loss |
|  | `'hard_projection'` | Force imaginary part to zero |
|  | `'hybrid'` | Both penalty and projection |
| `REAL_PRIOR_WEIGHT` | Float (e.g., `0.1`) | Weight λ for imaginary penalty |

---

## 🚀 Usage Examples

### Example 1: Unsupervised Training with Hybrid Strategy

```python
# In train.py or train_configurable.py
ENFORCE_REAL_PRIOR = True
REAL_PRIOR_STRATEGY = 'hybrid'
REAL_PRIOR_WEIGHT = 0.1

# Run training
python train.py
```

**Output:**
```
--- Real Prior: HYBRID (weight=0.1) ---
...
Training complete
Final imaginary magnitude ratio: 0.000001
```

---

### Example 2: Supervised Training with Hard Projection

```python
# In train_configurable.py
TRAINING_MODE = 'supervised'
ENFORCE_REAL_PRIOR = True
REAL_PRIOR_STRATEGY = 'hard_projection'
# No need to set REAL_PRIOR_WEIGHT (not used)

python train_configurable.py
```

**Result:** All outputs will have exactly zero imaginary part.

---

### Example 3: Curriculum Training with Loss Penalty

```python
# In train_denoiser_curriculum.py
ENFORCE_REAL_PRIOR = True
REAL_PRIOR_STRATEGY = 'loss_penalty'
REAL_PRIOR_WEIGHT = 0.05  # Lower weight for gentler enforcement

python train_denoiser_curriculum.py
```

**Result:** Network learns to minimize imaginary components gradually.

---

## 🔧 Hyperparameter Tuning

### Choosing `REAL_PRIOR_WEIGHT`

The penalty weight λ controls how strongly the imaginary part is penalized.

| Weight | Effect | Use Case |
|--------|--------|----------|
| `0.01` | Weak penalty | Mostly allow imaginary, slight encouragement |
| `0.05` | Mild penalty | Balance between real prior and main loss |
| `0.1` | **Recommended** | Good default for most cases |
| `0.5` | Strong penalty | Aggressive real enforcement |
| `1.0` | Very strong | Imaginary minimization becomes primary objective |

**Tuning guidelines:**
1. Start with `0.1` (works well for most cases)
2. If imaginary part is still large (ratio > 0.01), increase to `0.5`
3. If training is unstable, decrease to `0.05`
4. For unsupervised, use lower values (`0.05-0.1`)
5. For supervised, can use higher values (`0.1-0.5`)

---

## 📊 Monitoring Results

### During Training

The training scripts will print:
```
--- Real Prior: HYBRID (weight=0.1) ---
```

### After Training

Check the imaginary magnitude ratio:

```python
from real_prior import measure_imaginary_magnitude

ratio = measure_imaginary_magnitude(x_hat)
print(f"Imaginary/Real ratio: {ratio}")
```

**Interpretation:**
- `ratio < 0.001`: Excellent (imaginary ~ negligible)
- `ratio < 0.01`: Good (imaginary is small)
- `ratio < 0.1`: Acceptable (some imaginary component)
- `ratio > 0.1`: Poor (significant imaginary component)

---

## 🧪 Testing Real Prior Enforcement

### Standalone Test

```bash
python real_prior.py
```

**Output:**
```
[Test 1] Hard Projection
  Imag should be zero: True

[Test 2] Loss Penalty
  Penalty value: 0.094354

[Test 3] Hybrid
  Imag is zero: True
  
✅ ALL TESTS PASSED
```

### Integration Test

```python
import torch
from real_prior import enforce_real_prior

# Create test data
x_complex = torch.randn(4, 2, 51)  # [batch, channels, N_theta]

# Apply real prior
x_real, penalty = enforce_real_prior(
    x_complex,
    strategy='hybrid',
    penalty_weight=0.1
)

print(f"Imaginary part max: {x_real[:, 1, :].abs().max()}")  # Should be 0.0
print(f"Penalty: {penalty:.6f}")
```

---

## 🔬 Technical Details

### Data Representation

Complex values are represented as 2-channel tensors:
```
x_complex: [batch, 2, N_theta]
  x_complex[:, 0, :] = real part
  x_complex[:, 1, :] = imaginary part
```

### Implementation

**Loss Penalty:**
```python
imaginary_part = x[:, 1, :]
penalty = λ * mean(imaginary_part²)
total_loss = primary_loss + penalty
```

**Hard Projection:**
```python
x_real = x.clone()
x_real[:, 1, :] = 0.0  # Zero out imaginary channel
```

**Hybrid:**
```python
# 1. Compute penalty first (before modifying x)
penalty = λ * mean(x[:, 1, :]²)

# 2. Project
x[:, 1, :] = 0.0

# 3. Use projected x for primary loss, add penalty
total_loss = primary_loss(x) + penalty
```

### Gradient Flow

**Loss Penalty:**
- Gradients flow through both real and imaginary channels
- Imaginary gradient includes penalty term: `∂L/∂x_imag = ... + 2λx_imag`

**Hard Projection:**
- Only real channel receives gradients
- Imaginary channel gradient is zero (no update)

**Hybrid:**
- Both channels receive gradients during penalty computation
- Only real channel affects primary loss
- Smoothest gradient landscape

---

## 🆚 Comparison Table

| Feature | No Prior | Loss Penalty | Hard Projection | Hybrid |
|---------|----------|--------------|-----------------|--------|
| **Imaginary part** | Non-zero | Small | Exactly zero | Exactly zero |
| **Enforcement** | None | Soft (gradual) | Hard (immediate) | Hard + soft |
| **Gradient flow** | Both channels | Both channels | Real only | Both → Real |
| **Hyperparams** | 0 | 1 (weight) | 0 | 1 (weight) |
| **Training dynamics** | Standard | Smooth | May be slower | Smooth |
| **Use case** | Complex targets | Real targets (flexible) | Real targets (strict) | **Real targets (best)** |

**Recommendation:** Use **Hybrid** strategy with `REAL_PRIOR_WEIGHT = 0.1` as the default.

---

## 📚 Integration with Training Scripts

### Modified Scripts

All training scripts now support real prior enforcement:

1. **`train.py`** - Original unsupervised training
2. **`train_configurable.py`** - Configurable training (supervised/unsupervised/hybrid/two-stage)
3. **`train_denoiser_curriculum.py`** - Curriculum training for denoiser

### How It's Integrated

In all scripts, after the forward pass:

```python
# Forward pass
x_hat = model(y)

# Apply real prior (if enabled)
if ENFORCE_REAL_PRIOR:
    x_hat, penalty = enforce_real_prior(
        x_hat,
        strategy=REAL_PRIOR_STRATEGY,
        penalty_weight=REAL_PRIOR_WEIGHT
    )

# Compute loss (penalty is added automatically)
loss = primary_loss(x_hat, target) + penalty
```

---

## 🎓 When to Use Real Prior

### ✅ Use Real Prior When:

1. **Physical constraint**: Target reflectivity is physically real
2. **Prior knowledge**: You know imaginary part should be zero
3. **Simpler solution**: Want to reduce model complexity
4. **Better interpretability**: Real values are easier to interpret
5. **Noise reduction**: Imaginary part may capture noise

### ❌ Don't Use Real Prior When:

1. **Complex targets**: Reflectivity is actually complex
2. **Phase information**: Imaginary part contains useful information
3. **Model mismatch**: Forward model `A` may have errors
4. **Experimentation**: You want to see what network learns naturally

---

## 🛠️ Troubleshooting

### Issue 1: Large Imaginary Part After Training

**Symptoms:**
- Imaginary/Real ratio > 0.1
- Real prior seems not working

**Solutions:**
1. Increase `REAL_PRIOR_WEIGHT` (try `0.5` or `1.0`)
2. Switch to `'hybrid'` or `'hard_projection'` strategy
3. Train for more epochs (prior needs time to take effect)
4. Check if target is actually real in your data

---

### Issue 2: Training Unstable with Real Prior

**Symptoms:**
- Loss oscillates or diverges
- NaN values appear

**Solutions:**
1. Decrease `REAL_PRIOR_WEIGHT` (try `0.01` or `0.05`)
2. Switch to `'loss_penalty'` strategy (softer)
3. Reduce learning rate
4. Check for numerical issues in data

---

### Issue 3: No Improvement with Real Prior

**Symptoms:**
- Similar performance with/without real prior
- Imaginary part was already small

**Solutions:**
- This is actually good! Your network was already learning real outputs
- Real prior provides a guarantee and may help generalization
- Keep it enabled for robustness

---

## 📝 Summary

**Key Takeaways:**

1. **Enable real prior** if your target reflectivity is real-valued
2. **Use hybrid strategy** with `weight=0.1` as default (best balance)
3. **Monitor** imaginary magnitude ratio after training
4. **Tune** penalty weight if needed (0.05-0.5 range)
5. **Works seamlessly** with all training modes and strategies

**Quick Setup:**

```python
# Add to your training script
ENFORCE_REAL_PRIOR = True
REAL_PRIOR_STRATEGY = 'hybrid'
REAL_PRIOR_WEIGHT = 0.1
```

That's it! Your network will now learn real-valued outputs. 🎉

---

## 🔗 Related Documentation

- **`SUPERVISED_TRAINING_GUIDE.md`**: Supervised training with real prior
- **`TWO_STAGE_TRAINING_GUIDE.md`**: Two-stage training with real prior
- **`CURRICULUM_TRAINING_GUIDE.md`**: Curriculum training with real prior
- **`real_prior.py`**: Source code and utility functions

---

## 📧 Questions?

If you have questions about real prior enforcement or encounter issues, check:
1. This guide
2. Run `python real_prior.py` to test standalone
3. Check training logs for "Real Prior:" status
4. Monitor imaginary magnitude ratio

Happy training with real priors! 🚀✨

