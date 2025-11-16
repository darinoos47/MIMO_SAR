# Quick Reference: Supervised Training

## Choose Your Training Mode (1 minute)

Edit `train_configurable.py` line 27:

```python
TRAINING_MODE = 'supervised'    # Has ground truth? Use this
TRAINING_MODE = 'unsupervised'  # No ground truth? Use this  
TRAINING_MODE = 'hybrid'        # Want both? Use this (recommended!)
```

## Run Training (1 command)

```bash
python train_configurable.py
```

## Loss Functions At-a-Glance

| Mode | Formula | When to Use |
|------|---------|-------------|
| **Supervised** | `‖x̂ - x_gt‖²` | Have accurate ground truth |
| **Unsupervised** | `‖Ax̂ - y‖²` | No ground truth (original) |
| **Hybrid** | `α‖x̂ - x_gt‖² + β‖Ax̂ - y‖²` | Best of both! |

## Recommended Hybrid Weights

```python
# Balanced (default)
SUPERVISED_WEIGHT = 1.0
UNSUPERVISED_WEIGHT = 1.0

# Favor image quality
SUPERVISED_WEIGHT = 1.0
UNSUPERVISED_WEIGHT = 0.1  # ← Recommended!

# Favor measurement fit
SUPERVISED_WEIGHT = 0.1
UNSUPERVISED_WEIGHT = 1.0
```

## Compare All Modes (automatic)

```bash
python compare_training_modes.py
```

Trains all three modes and saves results for comparison.

## Files Generated

- `dbp_model.pth` - Trained model
- `training_loss_{mode}.png` - Loss curves
- `train_debug_plot_x.png` - Image domain
- `train_debug_plot_y.png` - Measurement domain

## Backward Compatible

Old code still works:
```bash
python train.py  # Original unsupervised training
```

## Need Help?

- Full guide: `SUPERVISED_TRAINING_GUIDE.md`
- Summary: `SUPERVISED_TRAINING_SUMMARY.md`

