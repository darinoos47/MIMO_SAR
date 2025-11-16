# Fix Summary: Dynamic Angle Bins

## Problem
The code had **hardcoded `NUM_ANGLE_BINS = 1001`** in multiple files, which caused a dimension mismatch error when training with datasets that had a different number of angle bins (e.g., 1002):

```
ValueError: x and y must have same first dimension, but have shapes (1001,) and (1002,)
```

## Root Cause
The number of angle bins (`N_theta`) depends on the MATLAB data generation script and can vary. However, the Python code assumed it was always 1001.

## Files Fixed

### 1. `train.py`
- **Before**: `NUM_ANGLE_BINS = 1001` (line 131)
- **After**: `NUM_ANGLE_BINS = x_hat_real.shape[0]` (dynamically determined from actual data)

### 2. `test_unsupervised_model.py`
- **Before**: `NUM_ANGLE_BINS = 1001` (line 27)
- **After**: `NUM_ANGLE_BINS = x_hat_np.shape[0]` (line 182, computed when needed)

### 3. `inference.py`
- **Before**: `NUM_ANGLE_BINS = 1001` (line 33)
- **After**: `NUM_ANGLE_BINS = A_tensor.shape[2]` (line 112, from steering matrix)

### 4. `test_full_dataset.py`
- **Before**: `NUM_ANGLE_BINS = 1001` (line 28)
- **After**: `NUM_ANGLE_BINS = A_tensor.shape[2]` (line 112, from steering matrix)

## Solution Approach
Instead of hardcoding the value, the number of angle bins is now determined dynamically from the actual data:
- From the **steering matrix A**: `A_tensor.shape[2]` gives `N_theta`
- From the **output**: `x_hat.shape[-1]` gives `N_theta`

## Verification
Tested with a dataset containing:
- 100 training samples
- 8 virtual antennas
- **1002 angle bins** (not 1001!)

Training now works correctly with any number of angle bins without code changes.

## Key Lesson
Always determine array dimensions from the actual data rather than hardcoding expected values, especially when the data comes from external sources (MATLAB).

