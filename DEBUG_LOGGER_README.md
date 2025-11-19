# Debug Logger Documentation

## Overview

The training script now includes a comprehensive debug logging system that writes structured diagnostic data to JSONL files. This allows you to run long training sessions and analyze the results afterwards.

## Usage

### Enable Debug Logging (default: ON)

```bash
python train.py \
  --datadir /path/to/data \
  --metapath /path/to/train.json \
  --val_metapath /path/to/val.json \
  --saving_path ./ckpt \
  --enable_debug_logger \
  --debug_log_dir ./logs/debug \
  ... other args ...
```

### Log Files

The debug logger creates two files in the debug log directory:

1. **`debug_log_<timestamp>.jsonl`** - Main log file with all events (JSONL format)
2. **`debug_summary_<timestamp>.json`** - Summary statistics (coming soon)

## Event Types Logged

### 1. `fit_start` (step 0)
- Trainable/frozen/total parameters
- Training configuration (epochs, steps, batch size, etc.)
- Optimizer configuration

### 2. `epoch_start` (every epoch)
- Current optimizer state (LR, betas, weight decay)

### 3. `training_step` (every 10 steps)
- Loss and accuracy
- Current learning rate
- Sample predictions vs. ground truth labels
- Tensor shapes and valid token counts

### 4. `gradient_stats` (every 10 steps)
- Total gradient norm
- Max/min gradient values
- Number of parameters with gradients
- Top 10 modules by gradient norm

### 5. `param_updates` (every 50 steps)
- Number of parameters that actually changed
- Top 10 parameters by change magnitude
- Warning flag if no parameters updated

### 6. `data_sanity_check` (step 0 only)
- Label range validation
- Checks if labels are within expected bounds

## Analyzing Logs

### Quick Analysis Script

```bash
python analyze_debug_logs.py logs/debug/debug_log_<timestamp>.jsonl
```

This will print:
- **Training progression**: Loss/acc by epoch, improvement check
- **Gradient analysis**: Gradient norms, top modules, zero-gradient warnings
- **Parameter updates**: Update counts, top changing params, no-update warnings
- **Data issues**: Label range validation

### Example Output

```
=== TRAINING PROGRESSION ===
Total training steps logged: 1000
Epochs covered: 0 to 9

Epoch 0:
  Steps: 100
  Loss:  6.8234 -> 6.2145 (avg: 6.5102)
  Acc:   0.0234 -> 0.0456 (avg: 0.0345)
  LR:    2.000000e-05 -> 2.000000e-04

=== IMPROVEMENT CHECK ===
First 10 steps:  Loss=6.9012, Acc=0.0201
Last 10 steps:   Loss=5.8234, Acc=0.0789
Loss change:     -1.0778 (IMPROVED)
Acc change:      +0.0588 (IMPROVED)

=== GRADIENT ANALYSIS ===
Total gradient norm:
  Range: 0.2341 to 45.6789
  Mean:  12.3456

⚠️  WARNING: 0 steps with near-zero gradients! (Good!)

=== PARAMETER UPDATES ===
Parameters updated per measurement: 234/234
⚠️  WARNING: 5 measurements with NO parameter updates! (Check optimizer!)
```

## What to Look For

### ✅ **Good Signs:**
- Loss decreasing over epochs
- Accuracy increasing over epochs
- Gradient norms in reasonable range (0.1 - 100)
- All parameters updating
- LR starting small and ramping up during warmup
- No zero-gradient warnings

### ⚠️ **Bad Signs & What They Mean:**

1. **Loss/Acc flat across epochs**
   - → Optimizer not updating weights
   - → Check learning rate schedule
   - → Check if gradients are flowing

2. **Zero gradient norms**
   - → No gradient flow
   - → Check if loss is connected to parameters
   - → Check for detached tensors

3. **NO PARAMETER UPDATES warnings**
   - → Learning rate might be too small
   - → Gradients might be zero
   - → Optimizer might not be stepping

4. **Labels out of range**
   - → Data preprocessing issue
   - → Check `n_codes` vs. actual codec vocabulary
   - → Regenerate metadata if needed

5. **Gradient norms exploding (>1000)**
   - → Gradient explosion
   - → Increase gradient clipping
   - → Reduce learning rate

## Manual Log Analysis

The JSONL format can be parsed with any tool:

### Python
```python
import json

events = []
with open('debug_log_1234567890.jsonl', 'r') as f:
    for line in f:
        events.append(json.loads(line))

# Filter specific event types
training_steps = [e for e in events if e['event_type'] == 'training_step']
losses = [e['data']['loss'] for e in training_steps]

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
```

### Shell (jq)
```bash
# Extract all losses
cat debug_log_*.jsonl | jq 'select(.event_type=="training_step") | .data.loss'

# Find steps with zero gradients
cat debug_log_*.jsonl | jq 'select(.event_type=="gradient_stats" and .data.total_norm < 0.001)'

# Get learning rate progression
cat debug_log_*.jsonl | jq 'select(.event_type=="training_step") | {step: .step, lr: .data.learning_rate}'
```

## Logging Frequency

To reduce overhead, events are logged at different frequencies:
- `training_step`: every 10 steps
- `gradient_stats`: every 10 steps
- `param_updates`: every 50 steps
- `epoch_start`: every epoch
- `fit_start`, `data_sanity_check`: once at start

Logs are buffered and written in batches of 100 events to minimize I/O overhead.

## Disabling Debug Logger

If you want to disable it (not recommended for debugging):

```bash
python train.py ... --no-enable_debug_logger
```

Or set in code:
```python
args.enable_debug_logger = False
```

## Tips

1. **Run for at least 5-10 epochs** to see trends
2. **Compare first vs. last epoch** statistics
3. **Look for sudden changes** in gradients or loss
4. **Check the first 100 steps carefully** - if nothing improves here, something is wrong
5. **Use the analysis script** - it's faster than manual inspection

