# Environment Variable Profiling - Quick Guide

This document shows how to enable profiling using environment variables - the simplest way to profile your training runs.

## Quick Start

```bash
# Profile step 100 (one time)
PROFILE_ENABLED=1 PROFILE_START_STEP=100 .venv/bin/python -m ueaj.train.train

# Profile every 1000 steps starting at step 100
PROFILE_ENABLED=1 PROFILE_START_STEP=100 PROFILE_INTERVAL=1000 .venv/bin/python -m ueaj.train.train

# Profile 3 consecutive steps starting at step 50
PROFILE_ENABLED=1 PROFILE_START_STEP=50 PROFILE_DURATION=3 PROFILE_INTERVAL=0 .venv/bin/python -m ueaj.train.train
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PROFILE_ENABLED` | `0` | Set to `1` to enable profiling |
| `PROFILE_DIR` | `./profiles` | Directory for profile outputs |
| `PROFILE_START_STEP` | `100` | Step to start profiling |
| `PROFILE_INTERVAL` | `1000` | Profile every N steps (0 = only once) |
| `PROFILE_DURATION` | `1` | Number of consecutive steps to profile |
| `PROFILE_MODE` | `tensorboard` | `tensorboard` or `perfetto` |

## Integration Pattern

Add this to your training loop:

```python
from ueaj.utils import maybe_profile

# Training loop
for step in range(max_steps):
    # This line handles all profiling logic based on env vars!
    with maybe_profile(step):
        loss = train_step(model, optimizer, batch)
        loss.block_until_ready()  # CRITICAL!

    # Your normal logging, checkpointing, etc.
    if step % 100 == 0:
        print(f"Step {step}: loss={loss:.4f}")
```

That's it! The `maybe_profile(step)` context manager:
- Checks if profiling is enabled
- Determines if this step should be profiled
- Starts/stops profiling automatically
- Saves traces to the correct directory

## Example Usage Patterns

### Pattern 1: One-Time Profile at Specific Step

Profile just step 100 to analyze steady-state performance:

```bash
PROFILE_ENABLED=1 \
PROFILE_START_STEP=100 \
PROFILE_INTERVAL=0 \
  .venv/bin/python -m ueaj.train.train
```

**Output:**
```
profiles/step_00000100/
```

### Pattern 2: Periodic Profiling

Profile steps 100, 1100, 2100, ... to track performance over training:

```bash
PROFILE_ENABLED=1 \
PROFILE_START_STEP=100 \
PROFILE_INTERVAL=1000 \
  .venv/bin/python -m ueaj.train.train
```

**Output:**
```
profiles/step_00000100/
profiles/step_00001100/
profiles/step_00002100/
...
```

### Pattern 3: Multi-Step Profile

Profile 5 consecutive steps to see step-to-step variation:

```bash
PROFILE_ENABLED=1 \
PROFILE_START_STEP=100 \
PROFILE_DURATION=5 \
PROFILE_INTERVAL=0 \
  .venv/bin/python -m ueaj.train.train
```

**Output:**
```
profiles/step_00000100/
profiles/step_00000101/
profiles/step_00000102/
profiles/step_00000103/
profiles/step_00000104/
```

### Pattern 4: Custom Profile Directory

Save profiles to a specific location:

```bash
PROFILE_ENABLED=1 \
PROFILE_DIR=~/experiment_profiles/run_001 \
PROFILE_START_STEP=50 \
  .venv/bin/python -m ueaj.train.train
```

**Output:**
```
~/experiment_profiles/run_001/step_00000050/
```

### Pattern 5: Perfetto Mode

Use Perfetto UI instead of TensorBoard:

```bash
PROFILE_ENABLED=1 \
PROFILE_MODE=perfetto \
PROFILE_START_STEP=100 \
  .venv/bin/python -m ueaj.train.train
```

**To view:** Upload `profiles/step_*/plugins/profile/*/perfetto_trace.json.gz` to https://ui.perfetto.dev

## Complete Training Example

```bash
# Full training command with profiling
OPTIMIZER=muon \
RUN_NAME=exp_muon_001 \
MODEL_PATH=./checkpoints \
BASE_LR=0.025 \
PROFILE_ENABLED=1 \
PROFILE_START_STEP=100 \
PROFILE_INTERVAL=1000 \
  .venv/bin/python -m ueaj.train.train
```

This will:
- Train with Muon optimizer
- Log to WANDB as "exp_muon_001"
- Save checkpoints to ./checkpoints
- Profile steps 100, 1100, 2100, ...
- Save profiles to ./profiles/

## Viewing the Profiles

### Method 1: TensorBoard (Default)

```bash
# Start TensorBoard
.venv/bin/tensorboard --logdir=./profiles

# Open browser to:
http://localhost:6006
```

Then:
1. Click **PROFILE** tab
2. Select a run (e.g., `step_00000100`)
3. Choose analysis tool from dropdown:
   - **Overview** - Performance summary
   - **Trace Viewer** - Timeline of operations
   - **Op Profile** - Operation statistics
   - **Memory Profile** - Memory usage

### Method 2: Perfetto UI (if using PROFILE_MODE=perfetto)

1. Go to: https://ui.perfetto.dev
2. Drag and drop: `profiles/step_*/plugins/profile/*/perfetto_trace.json.gz`
3. Analyze in interactive viewer

## What Gets Profiled

The profiler captures:

✅ **Included:**
- Model forward pass
- Loss computation
- Gradient computation (backward pass)
- Optimizer update
- Memory transfers
- Kernel launches
- XLA compilation (first run)

❌ **Not Included:**
- Data loading (unless inside `maybe_profile` block)
- WANDB logging
- Checkpoint saving
- Python overhead outside JIT functions

## Tips

### 1. Always Warmup Before Profiling

```python
# ✅ Good - warmup first
for _ in range(5):
    loss = train_step(...)

with maybe_profile(step):
    loss = train_step(...)
```

### 2. Use .block_until_ready()

```python
# ✅ Correct
with maybe_profile(step):
    loss = train_step(...)
    loss.block_until_ready()  # WAIT for completion!

# ❌ Wrong - will only capture dispatch overhead
with maybe_profile(step):
    loss = train_step(...)  # Returns immediately!
```

### 3. Profile Representative Steps

```bash
# ✅ Good - profile after warmup
PROFILE_START_STEP=100

# ❌ Bad - step 0 includes compilation
PROFILE_START_STEP=0
```

### 4. Limit Profile Duration

```bash
# ✅ Good - 1-3 steps
PROFILE_DURATION=1

# ❌ Bad - creates huge files
PROFILE_DURATION=100
```

### 5. Use Linux Filesystem (WSL)

```bash
# ✅ Fast - Linux filesystem
PROFILE_DIR=~/profiles

# ❌ Slow - Windows filesystem
PROFILE_DIR=/mnt/c/Users/devse/profiles
```

## Troubleshooting

### "No profiling output appears"

**Check:** Is `PROFILE_ENABLED=1` set?
```bash
echo $PROFILE_ENABLED  # Should print: 1
```

**Check:** Is your step >= PROFILE_START_STEP?
```bash
# If PROFILE_START_STEP=100, profiling starts at step 100
```

### "Profile directory is empty"

**Cause:** Forgot `.block_until_ready()`
```python
# Fix:
with maybe_profile(step):
    loss = train_step(...)
    loss.block_until_ready()  # Add this!
```

### "Profiling every step (too slow)"

**Cause:** `PROFILE_INTERVAL=0` with high `PROFILE_DURATION`
```bash
# Fix: Use interval mode
PROFILE_INTERVAL=1000  # Profile every 1000 steps
```

### "TensorBoard shows no data"

**Cause:** Plugin not installed
```bash
.venv/bin/pip install tensorboard-plugin-profile
# Restart TensorBoard
```

## Summary

**Minimal setup:**
```bash
PROFILE_ENABLED=1 your_script.py
```

**Recommended setup:**
```bash
PROFILE_ENABLED=1 \
PROFILE_START_STEP=100 \
PROFILE_INTERVAL=1000 \
  your_script.py
```

**View results:**
```bash
.venv/bin/tensorboard --logdir=./profiles
# Open: http://localhost:6006
```

**In code:**
```python
from ueaj.utils import maybe_profile

with maybe_profile(step):
    result = expensive_operation()
    result.block_until_ready()
```

See also:
- `docs/PROFILING_GUIDE.md` - Comprehensive profiling guide
- `docs/VIEWING_PROFILES.md` - Detailed viewing instructions
- `docs/PROFILING_QUICKREF.md` - Quick reference card
