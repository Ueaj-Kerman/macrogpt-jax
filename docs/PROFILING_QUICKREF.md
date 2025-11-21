# JAX Profiling Quick Reference

## One-Line Profiling Commands

```bash
# Profile training step
.venv/bin/python scripts/profile_training.py

# Profile forward pass only
.venv/bin/python scripts/profile_training.py forward

# Profile model components
.venv/bin/python scripts/profile_training.py components

# Benchmark performance
.venv/bin/python scripts/profile_training.py benchmark

# Start TensorBoard
.venv/bin/tensorboard --logdir=./profiles
# Then open: http://localhost:6006
```

## Code Snippets

### Basic Profiling

```python
from ueaj.utils import profile_trace

# Warmup (REQUIRED!)
for _ in range(3):
    result = my_function()
    result.block_until_ready()

# Profile
with profile_trace("./profiles", name="my_profile"):
    result = my_function()
    result.block_until_ready()  # CRITICAL!
```

### Profile with Scopes (for detailed analysis)

```python
from ueaj.utils import profile_trace, profile_scope

with profile_trace("./profiles", name="detailed"):
    with profile_scope("attention"):
        attn_out = attention(x)
    with profile_scope("mlp"):
        mlp_out = mlp(attn_out)
    mlp_out.block_until_ready()
```

### Profile a Function (decorator)

```python
from ueaj.utils import profile_function

@profile_function("./profiles", warmup_steps=5)
def train_step(model, batch):
    return updated_model, loss
```

### Benchmark Performance

```python
from ueaj.utils import benchmark_function

stats = benchmark_function(fn, args, num_iterations=100)
print(f"{stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
```

## Integration with Existing Training

### Method 1: Environment Variables (Easiest!)

```python
# In ueaj/train/train.py
from ueaj.utils import maybe_profile

# Inside training loop:
for step in range(max_steps):
    with maybe_profile(step):  # Handles everything automatically!
        loss = train_step(model, optimizer, batch)
        loss.block_until_ready()
```

Then control via environment variables:
```bash
# Profile step 100 only
PROFILE_ENABLED=1 PROFILE_START_STEP=100 .venv/bin/python -m ueaj.train.train

# Profile every 1000 steps
PROFILE_ENABLED=1 PROFILE_START_STEP=100 PROFILE_INTERVAL=1000 .venv/bin/python -m ueaj.train.train

# See docs/PROFILING_ENV_VARS.md for all options
```

### Method 2: Manual Control

```python
# In ueaj/train/train.py
from ueaj.utils import profile_trace

# Inside training loop:
should_profile = (step % 1000 == 0) and step > 0

if should_profile:
    with profile_trace(f"./profiles/step_{step}"):
        loss = train_step(...)
        loss.block_until_ready()
else:
    loss = train_step(...)
```

## TensorBoard Analysis Workflow

1. **Overview Page** → Identify slow operations
2. **Trace Viewer** → See timeline of execution
3. **Op Profile** → Table of operations sorted by time
4. **Memory Profile** → Find memory bottlenecks

### What to Look For

- ✅ **Good**: GPU utilization >80%, minimal CPU fallback
- ⚠️ **Warning**: Memory transfers, small kernels, CPU ops
- 🔴 **Bad**: GPU idle time, excessive host-device copies

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "No trace data" | Add `.block_until_ready()` |
| "Empty TensorBoard" | Install `tensorboard-plugin-profile` |
| Profile too large | Profile only 1-2 iterations |
| Slow I/O | Use Linux filesystem (`~/profiles` not `/mnt/c/...`) |

## File Locations

```
profiles/
├── forward_pass/           # Forward pass profile
│   └── plugins/profile/*/trace.json.gz
├── training_step/          # Full training step
│   └── plugins/profile/*/trace.json.gz
└── model_components/       # Component breakdown
    └── plugins/profile/*/trace.json.gz
```

Upload `.gz` files to https://ui.perfetto.dev for visualization.

## WSL-Specific Tips

```bash
# View TensorBoard from Windows
# Open: http://localhost:6006

# If localhost doesn't work, get WSL IP:
hostname -I
# Then use: http://<WSL_IP>:6006

# For best performance, profile to Linux filesystem:
profile_trace("~/profiles", ...)  # ✅ Good
profile_trace("/mnt/c/...", ...)  # ❌ Slow
```

## Performance Optimization Checklist

After profiling, check:

- [ ] GPU utilization >80%
- [ ] No unexpected CPU operations
- [ ] Minimal host-device memory transfers
- [ ] Attention operations using fused kernels
- [ ] No small (<1ms) kernels (indicates poor fusion)
- [ ] Memory usage within budget
- [ ] No excessive compilation time

## Advanced: Profiler Options

```python
import jax.profiler

# Full API
with jax.profiler.trace(
    log_dir="./profiles",
    create_perfetto_link=False,     # Interactive link (blocks)
    create_perfetto_trace=True,     # Save to file (TensorBoard)
):
    computation()
```

## Quick Wins

After profiling, common optimizations:

1. **Enable flash attention** (if not already)
2. **Use bfloat16** for matmuls on A100+
3. **Increase batch size** to improve GPU utilization
4. **Profile data loading** separately (might be bottleneck)
5. **Check for CPU fallbacks** in custom operations
