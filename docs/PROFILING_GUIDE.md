# JAX Profiling Guide for MacroGPT-JAX

This guide explains how to profile your JAX code using TensorBoard/XProf for detailed performance analysis.

## Quick Start

### 1. Install Dependencies

```bash
.venv/bin/pip install tensorboard tensorboard-plugin-profile
```

### 2. Run Example Profiling

```bash
# Profile a training step
.venv/bin/python scripts/profile_training.py

# Start TensorBoard
.venv/bin/tensorboard --logdir=./profiles

# Open http://localhost:6006 in your browser
```

## Profiling Methods

### Method 1: Profile Entire Training Step (Recommended)

```python
from ueaj.utils import profile_trace

# Warmup first (important!)
for _ in range(3):
    loss = train_step(model, batch)
    loss.block_until_ready()

# Profile
with profile_trace("./profiles", name="train_step"):
    loss = train_step(model, batch)
    loss.block_until_ready()
```

### Method 2: Profile with Named Scopes

Use `profile_scope` to annotate different parts of your code:

```python
from ueaj.utils import profile_trace, profile_scope

with profile_trace("./profiles", name="detailed_profile"):
    with profile_scope("data_loading"):
        batch = next(data_iterator)

    with profile_scope("forward_pass"):
        logits = model(batch['input_ids'])

    with profile_scope("loss_computation"):
        loss = compute_loss(logits, batch['labels'])

    with profile_scope("backward_pass"):
        grads = jax.grad(loss_fn)(model)

    loss.block_until_ready()
```

### Method 3: Decorator for Functions

```python
from ueaj.utils import profile_function

@profile_function("./profiles", warmup_steps=5)
def train_step(state, batch):
    # Your training logic
    return new_state, metrics
```

## Understanding TensorBoard Profiler UI

### Overview Page
- **Step Time Graph**: Shows breakdown of time spent in different operations
- **Device Placement**: Shows which operations run on GPU vs CPU
- **Memory Usage**: Track memory consumption over time

### Trace Viewer
- **Timeline**: See exact timing of every operation
- **Kernel Execution**: View GPU kernel launches and execution times
- **Memory Transfers**: Identify host-to-device and device-to-host copies

### Op Profile
- **Table View**: Sorted list of operations by time/memory
- **HLO Operations**: See XLA-compiled operations and their cost
- **Kernel Stats**: Details on GPU kernels (occupancy, grid size, etc.)

### Memory Profile
- **Peak Memory Usage**: Identify memory bottlenecks
- **Buffer Allocation**: See which tensors consume most memory
- **Memory Timeline**: Track allocation/deallocation patterns

## WSL-Specific Considerations

### GPU Profiling in WSL2

WSL2 supports NVIDIA GPU profiling through the Windows driver. Ensure:

1. **Latest NVIDIA drivers** (supporting WSL2)
2. **CUDA toolkit installed in WSL** (optional, but helpful for nsys)
3. **Proper permissions**: Add yourself to video group if needed

```bash
sudo usermod -a -G video $USER
```

### File System Performance

⚠️ **Important**: Profile output directories should be on the Linux filesystem (`/home/...`) not Windows filesystem (`/mnt/c/...`) for better I/O performance.

```python
# Good - Linux filesystem
profile_trace("/home/devse/profiles", ...)

# Bad - Windows filesystem (slower)
profile_trace("/mnt/c/Users/devse/profiles", ...)
```

### Viewing TensorBoard from Windows

Since TensorBoard runs in WSL, access it from Windows via:
- **Localhost**: `http://localhost:6006` (usually works automatically)
- **WSL IP**: If localhost doesn't work, get WSL IP with `hostname -I`

## Integration with Your Training Loop

### Periodic Profiling During Training

Add profiling at specific steps:

```python
# In ueaj/train/train.py

from ueaj.utils import profile_trace, profile_scope

def train(model, optimizer, data_loader, config):
    for step in range(config.max_steps):
        batch = next(data_loader)

        # Profile every 1000 steps
        should_profile = (step > 0 and step % 1000 == 0)

        if should_profile:
            profile_dir = f"{config.profile_dir}/step_{step}"
            with profile_trace(profile_dir, name=f"step_{step}"):
                loss, metrics = train_step(model, optimizer, batch)
                loss.block_until_ready()
        else:
            loss, metrics = train_step(model, optimizer, batch)

        # Logging, checkpointing, etc.
```

### Profile Specific Model Components

```python
from ueaj.utils import profile_trace, profile_scope

with profile_trace("./profiles", name="model_breakdown"):
    # Profile each transformer layer
    x = model.embed(input_ids)

    for i, layer in enumerate(model.layers):
        with profile_scope(f"layer_{i}"):
            x = layer(x)

    with profile_scope("lm_head"):
        logits = model.lm_head(x)

    logits.block_until_ready()
```

## Common Profiling Patterns

### 1. Find Bottlenecks in Training

```bash
# Profile single training step
.venv/bin/python scripts/profile_training.py step

# View in TensorBoard
.venv/bin/tensorboard --logdir=./profiles
```

**Look for:**
- Operations taking >10ms
- CPU-GPU transfer overhead
- Inefficient memory layouts
- Unoptimized attention patterns

### 2. Compare Optimizer Performance

```python
optimizers = ['adamw', 'muon', 'multiscale']

for opt_name in optimizers:
    optimizer = create_optimizer(opt_name)

    with profile_trace("./profiles", name=f"optimizer_{opt_name}"):
        loss = train_step(model, optimizer, batch)
        loss.block_until_ready()
```

### 3. Profile Data Loading Pipeline

```python
with profile_trace("./profiles", name="data_pipeline"):
    with profile_scope("fetch_batch"):
        batch = next(data_iterator)

    with profile_scope("to_device"):
        batch = jax.device_put(batch)

    with profile_scope("computation"):
        loss = train_step(model, batch)
        loss.block_until_ready()
```

### 4. Memory Profiling for Large Models

```python
import jax

# Enable memory profiling
jax.config.update('jax_log_compiles', True)

with profile_trace("./profiles", name="memory_analysis"):
    # Force compilation
    _ = model(dummy_input).block_until_ready()

    # Profile actual run
    output = model(real_input)
    output.block_until_ready()
```

## Performance Tips

### Before Profiling

1. **Always warmup** (3-5 iterations) to compile JIT functions
2. **Use representative data** matching your production workload
3. **Profile on target hardware** (same GPU model)
4. **Disable logging/checkpointing** during profiling runs

### During Analysis

1. **Start with Overview page** to identify major bottlenecks
2. **Use Trace Viewer** for detailed timing analysis
3. **Check Op Profile** for expensive operations
4. **Monitor Memory Profile** for OOM issues

### After Profiling

1. **Focus on top 5 expensive operations** (80/20 rule)
2. **Look for unexpected CPU fallbacks**
3. **Identify memory transfer overhead**
4. **Check for suboptimal kernel fusion**

## Troubleshooting

### "No trace data collected"

**Cause**: Operations not blocking properly
**Fix**: Always call `.block_until_ready()` on outputs

```python
# Wrong
with profile_trace("./profiles"):
    x = jax.jit(fn)(data)  # Returns immediately!

# Correct
with profile_trace("./profiles"):
    x = jax.jit(fn)(data)
    x.block_until_ready()  # Wait for computation
```

### "TensorBoard shows empty profile"

**Cause**: Missing `create_perfetto_trace=True` or plugin not installed
**Fix**:

```bash
.venv/bin/pip install tensorboard-plugin-profile
```

Or explicitly enable in code:

```python
with jax.profiler.trace("./profiles", create_perfetto_trace=True):
    ...
```

### "Profile data too large"

**Cause**: Profiling too many iterations
**Fix**: Profile only 1-2 iterations:

```python
# Bad - profiles 100 steps
for _ in range(100):
    with profile_trace("./profiles"):
        train_step(...)

# Good - profiles only 1 step
with profile_trace("./profiles"):
    train_step(...)
```

### WSL: "Permission denied" errors

**Fix**:
```bash
# Ensure profile directory exists and is writable
mkdir -p ~/profiles
chmod 755 ~/profiles
```

## Advanced: Programmatic Analysis

For automated performance regression testing:

```python
from ueaj.utils import benchmark_function

# Benchmark training step
stats = benchmark_function(
    train_step,
    model, batch,
    num_iterations=100,
    warmup=10
)

print(f"Mean: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")

# Assert performance regression
assert stats['mean'] < 150.0, f"Training step too slow: {stats['mean']:.2f}ms"
```

## Environment Variables

Set these for optimal profiling:

```bash
# Enable XLA compilation logging
export JAX_LOG_COMPILES=1

# Use compilation cache
export JAX_COMPILATION_CACHE_DIR=$HOME/tmp/jax_cache

# Reduce memory fragmentation
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Enable detailed profiling
export TF_CPP_MIN_LOG_LEVEL=0
```

## Resources

- [JAX Profiling Docs](https://jax.readthedocs.io/en/latest/profiling.html)
- [TensorBoard Profiler Guide](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
- [XProf Documentation](https://github.com/openxla/xprof)
- [Perfetto UI](https://ui.perfetto.dev)

## Summary

**For most use cases:**
1. Use `profile_trace()` context manager
2. Warmup 3-5 iterations before profiling
3. Profile single iterations for detailed analysis
4. View results in TensorBoard
5. Focus on top bottlenecks first
