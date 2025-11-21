# How to View JAX Profile Traces

This guide shows you exactly how to view and analyze your JAX profile traces.

## Table of Contents
- [TensorBoard Method (Recommended)](#tensorboard-method-recommended)
- [Perfetto UI Method](#perfetto-ui-method)
- [WSL-Specific Instructions](#wsl-specific-instructions)
- [Understanding the Profile Data](#understanding-the-profile-data)

---

## TensorBoard Method (Recommended)

TensorBoard provides the most detailed analysis with interactive UI, HLO operation breakdown, and memory profiling.

### 1. Start TensorBoard

```bash
# From your project directory
.venv/bin/tensorboard --logdir=./profiles

# Or specify a different port
.venv/bin/tensorboard --logdir=./profiles --port=6007

# To allow external access (useful for remote machines)
.venv/bin/tensorboard --logdir=./profiles --host=0.0.0.0 --port=6006
```

You should see output like:
```
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

### 2. Open in Browser

**From WSL:**
- Open your Windows browser (Chrome, Firefox, Edge)
- Navigate to: `http://localhost:6006`
- If localhost doesn't work, try `http://127.0.0.1:6006`

**From remote machine:**
```bash
# On WSL, find your IP
hostname -I
# Example output: 172.23.45.67

# Then from Windows browser:
http://172.23.45.67:6006
```

### 3. Navigate to Profile Tab

Once TensorBoard opens:

1. **Top navigation bar** → Click "PROFILE" (or "PROFILER")
2. **Left sidebar** → You'll see a list of profiling runs
3. **Select a run** → Click on the timestamp (e.g., `2025_11_19_22_37_11`)
4. **Tools dropdown** → Choose which view to use

### 4. TensorBoard Profile Views

#### Overview Page (Start Here!)

**What it shows:**
- Performance summary
- Top TensorFlow operations
- Run environment (GPU, CPU info)
- Step time breakdown

**Key metrics to check:**
- **Device step time**: How long computation takes
- **Host step time**: CPU overhead
- **Device idle time**: GPU waiting (should be <10%)

**Look for:**
- High "Kernel Launch" time → Too many small operations
- High "All Others" time → Non-optimized operations
- High "Input" time → Data loading bottleneck

#### Trace Viewer (Detailed Timeline)

**What it shows:**
- Exact timeline of every operation
- GPU kernel execution
- Memory transfers
- CPU-GPU synchronization

**How to use:**
1. Select "trace_viewer" from Tools dropdown
2. Use W/A/S/D to zoom and pan
3. Click on operations to see details

**What to look for:**
- **GPU utilization**: Should be solid, no gaps
- **Small kernels**: Many <0.1ms operations → poor fusion
- **Host-to-Device transfers**: Should be minimal
- **Idle gaps**: GPU waiting for CPU

**Timeline rows:**
- `/device:GPU:0` - GPU operations
- `/host:CPU` - CPU operations
- `Dataflow` - Data movement

#### Op Profile (Operation Statistics)

**What it shows:**
- Table of all operations sorted by time
- Self-time vs total-time
- Operation type and device placement

**How to use:**
1. Select "op_profile" from Tools dropdown
2. Sort by "Self time" (click column header)
3. Focus on top 5-10 operations

**Key columns:**
- **Operation**: HLO operation name
- **Self Time**: Time spent in this op (ms)
- **Total Time**: Including child ops
- **Device**: GPU/CPU/TPU

**Optimization targets:**
- Operations with >50ms self-time
- CPU operations that should be on GPU
- Repeated small operations

#### Memory Profile

**What it shows:**
- Peak memory usage
- Memory timeline
- Buffer allocations
- Memory bandwidth utilization

**How to use:**
1. Select "memory_profile" from Tools dropdown
2. Look at "Peak Memory Usage" graph
3. Check "Memory Timeline" for allocation patterns

**What to look for:**
- **Peak usage**: Should be <90% of GPU memory
- **Memory spikes**: Sudden allocations (possible OOM risk)
- **Fragmentation**: Many small allocations

#### Kernel Stats

**What it shows:**
- GPU kernel execution statistics
- Occupancy metrics
- Grid/block dimensions

**How to use:**
1. Select "kernel_stats" from Tools dropdown
2. Sort by "Duration" or "Occurrences"
3. Look for optimization opportunities

**Key metrics:**
- **Occupancy**: Should be >50% for compute-bound kernels
- **Duration**: Long kernels might need optimization
- **Occurrences**: High count might indicate loop unrolling issues

---

## Perfetto UI Method

Perfetto is a lightweight web-based trace viewer. Best for quick checks and sharing traces.

### 1. Find Your Trace Files

After profiling, traces are saved to:
```
profiles/
└── step_00000100/
    └── plugins/
        └── profile/
            └── 2025_11_19_22_37_11/
                ├── perfetto_trace.json.gz    <-- This one!
                └── ueajinator.trace.json.gz  <-- Or this one!
```

### 2. Open Perfetto UI

Go to: **https://ui.perfetto.dev**

### 3. Load Your Trace

**Method 1: Drag and Drop**
- Drag `perfetto_trace.json.gz` onto the Perfetto UI page
- No upload needed - everything stays local!

**Method 2: Click to Upload**
1. Click "Open trace file" on the Perfetto UI
2. Navigate to your trace file
3. Select and open

**Method 3: From WSL (if you have WSL-Windows filesystem access)**
```bash
# Copy trace to Windows desktop
cp profiles/step_*/plugins/profile/*/perfetto_trace.json.gz /mnt/c/Users/devse/Desktop/

# Then drag from Windows Explorer to browser
```

### 4. Navigate Perfetto UI

**Keyboard shortcuts:**
- `W/A/S/D` - Pan and zoom
- `M` - Mark area
- `F` - Zoom to selection
- `?` - Show all shortcuts

**Timeline sections:**
- Top rows: CPU threads
- Middle rows: GPU streams
- Bottom rows: Memory and other events

**Click on events to see:**
- Duration
- Arguments
- Call stack (if available)

### 5. Share Traces

Perfetto creates shareable links:

1. Click **"Share"** button (top right)
2. Choose upload option
3. Get permanent link to share with teammates

**Note:** This uploads your trace to Perfetto's servers. Don't share traces with proprietary code!

---

## WSL-Specific Instructions

### Issue: TensorBoard not accessible from Windows

**Solution 1: Check port forwarding**
```bash
# WSL2 should auto-forward ports, but check:
netstat -an | grep 6006

# Should show:
# tcp        0      0 0.0.0.0:6006      0.0.0.0:*     LISTEN
```

**Solution 2: Use WSL IP address**
```bash
# In WSL, get your IP
hostname -I
# Example: 172.23.45.67

# In Windows browser:
http://172.23.45.67:6006
```

**Solution 3: Windows port forwarding (if needed)**
```powershell
# In PowerShell (as Administrator)
netsh interface portproxy add v4tov4 listenport=6006 listenaddress=0.0.0.0 connectport=6006 connectaddress=172.23.45.67

# Replace 172.23.45.67 with your WSL IP from hostname -I
```

### Issue: Slow profile loading from /mnt/c/

**Solution:** Always save profiles to Linux filesystem
```bash
# ✅ Good - Linux filesystem
PROFILE_DIR=~/profiles

# ❌ Bad - Windows filesystem (slow I/O)
PROFILE_DIR=/mnt/c/Users/devse/profiles
```

### Issue: Permission denied when writing profiles

**Solution:**
```bash
# Create profile directory with correct permissions
mkdir -p ~/profiles
chmod 755 ~/profiles

# Or use current user's home
PROFILE_DIR=$HOME/profiles
```

---

## Understanding the Profile Data

### What the Trace Contains

A JAX profile includes:

1. **Python-level trace**: Your Python function calls
2. **JAX transformations**: jit, grad, vmap, etc.
3. **XLA HLO operations**: Compiled operations
4. **GPU kernels**: Actual CUDA/ROCm kernel launches
5. **Memory operations**: Allocations, transfers, deallocations

### Profile Structure

```
Step
├── Python execution (host)
│   ├── Data loading
│   ├── Preprocessing
│   └── JAX dispatch
├── XLA compilation (first run only)
│   ├── HLO optimization passes
│   └── Kernel code generation
└── Device execution (GPU/TPU)
    ├── Kernel launches
    ├── Memory transfers
    └── Synchronization
```

### Key Performance Indicators

| Metric | Good | Warning | Bad |
|--------|------|---------|-----|
| GPU Utilization | >80% | 50-80% | <50% |
| Kernel Duration | >1ms | 0.1-1ms | <0.1ms |
| Host Overhead | <10% | 10-20% | >20% |
| Memory Bandwidth | >60% | 40-60% | <40% |
| Compilation Time | <2s | 2-10s | >10s |

### Common Patterns and What They Mean

#### Pattern: Many Small Kernels (<0.1ms each)

**Cause:** XLA didn't fuse operations
**Fix:**
- Use jax.jit on larger functions
- Enable XLA optimization flags
- Reduce control flow branches

#### Pattern: GPU Idle Gaps

**Cause:** CPU bottleneck or synchronization
**Fix:**
- Reduce block_until_ready() calls
- Prefetch data to device
- Use asynchronous dispatch

#### Pattern: Large Memory Transfers

**Cause:** Data on wrong device or repeated transfers
**Fix:**
- Use jax.device_put() to pre-load data
- Keep intermediate results on device
- Avoid numpy() conversions in loop

#### Pattern: One Operation Dominates (>50% time)

**Cause:** Unoptimized operation or algorithm
**Fix:**
- Profile that operation separately
- Consider algorithm changes
- Use specialized libraries (cuBLAS, cuDNN)

---

## Quick Profiling Workflow

**1. Profile your code**
```bash
PROFILE_ENABLED=1 PROFILE_START_STEP=100 .venv/bin/python scripts/train_with_profiling.py
```

**2. Start TensorBoard**
```bash
.venv/bin/tensorboard --logdir=./profiles
```

**3. Open browser**
```
http://localhost:6006
```

**4. Analysis checklist**
- [ ] Check Overview → Device step time
- [ ] Trace Viewer → GPU utilization
- [ ] Op Profile → Sort by self-time, check top 5 ops
- [ ] Memory Profile → Check peak usage
- [ ] Kernel Stats → Look for low occupancy

**5. Identify bottleneck**
- Find operations >50ms
- Check for CPU fallback
- Look for memory transfer overhead

**6. Optimize and re-profile**
```bash
# Compare before/after
.venv/bin/tensorboard --logdir=./profiles
```

---

## Example: Complete Profile Analysis Session

```bash
# Step 1: Run training with profiling
PROFILE_ENABLED=1 PROFILE_START_STEP=100 PROFILE_INTERVAL=0 \
  .venv/bin/python scripts/train_with_profiling.py

# Output:
# ============================================================
# 📊 PROFILING ENABLED
# ============================================================
#   Profile Dir:    ./profiles
#   Start Step:     100
#   Interval:       once
#   Duration:       1 step(s)
#   Mode:           tensorboard
# ============================================================
# ...
# 📊 Profiling step 100...
# ✓ Profile saved: ./profiles/step_00000100

# Step 2: Start TensorBoard
.venv/bin/tensorboard --logdir=./profiles

# Output:
# TensorBoard 2.20.0 at http://localhost:6006/

# Step 3: Open browser
# Navigate to: http://localhost:6006
# Click: PROFILE tab
# Select: step_00000100 run

# Step 4: Analyze
# Overview → Device step time: 145ms ✓ (good)
# Trace Viewer → GPU utilization: 85% ✓ (good)
# Op Profile → Top op: dot_general (80ms) ⚠️ (check if expected)
# Memory Profile → Peak: 3.2GB / 8GB ✓ (good)

# Step 5: If you find issues, optimize and re-run
PROFILE_ENABLED=1 PROFILE_START_STEP=100 \
  .venv/bin/python scripts/train_with_profiling.py

# Step 6: Compare
# TensorBoard will show both runs - compare side-by-side
```

---

## Troubleshooting

### "No profile data found"

**Causes:**
- Forgot `.block_until_ready()`
- Profile directory doesn't exist
- TensorBoard looking at wrong directory

**Fix:**
```python
# Always block on outputs
with maybe_profile(step):
    result = train_step(...)
    result.block_until_ready()  # <-- Required!
```

### "Empty trace in TensorBoard"

**Cause:** Plugin not installed
**Fix:**
```bash
.venv/bin/pip install tensorboard-plugin-profile
# Restart TensorBoard
```

### "Trace too large to load"

**Cause:** Profiled too many steps
**Fix:**
```bash
# Profile only 1-2 steps
PROFILE_DURATION=1 PROFILE_ENABLED=1 ...
```

### "Can't open .json.gz file"

**Cause:** File corruption or wrong format
**Fix:**
```bash
# Check file exists and has size
ls -lh profiles/step_*/plugins/profile/*/*.json.gz

# Should show files >1KB
# If 0 bytes, re-run profiling
```

---

## Summary

**TensorBoard (Detailed Analysis):**
```bash
.venv/bin/tensorboard --logdir=./profiles
# Open: http://localhost:6006
```

**Perfetto (Quick View):**
```bash
# Upload trace to: https://ui.perfetto.dev
profiles/step_*/plugins/profile/*/perfetto_trace.json.gz
```

**Environment Variable Profiling:**
```bash
PROFILE_ENABLED=1 PROFILE_START_STEP=100 your_script.py
```

**Key Files:**
- `perfetto_trace.json.gz` - Perfetto format
- `*.xplane.pb` - TensorBoard format (auto-loaded)
