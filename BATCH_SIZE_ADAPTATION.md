# Batch Size Adaptation in Distributed Data Loading

## The Problem

When using a 2D mesh (e.g., 2×2 for data + tensor parallelism), different hosts may have different numbers of **data-loading devices** (devices at tensor=0). The batch size must adapt accordingly.

## Example: 2×2 Mesh (4 devices, 2 hosts)

### Setup
```
Mesh: (data=2, tensor=2) = 4 devices total
batch_size = 4 (per data-parallel shard)
```

### Device Layout (typical arrangement)

```
Host 0:                      Host 1:
Device (data=0, tensor=0)    Device (data=0, tensor=1)
Device (data=1, tensor=0)    Device (data=1, tensor=1)
```

### Without Batch Size Adaptation (WRONG ❌)

**Host 0:**
- Has 2 devices at tensor=0
- Needs data for **2 data shards**
- Would load only `batch_size=4` examples
- ❌ **Problem:** Not enough data! Each device gets 2 examples instead of 4

**Host 1:**
- Has 0 devices at tensor=0
- Creates dummy zeros
- ✓ Correct (but doesn't matter if Host 0 is broken)

### With Batch Size Adaptation (CORRECT ✅)

**Host 0:**
```python
should_load, local_shards = compute_batch_size(mesh)
# Returns: (True, 2)

local_batch_size = batch_size * local_shards
                 = 4 × 2
                 = 8 examples
```
- ✅ Loads 8 examples total
- ✅ 4 examples for (data=0, tensor=0)
- ✅ 4 examples for (data=1, tensor=0)

**Host 1:**
```python
should_load, local_shards = compute_batch_size(mesh)
# Returns: (False, 0)

local_batch_size = batch_size * local_shards
                 = 4 × 0
                 = 0 examples
```
- ✅ Creates dummy zeros
- ✅ Receives real data via broadcast

### Final Result

**Global array shape:** `(8, 128)`
- First 4 examples: data shard 0
- Last 4 examples: data shard 1

**Sharding:** `P('data', None)` - replicated across tensor dimension

## Implementation

### Key Code Changes

**Before (WRONG):**
```python
should_load, _ = compute_batch_size(mesh)  # Throwing away local_shards!

if should_load:
    dataset = batch_iterator(dataset, batch_size=batch_size)  # Always 4
```

**After (CORRECT):**
```python
should_load, local_shards = compute_batch_size(mesh)
local_batch_size = batch_size * local_shards  # Adapts per host!

if should_load:
    dataset = batch_iterator(dataset, batch_size=local_batch_size)  # 8 for Host 0
```

### Global Structure
```python
data_axis_size = mesh.shape[data_axis]  # 2
global_batch_size = batch_size * data_axis_size  # 4 × 2 = 8

tokens_struct = jax.ShapeDtypeStruct((global_batch_size, seq_len), jnp.int32)
# Shape: (8, 128)
```

## Different Mesh Configurations

### 1×2 Mesh (1 data × 2 tensor)
```
Host 0: (data=0, tensor=0) → local_shards=1 → load 4 examples
Host 1: (data=0, tensor=1) → local_shards=0 → create zeros

Global shape: (4, 128)
I/O savings: 2× reduction
```

### 2×1 Mesh (2 data × 1 tensor)
```
Host 0: (data=0, tensor=0) → local_shards=1 → load 4 examples
Host 1: (data=1, tensor=0) → local_shards=1 → load 4 examples

Global shape: (8, 128)
I/O savings: None (both hosts load, but different data)
```

### 2×2 Mesh (2 data × 2 tensor)
```
Host 0: (data=0, tensor=0), (data=1, tensor=0) → local_shards=2 → load 8 examples
Host 1: (data=0, tensor=1), (data=1, tensor=1) → local_shards=0 → create zeros

Global shape: (8, 128)
I/O savings: 2× reduction (only Host 0 loads)
```

### 4×4 Mesh (4 data × 4 tensor)
```
Hypothetical 4-host setup where each host has 4 devices:

Host 0: All at tensor=0 → local_shards=4 → load 16 examples
Host 1: All at tensor=1 → local_shards=0 → create zeros
Host 2: All at tensor=2 → local_shards=0 → create zeros
Host 3: All at tensor=3 → local_shards=0 → create zeros

Global shape: (16, 128)
I/O savings: 4× reduction!
```

## Testing

### Test 1×2 mesh:
```bash
bash examples/launch_1x2_test.sh
```

Expected output:
```
Host 0: Loading data from dataset (local_shards=1, local_batch_size=4)
Host 1: Creating dummy data iterator (will receive via broadcast)

✓ Host 0 loaded 4 examples
○ Host 1 created zeros, received via broadcast
```

### Verify batch size adaptation:
```bash
# Check the logs for local_batch_size
grep "local_batch_size" /tmp/test_1x2_host*.log
```

## Key Takeaways

`★ Insight ─────────────────────────────────────`
**Why Batch Size Adaptation Matters**

1. **Correctness**: Without adaptation, hosts with multiple data loaders would under-load, causing shape mismatches or missing data.

2. **Efficiency**: `compute_batch_size()` returns both whether to load AND how many shards, enabling proper per-host batching.

3. **Scalability**: For large tensor-parallel factors (8-way, 16-way), only 1/N hosts load data, but that host must load N× the base batch size.

4. **Automatic**: The implementation handles all mesh configurations automatically - users just specify `batch_size` per shard, and the system adapts.
`─────────────────────────────────────────────────`

## Related Files

- `ueaj/data/dataset.py:133-143` - Batch size calculation
- `ueaj/utils/distutil.py:116-143` - `compute_batch_size()` implementation
- `examples/test_1x2_mesh.py` - Working example
- `examples/demo_2x2_mesh.py` - Conceptual demonstration
