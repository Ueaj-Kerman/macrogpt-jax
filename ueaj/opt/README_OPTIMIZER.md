# Optimizer Configuration System

This directory contains the complete optimizer configuration system for the NanoLLaMA project.

## Core Files

### 1. `optimizer_config.py`
Main implementation of the optimizer configuration system. Provides:
- List access on trees: `opt[['mlp', 'attn']] = optimizer`
- List access on tensors: `opt['param', [1, 2, 3]] = optimizer`
- Efficient tensor slicing: `opt['param', :8k] = opt1`
- Shared optimizer instances with single init/update calls
- Complex nested patterns with tensor indexing

### 2. `optimizer_state.py`
Contains helper classes and functions:
- `TensorRegion`: Represents a region of a tensor with an optimizer
- `TensorSplitter`: Manages non-overlapping regions for tensors
- `map_state`: Transforms state to/from optimizer format for Ueajsum modules

### 3. `index_set.py`
Efficient index set representation for tensor slicing:
- `IndexSet`: Represents sets of indices with automatic optimization
- Converts between slices, lists, and integers
- Supports set operations (union, intersection, subtraction)

## Usage Example

```python
from ueaj.opt import OptimizerConfig

# Create configuration
model = create_model()
config = OptimizerConfig(model)

# List access on trees
config[['mlp', 'attn']] = optax.adam(1e-3)

# Tensor slicing
config['embeddings', :1000] = optax.adamw(2e-3)
config['embeddings', 1000:] = optax.sgd(1e-2)

# Create optimizer
optimizer = config.create_optimizer()

# Use in training
params = nnx.state(model, nnx.Param)
opt_state = optimizer.init(params)
```

## Tests

See `test/test_optimizer_config.py` for comprehensive tests of all features.