# CLAUDE.md

## Project Overview

**MacroGPT-JAX** - Configurable distributed pretraining framework on JAX/Flax for optimizer/precision research. Features custom transformers with HuggingFace LLaMA support.

## Running Commands

```bash
.venv/bin/python script.py           # Run script
.venv/bin/python -m pytest test/ -q  # Run tests
```

## Key Components

- **`ueaj/model/`** - Transformer components (attention, MLP, RMSNorm, RoPE, einsum)
- **`ueaj/model/ttt/`** - Test-Time Training layer with multi-query support (`q_heads` param)
- **`ueaj/llama/`** - HuggingFace weight loading, PEFT/LoRA compatibility
- **`ueaj/opt/`** - Optimizers (muon, multiscale) and loss functions
- **`ueaj/train/`** - Training loop, logging, LoRA utilities
- **`ueaj/data/`** - Dataset prep, document packing (`use_packing` toggle), batching
- **`ueaj/utils/`** - `@config` decorator, `compile_function`, gradient utilities

## Patterns

```python
# Configuration via @config decorator
from ueaj.utils.configurator import config

@config
class MyModule(nnx.Module):
    def __init__(self, model_d: int, rngs: rng.Rngs, **kwargs): ...
# Use: MyModule.override(param=value)

# Einsum weight format: (...batch_dims, reducing_dims, non_reducing_dims)
# Expression syntax: "bnd,dh->bnh"
```

## Environment

- **Virtual environment**: `.venv`
- **JAX cache**: `JAX_COMPILATION_CACHE_DIR=$HOME/tmp/jax_cache`
- Training env vars: `OPTIMIZER`, `RUN_NAME`, `MODEL_PATH`, `BASE_LR`
