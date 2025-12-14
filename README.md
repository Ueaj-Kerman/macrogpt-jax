# MacroGPT-JAX

A very opinionated distributed pretraining framework built on JAX/Flax, designed for OOD/creative research. 

## Branch Information
This branch implements a new surrogate gradient method for test time training. 
It can be found in `~/ueaj/model/ttt/impl.py`. The way it works is by accumulating the state deltas across the sequence
in one forward pass, and then doing a second forward pass, subtracting the recomputed local delta from the overall sum.
This is an approximation and has tradeoffs.

### Tradeoff
The fundamental tradeoff of the new method is that it does not teach the network multi-step composition.
In other words it does not learn to, given future changes to the state, how to adjust the current state to meet a future target state.
It does however learn single degree composition, that is, given a future state, and it's target how can I change my current state to get closer?
The transformer on the other hand, can do neither.

### TODO
 - [ ] Local blockwise pure backprop
   - During the first forward in the bwd pass, do true backprop to get the state delta for each *block*
   - For hidden dimension 768 the hidden state consumes 28,311,552 bytes. 
     - Targeting a total checkpointing volume of 1GB allows us to make 37 total checkpoints.
     - A block size of 32 is a good target
   - For a hidden dimension of 512 the hidden state consumes 12,582,912 bytes.
     - A block size of 64 is a good target
 - [ ] Integration with pararnn (double bloccy!! (I'm killing myself))
## Key Features

- **Highly Configurable**: Hierarchical configuration system using `@config` decorator for flexible model composition
- **Per-Parameter Optimizers**: Assign different optimizer instances per layer/parameter type
- **Mixed Precision**: FP8/FP16 support with custom precision control and gradient casting
- **WANDB Integration**: Comprehensive logging and experiment tracking

## Installation

### Prerequisites

- Python 3.10+
- JAX with GPU support (or CPU for testing)
- Virtual environment at `~/venvs/jax-packages` (or modify `scripts/run_python.sh`)

### Setup

```bash
# Clone the repository
git clone https://github.com/Ueaj-Kerman/macrogpt-jax.git
cd macrogpt-jax

# Create and activate virtual environment
python -m venv ~/venvs/jax-packages
source ~/venvs/jax-packages/bin/activate

# Install dependencies
pip install jax[cuda12] flax optax  # For GPU
# OR
pip install jax flax optax  # For CPU

# Install additional dependencies
pip install transformers tokenizers wandb pytest huggingface_hub

# Verify installation
./scripts/run_python.sh -c "from ueaj.llama import load_llama_from_hf; print('✓ Installation successful')"
```

## Quick Start

### Training from Scratch

```bash
# Launch distributed pretraining with Muon optimizer
OPTIMIZER=muon RUN_NAME=my_experiment ./scripts/run_python.sh -m ueaj.train.train

# Train with Multiscale optimizer and custom learning rate
OPTIMIZER=multiscale RUN_NAME=exp_001 BASE_LR=0.025 ./scripts/run_python.sh -m ueaj.train.train

# Available optimizers: multiscale, muon, adamw
```

## Architecture Overview

### Model Components

- **TransformerLayer**: Combines attention and MLP with residual connections
- **SoftmaxAttention**: Flash attention with RoPE, mixed precision support
- **GMLP**: Gated MLP with LeCun initialization
- **RMSNorm**: Configurable normalization (centered/uncentered, scalar/none)
- **Einsum**: Simplified 2-argument einsum with optimizer canonicalization

### LoRA Integration

```python
from ueaj.model import configs, apply_lora_to_model
from ueaj.train import make_lora_optimizer, print_lora_info
from ueaj.llama import save_lora_to_peft
from flax.nnx import rnglib as rng
from flax import nnx

# Create base model
model = configs.UEAJ_150M(rngs=rng.Rngs(0))

# Apply LoRA (default: all modules except lm_head)
model = apply_lora_to_model(
    model,
    rank=16,
    alpha=32,
    target_modules=['q', 'k', 'v', 'o'],  # Or None for all
    rngs=rng.Rngs(42)
)
print_lora_info(model)  # ~2-5% of total params are trainable

# Extract LoRA parameters for training
lora_state = nnx.state(model, nnx.LoRAParam)

# Train only LoRA parameters (base frozen automatically)
# ... training loop ...

# Save adapter in PEFT format (vLLM compatible)
save_lora_to_peft(model, "./my_lora_adapter")
```

## Advanced Features

### Per-Parameter Optimizers
```python
from ueaj.opt import OptimizerConfig
import optax

# Different optimizers for different layers
config = OptimizerConfig(model=...)
config['layers', 'attn'] = optax.adam(1e-3)
config['layers', 'mlp'] = optax.lion(1e-4)
config['embed'] = optax.sgd(1e-2)
```

### Configuration System

```python
from ueaj.utils.configurator import config
from flax import nnx

@config
class MyModule(nnx.Module):
    def __init__(self, model_d: int, hidden_d: int, rngs, **kwargs):
        self.linear = nnx.Linear(model_d, hidden_d, rngs=rngs)

# Create configured versions
MyLargeModule = MyModule.override(hidden_d=4096)
MySmallModule = MyModule.override(hidden_d=512)

# Instantiate
model = MyLargeModule(model_d=1024, rngs=rngs)
```

### Memory and Performance Analysis

```python
from ueaj.utils.compile import compile_function

compiled_fn = compile_function(
    my_function,
    sample_args=(args,),
    sample_kwargs={'key': value},
    name="Training Step"
)
# Outputs: memory usage, FLOPs, compilation time
```


## Project Structure

```
macrogpt-jax/
├── scripts/              # Executable scripts
│   ├── run_python.sh     # Python execution wrapper (always use this)
│   ├── train_lora.py     # LoRA fine-tuning
│   ├── sample_llama.py   # Text generation
│   └── sweep.sh          # Hyperparameter sweeping
├── ueaj/                 # Main package
│   ├── data/             # Data loading and preprocessing
│   ├── model/            # Transformer architecture
│   ├── llama/            # LLaMA loading and PEFT utilities
│   ├── opt/              # Custom optimizers
│   ├── train/            # Training loop and logging
│   └── utils/            # Configuration and utilities
├── test/                 # Test suite (pytest)
├── CLAUDE.md             # Developer guide for Claude Code
└── README.md             # This file
```

## Environment Variables

### Training Configuration
- `OPTIMIZER`: Optimizer choice (`multiscale`, `muon`, `adamw`)
- `RUN_NAME`: WANDB run name for logging
- `MODEL_PATH`: Directory for saving checkpoints (default: `./checkpoints`)
- `BASE_LR`: Base learning rate (default: `0.025`)

### JAX Configuration
- `JAX_COMPILATION_CACHE_DIR`: Compilation cache location (recommended: `$HOME/tmp/jax_cache`)
- `XLA_PYTHON_CLIENT_MEM_FRACTION`: GPU memory fraction (default: `0.95`)
- `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS`: Enable for kvax support (set to `1`)

## Contributing

Check out https://github.com/Ueaj-Kerman/macrogpt-jax/issues

## Citation

If you use this code in your research, please cite:

```bibtex
@software{macrogpt_jax,
  author = {Ueaj Kerman},
  title = {MacroGPT-JAX: A Configurable Distributed Pretraining Framework},
  year = {2025},
  url = {https://github.com/Ueaj-Kerman/macrogpt-jax}
}
```
