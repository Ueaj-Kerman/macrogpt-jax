# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MacroGPT-JAX** is a highly configurable distributed pretraining framework built on JAX/Flax. The project prioritizes configurability, resilience, and decent performance (~10% of modded-nanogpt) to serve as a foundation for research in optimizers, numerical precision, and reinforcement learning. It features custom transformer implementations with support for loading pre-trained LLaMA models from HuggingFace.

## Key Architecture Components

### Einsum System (Simplified from UeajSum)
- **Core Module**: `ueaj/model/einsum.py` - Simple einsum layer with 2-argument support
- **Weight Format**: `(...batch_dims, reducing_dims, non_reducing_dims)`
- **Expression Syntax**: Standard einsum notation (e.g., "bnd,dh->bnh")
- **Configuration**: Uses `@config` decorator from `ueaj/utils/configurator.py` for flexible configuration

### Model Architecture

#### Core Components (`ueaj/model/`)
- **layer.py**: TransformerLayer combining attention and MLP with residual connections
- **soft_attn.py**: SoftmaxAttention with kvax/flash attention support, RoPE, and mixed precision
- **mlp.py**: MLP and GMLP (gated MLP) implementations with LeCun initialization
- **rmsnorm.py**: RMSNorm with configurable scaling modes (centered, uncentered, scalar, none)
- **rope.py**: Rotary Position Embeddings supporting various tensor shapes
- **einsum.py**: Simplified 2-argument einsum implementation with EinsumMetadata for optimizer canonicalization
- **model.py**: LlamaModel implementation with HuggingFace compatibility
- **configs.py**: Model configurations (UEAJ_NH, UEAJ_150M, UEAJ_1B)
- **nn.py**: Custom activation functions (relu_squared, leaky_relu_squared, signed_sqrt)
- **lora.py**: LoRA (Low-Rank Adaptation) implementation for parameter-efficient fine-tuning
  - `LoRAEinsum`: Wraps Einsum layers with low-rank adaptation
  - `LoRAEmbed`: LoRA for embedding layers with attend() for tied weights
  - `apply_lora_to_model()`: Model surgery to add LoRA to specific modules
  - Use `nnx.state(model, nnx.LoRAParam)` to extract LoRA params, `nnx.update()` to merge

#### LLaMA Integration (`ueaj/llama/`)
- **llama_loader.py**: High-level loading of LLaMA models from HuggingFace
  - `load_llama_from_hf()`: Load pretrained LLaMA with optional LoRA
- **weight_loader.py**: Weight mapping and PEFT compatibility
  - `WeightMapper`: Maps HuggingFace weight names to model structure
  - `from_pretrained()`: Load model weights from HuggingFace
  - `save_lora_to_peft()`: Export LoRA to HuggingFace PEFT format (vLLM compatible)
  - `load_lora_from_peft()`: Import existing PEFT adapters as nnx.State

#### Utility Components (`ueaj/utils/`)
- **__init__.py**: Contains configuration values (LOW_PRECISION, DEFAULT_ACCUM_TYPE) and utility functions
- **gradutils.py**: Gradient utilities including `custom_astype` for flexible dtype casting
- **tensorutil.py**: Tensor operations like `chunked_scan` and `promote_fp8`
- **configurator.py**: `@config` decorator for configuration management
- **compile.py**: `compile_function` for JAX compilation with detailed memory/FLOP analysis
- **kvax_context.py**: Context manager for kvax operations
- **distutil.py**: Distributed utilities for JAX mesh operations
  - `MeshSlice`: NumPy-style slicing for JAX device meshes
  - `slice(mesh)`: Create a mesh slicer wrapper
  - `this_host_has_first()`: Check if current host has first device on an axis
  - `block_allocations()`: Compute block allocations for distributed training
  - `blockify()` / `deblockify()`: Pack/unpack parameters into blocked tensor arrays
  - `shard()`: Apply sharding constraints to blocked arrays

### Training and Optimization (`ueaj/train/`, `ueaj/opt/`)

#### Training (`ueaj/train/`)
- **train.py**: Main training loop with WANDB logging and checkpointing
- **optimizer_setup.py**: Creates optimizer configurations (multiscale, muon, adamw) with per-layer learning rates
- **training_utils.py**: Training utilities and helper functions
- **logging_utils.py**: Logging utilities for WANDB statistics
- **grad_transforms.py**: Gradient transformations and clipping utilities
- **lora_utils.py**: LoRA-specific training utilities
  - `make_lora_optimizer()`: Simple AdamW optimizer (base params frozen by differentiating w.r.t. LoRA state only)
  - `get_lora_param_count()`, `print_lora_info()`: Diagnostics

#### Optimizers (`ueaj/opt/`)
- **multiscale.py**: Advanced optimizers (muon, multiscale_muon) with Newton-Schulz orthogonalization
- **next_token_loss.py**: Loss functions for language modeling
- **canonicalize.py**: Einsum parameter canonicalization for optimizer chains (ensures consistent weight format)
- **__init__.py**: OptimizerConfig for per-parameter optimizer assignment

### Data Processing (`ueaj/data/`)
- **dataset.py**: Dataset preparation and tokenization
- **packing.py**: Document packing for efficient training
- **batching.py**: Batch iterators and collation functions
- **prefetch.py**: Device prefetching for performance

### Testing Infrastructure (`test/`)

The test suite focuses on optimizer research and numerical precision:

#### Optimizer Tests
- **test_unroll_*.py**: Tests for scan unrolling optimization (simple, practical, real-world scenarios)
- **test_optimizer_*.py**: Optimizer canonicalization and inlining tests
- **test_scan_unroll_optimization.py**: Scan operator optimization tests
- **test_weight_decay_clean.py**: Weight decay behavior verification
- **test_guaranteed_weight_decay.py**: Guaranteed weight decay implementation

#### Precision Tests
- **test_fp8_exhaustive.py**: Exhaustive FP8 precision testing
- **test_fp16_exhaustive.py**: Exhaustive FP16 precision testing
- **test_nextafter_*.py**: Edge cases and precision tests for nextafter operation

#### Other Tests
- **test_distutil.py**: Distributed utility tests

## Common Commands

**IMPORTANT**: Always use the `scripts/run_python.sh` script to run Python commands to ensure the correct virtual environment is used.

### Running Python Scripts
```bash
./scripts/run_python.sh script.py                     # Run a Python script
./scripts/run_python.sh -m module                     # Run a Python module
./scripts/run_python.sh -c "print('Hello')"          # Run Python command
```

### Testing
```bash
# Run all tests with pytest
./scripts/run_python.sh -m pytest test/ -q

# Run specific test category
./scripts/run_python.sh -m pytest test/ -k unroll
./scripts/run_python.sh -m pytest test/ -k fp8
./scripts/run_python.sh -m pytest test/ -k weight_decay

# Run a single test file
./scripts/run_python.sh -m pytest test/test_unroll_simple.py -v

# Quick smoke test for config system
./scripts/run_python.sh test.py
```

### Training
```bash
# Launch distributed pretraining with environment variables
OPTIMIZER=multiscale RUN_NAME=my_run ./scripts/run_python.sh -m ueaj.train.train

# Set model path and base learning rate
OPTIMIZER=muon RUN_NAME=exp_001 MODEL_PATH=./checkpoints BASE_LR=0.025 ./scripts/run_python.sh -m ueaj.train.train

# Available optimizers: multiscale, muon, adamw
```

### LoRA Fine-tuning
```bash
# Basic LoRA training (uses UEAJ_150M by default)
./scripts/run_python.sh scripts/train_lora.py --run-name my_lora

# LoRA training with customization
./scripts/run_python.sh scripts/train_lora.py \
    --rank 32 --alpha 64 --lr 1e-4 \
    --max-steps 5000 --output-dir ./my_adapter

# Load pretrained model and apply LoRA (when HF loading is implemented)
./scripts/run_python.sh scripts/train_lora.py \
    --model meta-llama/Llama-3.2-1B \
    --target-modules q k v o \
    --rank 16 --alpha 32
```

### Text Generation
```bash
# Sample from pretrained model
./scripts/run_python.sh scripts/sample_llama.py \
    --prompt "Once upon a time" --max-new-tokens 100

# With temperature and sampling parameters
./scripts/run_python.sh scripts/sample_llama.py \
    --prompt "The future of AI" --temperature 0.8 \
    --top-k 50 --top-p 0.9
```

## Key Development Patterns

### Configuration with @config Decorator
```python
from ueaj.utils.configurator import config

@config
class MyModule(nnx.Module):
    def __init__(self, model_d: int, rngs: rng.Rngs, **kwargs):
        # Use MyModule.override(param=value) to create configured versions
```

### Import Pattern
The codebase uses wildcard imports for extensibility:
```python
from ueaj.model import *
from ueaj.utils import *
```

### Mixed Precision with custom_astype
```python
from ueaj.utils import custom_astype

# Forward cast only (old astype_fwd_noop_bwd)
x = custom_astype(x, dtype, cast_forward=True, cast_backward=False)

# Backward cast only (old noop_fwd_astype_bwd)
x = custom_astype(x, dtype, cast_forward=False, cast_backward=True)
```

### Memory Analysis with compile_function
```python
from ueaj.utils.compile import compile_function

compiled_fn = compile_function(
    func,
    sample_args=(args,),
    sample_kwargs={'key': value},
    name="Function Name"
)
# Provides detailed memory usage, FLOPs, and compilation time
```

### LoRA Fine-tuning Pattern
```python
from ueaj.model import configs, apply_lora_to_model
from ueaj.train import make_lora_optimizer, print_lora_info
from ueaj.llama import save_lora_to_peft, load_lora_from_peft
from flax import nnx
from flax.nnx import rnglib as rng
import jax
import optax

# Create or load model
model = configs.UEAJ_150M(rngs=rng.Rngs(0))

# Apply LoRA (default: all modules except lm_head)
model = apply_lora_to_model(
    model,
    rank=16,
    alpha=32,
    target_modules=['q', 'k', 'v', 'o'],  # Or None for all
    rngs=rng.Rngs(42)
)
print_lora_info(model)  # Show trainable param count (~2-5% of total)

# Extract LoRA state for training
lora_state = nnx.state(model, nnx.LoRAParam)

# Setup optimizer (no masking needed - we differentiate w.r.t. LoRA state only)
optimizer = make_lora_optimizer(lr=1e-4)
opt_state = optimizer.init(lora_state)

# Training step
def loss_fn(lora_state):
    nnx.update(model, lora_state)  # Update model with current LoRA params
    logits = model(inputs)
    return compute_loss(logits, targets)

# Compute gradients w.r.t. LoRA state only (base params automatically frozen)
grads = jax.grad(loss_fn)(lora_state)

# Update LoRA params
updates, opt_state = optimizer.update(grads, opt_state, lora_state)
lora_state = optax.apply_updates(lora_state, updates)
nnx.update(model, lora_state)

# Save adapter in PEFT format (vLLM compatible)
save_lora_to_peft(model, "./my_lora_adapter")

# Later: load adapter back and merge
lora_state = load_lora_from_peft("./my_lora_adapter")
nnx.update(model, lora_state)
```

### Mesh Slicing with MeshSlice
```python
from ueaj.utils.distutil import slice as mesh_slice
import jax

# Create a 2D mesh (e.g., data parallel × model parallel)
devices = jax.devices()
mesh = jax.sharding.Mesh(
    devices.reshape(4, 2),
    ('data', 'model')
)

# Positional slicing (like NumPy array slicing)
# Get first 2 data parallel slices, all model parallel devices
sub_mesh = mesh_slice(mesh)[0:2, :]
# Result: Mesh with shape (data=2, model=2)

# Index with integer to remove an axis
sub_mesh = mesh_slice(mesh)[:, 0]
# Result: Mesh with shape (data=4,) - 'model' axis collapsed

# Dict-based slicing (more explicit, order-independent)
# Specify slices by axis name
sub_mesh = mesh_slice(mesh)[{'data': slice(0, 2), 'model': 0}]
# Result: Mesh with shape (data=2,) - 'model' axis collapsed

# Dict with step (every 2nd device)
sub_mesh = mesh_slice(mesh)[{'tensor': slice(None, None, 2)}]
# Unspecified axes default to slice(None) (select all)
```

**Key behaviors** (ueaj/utils/distutil.py:29-79):
- **Two syntaxes**: Positional `[0:2, :]` or dict-based `[{'data': slice(0, 2)}]`
- **Axis preservation**: Slice objects preserve the axis; integers remove it
- **Dict advantages**: Order-independent, explicit axis names, fewer errors
- **Auto-padding**: Missing dimensions/axes automatically default to `slice(None)`
- **Returns new Mesh**: Slices the underlying device array and creates fresh mesh

## Project Structure

```
nanollama/
├── scripts/              # Executable scripts
│   ├── run_python.sh     # Python execution wrapper
│   ├── train_lora.py     # LoRA fine-tuning script
│   ├── sample_llama.py   # Text generation script
│   ├── sweep.sh          # Hyperparameter sweeping
│   └── hf_token.py       # HuggingFace token placeholder (DO NOT commit real tokens)
├── ueaj/                 # Main package
│   ├── data/             # Data loading and preprocessing
│   ├── model/            # Transformer architecture and components
│   ├── llama/            # LLaMA model loading and PEFT utilities
│   ├── opt/              # Custom optimizers and loss functions
│   ├── train/            # Training loop and logging
│   ├── utils/            # Configuration and compilation utilities
│   └── BACKLOG.md        # Internal development roadmap (not for CLAUDE.md)
├── test/                 # Test suite (pytest-based)
├── CLAUDE.md             # This file
└── README.md             # Project overview
```

## Key Architectural Decisions

1. **Simplified Einsum**: Uses 2-argument einsum with EinsumMetadata for optimizer canonicalization
2. **Flattened Structure**: Model components are in flat `ueaj/model/` directory
3. **Configuration System**: The `@config` decorator enables hierarchical composition with `.override()` methods
4. **Per-Parameter Optimizers**: OptimizerConfig allows different optimizer instances per layer/parameter type
5. **Distributed Training**: Designed for multi-GPU training with mesh-based parallelization

## Environment Configuration

- **Virtual environment**: `~/venvs/jax-packages` (managed by `scripts/run_python.sh`)
- **JAX compilation cache**: `JAX_COMPILATION_CACHE_DIR=$HOME/tmp/jax_cache`
- **XLA memory fraction**: `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` (use 95% of available GPU memory)
- **Triton globals**: `TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1` (required for kvax)
- Supports both GPU (with fused kernels) and CPU execution

### Training Environment Variables
- `OPTIMIZER`: Optimizer choice (multiscale, muon, adamw)
- `RUN_NAME`: WANDB run name for logging
- `MODEL_PATH`: Directory for saving checkpoints
- `BASE_LR`: Base learning rate (default: 0.025)

## Development Practices

### Commit Guidelines
- Use short, imperative summaries (e.g., "fix optimizer bug", "add FP8 tests")
- Group related changes per commit
- Note major configuration shifts in commit body
- Before commits, run `./scripts/run_python.sh -m pytest test/ -q` to ensure tests pass

### Security
- **Never commit real credentials**: `scripts/hf_token.py` is a placeholder
- Store HuggingFace tokens via `huggingface-cli login` or environment variables
- Confirm JAX cache directories exist on shared systems before long jobs

## Current Focus Areas

The project is actively maintained with focus on:
- Optimizer research (Muon, multiscale optimizers)
- Numerical precision experiments (FP8, FP16, custom precision formats)
- Scan unrolling and compilation optimizations
- Distributed training infrastructure
- Integration with pre-trained LLaMA models from HuggingFace