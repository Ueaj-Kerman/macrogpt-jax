#!/usr/bin/env python3
"""
Example: Profile a training step with TensorBoard/XProf.

This script demonstrates profiling a complete training step including:
- Forward pass
- Loss computation
- Backward pass (gradient computation)
- Optimizer update

Usage:
    # Profile a single step
    .venv/bin/python scripts/profile_training.py

    # View in TensorBoard
    .venv/bin/tensorboard --logdir=./profiles

    # Then open http://localhost:6006 in your browser
"""

import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = os.path.expanduser('~/tmp/jax_cache')
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.95'
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
from flax.nnx import rnglib as rng
import optax

# Import your model
from ueaj.model import configs
from ueaj.opt import next_token_loss
from ueaj.utils import profile_trace, profile_scope, benchmark_function


def create_dummy_batch(model_config, batch_size=4, seq_len=512):
    """Create a dummy batch for profiling."""
    return {
        'input_ids': jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        'labels': jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }


def profile_forward_pass():
    """Profile just the model forward pass."""
    print("=" * 60)
    print("PROFILING: Forward Pass Only")
    print("=" * 60)

    # Create model
    model = configs.UEAJ_150M(rngs=rng.Rngs(42))
    batch = create_dummy_batch(model, batch_size=4, seq_len=512)

    # JIT compile
    @jax.jit
    def forward(input_ids):
        return model(input_ids)

    # Warmup
    print("Warming up...")
    _ = forward(batch['input_ids']).block_until_ready()

    # Profile
    with profile_trace("./profiles", name="forward_pass"):
        with profile_scope("model_forward"):
            logits = forward(batch['input_ids'])
            logits.block_until_ready()

    print("\n✓ Forward pass profiled!")
    print("  View: tensorboard --logdir=./profiles")


def profile_training_step():
    """Profile a complete training step (forward + backward + update)."""
    print("=" * 60)
    print("PROFILING: Complete Training Step")
    print("=" * 60)

    # Create model and optimizer
    model = configs.UEAJ_150M(rngs=rng.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=3e-4))
    batch = create_dummy_batch(model, batch_size=4, seq_len=512)

    # Define training step
    @jax.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            logits = model(batch['input_ids'])
            loss = next_token_loss(logits, batch['labels'])
            return loss

        # Compute gradients
        loss, grads = nnx.value_and_grad(loss_fn)(model)

        # Update parameters
        optimizer.update(grads)

        return loss

    # Warmup
    print("Warming up (JIT compilation)...")
    for _ in range(3):
        _ = train_step(model, optimizer, batch).block_until_ready()

    # Profile with detailed scopes
    print("\nProfiling training step...")
    with profile_trace("./profiles", name="training_step"):
        with profile_scope("full_step"):
            # Forward + backward happens inside train_step
            with profile_scope("train_step_computation"):
                loss = train_step(model, optimizer, batch)
                loss.block_until_ready()

    print(f"\n✓ Training step profiled! Loss: {loss:.4f}")
    print("  View: tensorboard --logdir=./profiles")


def profile_model_components():
    """Profile individual model components (attention, MLP, etc.)."""
    print("=" * 60)
    print("PROFILING: Model Components")
    print("=" * 60)

    model = configs.UEAJ_150M(rngs=rng.Rngs(42))
    batch_size, seq_len = 4, 512
    x = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    # Profile embedding
    @jax.jit
    def embed_tokens(x):
        return model.embed(x)

    # Profile first transformer layer
    @jax.jit
    def transformer_layer(x):
        embedded = model.embed(x)
        return model.layers[0](embedded)

    # Warmup
    print("Warming up...")
    _ = embed_tokens(x).block_until_ready()
    _ = transformer_layer(x).block_until_ready()

    # Profile components
    with profile_trace("./profiles", name="model_components"):
        with profile_scope("embedding"):
            emb = embed_tokens(x)
            emb.block_until_ready()

        with profile_scope("transformer_layer_0"):
            layer_out = transformer_layer(x)
            layer_out.block_until_ready()

    print("\n✓ Components profiled!")
    print("  View: tensorboard --logdir=./profiles")


def benchmark_performance():
    """Benchmark training step performance with statistics."""
    print("=" * 60)
    print("BENCHMARKING: Training Step Performance")
    print("=" * 60)

    model = configs.UEAJ_150M(rngs=rng.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=3e-4))
    batch = create_dummy_batch(model, batch_size=4, seq_len=512)

    @jax.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            logits = model(batch['input_ids'])
            loss = next_token_loss(logits, batch['labels'])
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    # Benchmark
    print("Running benchmark (100 iterations)...")
    stats = benchmark_function(
        train_step,
        model, optimizer, batch,
        num_iterations=100,
        warmup=10
    )

    print("\n📊 Performance Statistics:")
    print(f"  Mean:   {stats['mean']:>8.2f} ms")
    print(f"  Median: {stats['median']:>8.2f} ms")
    print(f"  Std:    {stats['std']:>8.2f} ms")
    print(f"  Min:    {stats['min']:>8.2f} ms")
    print(f"  Max:    {stats['max']:>8.2f} ms")
    print(f"\n  Throughput: {1000.0 / stats['mean']:.2f} steps/sec")


if __name__ == "__main__":
    import sys

    # Create profiles directory
    os.makedirs("./profiles", exist_ok=True)

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "forward":
            profile_forward_pass()
        elif mode == "step":
            profile_training_step()
        elif mode == "components":
            profile_model_components()
        elif mode == "benchmark":
            benchmark_performance()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: forward, step, components, benchmark")
    else:
        # Default: profile training step
        profile_training_step()
        print("\n" + "=" * 60)
        print("Try other modes:")
        print("  .venv/bin/python scripts/profile_training.py forward")
        print("  .venv/bin/python scripts/profile_training.py components")
        print("  .venv/bin/python scripts/profile_training.py benchmark")
        print("=" * 60)
