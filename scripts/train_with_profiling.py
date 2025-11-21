#!/usr/bin/env python3
"""
Example: Training loop with environment variable profiling.

Environment Variables:
    PROFILE_ENABLED=1       Enable profiling
    PROFILE_START_STEP=50   Start profiling at step 50
    PROFILE_INTERVAL=100    Profile every 100 steps
    PROFILE_DURATION=2      Profile 2 consecutive steps
    PROFILE_MODE=tensorboard  Use TensorBoard (or 'perfetto')

Examples:
    # Profile step 100 only
    PROFILE_ENABLED=1 PROFILE_START_STEP=100 .venv/bin/python scripts/train_with_profiling.py

    # Profile steps 50, 150, 250, ... (every 100 steps)
    PROFILE_ENABLED=1 PROFILE_START_STEP=50 PROFILE_INTERVAL=100 .venv/bin/python scripts/train_with_profiling.py

    # Profile 3 consecutive steps starting at step 100
    PROFILE_ENABLED=1 PROFILE_START_STEP=100 PROFILE_DURATION=3 PROFILE_INTERVAL=0 .venv/bin/python scripts/train_with_profiling.py

View Results:
    .venv/bin/tensorboard --logdir=./profiles
"""

import os
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', os.path.expanduser('~/tmp/jax_cache'))
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '.95')
os.environ.setdefault('TRITON_ALLOW_NON_CONSTEXPR_GLOBALS', '1')

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
import optax

from ueaj.model import configs
from ueaj.utils import maybe_profile  # <-- Simple env-var profiling!
import jax.nn as jnn


def create_dummy_batch(batch_size=4, seq_len=512):
    """Create dummy training batch."""
    return {
        'input_ids': jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        'labels': jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }


def simple_cross_entropy_loss(logits, labels):
    """Simple cross-entropy loss for demonstration."""
    # logits: (batch, seq, vocab)
    # labels: (batch, seq)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)
    log_probs = jnn.log_softmax(logits_flat, axis=-1)
    one_hot = jnn.one_hot(labels_flat, vocab_size)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


def main():
    print("Initializing model and optimizer...")
    model = configs.UEAJ_150M(rngs=rng.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=3e-4), wrt=nnx.Param)

    # Create training step
    @jax.jit
    def train_step(model, optimizer, batch):
        def loss_fn(model):
            logits = model(batch['input_ids'])
            return simple_cross_entropy_loss(logits, batch['labels'])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    # Dummy data
    batch = create_dummy_batch()

    # Warmup (compile)
    print("Warming up (JIT compilation)...")
    for _ in range(3):
        _ = train_step(model, optimizer, batch).block_until_ready()

    print("\nStarting training loop...")
    print("=" * 60)

    # Training loop with automatic profiling
    max_steps = 300
    for step in range(max_steps):
        # This automatically profiles based on env vars!
        with maybe_profile(step):
            loss = train_step(model, optimizer, batch)
            loss.block_until_ready()

        # Regular logging
        if step % 10 == 0:
            print(f"Step {step:4d} | Loss: {loss:.4f}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("\nTo view profiles:")
    print("  .venv/bin/tensorboard --logdir=./profiles")
    print("  Then open: http://localhost:6006")
    print("=" * 60)


if __name__ == "__main__":
    main()
