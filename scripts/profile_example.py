#!/usr/bin/env python3
"""
Example script demonstrating JAX profiling with Perfetto.

Usage:
    .venv/bin/python scripts/profile_example.py
"""

import jax
import jax.numpy as jnp
from jax import random
import os

# Configure profiling output directory
PROFILE_DIR = os.path.join(os.getcwd(), "profiles")
os.makedirs(PROFILE_DIR, exist_ok=True)


def example_computation():
    """Example computation to profile - simulates a small transformer forward pass."""
    key = random.key(42)

    # Simulate attention computation
    seq_len, d_model = 512, 768
    Q = random.normal(key, (seq_len, d_model))
    K = random.normal(random.fold_in(key, 1), (seq_len, d_model))
    V = random.normal(random.fold_in(key, 2), (seq_len, d_model))

    # Attention scores
    scores = Q @ K.T
    attn = jax.nn.softmax(scores, axis=-1)
    output = attn @ V

    return output


def profile_with_perfetto():
    """Profile using Perfetto (interactive web UI)."""
    print("Warming up JIT compilation...")
    # Warmup to compile
    jitted_fn = jax.jit(example_computation)
    _ = jitted_fn().block_until_ready()

    print(f"\nProfiling to: {PROFILE_DIR}/perfetto-trace")
    print("This will open a Perfetto UI link when done...")

    # Profile with Perfetto
    with jax.profiler.trace(
        os.path.join(PROFILE_DIR, "perfetto-trace"),
        create_perfetto_link=True  # Creates interactive link
    ):
        result = jitted_fn()
        result.block_until_ready()  # Ensure computation completes

    print("✓ Profiling complete! Click the link above to view trace.")


def profile_with_perfetto_file():
    """Profile using Perfetto (save to file for later viewing)."""
    print("Warming up JIT compilation...")
    jitted_fn = jax.jit(example_computation)
    _ = jitted_fn().block_until_ready()

    trace_file = os.path.join(PROFILE_DIR, "perfetto-trace-file")
    print(f"\nProfiling to: {trace_file}")

    # Profile and save to file
    with jax.profiler.trace(
        trace_file,
        create_perfetto_trace=True  # Saves .json.gz file
    ):
        result = jitted_fn()
        result.block_until_ready()

    print(f"✓ Profiling complete! View at https://ui.perfetto.dev")
    print(f"  Upload: {trace_file}/plugins/profile/*/trace.json.gz")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        profile_with_perfetto_file()
    else:
        profile_with_perfetto()
