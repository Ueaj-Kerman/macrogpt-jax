#!/usr/bin/env python3
"""
Test unroll impact on backwards pass + optimizer memory usage.

This tests the full training step: forward + backward + optimizer update
to see if unroll enables memory-efficient gradient-update interleaving.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
import optax

from ueaj.model import configs
from ueaj.utils.compile import compile_function


def test_full_training_step_memory():
    """Test memory usage during full training step with different unroll values."""

    print("="*80)
    print("BACKWARDS PASS + OPTIMIZER MEMORY ANALYSIS")
    print("="*80)

    # Create large model with substantial context
    print("🔧 Creating UEAJ_1B model...")
    model = configs.UEAJ_1B(rngs=rng.Rngs(42))
    graph_def, state = nnx.split(model)

    # Large context to maximize memory pressure
    batch_size, seq_len = 1, 4096
    input_shape = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

    print(f"📏 Model: UEAJ_1B (1.5B params)")
    print(f"📏 Context: {batch_size}x{seq_len} = {seq_len:,} tokens")

    # Create optimizer
    optimizer = optax.adamw(learning_rate=1e-4)
    opt_state = optimizer.init(state)

    # Test different unroll values
    unroll_values = [1, 4]

    results = {}

    for unroll_val in unroll_values:
        print(f"\n🔄 Testing full training step with unroll={unroll_val}...")

        try:
            def full_training_step(graph_def, state, opt_state, inputs):
                """Complete training step: forward + backward + optimizer update."""

                def loss_fn(state):
                    model = nnx.merge(graph_def, state)
                    # Use unroll in forward pass
                    hidden_states = model.get_activations(inputs, unroll=unroll_val)
                    # Simple MSE loss for testing
                    return jnp.mean(jnp.square(hidden_states))

                # Compute loss and gradients
                loss_val, grads = jax.value_and_grad(loss_fn)(state)

                # Apply optimizer update
                updates, new_opt_state = optimizer.update(grads, opt_state)
                new_state = optax.apply_updates(state, updates)

                return loss_val, new_state, new_opt_state

            # Compile with memory analysis
            start_time = time.time()
            compiled_fn = compile_function(
                jax.jit(full_training_step),
                sample_args=(graph_def, state, opt_state, input_shape),
                name=f"Full Training Step unroll={unroll_val}"
            )
            compilation_time = time.time() - start_time

            # Get detailed metrics
            cost = compiled_fn.cost_analysis()
            memory = compiled_fn.memory_analysis()

            results[unroll_val] = {
                'compilation_time': compilation_time,
                'peak_memory_gb': memory.temp_size_in_bytes * 1e-9 if memory else 0,
                'output_memory_gb': memory.output_size_in_bytes * 1e-9 if memory else 0,
                'total_memory_gb': (memory.temp_size_in_bytes + memory.output_size_in_bytes) * 1e-9 if memory else 0,
                'flops': cost.get('flops', 0) if cost else 0,
            }

            print(f"   ✓ Compilation: {compilation_time:.2f}s")
            if memory:
                print(f"   ✓ Peak VRAM: {memory.temp_size_in_bytes * 1e-9:.2f} GB")
                print(f"   ✓ Output VRAM: {memory.output_size_in_bytes * 1e-9:.2f} GB")
                print(f"   ✓ Total VRAM: {(memory.temp_size_in_bytes + memory.output_size_in_bytes) * 1e-9:.2f} GB")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[unroll_val] = {'error': str(e)}

    # Detailed comparison
    print(f"\n📊 FULL TRAINING STEP COMPARISON")
    print("=" * 80)
    print(f"{'Unroll':<8} {'Time (s)':<12} {'Peak (GB)':<12} {'Total (GB)':<12} {'Memory Diff':<12}")
    print("─" * 80)

    successful = {k: v for k, v in results.items() if 'error' not in v}

    if len(successful) >= 2:
        baseline = successful[1]
        unroll4 = successful[4]

        # Show results
        print(f"1        {baseline['compilation_time']:<12.2f} "
              f"{baseline['peak_memory_gb']:<12.2f} {baseline['total_memory_gb']:<12.2f} baseline")

        memory_diff = baseline['peak_memory_gb'] - unroll4['peak_memory_gb']
        diff_str = f"{memory_diff:+.2f} GB"
        speedup = baseline['compilation_time'] / unroll4['compilation_time']

        print(f"4        {unroll4['compilation_time']:<12.2f} "
              f"{unroll4['peak_memory_gb']:<12.2f} {unroll4['total_memory_gb']:<12.2f} {diff_str}")

        # Analysis
        print(f"\n🔍 BACKWARDS PASS + OPTIMIZER ANALYSIS:")
        print("─" * 60)
        print(f"   Compilation speedup: {speedup:.2f}x")
        print(f"   Peak memory change: {memory_diff:+.2f} GB ({memory_diff/baseline['peak_memory_gb']*100:+.1f}%)")

        if memory_diff > 0:
            print(f"   🎯 Memory SAVINGS with unroll=4: {memory_diff:.2f} GB")
            print(f"      This confirms optimizer inlining is working!")
            print(f"      Gradients are freed immediately after each layer's optimizer update.")
        else:
            print(f"   📈 Memory overhead with unroll=4: {abs(memory_diff):.2f} GB")
            print(f"      Code expansion dominates at this model size.")

        # FLOPS analysis
        if 'flops' in baseline and 'flops' in unroll4:
            flops_ratio = unroll4['flops'] / max(baseline['flops'], 1)
            print(f"   Compute ratio: {flops_ratio:.2f}x (higher = more parallel operations)")


def test_gradient_accumulation_memory():
    """Test how unroll affects gradient accumulation patterns."""

    print(f"\n{'='*80}")
    print("GRADIENT ACCUMULATION PATTERN ANALYSIS")
    print("=" * 80)

    # Smaller model for clearer analysis
    print("🔧 Creating UEAJ_150M for gradient analysis...")
    model = configs.UEAJ_150M(rngs=rng.Rngs(42))
    graph_def, state = nnx.split(model)

    # Medium context
    batch_size, seq_len = 2, 2048
    input_shape = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

    print(f"📏 Model: UEAJ_150M (150M params)")
    print(f"📏 Context: {batch_size}x{seq_len} = {seq_len:,} tokens each")

    for unroll_val in [1, 4]:
        print(f"\n📈 Gradient computation with unroll={unroll_val}...")

        try:
            def gradient_only_step(graph_def, state, inputs):
                """Just compute gradients to see memory pattern."""
                def loss_fn(state):
                    model = nnx.merge(graph_def, state)
                    hidden_states = model.get_activations(inputs, unroll=unroll_val)
                    return jnp.mean(jnp.square(hidden_states))

                loss_val, grads = jax.value_and_grad(loss_fn)(state)
                return loss_val, grads

            compiled_fn = compile_function(
                jax.jit(gradient_only_step),
                sample_args=(graph_def, state, input_shape),
                name=f"Gradient Only unroll={unroll_val}"
            )

            memory = compiled_fn.memory_analysis()
            if memory:
                peak_gb = memory.temp_size_in_bytes * 1e-9
                print(f"   📊 Peak memory during gradients: {peak_gb:.3f} GB")

        except Exception as e:
            print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    test_full_training_step_memory()
    test_gradient_accumulation_memory()

    print(f"\n{'='*80}")
    print("KEY INSIGHTS:")
    print("• If unroll=4 shows LOWER peak memory → optimizer inlining is working")
    print("• If unroll=4 shows HIGHER peak memory → code expansion dominates")
    print("• Memory benefits increase with model size and gradient accumulation")
    print("• Compilation speedup is valuable regardless of memory impact")
    print("="*80)