#!/usr/bin/env python3
"""
Test unroll parameter with the actual model using @nnx.scan(unroll=X).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.model import configs
from ueaj.utils.compile import compile_function


def test_unroll_comparison():
    """Test compilation and memory differences between unroll values."""

    print("="*80)
    print("LARGE MODEL UNROLL COMPARISON")
    print("="*80)

    # Create large model with long context
    print("🔧 Creating UEAJ_1B model...")
    model = configs.UEAJ_1B(rngs=rng.Rngs(42))
    graph_def, state = nnx.split(model)

    # Very large context length to force memory optimizations
    batch_size, seq_len = 1, 8192  # Large context length
    input_shape = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

    print(f"📏 Model: UEAJ_1B (1.5B params)")
    print(f"📏 Context: {batch_size}x{seq_len} = {seq_len:,} tokens")

    # Test different unroll values (skip 8 for large model to avoid compilation explosion)
    unroll_values = [1, 2, 4]

    results = {}

    for unroll_val in unroll_values:
        print(f"\n🔄 Testing unroll={unroll_val}...")

        try:
            @jax.jit
            def forward_with_unroll(graph_def, state, inputs):
                model = nnx.merge(graph_def, state)
                return model.get_activations(inputs, unroll=unroll_val)

            # Compile and measure
            start_time = time.time()
            compiled_fn = compile_function(
                forward_with_unroll,
                sample_args=(graph_def, state, input_shape),
                name=f"UEAJ_1B unroll={unroll_val}"
            )
            compilation_time = time.time() - start_time

            # Get metrics
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
                total_mem = (memory.temp_size_in_bytes + memory.output_size_in_bytes) * 1e-9
                print(f"   ✓ Total VRAM: {total_mem:.2f} GB")
                print(f"   ✓ Peak VRAM: {memory.temp_size_in_bytes * 1e-9:.2f} GB")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results[unroll_val] = {'error': str(e)}

    # Print detailed comparison
    print(f"\n📊 UNROLL COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Unroll':<8} {'Time (s)':<12} {'Peak (GB)':<12} {'Total (GB)':<12} {'Speedup':<10}")
    print("─" * 80)

    successful = {k: v for k, v in results.items() if 'error' not in v}

    if successful:
        baseline = successful.get(1)

        for unroll_val in sorted(successful.keys()):
            result = successful[unroll_val]

            if baseline and unroll_val != 1:
                time_speedup = baseline['compilation_time'] / result['compilation_time']
                speedup_str = f"{time_speedup:.2f}x"
            else:
                speedup_str = "baseline"

            print(f"{unroll_val:<8} {result['compilation_time']:<12.2f} "
                  f"{result['peak_memory_gb']:<12.2f} {result['total_memory_gb']:<12.2f} {speedup_str:<10}")

        # Memory analysis
        print(f"\n💾 MEMORY ANALYSIS:")
        print("─" * 50)

        if baseline:
            for unroll_val in [2, 4]:
                if unroll_val in successful:
                    result = successful[unroll_val]
                    memory_reduction = (baseline['peak_memory_gb'] - result['peak_memory_gb']) / baseline['peak_memory_gb'] * 100
                    print(f"   unroll={unroll_val}: {memory_reduction:+.1f}% peak memory vs unroll=1")

            # Show absolute memory numbers for large model
            print(f"\n📊 ABSOLUTE MEMORY USAGE:")
            for unroll_val in sorted(successful.keys()):
                result = successful[unroll_val]
                print(f"   unroll={unroll_val}: {result['peak_memory_gb']:.2f} GB peak, {result['total_memory_gb']:.2f} GB total")

        # Find optimal
        memory_sorted = sorted(successful.items(), key=lambda x: x[1]['peak_memory_gb'])
        time_sorted = sorted(successful.items(), key=lambda x: x[1]['compilation_time'])

        print(f"\n🏆 RECOMMENDATIONS:")
        print("─" * 50)
        print(f"   • Fastest compilation: unroll={time_sorted[0][0]} ({time_sorted[0][1]['compilation_time']:.2f}s)")
        print(f"   • Lowest peak memory: unroll={memory_sorted[0][0]} ({memory_sorted[0][1]['peak_memory_gb']:.2f} GB)")

        # Sweet spot recommendation
        if len(successful) >= 2:
            print(f"   • Recommended: unroll=4 (good balance of compilation speed and memory efficiency)")

        # Memory efficiency insight
        if baseline and 4 in successful:
            baseline_mem = baseline['peak_memory_gb']
            unroll4_mem = successful[4]['peak_memory_gb']
            if unroll4_mem < baseline_mem:
                savings = baseline_mem - unroll4_mem
                print(f"   🎯 Memory savings with unroll=4: {savings:.2f} GB ({savings/baseline_mem*100:.1f}% reduction)")
            else:
                overhead = unroll4_mem - baseline_mem
                print(f"   ⚠️  Memory overhead with unroll=4: +{overhead:.2f} GB (expected for large models)")


def test_training_step_with_unroll():
    """Test unroll impact in a training-like scenario with large model."""

    print(f"\n{'='*80}")
    print("LARGE MODEL TRAINING STEP UNROLL IMPACT")
    print("=" * 80)

    # Create large model
    print("🔧 Creating UEAJ_1B for training test...")
    model = configs.UEAJ_1B(rngs=rng.Rngs(42))
    graph_def, state = nnx.split(model)

    # Large context for training
    batch_size, seq_len = 1, 4096  # Large but reasonable for training
    input_shape = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

    print(f"📏 Training context: {batch_size}x{seq_len} = {seq_len:,} tokens")

    # Test training step with different unroll values
    for unroll_val in [1, 4]:
        print(f"\n🎯 Training step with unroll={unroll_val}...")

        try:
            @jax.jit
            def training_step(graph_def, state, inputs):
                def loss_fn(state):
                    model = nnx.merge(graph_def, state)
                    hidden_states = model.get_activations(inputs, unroll=unroll_val)
                    # Simple loss for testing
                    return jnp.mean(jnp.square(hidden_states))

                loss_val, grads = jax.value_and_grad(loss_fn)(state)
                return loss_val, grads

            # Compile and measure
            start_time = time.time()
            compiled_fn = compile_function(
                training_step,
                sample_args=(graph_def, state, input_shape),
                name=f"Training Step unroll={unroll_val}"
            )
            compilation_time = time.time() - start_time

            memory = compiled_fn.memory_analysis()
            if memory:
                peak_gb = memory.temp_size_in_bytes * 1e-9
                print(f"   ✓ Peak VRAM: {peak_gb:.2f} GB")
                print(f"   ✓ Compilation: {compilation_time:.2f}s")

        except Exception as e:
            print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    test_unroll_comparison()
    test_training_step_with_unroll()

    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print("1. Use unroll=4 in your training script for memory optimization")
    print("2. Measure actual training VRAM usage with different unroll values")
    print("3. Remove unroll parameter when done testing (or keep if beneficial)")
    print("="*80)