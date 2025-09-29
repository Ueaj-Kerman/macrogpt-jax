#!/usr/bin/env python3
"""
Test optimizer inlining with UEAJ_1B+ models.

This tests if unroll enables optimizer updates to be interleaved with
gradient computation, reducing peak memory usage.
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


def test_optimizer_inlining_1b():
    """Test optimizer inlining specifically with UEAJ_1B model."""

    print("="*80)
    print("OPTIMIZER INLINING TEST - UEAJ_15B")
    print("="*80)

    # Use 15B model to see optimizer inlining benefits (abstract evaluation)
    print("🔧 Creating UEAJ_15B model (abstract)...")

    def create_model():
        return configs.UEAJ_15B(rngs=rng.Rngs(42))

    # Use abstract evaluation to avoid memory allocation
    model_shape = nnx.eval_shape(create_model)
    graph_def, state = nnx.split(model_shape)

    # Context length
    batch_size, seq_len = 1, 1024
    input_shape = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

    print(f"📏 Model: UEAJ_15B (~15B params)")
    print(f"📏 Context: {batch_size}x{seq_len} = {seq_len:,} tokens")

    # Use Adam optimizer to see gradient accumulation and momentum effects
    lr = 1e-4

    unroll_values = [1, 4]
    results = {}

    for unroll_val in unroll_values:
        print(f"\n🎯 Testing optimizer inlining with unroll={unroll_val}...")

        try:
            def training_step_with_sgd(graph_def, state, inputs):
                """Training step with simple SGD optimizer."""

                def loss_fn(state):
                    model = nnx.merge(graph_def, state)
                    # Use unroll parameter in forward pass
                    hidden_states = model.get_activations(inputs, unroll=unroll_val)
                    # Simple loss
                    return jnp.mean(jnp.square(hidden_states))

                # Compute gradients
                loss_val, grads = jax.value_and_grad(loss_fn)(state)

                # Adam optimizer with momentum and variance accumulation
                # This should create memory pressure that can benefit from inlining

                # Simulate Adam state (momentum and variance)
                # In real Adam, these would persist across steps
                momentum = jax.tree.map(lambda x: jnp.zeros_like(x), grads)
                variance = jax.tree.map(lambda x: jnp.zeros_like(x), grads)

                # Adam update equations
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                step = 1

                # Update momentum and variance
                momentum = jax.tree.map(
                    lambda m, g: beta1 * m + (1 - beta1) * g,
                    momentum, grads
                )
                variance = jax.tree.map(
                    lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g),
                    variance, grads
                )

                # Bias correction
                m_hat = jax.tree.map(lambda m: m / (1 - beta1**step), momentum)
                v_hat = jax.tree.map(lambda v: v / (1 - beta2**step), variance)

                # Adam update
                updated_state = jax.tree.map(
                    lambda param, m, v: param - lr * m / (jnp.sqrt(v) + eps),
                    state, m_hat, v_hat
                )

                return loss_val, updated_state, (grads, momentum, variance)

            # Compile and measure
            start_time = time.time()
            compiled_fn = compile_function(
                jax.jit(training_step_with_sgd),
                sample_args=(graph_def, state, input_shape),
                name=f"UEAJ_15B Training unroll={unroll_val}"
            )
            compilation_time = time.time() - start_time

            # Get memory metrics
            memory = compiled_fn.memory_analysis()
            cost = compiled_fn.cost_analysis()

            results[unroll_val] = {
                'compilation_time': compilation_time,
                'peak_memory_gb': memory.temp_size_in_bytes * 1e-9 if memory else 0,
                'output_memory_gb': memory.output_size_in_bytes * 1e-9 if memory else 0,
                'total_memory_gb': (memory.temp_size_in_bytes + memory.output_size_in_bytes) * 1e-9 if memory else 0,
                'flops': cost.get('flops', 0) if cost else 0,
            }

            print(f"   ✓ Compilation: {compilation_time:.2f}s")
            if memory:
                peak_gb = memory.temp_size_in_bytes * 1e-9
                total_gb = (memory.temp_size_in_bytes + memory.output_size_in_bytes) * 1e-9
                print(f"   ✓ Peak VRAM: {peak_gb:.2f} GB")
                print(f"   ✓ Total VRAM: {total_gb:.2f} GB")

        except Exception as e:
            print(f"   ❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            results[unroll_val] = {'error': str(e)}

    # Analysis
    print(f"\n📊 OPTIMIZER INLINING ANALYSIS")
    print("=" * 80)

    successful = {k: v for k, v in results.items() if 'error' not in v}

    if len(successful) >= 2:
        baseline = successful[1]
        unroll4 = successful[4]

        print(f"{'Metric':<20} {'unroll=1':<15} {'unroll=4':<15} {'Change':<15}")
        print("─" * 75)

        # Compilation time
        comp_speedup = baseline['compilation_time'] / unroll4['compilation_time']
        print(f"{'Compilation (s)':<20} {baseline['compilation_time']:<15.2f} "
              f"{unroll4['compilation_time']:<15.2f} {comp_speedup:.2f}x faster")

        # Peak memory
        memory_diff = baseline['peak_memory_gb'] - unroll4['peak_memory_gb']
        memory_pct = (memory_diff / baseline['peak_memory_gb']) * 100
        print(f"{'Peak Memory (GB)':<20} {baseline['peak_memory_gb']:<15.2f} "
              f"{unroll4['peak_memory_gb']:<15.2f} {memory_diff:+.2f} GB")

        # Total memory
        total_diff = baseline['total_memory_gb'] - unroll4['total_memory_gb']
        print(f"{'Total Memory (GB)':<20} {baseline['total_memory_gb']:<15.2f} "
              f"{unroll4['total_memory_gb']:<15.2f} {total_diff:+.2f} GB")

        # FLOPs
        if baseline['flops'] > 0 and unroll4['flops'] > 0:
            flops_ratio = unroll4['flops'] / baseline['flops']
            print(f"{'Compute (TFLOPs)':<20} {baseline['flops']*1e-12:<15.2f} "
                  f"{unroll4['flops']*1e-12:<15.2f} {flops_ratio:.2f}x")

        print("\n🔍 OPTIMIZER INLINING ASSESSMENT:")
        print("─" * 50)

        if memory_diff > 0:
            print(f"✅ MEMORY SAVINGS: {memory_diff:.2f} GB ({memory_pct:+.1f}%)")
            print(f"   🎯 Optimizer inlining is working!")
            print(f"   🔄 Gradients freed immediately after each layer's update")
            print(f"   📉 Peak memory reduced despite code expansion")
        else:
            print(f"📈 MEMORY OVERHEAD: {abs(memory_diff):.2f} GB ({abs(memory_pct):.1f}%)")
            print(f"   ⚠️  Code expansion dominates at this scale")
            print(f"   🔧 Try larger model or longer sequence for inlining benefits")

        print(f"\n⚡ COMPILATION BENEFIT: {comp_speedup:.2f}x faster ({baseline['compilation_time'] - unroll4['compilation_time']:.1f}s saved)")

    else:
        print("❌ Insufficient results for comparison")


def test_gradient_vs_optimizer_memory():
    """Compare gradient-only vs full training step memory usage."""

    print(f"\n{'='*80}")
    print("GRADIENT VS OPTIMIZER MEMORY BREAKDOWN")
    print("=" * 80)

    def create_model():
        return configs.UEAJ_15B(rngs=rng.Rngs(42))

    # Use abstract evaluation
    model_shape = nnx.eval_shape(create_model)
    graph_def, state = nnx.split(model_shape)

    batch_size, seq_len = 1, 1024  # Smaller context for memory breakdown
    input_shape = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

    print(f"🔬 Isolating gradient vs optimizer memory patterns...")

    for unroll_val in [1, 4]:
        print(f"\n📈 Memory breakdown with unroll={unroll_val}:")

        # Test gradient computation only
        def gradient_only(graph_def, state, inputs):
            def loss_fn(state):
                model = nnx.merge(graph_def, state)
                hidden_states = model.get_activations(inputs, unroll=unroll_val)
                return jnp.mean(jnp.square(hidden_states))

            loss_val, grads = jax.value_and_grad(loss_fn)(state)
            return loss_val, grads

        # Test full training step
        def full_training(graph_def, state, inputs):
            def loss_fn(state):
                model = nnx.merge(graph_def, state)
                hidden_states = model.get_activations(inputs, unroll=unroll_val)
                return jnp.mean(jnp.square(hidden_states))

            loss_val, grads = jax.value_and_grad(loss_fn)(state)

            # Adam optimizer update
            momentum = jax.tree.map(lambda x: jnp.zeros_like(x), grads)
            variance = jax.tree.map(lambda x: jnp.zeros_like(x), grads)

            beta1, beta2, eps = 0.9, 0.999, 1e-8
            step = 1
            lr = 1e-4

            # Update momentum and variance (creates memory pressure)
            momentum = jax.tree.map(
                lambda m, g: beta1 * m + (1 - beta1) * g,
                momentum, grads
            )
            variance = jax.tree.map(
                lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g),
                variance, grads
            )

            # Bias correction
            m_hat = jax.tree.map(lambda m: m / (1 - beta1**step), momentum)
            v_hat = jax.tree.map(lambda v: v / (1 - beta2**step), variance)

            # Adam update
            updated_state = jax.tree.map(
                lambda param, m, v: param - lr * m / (jnp.sqrt(v) + eps),
                state, m_hat, v_hat
            )

            return loss_val, updated_state

        try:
            # Compile gradient-only version
            grad_compiled = compile_function(
                jax.jit(gradient_only),
                sample_args=(graph_def, state, input_shape),
                name=f"Gradient Only unroll={unroll_val}"
            )

            # Compile full training version
            full_compiled = compile_function(
                jax.jit(full_training),
                sample_args=(graph_def, state, input_shape),
                name=f"Full Training unroll={unroll_val}"
            )

            grad_memory = grad_compiled.memory_analysis()
            full_memory = full_compiled.memory_analysis()

            if grad_memory and full_memory:
                grad_peak = grad_memory.temp_size_in_bytes * 1e-9
                full_peak = full_memory.temp_size_in_bytes * 1e-9
                optimizer_overhead = full_peak - grad_peak

                print(f"   📊 Gradient peak: {grad_peak:.2f} GB")
                print(f"   📊 Full training peak: {full_peak:.2f} GB")
                print(f"   📊 Adam optimizer overhead: {optimizer_overhead:+.2f} GB")

        except Exception as e:
            print(f"   ❌ Failed: {e}")


if __name__ == "__main__":
    test_optimizer_inlining_1b()
    test_gradient_vs_optimizer_memory()

    print(f"\n{'='*80}")
    print("SUMMARY:")
    print("• Memory SAVINGS with unroll=4 → Optimizer inlining working ✅")
    print("• Memory OVERHEAD with unroll=4 → Need larger model/context")
    print("• Compilation speedup valuable regardless of memory impact")
    print("="*80)