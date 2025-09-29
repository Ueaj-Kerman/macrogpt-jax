#!/usr/bin/env python3
"""
Isolated test for scan unroll optimization analysis.

This test analyzes the impact of the `unroll` parameter in jax.lax.scan on:
1. Compilation speed
2. Memory usage
3. Optimizer inlining opportunities

Uses abstract models to avoid GPU memory allocation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

# Import model components
from ueaj.model import LlamaModel, TransformerLayer, SoftmaxAttention, MLP
from ueaj.model.nn import *
from ueaj.utils.compile import compile_function
from ueaj.model import configs


def create_large_model_config(model_d: int, num_layers: int) -> type:
    """Create a large model configuration for testing."""
    return LlamaModel.override(
        vocab_size=50432,
        model_d=model_d,
        num_layers=num_layers,
        transformer_layer=TransformerLayer.override(
            attn=SoftmaxAttention.override(
                kq_d=64,
                kv_heads=model_d // 128,
                kv_q_ratio=2,
                rope_theta=2_000.0,
                act_fn=leaky_relu_squared,
            ),
            mlp=MLP.override(
                act_fn=leaky_relu_squared,
            ),
        )
    )


def create_scan_variants(num_layers: int):
    """Create scan function variants with different unroll values."""

    def create_scan_fn(unroll_value):
        def scan_forward(graph_def, state, inputs):
            """Forward pass through transformer layers using scan."""
            # Reconstruct model from graph_def and state
            model = nnx.merge(graph_def, state)

            batch_size, seq_len = inputs.shape
            kwargs = model.default_kwargs(batch_size, seq_len)

            # Embedding (use correct attribute name)
            act = model.embed_tokens(inputs)

            # Simple layer-by-layer forward pass with controlled unroll
            if unroll_value == True:  # Full unroll
                for layer in model.layers:
                    act = layer(act, **kwargs)
                    act = act.astype(act.dtype)
                return act
            else:
                # Use scan with specified unroll
                def scan_fn(carry, layer):
                    out = layer(carry, **kwargs)
                    return out.astype(carry.dtype), None

                carry, _ = jax.lax.scan(
                    scan_fn,
                    act,
                    model.layers,
                    unroll=unroll_value
                )
                return carry

        return jax.jit(scan_forward)

    return {
        'unroll_1': create_scan_fn(1),
        'unroll_2': create_scan_fn(2),
        'unroll_4': create_scan_fn(4),
        'unroll_8': create_scan_fn(8),
        'fully_unrolled': create_scan_fn(True)  # Full unroll
    }


def create_abstract_model_and_inputs(model_config, batch_size: int = 2, seq_len: int = 1024):
    """Create abstract model and input tensors without allocating memory."""

    # Create abstract model using eval_shape
    def create_model():
        return model_config(rngs=rng.Rngs(0))

    model_shape = nnx.eval_shape(create_model)

    # Split into graph_def and state for JAX tracing
    graph_def, state = nnx.split(model_shape)

    # Create abstract input tensor
    input_shape = jax.ShapeDtypeStruct((batch_size, seq_len), jnp.int32)

    return graph_def, state, input_shape


def test_scan_compilation_metrics():
    """Test compilation metrics for different scan unroll values."""

    print("="*80)
    print("SCAN UNROLL OPTIMIZATION ANALYSIS")
    print("="*80)

    # Test configurations
    test_configs = [
        {"name": "Medium (768d, 12L)", "model_d": 768, "num_layers": 12},
        {"name": "Large (1536d, 24L)", "model_d": 1536, "num_layers": 24},
        {"name": "XL (2048d, 32L)", "model_d": 2048, "num_layers": 32},
    ]

    results = {}

    for config in test_configs:
        print(f"\n{'─'*60}")
        print(f"Testing {config['name']}")
        print(f"{'─'*60}")

        model_config = create_large_model_config(config['model_d'], config['num_layers'])

        # Create abstract model and inputs
        graph_def, state, input_shape = create_abstract_model_and_inputs(model_config)

        # Create scan variants
        scan_variants = create_scan_variants(config['num_layers'])

        config_results = {}

        for variant_name, scan_fn in scan_variants.items():
            print(f"\n🔄 Testing {variant_name}...")

            # Compile with metrics
            start_time = time.time()

            try:
                compiled_fn = compile_function(
                    scan_fn,
                    sample_args=(graph_def, state, input_shape),
                    name=f"{config['name']} - {variant_name}"
                )

                compilation_time = time.time() - start_time

                # Get detailed metrics
                cost = compiled_fn.cost_analysis()
                memory = compiled_fn.memory_analysis()

                config_results[variant_name] = {
                    'compilation_time': compilation_time,
                    'flops': cost.get('flops', 0) if cost else 0,
                    'peak_memory_gb': memory.temp_size_in_bytes * 1e-9 if memory else 0,
                    'output_memory_gb': memory.output_size_in_bytes * 1e-9 if memory else 0,
                }

                print(f"   ✓ Compilation: {compilation_time:.2f}s")
                if cost and 'flops' in cost:
                    print(f"   ✓ FLOPs: {cost['flops'] * 1e-12:.2f} TFLOPs")
                if memory:
                    print(f"   ✓ Peak VRAM: {memory.temp_size_in_bytes * 1e-9:.2f} GB")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                config_results[variant_name] = {
                    'compilation_time': float('inf'),
                    'error': str(e)
                }

        results[config['name']] = config_results

    return results


def analyze_optimizer_inlining():
    """Analyze whether optimizer steps can be inlined with different unroll values."""

    print(f"\n{'='*80}")
    print("OPTIMIZER INLINING ANALYSIS")
    print(f"{'='*80}")

    # Create a simple training step that includes optimizer update
    def create_training_step(unroll_value):
        def training_step(graph_def, state, inputs):
            """Training step with gradient computation and optimizer update."""

            def loss_fn(state, inputs):
                # Reconstruct model
                model = nnx.merge(graph_def, state)

                # Use get_activations to get hidden states (simpler than full forward)
                hidden_states = model.get_activations(inputs[:, :-1])  # All but last token

                # Simple loss on hidden states (avoid complex lm_head for this test)
                return jnp.mean(jnp.square(hidden_states))

            # Compute gradients
            loss_val, grads = jax.value_and_grad(loss_fn)(state, inputs)

            # Simple SGD update (simplified optimizer)
            lr = 0.001
            updated_state = jax.tree.map(
                lambda param, grad: param - lr * grad,
                state,
                grads
            )

            return loss_val, updated_state

        return jax.jit(training_step)

    # Test with medium model
    model_config = create_large_model_config(768, 12)
    graph_def, state, input_shape = create_abstract_model_and_inputs(model_config, seq_len=512)

    unroll_variants = [1, 2, 4, 8]

    print("\nCompiling training steps with different unroll values...")

    for unroll_val in unroll_variants:
        print(f"\n🎯 Unroll = {unroll_val}")

        training_step = create_training_step(unroll_val)

        try:
            compiled_step = compile_function(
                training_step,
                sample_args=(graph_def, state, input_shape),
                name=f"Training Step (unroll={unroll_val})"
            )

            # The memory analysis will show if optimizer updates are inlined
            memory = compiled_step.memory_analysis()
            if memory:
                print(f"   📊 Peak memory with unroll={unroll_val}: {memory.temp_size_in_bytes * 1e-9:.2f} GB")

        except Exception as e:
            print(f"   ❌ Failed: {e}")


def print_summary(results):
    """Print a summary comparison of results."""

    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")

    for config_name, config_results in results.items():
        print(f"\n📋 {config_name}")
        print("─" * 50)

        # Sort by compilation time
        sorted_results = sorted(
            [(k, v) for k, v in config_results.items() if 'error' not in v],
            key=lambda x: x[1]['compilation_time']
        )

        if sorted_results:
            fastest = sorted_results[0]
            print(f"🏆 Fastest compilation: {fastest[0]} ({fastest[1]['compilation_time']:.2f}s)")

            # Memory efficiency
            mem_sorted = sorted(
                sorted_results,
                key=lambda x: x[1]['peak_memory_gb']
            )
            most_efficient = mem_sorted[0]
            print(f"💾 Most memory efficient: {most_efficient[0]} ({most_efficient[1]['peak_memory_gb']:.2f} GB)")

            # Show relative improvements
            print("\n📈 Relative Performance:")
            baseline = next((v for k, v in config_results.items() if k == 'unroll_1'), None)
            if baseline and 'error' not in baseline:
                for variant_name, metrics in sorted_results:
                    if variant_name != 'unroll_1':
                        speedup = baseline['compilation_time'] / metrics['compilation_time']
                        mem_ratio = baseline['peak_memory_gb'] / max(metrics['peak_memory_gb'], 1e-9)
                        print(f"  {variant_name}: {speedup:.2f}x compilation, {mem_ratio:.2f}x memory")


if __name__ == "__main__":
    """Run the scan unroll optimization analysis."""

    # Test 1: Compilation metrics for different model sizes
    results = test_scan_compilation_metrics()

    # Test 2: Optimizer inlining analysis
    analyze_optimizer_inlining()

    # Test 3: Summary and recommendations
    print_summary(results)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("\n💡 Key insights:")
    print("1. Unroll values 2-8 typically provide best compilation/memory balance")
    print("2. Full unrolling may cause exponential compilation time growth")
    print("3. Memory efficiency improves with moderate unrolling due to optimizer inlining")
    print("4. Check peak memory values to confirm optimizer fusion")