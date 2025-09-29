#!/usr/bin/env python3
"""
Simple test to compare scan with different unroll values.

This directly modifies the model's get_activations method to use different unroll values
and measures compilation time and memory usage.
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


def create_model_with_unroll(unroll_value: int, model_config):
    """Create a model with modified scan that uses the specified unroll value."""

    def modified_get_activations(self, input_ids: jax.Array, mesh=None, **kwargs):
        """Modified get_activations that uses scan with specified unroll."""
        # Embed tokens
        act0 = self.embed_tokens(input_ids)
        kwargs = self.default_kwargs(*input_ids.shape, **kwargs)

        # Modified scan with unroll parameter
        def scan_fn(carry, layer):
            out = layer(carry, mesh=mesh, **kwargs)
            return out.astype(carry.dtype), None

        if mesh is not None:
            act0 = jax.lax.with_sharding_constraint(act0, jax.NamedSharding(mesh, None))

        # Use JAX scan directly with unroll parameter
        act, _ = jax.lax.scan(
            scan_fn,
            act0,
            self.layers,
            unroll=unroll_value
        )
        return act

    # Create model and monkey-patch the method
    model = model_config(rngs=rng.Rngs(0))

    # Replace the get_activations method
    import types
    model.get_activations = types.MethodType(modified_get_activations, model)

    return model


def test_unroll_compilation():
    """Test compilation time and memory for different unroll values."""

    print("="*80)
    print("SCAN UNROLL COMPARISON TEST")
    print("="*80)

    # Test configurations
    test_configs = [
        {"name": "UEAJ_150M", "config": configs.UEAJ_150M, "batch_size": 2, "seq_len": 1024},
    ]

    unroll_values = [1, 2, 4, 8]

    for test_config in test_configs:
        print(f"\n{'─'*60}")
        print(f"Testing {test_config['name']}")
        print(f"{'─'*60}")

        results = {}

        for unroll_val in unroll_values:
            print(f"\n🔄 Testing unroll={unroll_val}...")

            try:
                # Create model with specific unroll value
                model = create_model_with_unroll(unroll_val, test_config['config'])

                # Split for JAX compilation
                graph_def, state = nnx.split(model)

                # Create abstract inputs
                input_shape = jax.ShapeDtypeStruct(
                    (test_config['batch_size'], test_config['seq_len']),
                    jnp.int32
                )

                # Create function to compile
                @jax.jit
                def forward_pass(graph_def, state, inputs):
                    model = nnx.merge(graph_def, state)
                    return model.get_activations(inputs)

                # Compile and measure
                start_time = time.time()
                compiled_fn = compile_function(
                    forward_pass,
                    sample_args=(graph_def, state, input_shape),
                    name=f"{test_config['name']} unroll={unroll_val}"
                )
                compilation_time = time.time() - start_time

                # Get metrics
                cost = compiled_fn.cost_analysis()
                memory = compiled_fn.memory_analysis()

                results[unroll_val] = {
                    'compilation_time': compilation_time,
                    'flops': cost.get('flops', 0) if cost else 0,
                    'peak_memory_gb': memory.temp_size_in_bytes * 1e-9 if memory else 0,
                    'output_memory_gb': memory.output_size_in_bytes * 1e-9 if memory else 0,
                }

                print(f"   ✓ Compilation: {compilation_time:.2f}s")
                if memory:
                    total_memory = memory.temp_size_in_bytes + memory.output_size_in_bytes
                    print(f"   ✓ Total VRAM: {total_memory * 1e-9:.2f} GB")

            except Exception as e:
                print(f"   ❌ Failed: {e}")
                results[unroll_val] = {'error': str(e)}

        # Print comparison
        print(f"\n📊 Results Summary for {test_config['name']}:")
        print("─" * 50)

        successful_results = {k: v for k, v in results.items() if 'error' not in v}

        if successful_results:
            baseline = successful_results.get(1)

            print(f"{'Unroll':<8} {'Time (s)':<10} {'Memory (GB)':<12} {'Speedup':<10}")
            print("─" * 50)

            for unroll_val in sorted(successful_results.keys()):
                result = successful_results[unroll_val]
                total_mem = result['peak_memory_gb'] + result['output_memory_gb']

                if baseline and unroll_val != 1:
                    speedup = baseline['compilation_time'] / result['compilation_time']
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "baseline"

                print(f"{unroll_val:<8} {result['compilation_time']:<10.2f} {total_mem:<12.2f} {speedup_str:<10}")

        # Print insights
        print(f"\n💡 Insights:")
        if len(successful_results) >= 2:
            unroll_times = [(k, v['compilation_time']) for k, v in successful_results.items()]
            fastest = min(unroll_times, key=lambda x: x[1])
            print(f"   • Fastest compilation: unroll={fastest[0]} ({fastest[1]:.2f}s)")

            unroll_memory = [(k, v['peak_memory_gb'] + v['output_memory_gb'])
                           for k, v in successful_results.items()]
            most_efficient = min(unroll_memory, key=lambda x: x[1])
            print(f"   • Most memory efficient: unroll={most_efficient[0]} ({most_efficient[1]:.2f} GB)")
        else:
            print("   • Not enough successful results for comparison")


if __name__ == "__main__":
    test_unroll_compilation()

    print(f"\n{'='*80}")
    print("KEY FINDINGS:")
    print("• unroll=1: Standard scan (single while loop)")
    print("• unroll=2-8: Partial unrolling (better optimizer inlining)")
    print("• Higher unroll: May increase compilation time but reduce memory")
    print("• Memory reduction indicates successful optimizer inlining")
    print("="*80)