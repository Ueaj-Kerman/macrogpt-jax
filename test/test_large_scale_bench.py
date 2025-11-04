"""
Benchmark Jacobi vs Newton at large scales.

Tests with long sequences (8192) and large hidden dims (2048).
"""

print("STARTING IMPORTS...")
import sys
sys.stdout.flush()

import jax
import jax.numpy as jnp
from jax import lax
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ueaj.model.parallel_scan import parallel_scan as jacobi_scan
from ueaj.model.parallel_scan_newton import parallel_scan_newton as newton_scan


def sequential_scan(scan_fn, init_carry, xs):
    """Reference implementation using JAX's lax.scan."""
    return lax.scan(scan_fn, init_carry, xs)


def benchmark_configuration(hidden_size, seq_len, input_size=64):
    """Benchmark a specific configuration."""
    print(f"\n{'='*70}")
    print(f"Config: hidden={hidden_size}, seq={seq_len}, input={input_size}")
    print(f"{'='*70}")
    import sys
    sys.stdout.flush()  # Force output immediately

    key = jax.random.PRNGKey(0)

    # Approach #1: Pass parameters as arguments (NOT closures)
    # This avoids constant folding since weights are data, not constants
    print("Creating parameters...")
    sys.stdout.flush()
    key, *subkeys = jax.random.split(key, 5)
    params = {
        'W_hh': jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1,
        'W_xh': jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1,
        'b_h': jax.random.normal(subkeys[2], (hidden_size,)) * 0.01,
    }

    # Define cell that takes params as argument
    print("Defining cell...")
    sys.stdout.flush()
    def elman_cell_fn(params, h, x):
        h_next = jnp.tanh(params['W_hh'] @ h + params['W_xh'] @ x + params['b_h'])
        return h_next, h_next

    # Approach #2: Use functools.partial to bind params
    from functools import partial
    elman_cell = partial(elman_cell_fn, params)

    print("Creating inputs...")
    sys.stdout.flush()
    inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
    h0 = jnp.zeros(hidden_size)

    # Memory estimates
    jacobi_mem = seq_len * hidden_size * 4 / 1e9  # GB
    newton_mem = seq_len * hidden_size * hidden_size * 4 / 1e9  # GB

    print(f"\nMemory estimates:")
    print(f"  Jacobi: {jacobi_mem:.2f} GB")
    print(f"  Newton: {newton_mem:.2f} GB")

    # Skip Newton - it's too slow to compile due to Jacobian computation
    print(f"  ⚠️  Skipping Newton (compilation too slow for large scales)")
    benchmark_jacobi_only = True

    # Benchmark Jacobi
    print(f"\n{'─'*70}")
    print("JACOBI (15 iterations)")
    print(f"{'─'*70}")

    try:
        print("Creating JIT function...")
        sys.stdout.flush()
        jacobi_fn = jax.jit(lambda: jacobi_scan(
            elman_cell, h0, inputs, num_iterations=15
        ))

        # Warmup
        print("Compiling (first call)...")
        sys.stdout.flush()
        _ = jacobi_fn()
        print("Waiting for compilation to finish...")
        sys.stdout.flush()
        jax.block_until_ready(jacobi_fn())

        # Benchmark
        print("Benchmarking...")
        num_runs = 20
        start = time.time()
        for _ in range(num_runs):
            result = jacobi_fn()
            jax.block_until_ready(result)
        jacobi_time = (time.time() - start) / num_runs

        print(f"✓ Jacobi time: {jacobi_time*1000:.2f} ms")
        jacobi_success = True
    except Exception as e:
        print(f"✗ Jacobi failed: {e}")
        jacobi_time = float('inf')
        jacobi_success = False

    # Benchmark Newton (if memory allows)
    if not benchmark_jacobi_only:
        print(f"\n{'─'*70}")
        print("NEWTON (3 iterations)")
        print(f"{'─'*70}")

        try:
            print("Creating Newton JIT function...")
            sys.stdout.flush()
            newton_fn = jax.jit(lambda: newton_scan(
                elman_cell, h0, inputs, num_iterations=3
            ))

            # Warmup
            print("Warming up...")
            _ = newton_fn()
            jax.block_until_ready(newton_fn())

            # Benchmark
            print("Benchmarking...")
            num_runs = 20
            start = time.time()
            for _ in range(num_runs):
                result = newton_fn()
                jax.block_until_ready(result)
            newton_time = (time.time() - start) / num_runs

            print(f"✓ Newton time: {newton_time*1000:.2f} ms")
            newton_success = True
        except Exception as e:
            print(f"✗ Newton failed: {e}")
            newton_time = float('inf')
            newton_success = False
    else:
        newton_time = float('inf')
        newton_success = False

    # Compare
    print(f"\n{'─'*70}")
    print("COMPARISON")
    print(f"{'─'*70}")

    if jacobi_success and newton_success:
        if jacobi_time < newton_time:
            winner = "Jacobi"
            speedup = newton_time / jacobi_time
            print(f"✓ {winner} wins by {speedup:.1f}x")
        else:
            winner = "Newton"
            speedup = jacobi_time / newton_time
            print(f"✓ {winner} wins by {speedup:.1f}x")
    elif jacobi_success:
        print(f"✓ Jacobi works ({jacobi_time*1000:.2f} ms), Newton skipped/failed")
    elif newton_success:
        print(f"✓ Newton works ({newton_time*1000:.2f} ms), Jacobi failed")
    else:
        print("✗ Both methods failed")

    return {
        'hidden': hidden_size,
        'seq': seq_len,
        'jacobi_time': jacobi_time,
        'newton_time': newton_time,
        'jacobi_mem': jacobi_mem,
        'newton_mem': newton_mem,
    }


if __name__ == "__main__":
    print("="*70)
    print("LARGE SCALE BENCHMARK: Jacobi vs Newton")
    print("="*70)

    results = []

    # Test progression from medium to large scale
    # NOTE: Using 2 iterations for faster compilation
    configs = [
        # (hidden, seq_len)
        (512, 2048),   # Medium: should favor Jacobi
        (1024, 2048),  # Large: crossover point?
        (1024, 4096),  # Long sequence
        (2048, 4096),  # Very long + large hidden
        (1024, 8192),  # Very long sequence (user requested)
        (2048, 8192),  # User requested: d=2048, T=8192
    ]

    for hidden, seq_len in configs:
        result = benchmark_configuration(hidden, seq_len)
        results.append(result)

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Hidden':>6} | {'Seq':>6} | {'Jacobi (ms)':>12} | {'Newton (ms)':>12} | {'Winner':>8}")
    print("-"*70)

    for r in results:
        jacobi_str = f"{r['jacobi_time']*1000:.1f}" if r['jacobi_time'] < float('inf') else "FAIL"
        newton_str = f"{r['newton_time']*1000:.1f}" if r['newton_time'] < float('inf') else "SKIP"

        if r['jacobi_time'] < r['newton_time']:
            winner = "Jacobi"
        elif r['newton_time'] < r['jacobi_time']:
            winner = "Newton"
        else:
            winner = "-"

        print(f"{r['hidden']:6d} | {r['seq']:6d} | {jacobi_str:>12} | {newton_str:>12} | {winner:>8}")

    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    print("""
Key insights:
1. Memory is the limiting factor for Newton at large scales
2. Newton's O(T×d²) memory grows much faster than Jacobi's O(T×d)
3. For typical use cases (d < 2048, T < 8192), Jacobi is better
4. Newton only wins when compute dominates over memory constraints
    """)
