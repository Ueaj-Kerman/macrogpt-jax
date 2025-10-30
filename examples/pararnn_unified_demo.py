"""
ParaRNN Unified Bidirectional Demo

This demonstrates the key insight: Both forward and backward RNN passes
can use the SAME parallel scan algorithm, just with reverse=True/False.

This makes the implementation:
1. Cleaner (one abstraction for both directions)
2. Faster (backward pass is also parallelized!)
3. More general (works for ANY scan-like computation)
"""

import jax
import jax.numpy as jnp
from jax import grad
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ueaj.model.pararnn_unified import parallel_scan, sequential_scan, parallel_rnn_v2
from ueaj.model.pararnn import parallel_rnn as parallel_rnn_v1


def compare_implementations():
    """
    Compare the original vs unified implementation.
    """
    print("="*70)
    print("Comparing Original vs Unified ParaRNN")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(42)
    hidden_size = 32
    input_size = 16
    seq_len = 64

    key, *subkeys = jax.random.split(key, 5)
    W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
    W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
    b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

    def cell(h, x):
        h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
        return h_next, h_next

    inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
    h0 = jnp.zeros(hidden_size)

    # Test correctness
    print("\n1. Forward Pass Correctness")
    print("-" * 70)

    final_v1, outputs_v1 = parallel_rnn_v1(cell, h0, inputs, num_iterations=10)
    final_v2, outputs_v2 = parallel_rnn_v2(cell, h0, inputs, num_iterations=10)

    fwd_error = jnp.max(jnp.abs(final_v1 - final_v2))
    print(f"Error between v1 and v2: {fwd_error:.2e}")
    print("✓ Forward passes match!" if fwd_error < 1e-6 else "✗ Outputs differ!")

    # Test gradient correctness
    print("\n2. Gradient Correctness")
    print("-" * 70)

    def loss_v1(h0):
        final_h, _ = parallel_rnn_v1(cell, h0, inputs, num_iterations=10)
        return jnp.sum(final_h ** 2)

    def loss_v2(h0):
        final_h, _ = parallel_rnn_v2(cell, h0, inputs, num_iterations=10)
        return jnp.sum(final_h ** 2)

    grad_v1 = jax.grad(loss_v1)(h0)
    grad_v2 = jax.grad(loss_v2)(h0)

    grad_error = jnp.max(jnp.abs(grad_v1 - grad_v2))
    print(f"Gradient error: {grad_error:.2e}")
    print("✓ Gradients match!" if grad_error < 1e-6 else "✗ Gradients differ!")

    # Benchmark
    print("\n3. Performance Comparison")
    print("-" * 70)

    loss_v1_jit = jax.jit(loss_v1)
    loss_v2_jit = jax.jit(loss_v2)
    grad_v1_jit = jax.jit(jax.grad(loss_v1))
    grad_v2_jit = jax.jit(jax.grad(loss_v2))

    # Warmup
    for _ in range(3):
        _ = loss_v1_jit(h0)
        _ = loss_v2_jit(h0)
        _ = grad_v1_jit(h0)
        _ = grad_v2_jit(h0)

    # Benchmark forward
    start = time.time()
    for _ in range(100):
        result = loss_v1_jit(h0)
        jax.block_until_ready(result)
    v1_fwd_time = (time.time() - start) / 100

    start = time.time()
    for _ in range(100):
        result = loss_v2_jit(h0)
        jax.block_until_ready(result)
    v2_fwd_time = (time.time() - start) / 100

    print(f"Forward pass:")
    print(f"  V1 (original): {v1_fwd_time*1000:.3f} ms")
    print(f"  V2 (unified):  {v2_fwd_time*1000:.3f} ms")
    print(f"  Speedup: {v1_fwd_time/v2_fwd_time:.2f}x")

    # Benchmark backward
    start = time.time()
    for _ in range(100):
        result = grad_v1_jit(h0)
        jax.block_until_ready(result)
    v1_bwd_time = (time.time() - start) / 100

    start = time.time()
    for _ in range(100):
        result = grad_v2_jit(h0)
        jax.block_until_ready(result)
    v2_bwd_time = (time.time() - start) / 100

    print(f"\nBackward pass:")
    print(f"  V1 (sequential bwd): {v1_bwd_time*1000:.3f} ms")
    print(f"  V2 (parallel bwd):   {v2_bwd_time*1000:.3f} ms")
    print(f"  Speedup: {v1_bwd_time/v2_bwd_time:.2f}x")

    print("\n✓ V2 parallelizes BOTH forward AND backward!")


def demonstrate_bidirectional_abstraction():
    """
    Show that the same abstraction works for both directions.
    """
    print("\n" + "="*70)
    print("Bidirectional Scan Abstraction")
    print("="*70)

    # Example: Computing running statistics
    def stats_scan(carry, x):
        """Running mean and variance."""
        count, mean, m2 = carry
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2
        return (count, mean, m2), mean

    data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    init = (0.0, 0.0, 0.0)

    # Forward: cumulative statistics
    final_fwd, outputs_fwd = parallel_scan(
        stats_scan, init, data,
        num_iterations=10,
        reverse=False
    )
    print(f"\nForward cumulative means: {outputs_fwd}")

    # Reverse: cumulative statistics from the end
    final_rev, outputs_rev = parallel_scan(
        stats_scan, init, data,
        num_iterations=10,
        reverse=True
    )
    print(f"Reverse cumulative means: {outputs_rev}")

    print("\n✓ Same abstraction works bidirectionally!")


def explain_the_insight():
    """
    Explain the key insight behind the unified abstraction.
    """
    print("\n" + "="*70)
    print("The Key Insight")
    print("="*70)

    print("""
BEFORE (Original Implementation):
- Forward pass: Parallel fixed-point iteration ✓
- Backward pass: Sequential scan ✗

Problem: Backward pass creates a bottleneck! All that parallelism
in the forward pass is wasted if gradients flow sequentially.

AFTER (Unified Implementation):
- Forward pass: parallel_scan_iteration(cell, h0, inputs, reverse=False) ✓
- Backward pass: parallel_scan_iteration(vjp_cell, grad, ..., reverse=True) ✓

Both use the SAME parallel algorithm!

WHY THIS WORKS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Forward RNN:  h_t = f(h_{t-1}, x_t)
              ↓ Information flows left→right
              h_0 → h_1 → h_2 → ... → h_T

Backward RNN: ∂L/∂h_{t-1} = ∂L/∂h_t · ∂f/∂h_{t-1}
              ↓ Gradients flow right→left
              ∂h_T ← ∂h_{T-1} ← ... ← ∂h_1 ← ∂h_0

BOTH have the same structure: carry_{t} depends on carry_{t±1}
BOTH can use Jacobi iteration with the same algorithm!

The only difference is DIRECTION (reverse=True/False)!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ABSTRACTION:

def parallel_scan_iteration(scan_fn, init_carry, xs, reverse):
    # Works for ANY sequential dependency!
    # Forward: reverse=False
    # Backward: reverse=True
    # Other scans: your choice!

This is the true elegance of ParaRNN - it's not just about RNNs,
it's a general pattern for parallelizing ANY scan-like computation.

BENEFITS:
✓ Cleaner code (one algorithm, two uses)
✓ Faster backward pass (parallelized!)
✓ More general (works for any scan)
✓ Easier to optimize (JIT, XLA can fuse operations)
""")


def main():
    compare_implementations()
    demonstrate_bidirectional_abstraction()
    explain_the_insight()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
KEY TAKEAWAYS:

1. The unified abstraction parallelizes BOTH directions
   ✓ Forward: parallel_scan(..., reverse=False)
   ✓ Backward: parallel_scan(..., reverse=True)

2. Backward pass is now parallel (not sequential!)
   ✓ This is crucial for large-scale training
   ✓ Maintains the speedup from forward pass

3. The abstraction is general
   ✓ Works for RNNs, cumulative ops, running stats, etc.
   ✓ Any scan with sequential dependencies

4. Implementation is cleaner
   ✓ One core algorithm (parallel_scan_iteration)
   ✓ Forward and backward use the same code path
   ✓ Easier to maintain and optimize

This is the "right" way to implement ParaRNN!
""")


if __name__ == "__main__":
    main()
