"""
ParaRNN Demo: Comparing parallel and sequential RNN execution.

This script demonstrates how to use the ParaRNN implementation with
various RNN cell types (Elman, GRU, LSTM) and compares:
1. Correctness vs sequential scan
2. Gradient computation
3. Potential speedups
"""

import jax
import jax.numpy as jnp
from jax import grad, jit
import time
from typing import Tuple

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ueaj.model.pararnn import parallel_rnn, sequential_rnn


# ============================================================================
# Example RNN Cells
# ============================================================================

def make_elman_cell(W_hh: jax.Array, W_xh: jax.Array, b_h: jax.Array):
    """
    Create a simple Elman RNN cell.

    h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b_h)
    """
    def cell(h, x):
        h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
        return h_next, h_next  # output = hidden state
    return cell


def make_gru_cell(
    W_xz: jax.Array, W_hz: jax.Array, b_z: jax.Array,
    W_xr: jax.Array, W_hr: jax.Array, b_r: jax.Array,
    W_xh: jax.Array, W_hh: jax.Array, b_h: jax.Array,
):
    """
    Create a GRU cell.

    z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1} + b_z)  # update gate
    r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1} + b_r)  # reset gate
    h_tilde = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}) + b_h)
    h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    """
    def cell(h, x):
        z = jax.nn.sigmoid(W_xz @ x + W_hz @ h + b_z)
        r = jax.nn.sigmoid(W_xr @ x + W_hr @ h + b_r)
        h_tilde = jnp.tanh(W_xh @ x + W_hh @ (r * h) + b_h)
        h_next = (1 - z) * h + z * h_tilde
        return h_next, h_next
    return cell


def make_linear_cell(A: jax.Array, B: jax.Array):
    """
    Create a linear RNN cell (for testing).

    h_t = A @ h_{t-1} + B @ x_t

    This should be exactly solvable in one iteration.
    """
    def cell(h, x):
        h_next = A @ h + B @ x
        return h_next, h_next
    return cell


# ============================================================================
# Test Functions
# ============================================================================

def test_correctness(cell, h0, inputs, num_iterations=10):
    """Compare parallel and sequential RNN outputs."""
    print("\n" + "="*70)
    print("Testing Correctness")
    print("="*70)

    # Sequential
    final_h_seq, outputs_seq = sequential_rnn(cell, h0, inputs)

    # Parallel
    final_h_par, outputs_par = parallel_rnn(
        cell, h0, inputs,
        method="iterative",
        num_iterations=num_iterations
    )

    # Compare
    h_error = jnp.max(jnp.abs(final_h_seq - final_h_par))
    output_error = jnp.max(jnp.abs(outputs_seq - outputs_par))

    print(f"Final hidden state error: {h_error:.2e}")
    print(f"Output sequence error:    {output_error:.2e}")
    print(f"Number of iterations:     {num_iterations}")

    if h_error < 1e-3:
        print("✓ Outputs match!")
    else:
        print("✗ Outputs differ significantly")

    return h_error, output_error


def test_gradients(cell, h0, inputs):
    """Compare gradients from parallel and sequential versions."""
    print("\n" + "="*70)
    print("Testing Gradients")
    print("="*70)

    # Define a simple loss: sum of final hidden state
    def loss_fn(h0, inputs, rnn_fn):
        final_h, _ = rnn_fn(cell, h0, inputs)
        return jnp.sum(final_h)

    # Compute gradients
    grad_h0_seq = grad(lambda h0: loss_fn(h0, inputs, sequential_rnn))(h0)
    grad_h0_par = grad(lambda h0: loss_fn(h0, inputs, parallel_rnn))(h0)

    grad_error = jnp.max(jnp.abs(grad_h0_seq - grad_h0_par))
    print(f"Gradient error (w.r.t. h0): {grad_error:.2e}")

    if grad_error < 1e-3:
        print("✓ Gradients match!")
    else:
        print("✗ Gradients differ significantly")

    return grad_error


def benchmark_speed(cell, h0, inputs, num_runs=10):
    """Benchmark sequential vs parallel execution."""
    print("\n" + "="*70)
    print("Benchmarking Speed")
    print("="*70)

    # JIT compile both versions
    seq_fn = jit(lambda: sequential_rnn(cell, h0, inputs))
    par_fn = jit(lambda: parallel_rnn(cell, h0, inputs, num_iterations=5))

    # Warmup
    _ = seq_fn()
    _ = par_fn()
    jax.block_until_ready(seq_fn())
    jax.block_until_ready(par_fn())

    # Benchmark sequential
    start = time.time()
    for _ in range(num_runs):
        result = seq_fn()
        jax.block_until_ready(result)
    seq_time = (time.time() - start) / num_runs

    # Benchmark parallel
    start = time.time()
    for _ in range(num_runs):
        result = par_fn()
        jax.block_until_ready(result)
    par_time = (time.time() - start) / num_runs

    print(f"Sequential time: {seq_time*1000:.3f} ms")
    print(f"Parallel time:   {par_time*1000:.3f} ms")
    print(f"Speedup:         {seq_time/par_time:.2f}x")

    # Note: The current implementation may not show speedup because:
    # 1. The iteration_step still uses scan (not fully parallel)
    # 2. Overhead of multiple iterations
    # 3. Small problem sizes
    print("\nNote: True speedups require longer sequences and/or")
    print("      optimized parallel implementation with associative_scan")


# ============================================================================
# Main Demo
# ============================================================================

def main():
    print("ParaRNN Demo")
    print("="*70)

    # Random seed
    key = jax.random.PRNGKey(42)

    # Problem dimensions
    input_size = 8
    hidden_size = 16
    seq_len = 32

    # Generate random inputs
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (seq_len, input_size))
    h0 = jnp.zeros(hidden_size)

    # ========================================================================
    # Test 1: Linear RNN (should converge in 1 iteration)
    # ========================================================================
    print("\n" + "#"*70)
    print("# Test 1: Linear RNN")
    print("#"*70)

    key, *subkeys = jax.random.split(key, 3)
    A = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
    B = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
    linear_cell = make_linear_cell(A, B)

    print("\nConvergence with increasing iterations:")
    for num_iter in [1, 5, 10, 20]:
        h_err, out_err = test_correctness(linear_cell, h0, inputs, num_iterations=num_iter)

    print("\nNote: Even linear RNNs need O(seq_len) iterations for Jacobi method")
    print("      to fully propagate information through the sequence.")
    print("      ParaRNN's speedup comes from parallelizing WITHIN each iteration.")

    # ========================================================================
    # Test 2: Elman RNN (nonlinear)
    # ========================================================================
    print("\n" + "#"*70)
    print("# Test 2: Elman RNN (Nonlinear)")
    print("#"*70)

    key, *subkeys = jax.random.split(key, 4)
    W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
    W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
    b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01
    elman_cell = make_elman_cell(W_hh, W_xh, b_h)

    for num_iter in [1, 5, 10]:
        test_correctness(elman_cell, h0, inputs, num_iterations=num_iter)

    # ========================================================================
    # Test 3: GRU (more complex nonlinear)
    # ========================================================================
    print("\n" + "#"*70)
    print("# Test 3: GRU (Complex Nonlinear)")
    print("#"*70)

    key, *subkeys = jax.random.split(key, 10)
    W_xz = jax.random.normal(subkeys[0], (hidden_size, input_size)) * 0.1
    W_hz = jax.random.normal(subkeys[1], (hidden_size, hidden_size)) * 0.1
    b_z = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01
    W_xr = jax.random.normal(subkeys[3], (hidden_size, input_size)) * 0.1
    W_hr = jax.random.normal(subkeys[4], (hidden_size, hidden_size)) * 0.1
    b_r = jax.random.normal(subkeys[5], (hidden_size,)) * 0.01
    W_xh = jax.random.normal(subkeys[6], (hidden_size, input_size)) * 0.1
    W_hh = jax.random.normal(subkeys[7], (hidden_size, hidden_size)) * 0.1
    b_h = jax.random.normal(subkeys[8], (hidden_size,)) * 0.01

    gru_cell = make_gru_cell(W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h)

    test_correctness(gru_cell, h0, inputs, num_iterations=10)

    # ========================================================================
    # Test 4: Gradient Correctness
    # ========================================================================
    print("\n" + "#"*70)
    print("# Test 4: Gradient Correctness")
    print("#"*70)

    test_gradients(elman_cell, h0, inputs)

    # ========================================================================
    # Test 5: Speed Benchmark
    # ========================================================================
    print("\n" + "#"*70)
    print("# Test 5: Speed Benchmark")
    print("#"*70)

    benchmark_speed(elman_cell, h0, inputs, num_runs=100)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
The current implementation demonstrates:
✓ Correct fixed-point iteration for parallel RNN execution
✓ Custom VJP for gradient computation
✓ Works with arbitrary RNN cells (Elman, GRU, linear)

Limitations and future improvements:
- Need true parallel implementation using associative_scan
- Current version uses scan inside iterations (still sequential)
- For full speedup, need:
  1. Linearization of nonlinear operations
  2. Parallel prefix scan with associative operator
  3. Longer sequences (>1000s of timesteps)
  4. Multi-GPU/TPU execution

The paper's 665x speedup comes from:
- True parallel execution with associative_scan
- Optimized Newton iterations
- Large-scale distributed training (7B parameter models)
""")


if __name__ == "__main__":
    main()
