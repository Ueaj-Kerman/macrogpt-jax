"""
ParaRNN Quick Start Example

Minimal example showing how to use ParaRNN with your own RNN cell.
"""

import jax
import jax.numpy as jnp
from functools import partial
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ueaj.model.pararnn import parallel_rnn, sequential_rnn


def main():
    # ========================================================================
    # 1. Define your RNN cell
    # ========================================================================
    # The cell takes (hidden_state, input) -> (next_hidden, output)
    # Here's a simple Elman RNN cell

    # Initialize parameters
    key = jax.random.PRNGKey(0)
    hidden_size = 32
    input_size = 16

    key, *subkeys = jax.random.split(key, 4)
    W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
    W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
    b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

    def rnn_cell(h, x):
        """h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t + b)"""
        h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
        return h_next, h_next  # (next_state, output)

    # ========================================================================
    # 2. Prepare inputs
    # ========================================================================
    seq_len = 64
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (seq_len, input_size))
    h0 = jnp.zeros(hidden_size)

    # ========================================================================
    # 3. Run with ParaRNN
    # ========================================================================
    print("Running ParaRNN...")
    final_state_par, outputs_par = parallel_rnn(
        cell=rnn_cell,
        h0=h0,
        inputs=inputs,
        num_iterations=10  # More iterations = more accurate
    )

    print(f"Final state shape: {final_state_par.shape}")
    print(f"Outputs shape: {outputs_par.shape}")

    # ========================================================================
    # 4. Compare with sequential (for verification)
    # ========================================================================
    print("\nComparing with sequential scan...")
    final_state_seq, outputs_seq = sequential_rnn(rnn_cell, h0, inputs)

    error = jnp.max(jnp.abs(final_state_seq - final_state_par))
    print(f"Error vs sequential: {error:.2e}")

    if error < 1e-3:
        print("✓ ParaRNN matches sequential!")
    else:
        print("✗ Consider increasing num_iterations")

    # ========================================================================
    # 5. Gradients work automatically!
    # ========================================================================
    print("\nTesting gradients...")

    def loss(h0):
        final_h, _ = parallel_rnn(rnn_cell, h0, inputs, num_iterations=10)
        return jnp.sum(final_h ** 2)

    grad_h0 = jax.grad(loss)(h0)
    print(f"Gradient shape: {grad_h0.shape}")
    print(f"Gradient norm: {jnp.linalg.norm(grad_h0):.4f}")
    print("✓ Gradients computed successfully!")

    # ========================================================================
    # 6. JIT compilation works too!
    # ========================================================================
    print("\nCompiling with JIT...")

    @jax.jit
    def run_pararnn(h0, inputs):
        return parallel_rnn(rnn_cell, h0, inputs, num_iterations=10)

    final_state_jit, outputs_jit = run_pararnn(h0, inputs)
    print("✓ JIT compilation successful!")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("Quick Start Summary")
    print("="*70)
    print("""
    ParaRNN is easy to use:

    1. Define your RNN cell: (h, x) -> (h_next, output)
    2. Call parallel_rnn(cell, h0, inputs, num_iterations)
    3. Enjoy parallel execution across time!

    Tips:
    - Start with num_iterations=10 for nonlinear RNNs
    - Use more iterations for longer sequences
    - Error < 1e-3 is typically good enough
    - Works with JAX transformations (jit, grad, vmap, etc.)
    """)


if __name__ == "__main__":
    main()
