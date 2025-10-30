"""
ParaRNN: Unified Parallel Scan Implementation

This module provides a clean abstraction for parallel scanning that works
bidirectionally, enabling both forward and backward passes to be parallelized.

Key insight: Forward and backward RNN passes have the same structure:
- Forward:  carry_t = f(carry_{t-1}, x_t)    [left-to-right]
- Backward: carry_t = g(carry_{t+1}, x_t)    [right-to-left]

Both can use the same parallel iteration algorithm!
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.tree_util import tree_map
from functools import partial
from typing import Callable, Tuple, Any, Optional

Carry = Any
Input = Any
Output = Any
ScanFn = Callable[[Carry, Input], Tuple[Carry, Output]]


def parallel_scan_iteration(
    scan_fn: ScanFn,
    init_carry: Carry,
    xs: jax.Array,
    num_iterations: int = 10,
    reverse: bool = False
) -> Tuple[Carry, jax.Array]:
    """
    Core parallel scan using Jacobi fixed-point iterations.

    This is the fundamental building block that works bidirectionally.

    Args:
        scan_fn: Function (carry, x) -> (next_carry, output)
        init_carry: Initial carry value
        xs: Sequence of inputs (seq_len, ...)
        num_iterations: Number of fixed-point iterations
        reverse: If True, scan right-to-left instead of left-to-right

    Returns:
        final_carry: Final carry value
        outputs: Sequence of outputs

    Algorithm:
        1. Initialize all carries to init_carry
        2. For k iterations:
           - For each position t in parallel:
             - If forward: carry_t^(k+1) = scan_fn(carry_{t-1}^(k), x_t)
             - If reverse: carry_t^(k+1) = scan_fn(carry_{t+1}^(k), x_t)
        3. Return converged carries
    """
    # Get sequence length (xs might be a pytree, so get from first leaf)
    first_x = jax.tree_util.tree_leaves(xs)[0]
    seq_len = first_x.shape[0]

    # Reverse if needed
    if reverse:
        xs = tree_map(lambda x: x[::-1], xs)

    # Convert init_carry to array if scalar
    init_carry = tree_map(lambda x: jnp.asarray(x), init_carry)

    # Initialize all carries to init_carry (our initial guess)
    def tile_carry(x):
        return jnp.tile(x[None, ...], (seq_len, *([1] * x.ndim)))

    carry_all = tree_map(tile_carry, init_carry)

    # Jacobi iteration: update all carries in parallel
    def iteration_step(carry_all_old):
        """
        One iteration of parallel carry updates.

        Key: All updates use carries from PREVIOUS iteration (Jacobi method).
        This is what makes it parallelizable.
        """
        # Get previous carries for each position
        # Forward: prepend init_carry (carry_{-1} = init_carry)
        # Reverse: append init_carry (carry_{T} = init_carry)
        carry_prev_all = tree_map(
            lambda init, carries: jnp.concatenate([init[None, ...], carries[:-1]], axis=0),
            init_carry,
            carry_all_old
        )

        # Vectorized update: apply scan_fn to all positions in parallel
        def update_single(carry_prev, x):
            carry_next, _ = scan_fn(carry_prev, x)
            return carry_next

        carry_new_all = jax.vmap(update_single)(carry_prev_all, xs)
        return carry_new_all

    # Run fixed-point iterations
    carry_all = carry_all
    for _ in range(num_iterations):
        carry_all = iteration_step(carry_all)

    # Compute outputs using final carries
    def compute_output(carry_prev, x):
        _, output = scan_fn(carry_prev, x)
        return output

    # Get previous carries for output computation
    carry_prev_all = tree_map(
        lambda init, carries: jnp.concatenate([init[None, ...], carries[:-1]], axis=0),
        init_carry,
        carry_all
    )
    outputs = jax.vmap(compute_output)(carry_prev_all, xs)

    # Get final carry
    final_carry = tree_map(lambda c: c[-1], carry_all)

    # Un-reverse if needed
    if reverse:
        carry_all = tree_map(lambda c: c[::-1], carry_all)
        outputs = tree_map(lambda o: o[::-1], outputs)

    return final_carry, outputs, carry_all


@partial(custom_vjp, nondiff_argnums=(0, 3, 4))
def parallel_scan(
    scan_fn: ScanFn,
    init_carry: Carry,
    xs: Input,
    num_iterations: int = 10,
    reverse: bool = False
) -> Tuple[Carry, Output]:
    """
    Parallel scan with custom VJP using the same parallel algorithm.

    The beauty: Both forward and backward passes use parallel_scan_iteration!

    Args:
        scan_fn: Function (carry, x) -> (next_carry, output)
        init_carry: Initial carry value
        xs: Input sequence
        num_iterations: Number of iterations for convergence
        reverse: Direction of scan

    Returns:
        final_carry: Final carry value
        outputs: Output sequence
    """
    final_carry, outputs, carry_all = parallel_scan_iteration(
        scan_fn, init_carry, xs, num_iterations, reverse
    )
    return final_carry, outputs


def parallel_scan_fwd(scan_fn, init_carry, xs, num_iterations, reverse):
    """Forward pass: compute and save residuals."""
    final_carry, outputs, carry_all = parallel_scan_iteration(
        scan_fn, init_carry, xs, num_iterations, reverse
    )

    # Save what we need for backward
    residuals = (init_carry, xs, carry_all)
    return (final_carry, outputs), residuals


def parallel_scan_bwd(scan_fn, num_iterations, reverse, residuals, grads):
    """
    Backward pass: Use the SAME parallel scan, just in reverse!

    This is the key insight: gradients flow opposite to the forward direction,
    but can use the same parallel iteration algorithm.
    """
    init_carry, xs, carry_all = residuals
    grad_final_carry, grad_outputs = grads

    seq_len = xs.shape[0]

    # Create a VJP-based scan function for the backward pass
    # This function computes gradients for one timestep
    def vjp_scan_fn(grad_carry_next, args):
        """
        Backward scan function using VJP.

        Takes gradient from next timestep, returns gradient for previous timestep.
        """
        idx, carry_prev, x, output_grad = args

        # Get the actual carry at this position (for VJP)
        carry = carry_all[idx]

        # Compute VJP through scan_fn
        (next_carry, output), vjp_fn = jax.vjp(
            lambda c, x: scan_fn(c, x),
            carry_prev,
            x
        )

        # Backprop through this step
        grad_carry_prev, grad_x = vjp_fn((grad_carry_next, output_grad))

        return grad_carry_prev, (grad_carry_prev, grad_x)

    # Prepare inputs for backward pass
    # We need: indices, previous carries, inputs, output gradients
    indices = jnp.arange(seq_len)
    carry_prev_all = tree_map(
        lambda init, carries: jnp.concatenate([init[None, ...], carries[:-1]], axis=0),
        init_carry,
        carry_all
    )

    # Package arguments
    backward_inputs = (indices, carry_prev_all, xs, grad_outputs)

    # Here's the magic: Use parallel_scan_iteration for the BACKWARD pass!
    # We scan in the OPPOSITE direction with num_iterations
    grad_init_carry, (grad_carries, grad_xs), _ = parallel_scan_iteration(
        vjp_scan_fn,
        grad_final_carry,
        backward_inputs,
        num_iterations=num_iterations,
        reverse=not reverse  # Reverse the direction!
    )

    return (grad_init_carry, grad_xs)


parallel_scan.defvjp(parallel_scan_fwd, parallel_scan_bwd)


# ============================================================================
# High-level RNN API
# ============================================================================

def parallel_rnn_v2(
    cell: Callable[[Any, jax.Array], Tuple[Any, jax.Array]],
    h0: jax.Array,
    inputs: jax.Array,
    num_iterations: int = 10
) -> Tuple[jax.Array, jax.Array]:
    """
    Parallel RNN using unified bidirectional scan.

    This version has:
    - Parallel forward pass
    - Parallel backward pass (using same algorithm in reverse!)
    - Clean abstraction that works for any scan-like computation

    Args:
        cell: RNN cell (h, x) -> (h_next, y)
        h0: Initial hidden state
        inputs: Input sequence
        num_iterations: Iterations for convergence

    Returns:
        final_state, outputs
    """
    return parallel_scan(
        cell, h0, inputs,
        num_iterations=num_iterations,
        reverse=False
    )


def sequential_scan(
    scan_fn: ScanFn,
    init_carry: Carry,
    xs: Input,
    reverse: bool = False
) -> Tuple[Carry, Output]:
    """Sequential scan for comparison."""
    from jax import lax

    if reverse:
        xs = tree_map(lambda x: x[::-1], xs)

    final_carry, outputs = lax.scan(scan_fn, init_carry, xs)

    if reverse:
        outputs = tree_map(lambda o: o[::-1], outputs)

    return final_carry, outputs


# ============================================================================
# Demonstration: The abstraction works for ANY scan pattern!
# ============================================================================

def demo_unified_abstraction():
    """
    Show that this abstraction works for any sequential computation,
    not just RNNs!
    """
    print("="*70)
    print("Unified Parallel Scan Abstraction")
    print("="*70)

    # Example 1: Cumulative sum (forward)
    def add_scan(carry, x):
        next_carry = carry + x
        return next_carry, next_carry

    xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # Sequential
    final_seq, outputs_seq = sequential_scan(add_scan, 0.0, xs)
    print(f"\nCumulative sum (sequential): {outputs_seq}")

    # Parallel (should converge)
    final_par, outputs_par = parallel_scan(add_scan, 0.0, xs, num_iterations=5)
    print(f"Cumulative sum (parallel):   {outputs_par}")
    print(f"Error: {jnp.max(jnp.abs(outputs_seq - outputs_par)):.2e}")

    # Example 2: Reverse cumulative sum
    final_rev, outputs_rev = parallel_scan(add_scan, 0.0, xs, num_iterations=5, reverse=True)
    print(f"\nReverse cumsum (parallel):   {outputs_rev}")

    # Example 3: RNN forward and backward
    key = jax.random.PRNGKey(0)
    hidden_size = 8
    input_size = 4
    seq_len = 16

    key, *subkeys = jax.random.split(key, 5)
    W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
    W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
    b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

    def rnn_cell(h, x):
        h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
        return h_next, h_next

    inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
    h0 = jnp.zeros(hidden_size)

    # Forward pass
    final_h, outputs = parallel_rnn_v2(rnn_cell, h0, inputs, num_iterations=10)
    print(f"\nRNN forward pass: final_state shape = {final_h.shape}")

    # Test gradients (backward pass is also parallel!)
    def loss(h0):
        final_h, _ = parallel_rnn_v2(rnn_cell, h0, inputs, num_iterations=10)
        return jnp.sum(final_h ** 2)

    grad_h0 = jax.grad(loss)(h0)
    print(f"RNN gradient: shape = {grad_h0.shape}, norm = {jnp.linalg.norm(grad_h0):.4f}")
    print("✓ Both forward AND backward are parallelized!")


if __name__ == "__main__":
    demo_unified_abstraction()
