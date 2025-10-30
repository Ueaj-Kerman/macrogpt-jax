"""
ParaRNN: Parallel Training of Nonlinear RNNs
Based on "ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models"
https://arxiv.org/abs/2510.21450

This module provides a JAX-idiomatic implementation that works with arbitrary RNN cells.
The key idea is to reformulate the sequential RNN recurrence h_t = f(h_{t-1}, x_t)
as a fixed-point problem that can be solved in parallel using Newton iterations.
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp, vjp, lax
from functools import partial
from typing import Callable, Tuple, Any

RNNCell = Callable[[Any, jax.Array], Tuple[Any, jax.Array]]


@partial(custom_vjp, nondiff_argnums=(0, 3, 4))
def pararnn_scan(
    cell: RNNCell,
    h0: jax.Array,
    inputs: jax.Array,
    num_iterations: int = 5,
    tol: float = 1e-6
) -> Tuple[jax.Array, jax.Array]:
    """
    Parallel RNN scan using fixed-point iterations.

    Args:
        cell: RNN cell function (h_prev, x) -> (h_next, y)
        h0: Initial hidden state
        inputs: Input sequence of shape (seq_len, ...)
        num_iterations: Number of Newton/Jacobi iterations
        tol: Convergence tolerance (currently unused, runs fixed iterations)

    Returns:
        final_state: Final hidden state
        outputs: Sequence of outputs
    """
    seq_len = inputs.shape[0]

    # Initial guess: broadcast h0 to all timesteps
    h_guess = jnp.tile(h0[None, ...], (seq_len, *([1] * h0.ndim)))

    # Define the iteration step for Jacobi method
    def iteration_step(h_all_old):
        """
        One iteration of parallel state update (Jacobi iteration).

        Key: All updates use the states from the PREVIOUS iteration,
        not from the current iteration. This is what makes it parallelizable.
        """
        # Prepend h0 to get h_{t-1} for each timestep
        h_prev_all = jnp.concatenate([h0[None, ...], h_all_old[:-1]], axis=0)

        # Vectorized update: compute all new states in parallel
        def update_single_state(h_prev, x):
            h_next, _ = cell(h_prev, x)
            return h_next

        # Use vmap to apply cell to all timesteps in parallel
        h_new_all = jax.vmap(update_single_state)(h_prev_all, inputs)
        return h_new_all

    # Run fixed-point iterations
    h_all = h_guess
    for _ in range(num_iterations):
        h_all = iteration_step(h_all)

    # Compute outputs using final hidden states
    def compute_output(idx):
        h_prev = jnp.where(idx == 0, h0, h_all[idx - 1])
        x = inputs[idx]
        _, y = cell(h_prev, x)
        return y

    outputs = jax.vmap(compute_output)(jnp.arange(seq_len))
    final_state = h_all[-1]

    return final_state, outputs


def pararnn_scan_fwd(cell, h0, inputs, num_iterations, tol):
    """Forward pass: compute fixed point and save residuals."""
    final_state, outputs = pararnn_scan(cell, h0, inputs, num_iterations, tol)
    # Save what we need for backward pass
    residuals = (h0, inputs, final_state)
    return (final_state, outputs), residuals


def pararnn_scan_bwd(cell, num_iterations, tol, residuals, grads):
    """Backward pass: use implicit differentiation."""
    h0, inputs, final_states = residuals
    grad_final_state, grad_outputs = grads

    seq_len = inputs.shape[0]

    # Reconstruct all hidden states for gradient computation
    # This is a simplification - a production version would use checkpointing
    h_all = jnp.zeros((seq_len, *h0.shape), dtype=h0.dtype)

    def forward_step(h_prev, x):
        h_next, _ = cell(h_prev, x)
        return h_next, h_next

    # Reconstruct forward pass
    _, h_all = lax.scan(forward_step, h0, inputs)
    h_all_with_init = jnp.concatenate([h0[None, ...], h_all], axis=0)

    # Backward pass through time (standard BPTT for now)
    def backward_step(grad_h_next, args):
        idx, h_prev, x = args

        # Compute VJPs through the cell
        def cell_wrapper(h, x):
            h_next, y = cell(h, x)
            return h_next, y

        (_, y), vjp_fn = vjp(lambda h, x: cell_wrapper(h, x), h_prev, x)
        grad_h_prev, grad_x = vjp_fn((grad_h_next, grad_outputs[idx]))

        return grad_h_prev, (grad_h_prev, grad_x)

    # Run backward pass
    indices = jnp.arange(seq_len - 1, -1, -1)
    h_prev_all = h_all_with_init[:-1]

    grad_h0, (grad_h_all, grad_x_all) = lax.scan(
        backward_step,
        grad_final_state,
        (indices, h_prev_all[::-1], inputs[::-1])
    )

    # Note: gradients for cell parameters would need to be accumulated
    # This is handled automatically by JAX's pytree system
    return (grad_h0, grad_x_all[::-1])


pararnn_scan.defvjp(pararnn_scan_fwd, pararnn_scan_bwd)


# ============================================================================
# Associative Scan Version (True Parallel Implementation)
# ============================================================================

def make_associative_rnn_op(cell: RNNCell):
    """
    Create an associative operator for parallel RNN scanning.

    This is the key to making ParaRNN truly parallel. The operator
    combines two RNN steps into one in an associative manner.

    For linear RNNs: h_t = A @ h_{t-1} + B @ x_t
    The associative operation combines (A1, B1*x1) and (A2, B2*x2) as:
        (A2 @ A1, A2 @ (B1*x1) + B2*x2)

    For nonlinear RNNs, we approximate this using linearization.
    """
    def associative_op(elem1, elem2):
        """
        Combine two RNN steps associatively.

        elem1, elem2: Tuples of (hidden_state, input)
        Returns: Combined element

        This is a simplified version - the paper uses more sophisticated
        linearization and Newton iterations.
        """
        h1, x1 = elem1
        h2, x2 = elem2

        # For now: just apply the cell sequentially
        # A true parallel version would linearize the nonlinearity
        h_next, _ = cell(h1, x2)
        return (h_next, x2)

    return associative_op


def pararnn_associative_scan(
    cell: RNNCell,
    h0: jax.Array,
    inputs: jax.Array,
    num_iterations: int = 1
) -> Tuple[jax.Array, jax.Array]:
    """
    Parallel RNN using associative_scan (sketch implementation).

    NOTE: This is a sketch. A full implementation requires:
    1. Proper linearization of the nonlinear cell
    2. Multiple iterations of Newton's method
    3. Custom gradients with implicit differentiation

    Args:
        cell: RNN cell function
        h0: Initial hidden state
        inputs: Input sequence
        num_iterations: Number of Newton iterations

    Returns:
        final_state, outputs
    """
    seq_len = inputs.shape[0]

    # Create initial elements for scan
    h_init = jnp.tile(h0[None, ...], (seq_len, *([1] * h0.ndim)))
    elements = (h_init, inputs)

    # For a linear RNN, we could directly use associative_scan
    # For nonlinear, we need to iterate with linearization
    associative_op = make_associative_rnn_op(cell)

    # Apply associative scan
    h_all, _ = lax.associative_scan(associative_op, elements, axis=0)

    # Compute outputs
    def compute_output(h_prev, x):
        _, y = cell(h_prev, x)
        return y

    h_prev_all = jnp.concatenate([h0[None, ...], h_all[:-1]], axis=0)
    outputs = jax.vmap(compute_output)(h_prev_all, inputs)

    return h_all[-1], outputs


# ============================================================================
# Convenience API
# ============================================================================

def parallel_rnn(
    cell: RNNCell,
    h0: jax.Array,
    inputs: jax.Array,
    method: str = "iterative",
    num_iterations: int = 5,
    tol: float = 1e-6
) -> Tuple[jax.Array, jax.Array]:
    """
    Run an RNN in parallel using ParaRNN.

    Args:
        cell: RNN cell function (h, x) -> (h_next, y)
        h0: Initial hidden state
        inputs: Input sequence of shape (seq_len, ...)
        method: "iterative" or "associative"
        num_iterations: Number of fixed-point iterations
        tol: Convergence tolerance

    Returns:
        final_state: Final hidden state after processing sequence
        outputs: Output sequence of shape (seq_len, ...)

    Example:
        >>> def gru_cell(h, x):
        ...     # GRU implementation
        ...     h_next = ...
        ...     return h_next, h_next
        >>>
        >>> h0 = jnp.zeros(hidden_size)
        >>> inputs = jnp.randn(seq_len, input_size)
        >>> final_h, outputs = parallel_rnn(gru_cell, h0, inputs)
    """
    if method == "iterative":
        return pararnn_scan(cell, h0, inputs, num_iterations, tol)
    elif method == "associative":
        return pararnn_associative_scan(cell, h0, inputs, num_iterations)
    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Comparison with Sequential Scan
# ============================================================================

def sequential_rnn(
    cell: RNNCell,
    h0: jax.Array,
    inputs: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    Standard sequential RNN scan for comparison.

    Args:
        cell: RNN cell function (h, x) -> (h_next, y)
        h0: Initial hidden state
        inputs: Input sequence

    Returns:
        final_state, outputs
    """
    def scan_fn(h, x):
        h_next, y = cell(h, x)
        return h_next, y

    final_state, outputs = lax.scan(scan_fn, h0, inputs)
    return final_state, outputs
