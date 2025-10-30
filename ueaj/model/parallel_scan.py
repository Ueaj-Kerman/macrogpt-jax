"""
Parallel Scan via Fixed-Point Iteration

Implements bidirectional parallel scan using Jacobi iteration for solving
fixed-point equations. Works for any scan-like computation including RNNs.

Based on: ParaRNN (https://arxiv.org/abs/2510.21450)
Extended with bidirectional support for efficient gradient computation.
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.tree_util import tree_map
from typing import Callable, Tuple, Any, TypeVar
from functools import partial

# Type variables for generic scan function
C = TypeVar('C')  # Carry type
X = TypeVar('X')  # Input type
Y = TypeVar('Y')  # Output type

ScanFn = Callable[[C, X], Tuple[C, Y]]


def _get_seq_len(xs: Any) -> int:
    """Get sequence length from potentially nested pytree."""
    first_leaf = jax.tree_util.tree_leaves(xs)[0]
    return first_leaf.shape[0]


def _reverse_tree(xs: Any) -> Any:
    """Reverse all arrays in a pytree along axis 0."""
    return tree_map(lambda x: x[::-1], xs)


def parallel_scan_iteration(
    scan_fn: ScanFn,
    init_carry: C,
    xs: X,
    num_iterations: int,
    reverse: bool = False
) -> Tuple[C, Y, Any]:
    """
    Core parallel scan using Jacobi fixed-point iteration.

    Solves the recurrence relation:
        carry[t] = scan_fn(carry[t-1], xs[t])  (forward)
        carry[t] = scan_fn(carry[t+1], xs[t])  (reverse)

    via parallel fixed-point iteration.

    Args:
        scan_fn: Function (carry, x) -> (next_carry, output)
        init_carry: Initial carry value (h0 for forward, final grad for backward)
        xs: Sequence of inputs [T, ...]
        num_iterations: Number of Jacobi iterations (typically 10-20)
        reverse: If True, scan right-to-left; if False, scan left-to-right

    Returns:
        final_carry: Carry at final position
        outputs: Sequence of outputs [T, ...]
        carry_all: All intermediate carries [T, ...] (for backward pass)

    Algorithm:
        1. Initialize all carries to init_carry
        2. For k iterations:
           - Get previous carries (shift by 1, prepend/append init_carry)
           - Update all carries in parallel: carry[t] = scan_fn(prev[t], xs[t])
        3. Compute outputs using final converged carries
    """
    seq_len = _get_seq_len(xs)

    # Reverse inputs if scanning backward
    if reverse:
        xs = _reverse_tree(xs)

    # Ensure init_carry elements are arrays (not scalars)
    init_carry = tree_map(lambda x: jnp.asarray(x), init_carry)

    # Initialize all carries to init_carry (initial guess for fixed point)
    def tile_init(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tile(x[None, ...], (seq_len, *([1] * x.ndim)))

    carry_all = tree_map(tile_init, init_carry)

    # Jacobi iteration: update all carries in parallel
    def iteration_step(carry_all_old: Any) -> Any:
        """One iteration of parallel carry update (Jacobi method)."""
        # Prepend init_carry to get previous carries for each position
        # Forward: carry[t] depends on carry[t-1], so prepend init at start
        # Reverse: carry[t] depends on carry[t+1], handled by reversed xs
        carry_prev_all = tree_map(
            lambda init, carries: jnp.concatenate([init[None, ...], carries[:-1]], axis=0),
            init_carry,
            carry_all_old
        )

        # Vectorized update: apply scan_fn to all positions in parallel
        def update_single(carry_prev: C, x: X) -> C:
            next_carry, _ = scan_fn(carry_prev, x)
            return next_carry

        return jax.vmap(update_single)(carry_prev_all, xs)

    # Run fixed-point iterations
    for _ in range(num_iterations):
        carry_all = iteration_step(carry_all)

    # Compute outputs using converged carries
    def compute_output(carry_prev: C, x: X) -> Y:
        _, output = scan_fn(carry_prev, x)
        return output

    carry_prev_all = tree_map(
        lambda init, carries: jnp.concatenate([init[None, ...], carries[:-1]], axis=0),
        init_carry,
        carry_all
    )
    outputs = jax.vmap(compute_output)(carry_prev_all, xs)

    # Get final carry
    final_carry = tree_map(lambda c: c[-1], carry_all)

    # Un-reverse outputs if needed
    if reverse:
        carry_all = _reverse_tree(carry_all)
        outputs = _reverse_tree(outputs)

    return final_carry, outputs, carry_all


@partial(custom_vjp, nondiff_argnums=(0, 3, 4))
def parallel_scan(
    scan_fn: ScanFn,
    init_carry: C,
    xs: X,
    num_iterations: int = 10,
    reverse: bool = False
) -> Tuple[C, Y]:
    """
    Parallel scan with bidirectional support and custom VJP.

    Both forward and backward passes use parallel fixed-point iteration,
    enabling efficient parallel gradient computation.

    Args:
        scan_fn: Scan function (carry, x) -> (next_carry, output)
        init_carry: Initial carry value
        xs: Input sequence [T, ...]
        num_iterations: Number of iterations for convergence (default: 10)
        reverse: Scan direction (False=forward, True=backward)

    Returns:
        final_carry: Final carry value
        outputs: Output sequence [T, ...]

    Example:
        >>> # Cumulative sum
        >>> def add_fn(carry, x):
        ...     next_carry = carry + x
        ...     return next_carry, next_carry
        >>> final, outputs = parallel_scan(add_fn, 0.0, xs)

        >>> # RNN forward pass
        >>> def rnn_cell(h, x):
        ...     h_next = jnp.tanh(W @ h + U @ x + b)
        ...     return h_next, h_next
        >>> final_h, h_seq = parallel_scan(rnn_cell, h0, inputs)
    """
    final_carry, outputs, _ = parallel_scan_iteration(
        scan_fn, init_carry, xs, num_iterations, reverse
    )
    return final_carry, outputs


def _parallel_scan_fwd(
    scan_fn: ScanFn,
    init_carry: C,
    xs: X,
    num_iterations: int,
    reverse: bool
) -> Tuple[Tuple[C, Y], Tuple[C, X, Any]]:
    """Forward pass: compute outputs and save residuals for backward."""
    final_carry, outputs, carry_all = parallel_scan_iteration(
        scan_fn, init_carry, xs, num_iterations, reverse
    )
    residuals = (init_carry, xs, carry_all)
    return (final_carry, outputs), residuals


def _parallel_scan_bwd(
    scan_fn: ScanFn,
    num_iterations: int,
    reverse: bool,
    residuals: Tuple[C, X, Any],
    grads: Tuple[Any, Any]
) -> Tuple[Any, Any]:
    """
    Backward pass: use parallel iteration in REVERSE direction.

    Key insight: Gradients flow opposite to forward direction but have
    the same recurrence structure, so we can use the same parallel algorithm!
    """
    init_carry, xs, carry_all = residuals
    grad_final_carry, grad_outputs = grads

    seq_len = _get_seq_len(xs)

    # Build VJP scan function
    def vjp_scan_fn(grad_carry_next: Any, args: Tuple) -> Tuple[Any, Tuple[Any, Any]]:
        """VJP function for one timestep."""
        idx, carry_prev, x, grad_out = args

        # Get actual carry at this position (from forward pass)
        carry = tree_map(lambda c: c[idx], carry_all)

        # Compute VJP through scan_fn
        _, vjp_fn = jax.vjp(lambda c, x: scan_fn(c, x), carry_prev, x)
        grad_carry_prev, grad_x = vjp_fn((grad_carry_next, grad_out))

        return grad_carry_prev, (grad_carry_prev, grad_x)

    # Prepare backward pass inputs
    indices = jnp.arange(seq_len)
    carry_prev_all = tree_map(
        lambda init, carries: jnp.concatenate([init[None, ...], carries[:-1]], axis=0),
        init_carry,
        carry_all
    )
    backward_inputs = (indices, carry_prev_all, xs, grad_outputs)

    # Use parallel_scan_iteration in REVERSE direction!
    grad_init_carry, (_, grad_xs), _ = parallel_scan_iteration(
        vjp_scan_fn,
        grad_final_carry,
        backward_inputs,
        num_iterations=num_iterations,
        reverse=not reverse  # Flip direction for backward pass
    )

    return (grad_init_carry, grad_xs)


parallel_scan.defvjp(_parallel_scan_fwd, _parallel_scan_bwd)
