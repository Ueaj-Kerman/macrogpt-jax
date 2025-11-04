"""
Parallel Scan via Newton's Method

Implements bidirectional parallel scan using Newton iteration for solving
fixed-point equations. Achieves quadratic convergence (3-5 iterations).

Based on: DEER (Lim et al., 2024) and ParaRNN extensions.
Tradeoff: Faster convergence but higher memory/compute per iteration.
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp, jacrev, lax
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


def parallel_tridiagonal_solve(
    a_diag: jnp.ndarray,
    b_lower: jnp.ndarray,
    c_upper: jnp.ndarray,
    d: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve block tridiagonal system Ax = d using sequential Thomas algorithm.

    For now, use simple sequential solve. A true parallel version would use
    cyclic reduction or parallel prefix, but that's complex for block matrices.

    System structure:
        [a[0]  c[0]    0     0   ...] [x[0]]   [d[0]]
        [b[0]  a[1]  c[1]    0   ...] [x[1]]   [d[1]]
        [ 0    b[1]  a[2]  c[2]  ...] [x[2]] = [d[2]]
        ...

    Args:
        a_diag: Diagonal blocks [T, d, d]
        b_lower: Lower diagonal blocks [T-1, d, d]
        c_upper: Upper diagonal blocks [T-1, d, d]
        d: Right-hand side vectors [T, d]

    Returns:
        x: Solution [T, d]

    Algorithm: Thomas algorithm (block tridiagonal variant)
    TODO: Implement parallel version using associative scan
    """
    T = a_diag.shape[0]

    # Forward elimination
    c_prime = []
    d_prime = []

    # First row
    c_prime.append(jnp.linalg.solve(a_diag[0], c_upper[0]))
    d_prime.append(jnp.linalg.solve(a_diag[0], d[0]))

    # Middle rows
    for i in range(1, T - 1):
        denom = a_diag[i] - b_lower[i-1] @ c_prime[i-1]
        c_prime.append(jnp.linalg.solve(denom, c_upper[i]))
        d_prime.append(jnp.linalg.solve(denom, d[i] - b_lower[i-1] @ d_prime[i-1]))

    # Last row
    denom = a_diag[T-1] - b_lower[T-2] @ c_prime[T-2]
    d_prime.append(jnp.linalg.solve(denom, d[T-1] - b_lower[T-2] @ d_prime[T-2]))

    # Back substitution
    x = [None] * T
    x[T-1] = d_prime[T-1]

    for i in range(T-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] @ x[i+1]

    return jnp.stack(x)


def newton_scan_iteration(
    scan_fn: ScanFn,
    init_carry: C,
    xs: X,
    num_iterations: int,
    reverse: bool = False
) -> Tuple[C, Y, Any]:
    """
    Core parallel scan using Newton's method.

    Solves the fixed-point equation via Newton iteration:
        Find h such that: h[t] = scan_fn(h[t-1], xs[t])

    Newton update:
        J(h^k) · Δh = -F(h^k)
        h^(k+1) = h^k + Δh

    Where J is the Jacobian of the residual F(h) = h - scan_fn(shift(h), xs)

    Args:
        scan_fn: Function (carry, x) -> (next_carry, output)
        init_carry: Initial carry value
        xs: Sequence of inputs [T, ...]
        num_iterations: Number of Newton iterations (typically 3-5)
        reverse: If True, scan right-to-left

    Returns:
        final_carry: Carry at final position
        outputs: Sequence of outputs [T, ...]
        carry_all: All intermediate carries [T, ...]

    Complexity:
        Per iteration: O(T × d³) where d = hidden_dim
        Memory: O(T × d²) for Jacobian
    """
    seq_len = _get_seq_len(xs)

    # Reverse inputs if scanning backward
    if reverse:
        xs = _reverse_tree(xs)

    # Ensure init_carry elements are arrays
    init_carry = tree_map(lambda x: jnp.asarray(x), init_carry)

    # Initialize all carries (Jacobi-style initialization)
    def tile_init(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tile(x[None, ...], (seq_len, *([1] * x.ndim)))

    carry_all = tree_map(tile_init, init_carry)

    # Get hidden dimension (assume single array for now)
    hidden_dim = jax.tree_util.tree_leaves(carry_all)[0].shape[-1]

    # Newton iterations using fori_loop (not Python for loop!)
    # This prevents unrolling and keeps compilation fast
    def newton_body_fn(iteration, carry_all):
        # 1. Compute residual: F(h) = h - scan_fn(shift(h), xs)
        carry_prev_all = tree_map(
            lambda init, carries: jnp.concatenate([init[None, ...], carries[:-1]], axis=0),
            init_carry,
            carry_all
        )

        def residual_fn(carry: C, carry_prev: C, x: X) -> C:
            """Residual at position t: h[t] - f(h[t-1], x[t])"""
            next_carry, _ = scan_fn(carry_prev, x)
            return tree_map(lambda h, f_h: h - f_h, carry, next_carry)

        # Compute residuals for all positions
        residuals = jax.vmap(residual_fn)(
            carry_all,
            carry_prev_all,
            xs
        )  # [T, d]

        # 2. Compute Jacobian blocks
        # J has block tridiagonal structure:
        #   J[t,t] = I - ∂f/∂h[t-1]  (diagonal)
        #   J[t,t-1] = -∂f/∂h[t-1]   (lower)

        def compute_jacobian_block(carry_prev: C, x: X) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Compute Jacobian block for position t.

            Returns:
                diag_block: I - ∂f/∂h_prev [d, d]
                lower_block: ∂f/∂h_prev [d, d]
            """
            # Jacobian of scan_fn w.r.t. carry_prev
            def extract_carry(c, x):
                next_c, _ = scan_fn(c, x)
                return jax.tree_util.tree_leaves(next_c)[0]

            jac = jacrev(extract_carry, argnums=0)(carry_prev, x)  # [d, d]

            # For tridiagonal structure:
            # h[t] - f(h[t-1], x[t]) = 0
            # Jacobian: ∂/∂h[t] = I, ∂/∂h[t-1] = -∂f/∂h[t-1]

            diag_block = jnp.eye(hidden_dim)  # I
            lower_block = -jac  # -∂f/∂h[t-1]

            return diag_block, lower_block

        # Compute all Jacobian blocks
        diag_blocks, lower_blocks = jax.vmap(compute_jacobian_block)(
            carry_prev_all, xs
        )  # [T, d, d]

        # Upper blocks: for forward scan, c[t,t+1] affects equation at t+1
        # But our formulation has structure where only lower diagonal matters
        # (each equation depends on previous state only)
        upper_blocks = jnp.zeros((seq_len - 1, hidden_dim, hidden_dim))

        # 3. Solve tridiagonal system: J · Δh = -residuals
        residuals_flat = jax.tree_util.tree_leaves(residuals)[0]  # [T, d]

        delta_h = parallel_tridiagonal_solve(
            diag_blocks,
            lower_blocks[1:],  # Lower diagonal (T-1 blocks)
            upper_blocks,
            -residuals_flat
        )  # [T, d]

        # 4. Update
        return tree_map(lambda h: h + delta_h, carry_all)

    carry_all = lax.fori_loop(0, num_iterations, newton_body_fn, carry_all)

    # Compute outputs using final converged carries
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
def parallel_scan_newton(
    scan_fn: ScanFn,
    init_carry: C,
    xs: X,
    num_iterations: int = 3,
    reverse: bool = False
) -> Tuple[C, Y]:
    """
    Parallel scan using Newton's method with bidirectional support.

    Achieves quadratic convergence (3-5 iterations vs 10-20 for Jacobi).

    Args:
        scan_fn: Scan function (carry, x) -> (next_carry, output)
        init_carry: Initial carry value
        xs: Input sequence [T, ...]
        num_iterations: Number of Newton iterations (default: 3)
        reverse: Scan direction (False=forward, True=backward)

    Returns:
        final_carry: Final carry value
        outputs: Output sequence [T, ...]

    Complexity:
        Per iteration: O(T × d³) vs O(T × d²) for Jacobi
        Memory: O(T × d²) vs O(T × d) for Jacobi
        Convergence: Quadratic vs Linear

    Example:
        >>> # RNN with Newton (faster convergence)
        >>> def rnn_cell(h, x):
        ...     h_next = jnp.tanh(W @ h + U @ x + b)
        ...     return h_next, h_next
        >>> final_h, outputs = parallel_scan_newton(
        ...     rnn_cell, h0, inputs, num_iterations=3
        ... )
    """
    final_carry, outputs, _ = newton_scan_iteration(
        scan_fn, init_carry, xs, num_iterations, reverse
    )
    return final_carry, outputs


def _parallel_scan_newton_fwd(
    scan_fn: ScanFn,
    init_carry: C,
    xs: X,
    num_iterations: int,
    reverse: bool
) -> Tuple[Tuple[C, Y], Tuple[C, X, Any]]:
    """Forward pass: compute outputs and save residuals for backward."""
    final_carry, outputs, carry_all = newton_scan_iteration(
        scan_fn, init_carry, xs, num_iterations, reverse
    )
    residuals = (init_carry, xs, carry_all)
    return (final_carry, outputs), residuals


def _parallel_scan_newton_bwd(
    scan_fn: ScanFn,
    num_iterations: int,
    reverse: bool,
    residuals: Tuple[C, X, Any],
    grads: Tuple[Any, Any]
) -> Tuple[Any, Any]:
    """
    Backward pass: use Newton iteration in REVERSE direction.

    Same structure as Jacobi backward, but with Newton's method.
    """
    init_carry, xs, carry_all = residuals
    grad_final_carry, grad_outputs = grads

    # Ensure grads are arrays
    init_carry = tree_map(lambda x: jnp.asarray(x), init_carry)
    grad_final_carry = tree_map(lambda x: jnp.asarray(x), grad_final_carry)

    seq_len = _get_seq_len(xs)

    # Build VJP scan function
    def vjp_scan_fn(grad_carry_next: Any, args: Tuple) -> Tuple[Any, Tuple[Any, Any]]:
        """VJP function for one timestep."""
        idx, carry_prev, x, grad_out = args

        # Get actual carry at this position
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

    # Use Newton iteration in REVERSE direction for backward pass
    grad_init_carry, (_, grad_xs), _ = newton_scan_iteration(
        vjp_scan_fn,
        grad_final_carry,
        backward_inputs,
        num_iterations=num_iterations,
        reverse=not reverse
    )

    return (grad_init_carry, grad_xs)


parallel_scan_newton.defvjp(_parallel_scan_newton_fwd, _parallel_scan_newton_bwd)
