"""Tensor statistics gathering utilities."""

import jax
import jax.numpy as jnp
from typing import Union, Tuple, Optional


def k_eff(W: jax.Array) -> jax.Array:
    """Compute effective rank (k_eff) of a tensor.
    
    The effective rank is computed as:
    k_eff = m * ||W^T W||_F^2 / ||W||_F^4
    
    With map_state, tensors are in the format:
    (...batch_dims, input_dim, output_dim)
    
    Args:
        W: Input tensor with shape (..., input_dim, output_dim).
           Should have at least 2 dimensions.
    
    Returns:
        Array of k_eff values with shape (...,) for batch dimensions.
    """
    if W.ndim < 2:
        return jnp.array(1.0)
    
    # Get dimensions - we work with the last two dimensions
    # In the mapped format, batch/vmap dims come first, then input, then output
    in_axis = W.ndim - 2
    out_axis = W.ndim - 1
    
    n = W.shape[in_axis]  # input dimension
    m = W.shape[out_axis]  # output dimension
    
    # Cast to bfloat16 for the computation
    # (squaring numbers only requires slightly more exponent bits)
    W = W.astype(jnp.bfloat16)
    
    # Compute Frobenius norm squared: ||W||_F^2 = sum(W^2)
    # Use float32 for accumulation to get more mantissa bits
    f2 = jnp.sum(jnp.square(W), axis=[in_axis, out_axis], dtype=jnp.float32)
    
    # Compute ||W^T W||_F^2 using dot_general
    # W^T W shape: (m, m)
    # For dot_general: specify which axes to contract and which are batch
    batch_dims = tuple(range(W.ndim - 2))
    
    # Compute W^T @ W
    # Contract over the input dimension (axis -2)
    # Batch dimensions are all dimensions except the last 2
    WtW = jax.lax.dot_general(
        W, W,
        dimension_numbers=(
            ([in_axis,], [in_axis,]),  # Contract over input dimension
            (batch_dims, batch_dims)    # Batch dimensions
        ),
        preferred_element_type=jnp.float32  # Accumulate in float32
    )
    
    # Compute ||W^T W||_F^2 = sum((W^T W)^2)
    s4 = jnp.sum(jnp.square(WtW), axis=[in_axis, out_axis], dtype=jnp.float32)
    
    # Return k_eff = m * s4 / f2^2
    return m * s4 / (f2 ** 2)

def tensor_stats(W: jax.Array) -> dict:
    """Compute various statistics for a tensor.
    
    Args:
        W: Input tensor
        
    Returns:
        Dictionary containing:
        - l1_norm: L1 norm (for precision monitoring)
        - l2_norm: L2 norm  
        - log_l1_norm: Log of L1 norm
        - variance: Variance of the tensor
        - k_eff: Effective rank (if tensor has 2+ dimensions)
    """
    stats = {}
    
    # L1 norm (useful for low precision monitoring)
    stats['l1_norm'] = jnp.mean(jnp.abs(W), dtype=jnp.float32)
    stats['log_l1_norm'] = jnp.log2(stats['l1_norm'] + 1e-10)
    
    # L2 norm
    stats['l2_norm'] = jnp.sqrt(jnp.mean(jnp.square(W), dtype=jnp.float32))
    
    # Effective rank (only for 2D+ tensors)
    if W.ndim >= 2:
        stats['k_eff'] = k_eff(W)
    
    return stats