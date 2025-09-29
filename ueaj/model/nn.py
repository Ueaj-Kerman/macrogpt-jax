"""Custom activation functions."""
import jax.numpy as jnp

def relu_squared(x):
    return jnp.where(x >= 0, jnp.square(x), 0)

def leaky_relu_squared(x):
    """Leaky ReLU squared activation function.
    
    Applies x^2 for positive values and -0.0625 * x^2 for negative values.
    
    Args:
        x: Input tensor
        
    Returns:
        Activated tensor
    """
    return jnp.where(x < 0, -0.0625, 1) * jnp.square(x)

def signed_sqrt(x):
    """Signed square root activation function.

    Applies sqrt(x) for positive values and -sqrt(-x) for negative values.

    Args:
        x: Input tensor

    Returns:
        Activated tensor
    """
    return jnp.where(x < 0, -jnp.sqrt(-x), jnp.sqrt(x))
