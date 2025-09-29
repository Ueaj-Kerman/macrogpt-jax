"""Gradient transformations for optimizers with precision guarantees."""

from typing import Optional, Union, Any
import jax
import jax.numpy as jnp
import optax
from optax._src import base


def guaranteed_weight_decay(weight_decay: float) -> base.GradientTransformation:
    """Weight decay that uses nextafter when standard decay rounds to zero.

    Args:
        weight_decay: Weight decay coefficient

    Returns:
        A gradient transformation that guarantees weight decay.
    """

    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError("`params` were not provided!")

        def apply_decay(g, p):
            decay = weight_decay * p
            return g + jnp.where(p == (p - decay), p - jax.lax.nextafter(p, -p), decay)

        updates = jax.tree.map(apply_decay, updates, params)
        return updates, state

    return base.GradientTransformation(init_fn, update_fn)
