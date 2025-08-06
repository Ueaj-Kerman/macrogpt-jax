"""Optimizer configuration and setup utilities."""

import os
from typing import Any, Dict

import jax.numpy as jnp
import optax
from flax import nnx

import ueaj.opt.multiscale as ms
from ueaj.opt import OptimizerConfig


def make_optimizer(lr: float, warmup: float, model: nnx.Module) -> optax.GradientTransformation:
    """Create optimizer configuration based on environment settings.
    
    Args:
        lr: Base learning rate
        warmup: Warmup factor (0 to 1)
        model: Model to optimize
        
    Returns:
        Configured optimizer
    """
    lr *= warmup

    opt = OptimizerConfig(model)
    
    # Norm optimizer
    norm = optax.lion(learning_rate=0.015625 * lr, b1=.95, b2=.95, weight_decay=1e-2)

    # lm_head optimizer
    default = optax.adamw(learning_rate=0.5 * lr, b1=.95, b2=.999, weight_decay=1e-3)

    # Embed optimizer (no weight decay)
    embed = optax.adam(learning_rate=lr, b1=.95, b2=.999)

    # Select optimizer with env_var
    opt_name = os.environ.get('OPTIMIZER', 'multiscale')
    if opt_name == 'multiscale':
        tensor = ms.einsum_aware(ms.multiscale_muon(lr=lr / 8, wd=1e-3, warmup_frac=warmup**2), model)
    elif opt_name == 'muon':
        tensor = ms.einsum_aware(ms.muon(lr=lr / 8, wd=1e-3), model)
    elif opt_name == 'adamw':
        tensor = optax.adamw(learning_rate=0.5 * lr, b1=.95, b2=0.999, weight_decay=1e-3)
    else:
        raise ValueError(f'Unrecognized optimizer name: {opt_name}')

    # Configure optimizer assignments
    opt[...] = default  # lm_head
    opt['norm'] = norm
    opt['layers', ['mlp_norm', 'attn_norm']] = norm
    opt['embed_tokens'] = embed
    opt['layers', ['mlp', 'attn']] = tensor

    return opt.create_optimizer()


def get_optimizer_name() -> str:
    """Get optimizer name from environment variable."""
    return os.environ.get('OPTIMIZER', 'multiscale')