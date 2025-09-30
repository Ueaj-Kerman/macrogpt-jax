"""Utilities for LoRA training.

LoRA training works by:
1. Extracting only LoRA parameters: lora_state = nnx.state(model, nnx.LoRAParam)
2. Computing gradients w.r.t. LoRA state: grads = jax.grad(loss_fn)(lora_state)
3. Updating LoRA parameters: updates = optimizer.update(grads, opt_state, lora_state)
4. Merging back: nnx.update(model, updated_lora_state)

This naturally freezes base parameters since we never compute gradients for them.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from typing import Optional

from ueaj.utils.configurator import config


@config
def make_lora_optimizer(
    lr: float,
    weight_decay: float = 1e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    dtype=jnp.float32
) -> optax.GradientTransformation:
    """Create optimizer for LoRA fine-tuning.

    Returns a simple AdamW optimizer. Base parameters are automatically frozen
    by only computing gradients w.r.t. the LoRA state in the training loop.

    Args:
        lr: Learning rate for LoRA parameters
        weight_decay: Weight decay for LoRA parameters
        b1: Adam beta1
        b2: Adam beta2
        dtype: Optimizer state dtype

    Returns:
        Configured AdamW optimizer

    Example:
        >>> # In training loop
        >>> def loss_fn(lora_state):
        ...     nnx.update(model, lora_state)
        ...     logits = model(inputs)
        ...     return compute_loss(logits, targets)
        >>>
        >>> lora_state = nnx.state(model, nnx.LoRAParam)
        >>> optimizer = make_lora_optimizer(lr=1e-4)
        >>> opt_state = optimizer.init(lora_state)
        >>>
        >>> # Training step
        >>> grads = jax.grad(loss_fn)(lora_state)
        >>> updates, opt_state = optimizer.update(grads, opt_state, lora_state)
        >>> lora_state = optax.apply_updates(lora_state, updates)
        >>> nnx.update(model, lora_state)
    """
    return optax.adamw(
        learning_rate=lr,
        b1=b1,
        b2=b2,
        weight_decay=weight_decay,
        mu_dtype=dtype
    )


def get_lora_param_count(model: nnx.Module) -> dict:
    """Count LoRA parameters in model.

    Args:
        model: Model with LoRA adaptations

    Returns:
        Dict with parameter counts:
        - lora_params: Number of trainable LoRA parameters
        - base_params: Number of frozen base parameters
        - total_params: Total parameters
        - lora_fraction: Fraction of params that are LoRA

    Example:
        >>> counts = get_lora_param_count(model)
        >>> print(f"Training {counts['lora_params']:,} LoRA params")
        >>> print(f"({counts['lora_fraction']:.2%} of total)")
    """
    # Get all parameters
    all_params = nnx.state(model, nnx.Param)
    lora_params = nnx.state(model, nnx.LoRAParam)

    def count_params(tree):
        """Count parameters in tree."""
        total = 0
        if hasattr(tree, 'value'):
            # This is a parameter
            if hasattr(tree.value, 'size'):
                total += tree.value.size
        elif hasattr(tree, 'items'):
            # This is a tree node
            for subtree in tree.items():
                total += count_params(subtree[1])
        return total

    lora_count = count_params(lora_params)
    total_count = count_params(all_params)
    base_count = total_count - lora_count

    return {
        'lora_params': lora_count,
        'base_params': base_count,
        'total_params': total_count,
        'lora_fraction': lora_count / total_count if total_count > 0 else 0.0
    }


def print_lora_info(model: nnx.Module):
    """Print information about LoRA adaptation.

    Args:
        model: Model with LoRA adaptations

    Example:
        >>> model = apply_lora_to_model(model, rank=16, rngs=rngs)
        >>> print_lora_info(model)
        LoRA Configuration:
          Trainable params: 1,234,567 (2.3% of total)
          Frozen params:    52,345,678
          Total params:     53,580,245
    """
    counts = get_lora_param_count(model)

    print("LoRA Configuration:")
    print(f"  Trainable params: {counts['lora_params']:,} ({counts['lora_fraction']:.1%} of total)")
    print(f"  Frozen params:    {counts['base_params']:,}")
    print(f"  Total params:     {counts['total_params']:,}")