"""Optimization utilities: loss functions and optimizer management."""

from .next_token_loss import (
    ntp_loss_mask,
    ntp_args,
    chunked_softmax_cross_entropy,
)

from .optimizer_config import OptimizerConfig
from .optimizer_state import map_state


__all__ = [
    # Loss functions
    "ntp_loss_mask",
    "ntp_args",
    "chunked_softmax_cross_entropy",
    # Optimizer management
    "OptimizerConfig",
    "map_state",
]