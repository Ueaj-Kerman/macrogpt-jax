"""Logging utilities for training metrics and wandb integration."""

from typing import Any, Dict, Optional, Tuple
import uuid

import wandb
import jax
import jax.numpy as jnp
from flax import nnx


def log_training_metrics(
    stats: Dict[str, Any],
    step: int,
    trained_tokens: int,
    opt_arg: Dict[str, Any],
    train_time: float,
    wait_time: float,
    batch_size: int,
    seq_len: int,
    run_name: str,
    test_loss: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Log training metrics to wandb and return values for console logging.
    
    Args:
        stats: Dictionary of training statistics
        step: Current training step
        trained_tokens: Current number of tokens trained
        opt_arg: Optimizer arguments (contains learning rate)
        train_time: Time taken for training step
        wait_time: Time spent waiting for data
        batch_size: Batch size for training
        seq_len: Sequence length for training
        run_name: Name for the wandb run
        test_loss: Optional tuple of (mean_loss, std_loss) from test evaluation
        
    Returns:
        Dictionary with values for console logging:
            - mean_loss: Mean training loss
            - std_loss: Standard deviation of training loss
            - test_mean: Mean test loss (if provided)
            - test_std: Standard deviation of test loss (if provided)
            - trained_tokens: Updated total tokens trained
            - tokens_per_second: Training throughput
    """
    # Calculate tokens per second and update trained tokens
    tokens_per_second = (batch_size * seq_len) / train_time
    trained_tokens += (batch_size * seq_len)
    
    # Initialize wandb on first call
    if step == 0:
        run_id = str(uuid.uuid4())[:5]
        wandb.init(project="experimental", name=f"{run_name}-{run_id}")
    
    # Build wandb dictionary with basic metrics
    wandb_dict = {
        "step": step,
        "tokens": trained_tokens,
        "lr": float(opt_arg['lr']),
        "train_time": train_time,
        "wait_time": wait_time,
        "tokens_per_second": tokens_per_second,
    }
    
    # Add test loss if provided
    if test_loss is not None:
        test_mean, test_std = test_loss
        wandb_dict["test_loss"] = float(test_mean)
        wandb_dict["test_loss_std"] = float(test_std)
    
    # Process and add all statistics from the stats dict
    for stat_name, stat_value in stats.items():
        if isinstance(stat_value, dict) or hasattr(stat_value, 'items'):
            # Handle nested statistics (like grad_norm, param stats)
            for key, value in nnx.to_flat_state(stat_value):
                if hasattr(value, 'value'):
                    value = value.value
                wandb_dict[f"{stat_name}-" + ".".join(key)] = float(jnp.mean(value))
        else:
            # Handle scalar statistics
            wandb_dict[stat_name] = float(stat_value)
    
    # Log to wandb
    wandb.log(wandb_dict)
    
    # Return values needed for console logging
    return_dict = {
        'mean_loss': float(stats.get('mean_loss', 0.0)),
        'std_loss': float(stats.get('std_loss', 0.0)),
        'trained_tokens': trained_tokens,
        'tokens_per_second': tokens_per_second,
    }
    
    if test_loss is not None:
        return_dict['test_mean'] = float(test_mean)
        return_dict['test_std'] = float(test_std)
    
    return return_dict