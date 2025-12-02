"""Optimizer configuration and setup utilities."""

import os
from typing import Any, Dict

import jax.lax
import jax.numpy as jnp
import optax
from flax import nnx

import ueaj.opt.multiscale as ms
from ueaj.opt import OptimizerConfig
from ueaj.opt.opt_utils import project_rms_rows
from ueaj.utils.configurator import config


@config
def make_optimizer(lr: float, warmup: float, model: nnx.Module, dtype=jnp.float32) -> optax.GradientTransformation:
	"""Create optimizer configuration based on environment settings.

	Args:
		lr: Base learning rate
		warmup: Warmup factor (0 to 1)
		model: Model to optimize
		dtype: Data type

	Returns:
		Configured optimizer
	"""

	opt = OptimizerConfig(model)

	layer_lr = lr*jax.lax.rsqrt(model.num_layers)

	# Norm optimizer
	norm = optax.lion(learning_rate=0.0625 * layer_lr, b1=.95, b2=.95, weight_decay=1e-2, mu_dtype=dtype)

	# lm_head optimizer
	lm_head = optax.adamw(learning_rate=0.5 * lr, b1=.95, b2=.999, weight_decay=1e-3, mu_dtype=dtype)

	default = optax.adamw(learning_rate=0.5 * lr, b1=.95, b2=.999, weight_decay=1e-3, mu_dtype=dtype)

	# Embed optimizer (no weight decay, with RMS projection to prevent unbounded growth)
	embed = optax.chain(
		optax.adam(learning_rate=4*layer_lr, b1=.95, b2=.999, mu_dtype=dtype),
		project_rms_rows(max_rms=1.25),
	)

	# Select optimizer with env_var
	opt_name = get_optimizer_name()
	if opt_name == 'multiscale':
		tensor = ms.multiscale_muon(model, lr=0.4 * layer_lr, wd=1e-3, warmup_frac=warmup ** 2, dtype=dtype)
	elif opt_name == 'muon':
		tensor = ms.muon(model, lr=0.5 * layer_lr, wd=1e-3, dtype=dtype)
	elif opt_name == 'adamw':
		tensor = optax.adamw(learning_rate=0.125 * layer_lr, b1=.95, b2=0.999, weight_decay=1e-3, mu_dtype=dtype)
	elif opt_name == 'lion':
		tensor = optax.lion(learning_rate=0.0625 * layer_lr, b1=.95, b2=0.95, weight_decay=1e-3, mu_dtype=dtype)
	else:
		raise ValueError(f'Unrecognized optimizer name: {opt_name}')

	# Configure optimizer assignments
	opt[...] = default
	opt['lm_head'] = lm_head
	opt['norm'] = norm
	opt['layers', ['mlp_norm', 'attn_norm']] = norm
	opt['embed_tokens'] = embed
	opt['layers', ['mlp', 'attn']] = tensor

	return opt.create_optimizer()


def get_optimizer_name() -> str:
	"""Get optimizer name from environment variable."""
	return os.environ.get('OPTIMIZER', 'muon')
