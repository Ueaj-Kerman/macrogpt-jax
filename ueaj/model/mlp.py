from dataclasses import dataclass, replace
from typing import Callable

import jax
from jax import numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
from ueaj.model import ueajsum as us
from ueaj.utils.argutils import either
from ueaj.utils.collections import LOW_PRECISION
from ueaj.utils.gradutils import debug_dtype, custom_grad_vectors
from jax.ad_checkpoint import checkpoint_name

@dataclass
class MLPConfig:
	model_d: int
	hidden_d: int

	activation_fn: Callable[[jax.Array], jax.Array] = nnx.swish

	param_config: us.ParamConfig = us.ParamConfig("", group=nnx.Param)

	_gate_config: us.ParamConfig | None = None
	_up_config: us.ParamConfig | None = None
	_down_config: us.ParamConfig | None = None

	@property
	def up_config(self):
		return either(self._up_config, self.param_config)

	@property
	def down_config(self):
		return either(self._down_config, self.param_config.with_initializer(nnx.initializers.zeros))

	@property
	def gate_config(self):
		return either(self._gate_config, self.param_config)

	def with_gate(self, config: us.ParamConfig):
		# Check for redundancy - if config is same as param_config, raise error
		if config == self.param_config:
			raise ValueError("Redundant specification: gate_config is same as param_config")
		return replace(self, _gate_config=config)

	def with_up(self, config: us.ParamConfig):
		# Check for redundancy - if config is same as param_config, raise error
		if config == self.param_config:
			raise ValueError("Redundant specification: up_config is same as param_config")
		return replace(self, _up_config=config)

	def with_down(self, config: us.ParamConfig):
		# Check for redundancy - if config is same as param_config with zeros init, raise error
		if config == self.param_config.with_initializer(nnx.initializers.zeros):
			raise ValueError("Redundant specification: down_config is same as param_config with zeros initializer")
		return replace(self, _down_config=config)

	def with_param(self, config: us.ParamConfig):
		return replace(self, param_config=config)


class MLP(nnx.Module):
	def __init__(self, config: MLPConfig, rngs: rng.Rngs):
		super().__init__()
		size_dict = {'d': config.model_d, 'h': config.hidden_d}

		make_ueajsum = lambda c: us.Ueajsum(c, size_dict, rngs=rngs)
		self.up_proj = make_ueajsum(
			us.parse("bnd,*dh->bnh").param(config.up_config).in_axes_zero()
		)
		self.down_proj = make_ueajsum(
			us.parse("bnh,*hd->bnd").param(config.down_config).in_axes_zero()
		)

		self.activation_fn = config.activation_fn

	@nnx.jit
	def __call__(self, x):
		x = self.up_proj(x)
		x = self.activation_fn(x)
		x = self.down_proj(x)
		return x

	def invoke_fp32_backprop(self, x):
		x = self.up_proj.invoke(self.up_proj.config.map_arg(0, lambda x: x.with_grad_dtype(jnp.float32)), False, x)
		x = self.activation_fn(x)
		x = self.down_proj.invoke(self.down_proj.config.map_arg(0, lambda x: x.with_grad_dtype(jnp.float32)), False, x)
		return x


class GMLP(nnx.Module):
	def __init__(self, config: MLPConfig, rngs: rng.Rngs):
		super().__init__()
		size_dict = {'d': config.model_d, 'h': config.hidden_d, 'i': 2}

		make_ueajsum = lambda c: us.Ueajsum(c, size_dict, rngs=rngs)
		self.fused_proj = make_ueajsum(
			us.parse("bnd,*idh->ibnh").param(config.up_config).in_axes({1: (1,)}).batch_axes({1: (0,)})
		)

		self.down_proj = make_ueajsum(
			us.parse("bnh,bnh,*hd->bnd").param(config.down_config).in_axes_zero()
		)
		self.activation_fn = config.activation_fn

	def __call__(self, x):
		up, gate = self.fused_proj(x)

		gate = self.activation_fn(gate)

		if x.dtype not in LOW_PRECISION:
			x = self.down_proj(up, gate)
		else:
			s = jnp.max(jnp.abs(up), axis=(0, 1), keepdims=True) + 1
			x = (s * up) * gate
			x = x / s
			x = self.down_proj.parse_and_call("bnh,[1]->bnd", x)

		return x


class BMLP(nnx.Module):
	"""Bayesian MLP with mean and variance projections"""
	def __init__(self, config: MLPConfig, rngs: rng.Rngs):
		super().__init__()
		size_dict = {'d': config.model_d, 'h': config.hidden_d}
		
		make_ueajsum = lambda c: us.Ueajsum(c, size_dict, rngs=rngs)
		
		# Up projection for mean
		self.up_proj = make_ueajsum(
			us.parse("bnd,*dh->bnh").param(config.up_config).in_axes_zero()
		)
		
		# Variance projection - uses gate_config for custom learning rate
		self.var_proj = make_ueajsum(
			us.parse("bnd,*dh->bnh").param(config.gate_config).in_axes_zero()
		)
		
		# Down projection
		self.down_proj = make_ueajsum(
			us.parse("bnh,*hd->bnd").param(config.down_config).in_axes_zero()
		)
		
		self.activation_fn = config.activation_fn
		self.var_lr = .1  # Learning rate multiplier for variance
	
	def __call__(self, x):
		# Project to get mean and variance
		mean = self.up_proj(x)

		var = self.var_proj(x)
		var = nnx.softplus(var)
		
		h = custom_grad_vectors(mean, var, self.var_lr)
		
		h = self.activation_fn(h)
		
		x = self.down_proj(h)
		return x


if __name__ == "__main__":
	config = MLPConfig(
		model_d=16,
		hidden_d=32,
		activation_fn=nnx.relu,
	)
	x = jnp.ones((1, 1, 16))
	m = MLP(config.with_down(config.param_config.with_initializer(nnx.initializers.lecun_normal())), rngs=rng.Rngs(0))
	gm = GMLP(config.with_down(config.param_config.with_initializer(nnx.initializers.lecun_normal())),
		rngs=rng.Rngs(0))
	bm = BMLP(config.with_down(config.param_config.with_initializer(nnx.initializers.lecun_normal())),
		rngs=rng.Rngs(0))

	print(m(x))
	print(gm(x))
	print(bm(x))
