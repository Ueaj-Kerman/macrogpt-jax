from typing import Optional, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.model import GMLP
from ueaj.model.einsum import Einsum, lecun_normal_init, zeros_init
from ueaj.model.rmsnorm import RMSNorm
from ueaj.utils.configurator import config
from .impl import ttt


@config
class TTTModel(nnx.Module):
	"""Test-Time Training layer that learns to adapt its hidden state during inference.

	The TTT layer maintains a hidden state that is updated at each sequence position
	using gradient descent on a self-supervised objective. The state is used to
	produce outputs via an inner learnable module.

	Args:
		model_d: Model dimension (input/output dimension)
		hidden_d: Hidden dimension for the state (defaults to model_d)
		module: Inner module class to use as fwd_fn (default: GMLP)
		module_kwargs: Additional kwargs to pass to the inner module
		param_dtype: Parameter dtype
		surrogate: Whether to use surrogate gradients (custom VJP) for backprop
		rngs: Random number generators
		mesh: Optional JAX mesh for distributed training
	"""
	def __init__(
		self,
		model_d: int,
		hidden_d: int | None = None,
		module: Callable = GMLP,
		module_kwargs: dict | None = None,
		param_dtype: jnp.dtype = jnp.bfloat16,
		surrogate: bool = True,
		*,
		rngs: rng.Rngs,
		mesh: Optional[jax.sharding.Mesh] = None
	):
		super().__init__()

		if hidden_d is None:
			hidden_d = model_d

		if module_kwargs is None:
			module_kwargs = {}

		self.model_d = model_d
		self.hidden_d = hidden_d
		self.surrogate = surrogate

		# Create fused k, v, q projection
		size_dict = {'d': model_d, 'h': hidden_d, 'i': 3}
		self.kvq_proj = Einsum(
			"bnd,idh->ibnh",
			size_dict=size_dict,
			batch_dims="i",
			rngs=rngs,
			dtype=param_dtype,
			mesh=mesh,
			sharding=(None, None, 'tensor') if mesh is not None else None
		)

		# Create inner module - its parameters will be the TTT state
		# This module takes (batch, seq, hidden_d) input and produces (batch, seq, hidden_d) output
		self.inner_module = module(
			model_d=hidden_d,
			rngs=rngs,
			mesh=mesh,
			**module_kwargs
		)

		# RMSNorm after TTT, before output projection
		self.norm = RMSNorm(hidden_d, rngs=rngs, mesh=mesh)

		# Output projection to map from hidden_d back to model_d
		size_dict_out = {'d': model_d, 'h': hidden_d}
		self.out_proj = Einsum(
			"bnh,hd->bnd",
			size_dict=size_dict_out,
			# initializer=zeros_init,
			rngs=rngs,
			dtype=param_dtype,
			mesh=mesh,
			sharding=('tensor', None) if mesh is not None else None
		)

		# Create the TTT forward function
		self.ttt_fn = ttt(self._fwd_fn, surrogate=surrogate)
		self.inner_module_gdef = nnx.graphdef(self.inner_module)

	def _fwd_fn(self, module_state: nnx.State, x: jax.Array) -> jax.Array:
		"""Forward function for TTT: reconstructs module from state and applies it.

		Args:
			module_state: NNX State containing the module's parameters (this is the TTT state)
			x: Input of shape (batch, hidden_d)

		Returns:
			Output of shape (batch, hidden_d)
		"""
		# Reconstruct the module from graph definition and state
		module = nnx.merge(self.inner_module_gdef, module_state)

		# Apply inner module
		# Add dummy sequence dimension since modules expect (batch, seq, d)
		x = x[None, None, ...]  # (batch, 1, hidden_d)
		x = module(x)
		x = x[0, 0]  # Remove sequence dimension -> (batch, hidden_d)
		return x

	def __call__(self, x: jax.Array) -> jax.Array:
		"""Apply TTT layer.

		Args:
			x: Input of shape (batch, seq_len, model_d)

		Returns:
			Output of shape (batch, seq_len, model_d)
		"""
		# Project input to k, v, q
		k, v, q = self.kvq_proj(x)  # Each: (batch, seq_len, hidden_d)

		# Apply TTT algorithm - returns (output, final_state)
		hidden, final_state = self.ttt_fn(k, v, q, nnx.state(self.inner_module))

		# Normalize and project back to model dimension
		hidden = self.norm(hidden)
		output = self.out_proj(hidden)
		return output

	def apply_ttt(self, k, v, q):
		"""Apply TTT algorithm directly on k, v, q.

		Returns:
			(hidden_output, final_state) tuple
		"""
		return self.ttt_fn(k, v, q, nnx.state(self.inner_module))