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
	"""Test-Time Training layer with multi-query support.

	The TTT layer maintains a hidden state that is updated at each sequence position
	using gradient descent on a self-supervised objective. Multiple query heads can
	share the same K/V state via the q_heads parameter.

	Args:
		model_d: Model dimension (input/output dimension)
		hidden_d: Hidden dimension for the state (defaults to model_d)
		q_heads: Number of query heads (multiquery uses shared k/v)
		module: Inner module class to use as fwd_fn (default: GMLP)
		module_kwargs: Additional kwargs to pass to the inner module
		param_dtype: Parameter dtype
		surrogate: Whether to use surrogate gradients (custom VJP) for backprop
		n_iters: Number of gradient descent iterations per token
		wd: Weight decay coefficient for state updates
		lr: Learning rate for state updates (if None, uses muP scaling: 0.005 * sqrt(hidden_d/768))
		block_size: If set, process tokens in blocks of this size (reduces memory)
		rngs: Random number generators
		mesh: Optional JAX mesh for distributed training
	"""
	def __init__(
		self,
		model_d: int,
		hidden_d: int | None = None,
		q_heads: int = 1,
		module: Callable = GMLP,
		module_kwargs: dict | None = None,
		param_dtype: jnp.dtype = jnp.bfloat16,
		surrogate: bool = True,
		n_iters: int = 1,
		wd: float = 0.01,
		lr: float = 0.01,
		block_size: int | None = None,
		*,
		rngs: rng.Rngs,
		mesh: Optional[jax.sharding.Mesh] = None
	):
		super().__init__()

		if hidden_d is None:
			hidden_d = model_d

		# muP scaling: lr scales as sqrt(hidden_d / base_dim)
		# Base lr=0.005 was tuned for hidden_d=768
		lr = lr * (hidden_d / 768) ** 0.5

		if q_heads < 1:
			raise ValueError("q_heads must be >= 1")

		if module_kwargs is None:
			module_kwargs = {}

		self.model_d = model_d
		self.hidden_d = hidden_d
		self.q_heads = q_heads
		self.surrogate = surrogate

		# Create k, v projections (shared across q heads)
		size_dict_kv = {'d': model_d, 'h': hidden_d}
		self.k_proj = Einsum(
			"bnd,dh->bnh",
			size_dict=size_dict_kv,
			rngs=rngs,
			dtype=param_dtype,
			mesh=mesh,
			sharding=(None, 'tensor') if mesh is not None else None
		)
		self.v_proj = Einsum(
			"bnd,dh->bnh",
			size_dict=size_dict_kv,
			rngs=rngs,
			dtype=param_dtype,
			mesh=mesh,
			sharding=(None, 'tensor') if mesh is not None else None
		)

		# Q projection with q_heads dimension
		size_dict_q = {'d': model_d, 'h': hidden_d, 'i': q_heads}
		self.q_proj = Einsum(
			"bnd,dih->bnih",
			size_dict=size_dict_q,
			rngs=rngs,
			dtype=param_dtype,
			mesh=mesh,
			sharding=(None, None, 'tensor') if mesh is not None else None
		)

		# Create inner module - its parameters will be the TTT state
		self.inner_module = module(
			model_d=hidden_d,
			rngs=rngs,
			mesh=mesh,
			param_dtype=param_dtype,
			**module_kwargs
		)

		# RMSNorm after TTT, before output projection
		self.norm = RMSNorm(hidden_d, rngs=rngs, mesh=mesh, scale_mode='none')

		# Output projection combines q_heads
		size_dict_out = {'d': model_d, 'h': hidden_d, 'i': q_heads}
		self.out_proj = Einsum(
			"bnih,ihd->bnd",
			size_dict=size_dict_out,
			rngs=rngs,
			dtype=param_dtype,
			mesh=mesh,
			sharding=(None, 'tensor', None) if mesh is not None else None,
			initializer=zeros_init
		)

		# Create the TTT forward function
		self.ttt_fn = ttt(
			self._fwd_fn,
			surrogate=surrogate,
			n_iters=n_iters,
			wd=wd,
			lr=lr,
			block_size=block_size
		)
		self.inner_module_gdef = nnx.graphdef(self.inner_module)

	def _fwd_fn(self, module_state: nnx.State, x: jax.Array) -> jax.Array:
		"""Forward function for TTT: reconstructs module from state and applies it.

		Args:
			module_state: NNX State containing the module's parameters (this is the TTT state)
			x: Input of shape (hidden_d,) or (q_heads, hidden_d)

		Returns:
			Output of shape (hidden_d,) or (q_heads, hidden_d)
		"""
		module = nnx.merge(self.inner_module_gdef, module_state)

		# Handle both single-query and multi-query cases
		if x.ndim == 1:
			# Single query: (hidden_d,) -> add batch and seq dims
			x = x[None, None, ...]  # (1, 1, hidden_d)
			x = module(x)
			return x[0, 0]
		elif x.ndim == 2:
			# Multi-query: (q_heads, hidden_d) -> treat q_heads as batch
			x = x[:, None, ...]  # (q_heads, 1, hidden_d)
			x = module(x)
			return x[:, 0]
		else:
			raise ValueError(f"Unsupported input rank for TTT fwd: {x.shape}")

	def __call__(self, x: jax.Array, **_) -> jax.Array:
		"""Apply TTT layer.

		Args:
			x: Input of shape (batch, seq_len, model_d)

		Returns:
			Output of shape (batch, seq_len, model_d)
		"""
		# Project input to k, v, q
		k = self.norm(self.k_proj(x))  # (batch, seq_len, hidden_d)
		v = self.norm(self.v_proj(x))  # (batch, seq_len, hidden_d)
		q = self.norm(self.q_proj(x))  # (batch, seq_len, q_heads, hidden_d)


		# Apply TTT algorithm
		hidden, final_state = self.ttt_fn(k, v, q, nnx.state(self.inner_module))
		# hidden: (batch, seq_len, q_heads, hidden_d)

		# Normalize and project back to model dimension
		hidden = self.norm(hidden)
		output = self.out_proj(hidden)
		return output

	def apply_ttt(self, k, v, q):
		"""Apply TTT algorithm directly on k, v, q.

		Args:
			k: (batch, seq_len, hidden_d)
			v: (batch, seq_len, hidden_d)
			q: (batch, seq_len, q_heads, hidden_d)

		Returns:
			(hidden_output, final_state) tuple
		"""
		return self.ttt_fn(k, v, q, nnx.state(self.inner_module))
