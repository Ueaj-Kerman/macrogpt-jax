from typing import *

import jax
from flax import nnx
from jax import numpy as jnp

from ueaj.utils import *
from ueaj.utils.configurator import *


@config
class RMSNorm(nnx.Module):
	"""Root Mean Square Layer Normalization.
	
	A simplified and configurable RMSNorm implementation that supports multiple
	scaling modes.
	"""

	def __init__(self,
		model_d: int,
		scale_mode: Literal['centered', 'uncentered', 'uncentered_scalar', 'centered_scalar', 'none'] = 'uncentered',
		eps: float = 1e-6,
		param_dtype: jnp.dtype = jnp.bfloat16,
		initializer: Optional[Callable] = None,
		*,
		rngs: nnx.Rngs,
		mesh: Optional[jax.sharding.Mesh] = None,
		sharding: Optional[str | None] = 'tensor',
	):
		super().__init__()
		self.eps = eps
		self.recenter = False
		self.sharding = sharding

		if scale_mode == 'none':
			self.scale = None
			return

		# Parse scale_mode to determine configuration
		is_scalar = scale_mode.endswith('_scalar')
		is_centered = scale_mode.startswith('centered')

		# Validate mode
		valid_modes = {'centered', 'uncentered', 'centered_scalar', 'uncentered_scalar', 'none'}
		if scale_mode not in valid_modes:
			raise ValueError(f"Unknown scale_mode: {scale_mode}")

		# Set configuration based on mode
		self.recenter = is_centered
		shape = () if is_scalar else (model_d,)
		default_init = nnx.initializers.zeros if is_centered else nnx.initializers.ones

		# Use provided initializer or the default
		if initializer is None:
			initializer = default_init

		# Initialize scale parameter
		scale_value = initializer(rngs.param(), shape, param_dtype)
		
		# Apply sharding if mesh is provided
		if mesh is not None and shape != ():  # Can't shard scalar
			partition_spec = jax.sharding.PartitionSpec(sharding)
			named_sharding = jax.NamedSharding(mesh, partition_spec)
			scale_value = jax.lax.with_sharding_constraint(scale_value, named_sharding)
		
		self.scale = nnx.Param(scale_value)

	def __call__(self, x):
		input_dtype = x.dtype
		
		# RMS normalization in float32 for stability
		x = x.astype(jnp.float32)
		var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
		x = x * jax.lax.rsqrt(var + self.eps)
		
		# Apply scale if present
		if self.scale is not None:
			x, scale = promote_fp8(x, self.scale.value)
			if self.recenter:
				x = x * (1 + scale)
			else:
				x = x * scale
		
		return x.astype(input_dtype)
	
	def reshard(self, mesh: jax.sharding.Mesh) -> None:
		"""Reshard the scale parameter on the given mesh.
		
		Args:
			mesh: JAX mesh to shard on
		"""
		if self.scale is None:
			# No scale parameter to reshard (scale_mode='none')
			return
		
		# Handle scalar scale (shape=())
		if self.scale.value.shape == ():
			# Scalar can't be sharded, just replicate
			partition_spec = jax.sharding.PartitionSpec()
		else:
			# Vector scale (shape=(model_d,))
			partition_spec = jax.sharding.PartitionSpec(self.sharding)
		
		# Apply sharding constraint
		named_sharding = jax.NamedSharding(mesh, partition_spec)
		self.scale.value = jax.lax.with_sharding_constraint(self.scale.value, named_sharding)


