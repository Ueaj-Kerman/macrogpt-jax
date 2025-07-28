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
		rngs: nnx.Rngs,
		scale_mode: Literal['centered', 'uncentered', 'scalar', 'none'] = 'centered',
		eps: float = 1e-6,
		param_dtype: jnp.dtype = jnp.bfloat16,
		initializer: Optional[Callable] = None,
	):
		super().__init__()
		self.eps = eps
		self.scale = None
		self.recenter = False

		if scale_mode == 'none':
			return

		# Determine shape and default initializer based on scale_mode
		if scale_mode == 'centered':
			self.recenter = True
			shape = (model_d,)
			default_init = nnx.initializers.zeros
		elif scale_mode == 'uncentered':
			self.recenter = False
			shape = (model_d,)
			default_init = nnx.initializers.ones
		elif scale_mode == 'scalar':
			self.recenter = True
			shape = ()
			default_init = nnx.initializers.zeros
		else:
			raise ValueError(f"Unknown scale_mode: {scale_mode}")

		# Use provided initializer or the default
		if initializer is None:
			initializer = default_init

		self.scale = nnx.Param(initializer(rngs.param(), shape, param_dtype))

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


