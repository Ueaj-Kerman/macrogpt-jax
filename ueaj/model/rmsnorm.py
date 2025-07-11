from dataclasses import dataclass, replace
from typing import Literal

import jax
from flax import nnx
from jax import typing
from jax import numpy as jnp

from ueaj.utils.argutils import promote_fp8, either
from ueaj.utils.gradutils import astype_fwd_noop_bwd, te_gradient_workaround
from ueaj.utils.config import DEFAULT_ACCUM_TYPE
from ueaj.utils.gradutils import debug_dtype

@dataclass(frozen=True)
class RMSNormConfig:
	model_d: int
	_scale_dtype: typing.DTypeLike = None
	scale: Literal["uncentered", "centered", "none", "single"] = "centered"

	_accum_dtype: typing.DTypeLike = None

	@property
	def accum_dtype(self):
		return self._accum_dtype or DEFAULT_ACCUM_TYPE.value

	@property
	def scale_dtype(self):
		return either(self._scale_dtype, jnp.bfloat16)

	def with_accum_dtype(self, accum_dtype):
		return replace(self, _accum_dtype=accum_dtype)

	def with_scale(self, scale):
		return replace(self, scale=scale)

	def with_model_d(self, model_d):
		return replace(self, model_d=model_d)

	def with_scale_dtype(self, scale_dtype):
		return replace(self, _scale_dtype=scale_dtype)


class RMSNorm(nnx.Module):
	"""Root Mean Square Layer Normalization.
	
	Supports multiple scaling methods:
	- "none": No scaling applied after normalization
	- "uncentered": Multiply by learned weights (initialized to ones)
	- "centered": Multiply by (1 + learned weights) where weights are initialized to zeros
	- "single": Multiply by a single learned scalar (initialized to one)
	
	Note: The Llama 3 implementation uses "uncentered" scaling with epsilon=1e-5
	(we use 1e-6 by default).
	"""

	def __init__(self, config):
		super().__init__()
		self.method = config.scale
		self.accum_dtype = config.accum_dtype

		initializer: nnx.Initializer | None = None
		shape = None
		
		if config.scale == "uncentered":
			initializer = nnx.initializers.ones
			shape = (config.model_d,)
		elif config.scale == "centered":
			initializer = nnx.initializers.zeros
			shape = (config.model_d,)
		elif config.scale == "single":
			initializer = nnx.initializers.ones
			shape = ()  # scalar
		elif config.scale == "none":
			initializer = None

		if initializer is not None:
			self.scale = nnx.Param(
				initializer(key=jax.random.PRNGKey(0), shape=shape, dtype=config.scale_dtype)
			)
		else:
			self.scale = None

	def __call__(self, x, downcast_grads: bool = True, te_workaround: bool = False):
		input_dtype = x.dtype
		cast_fn = jax.lax.convert_element_type if downcast_grads else astype_fwd_noop_bwd
		# indexing into list test method for memory and gradients and stuff
		# x = debug_dtype(x, "input")

		x = cast_fn(x, self.accum_dtype)
		# x = debug_dtype(x, "accum")

		var = jnp.mean(jnp.square(x), axis=-1, keepdims=True, dtype=self.accum_dtype)
		# Note: Llama 3 uses 1e-5, but we use 1e-6 for better numerical stability
		# var = debug_dtype(var, "var")
		x = x * jax.lax.rsqrt(var + 1e-06)

		# Workaround for TransformerEngine CUDA illegal memory access
		if te_workaround:
			x = te_gradient_workaround(x)

		if self.method == "none":
			return cast_fn(x, input_dtype)

		x, scale = promote_fp8(x, self.scale.value)

		if self.method == "uncentered":
			return cast_fn(x * scale, input_dtype)
		elif self.method == "centered":
			return cast_fn(x * (1 + scale), input_dtype)
		elif self.method == "single":
			return cast_fn(x * scale, input_dtype)
		else:
			raise NotImplementedError(f"Unknown scaling method: {self.method}")


if __name__ == "__main__":
	# Test RMSNorm with different configurations
	config = RMSNormConfig(
		model_d=16,
		_scale_dtype=jnp.bfloat16,
		scale="centered"
	)

	# Create test input
	x = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 16), dtype=jnp.float16)

	# Test centered scaling
	rms_centered = RMSNorm(config)
	print("Centered RMSNorm:")
	print("Input shape:", x.shape, "dtype:", x.dtype)
	output_centered = rms_centered(x)
	print("Output shape:", output_centered.shape, "dtype:", output_centered.dtype)
	print("Mean:", jnp.mean(output_centered))
	print("Std:", jnp.std(output_centered))
	print()

	# Test uncentered scaling
	config_uncentered = RMSNormConfig(
		model_d=16,
		_scale_dtype=jnp.bfloat16,
		scale="uncentered"
	)
	rms_uncentered = RMSNorm(config_uncentered)
	print("Uncentered RMSNorm:")
	output_uncentered = rms_uncentered(x)
	print("Output shape:", output_uncentered.shape, "dtype:", output_uncentered.dtype)
	print("Mean:", jnp.mean(output_uncentered))
	print("Std:", jnp.std(output_uncentered))
	print()

	# Test no scaling
	config_none = RMSNormConfig(
		model_d=16,
		_scale_dtype=jnp.bfloat16,
		scale="none"
	)
	rms_none = RMSNorm(config_none)
	print("No scaling RMSNorm:")
	output_none = rms_none(x)
	print("Output shape:", output_none.shape, "dtype:", output_none.dtype)
	print("Mean:", jnp.mean(output_none))
	print("Std:", jnp.std(output_none))
	print()

	# Test single scalar scaling
	config_single = RMSNormConfig(
		model_d=16,
		_scale_dtype=jnp.bfloat16,
		scale="single"
	)
	rms_single = RMSNorm(config_single)
	print("Single scalar RMSNorm:")
	output_single = rms_single(x)
	print("Output shape:", output_single.shape, "dtype:", output_single.dtype)
	print("Mean:", jnp.mean(output_single))
	print("Std:", jnp.std(output_single))
	print("Scale shape:", rms_single.scale.value.shape)
	print("Scale value:", rms_single.scale.value)
