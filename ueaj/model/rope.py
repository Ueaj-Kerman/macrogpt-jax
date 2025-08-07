from typing import Any, Tuple, Optional

from jax import numpy as jnp
import jax
from flax import nnx
from ueaj.utils import gradutils as gu
from ueaj.utils.configurator import config


def compute_rope_embedding(pos: jax.Array, rope_theta: float, rope_d: int, 
                          value_dtype: Any) -> Tuple[jax.Array, jax.Array]:
	"""Compute RoPE sine and cosine embeddings for given positions."""
	half = rope_d // 2
	freq_seq = jnp.arange(half)
	inv_freq = 1.0 / (rope_theta ** (freq_seq / half))
	angles = jnp.einsum("...l,h->...lh", pos, inv_freq)
	sin_part = jnp.sin(angles).astype(dtype=value_dtype)
	cos_part = jnp.cos(angles).astype(dtype=value_dtype)
	return sin_part, cos_part


@config
class RoPE(nnx.Module):
	def __init__(self, 
		rope_theta: float,
		rope_d: int,
		value_dtype: Any = jnp.float32
	):
		super().__init__()
		self.rope_theta = rope_theta
		self.rope_d = rope_d
		self.value_dtype = value_dtype

	def __call__(self, x: jax.Array, position_ids: Optional[jax.Array] = None, 
	             rope_freqs: Optional[Tuple[jax.Array, jax.Array]] = None) -> jax.Array:
		"""Apply RoPE to a tensor.
		
		Args:
			x: Input tensor
			position_ids: Optional position IDs for RoPE computation
			rope_freqs: Optional pre-computed RoPE frequencies (sin_part, cos_part)
			
		Returns:
			Rotated tensor
		"""
		if position_ids is not None:
			# Use provided position IDs for RoPE
			sin_part, cos_part = compute_rope_embedding(position_ids, self.rope_theta, self.rope_d, self.value_dtype)
		elif rope_freqs is not None:
			# Use pre-computed frequencies
			sin_part, cos_part = rope_freqs
			# Validate shape
			if sin_part.shape[:2] != x.shape[:2] and sin_part.shape[0] != x.shape[1]:
				raise ValueError(f"RoPE frequencies must match input shape, got {sin_part.shape[:2]} and {x.shape[:2]}")
			# Ensure rope embeddings match the computation dtype
			if sin_part.dtype != x.dtype:
				sin_part = sin_part.astype(x.dtype)
				cos_part = cos_part.astype(x.dtype)
		else:
			# Default: sequential positions
			sin_part, cos_part = compute_rope_embedding(jnp.arange(x.shape[1]), self.rope_theta, self.rope_d, self.value_dtype)
		
		return self._apply_rope(x, sin_part, cos_part)
	
	def compute_freqs(self, position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
		"""Compute RoPE frequencies for given position IDs."""
		return compute_rope_embedding(position_ids, self.rope_theta, self.rope_d, self.value_dtype)

	def _apply_rope(self, x: jax.Array, sin_part: jax.Array, cos_part: jax.Array) -> jax.Array:
		"""Apply rotary position embedding to a single tensor."""
		batch_size, seq_len, *rest_dims = x.shape
		model_d = x.shape[-1]
		half = model_d // 2
		x1 = x[..., :half]
		x2 = x[..., half:]

		# Handle different embed shapes
		if sin_part.ndim == 2:  # Shape: (seq_len, half)
			if seq_len < sin_part.shape[0]:
				sin_part = sin_part[:seq_len, ...]
				cos_part = cos_part[:seq_len, ...]
			
			extra_dims = len(x.shape) - 3
			cos_part = cos_part.reshape((1, seq_len) + (1,) * extra_dims + (half,))
			sin_part = sin_part.reshape((1, seq_len) + (1,) * extra_dims + (half,))
		elif sin_part.ndim == 3:  # Shape: (batch_size, seq_len, half)
			# Already in the right shape for broadcasting
			extra_dims = len(x.shape) - 3
			if extra_dims > 0:
				cos_part = cos_part.reshape(cos_part.shape[:2] + (1,) * extra_dims + (half,))
				sin_part = sin_part.reshape(sin_part.shape[:2] + (1,) * extra_dims + (half,))

		# Cast rope frequencies to match input dtype to avoid dtype mismatch
		target_dtype = x.dtype
		cos_part = cos_part.astype(target_dtype)
		sin_part = sin_part.astype(target_dtype)
		
		# Apply rotation
		out1 = x1 * cos_part - x2 * sin_part
		out2 = x2 * cos_part + x1 * sin_part

		rope = jnp.concatenate([out1, out2], axis=-1)
		# Cast gradients to input dtype to handle kvax returning fp32 gradients
		rope = gu.custom_astype(rope, x.dtype, cast_forward=False, cast_backward=True)
		return rope

if __name__ == "__main__":
	# Test with direct instantiation
	rope = RoPE(rope_theta=0.1, rope_d=512, value_dtype=jnp.float32)
	rope(jnp.zeros((2, 3, 512)))
	rope(jnp.zeros((2, 3, 4, 512)))
	rope(jnp.zeros((2, 3, 4, 5, 512)))
	
	# Test with override
	RoPECustom = RoPE.override(rope_theta=10000.0, value_dtype=jnp.bfloat16)
	rope_custom = RoPECustom(rope_d=128)
	print(f"Custom RoPE: theta={rope_custom.rope_theta}, d={rope_custom.rope_d}, dtype={rope_custom.value_dtype}")
