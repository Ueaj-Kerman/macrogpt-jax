from abc import ABC
from typing import *

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from flax.nnx import rnglib as rng

from kvax.ops.flash_attention_clean import flash_attention

from ueaj.model.einsum import *
from ueaj.model import rope
from ueaj.utils import gradutils as gu
from ueaj.utils.configurator import *


@config
class SoftmaxAttention(nnx.Module, ABC):
	"""Multi-head attention with minimal configuration.
	
	Only necessary parameters:
	- model_d: Model dimension
	- kq_d: Key/query dimension per head
	- kv_heads: Number of key/value heads
	- rngs: Random number generators
	
	Optional features:
	- kv_q_ratio: Ratio of query heads to kv heads (for grouped query attention)
	- rope_theta: RoPE theta for positional encoding
	- window_size: Sliding window attention
	- act_fn: Activation function after attention
	"""
	
	def __init__(self, 
		model_d: int,
		kq_d: int = 64,  # Good default head dimension
		kv_heads: int | None = None,  # Will default to model_d // kq_d
		kv_q_ratio: int = 1,
		rope_theta: float | None = 10000.0,
		window_size: int | None = None,
		act_fn: Callable[[jax.Array], jax.Array] | None = None,
		k: Callable = Einsum,
		v: Callable = Einsum,
		q: Callable = Einsum,
		o: Callable = Einsum,
		attn_scale: float | Literal['mup', 'sp'] = 'sp',
		*,
		rngs: rng.Rngs,
		mesh: Optional[jax.sharding.Mesh] = None
	):
		super().__init__()
		
		# Default kv_heads if not specified
		if kv_heads is None:
			kv_heads = model_d // kq_d
		
		# Store necessary config
		self.model_d = model_d
		self.kq_d = kq_d
		self.v_head_d = kq_d  # Always same as kq_d for simplicity
		self.kv_heads = kv_heads
		self.kv_q_ratio = kv_q_ratio
		self.rope_theta = rope_theta
		self.window_size = window_size
		self.act_fn = act_fn

		# Size dictionary for projections
		size_dict = {
			'd': model_d,
			'k': kq_d,
			'v': kq_d,  # Same as k for simplicity
			'h': kv_heads,
			'i': kv_q_ratio,
			'f': 2
		}

		# Create projections with LeCun initialization
		# LeCun init: stddev = sqrt(1/fan_in) where fan_in = model_d
		self.k = k(
			"bnd,dhk->bnhk",
			size_dict=size_dict,
			rngs=rngs,
			dtype=jnp.bfloat16,
			mesh=mesh,
			sharding=(None, 'tensor', None)
		)
		self.v = v(
			"bnd,dhv->bnhv",
			size_dict=size_dict,
			rngs=rngs,
			dtype=jnp.bfloat16,
			mesh=mesh,
			sharding=(None, 'tensor', None)
		)
		# Q projection with group query attention (i dimension)
		self.q = q(
			"bnd,dhik->bnhik",
			size_dict=size_dict,
			rngs=rngs,
			dtype=jnp.bfloat16,
			mesh=mesh,
			sharding=(None, 'tensor', None, None)
		)


		if mesh and size_dict['i']*size_dict['v']*(size_dict['h'] // mesh.shape['tensor']) > size_dict['d']:
			# down project first to save memory then ring reduce
			down_shards = ('tensor', None, None, None)
		else:
			# perform attention first then ring matmul
			down_shards = (None, None, None, 'tensor')

		# Output projection with zero initialization and batch dims
		self.o = o(
			"bnhiv,hivd->bnd",
			initializer=zeros_init,
			size_dict=size_dict,
			rngs=rngs,
			dtype=jnp.bfloat16,
			mesh=mesh,
			sharding=down_shards
		)

		self.attn_scale = attn_scale

		# Create RoPE if theta provided
		if rope_theta is not None:
			self.rope = rope.RoPE(rope_theta=rope_theta, rope_d=kq_d, value_dtype=jnp.bfloat16)
		else:
			self.rope = None


	@property
	def config(self):
		"""For backwards compatibility with code that accesses self.config"""
		return self
	
	def window_tuple(self) -> tuple[int, int] | None:
		return (0, self.window_size) if self.window_size else None

	def __call__(self, x, **kwargs):
		position_ids = kwargs.get('position_ids', None)
		if position_ids is not None:
			assert position_ids.shape[:2] == x.shape[:2], "Position IDs must match input shape"
		else:
			position_ids = jnp.arange(x.shape[1])

		k = self.k(x)
		v = self.v(x)
		q = self.q(x)

		if self.rope:
			# Apply RoPE to k and q
			k = self.rope(k, position_ids=position_ids, rope_freqs=kwargs.get('rope'))
			q = self.rope(q, position_ids=position_ids, rope_freqs=kwargs.get('rope'))

		# Reshape tensors for attention
		b, n, h, i, k_dim = q.shape
		q_reshaped = q.reshape(b, n, h * i, k_dim).astype(jnp.bfloat16)
		k = k.astype(jnp.bfloat16)
		v = v.astype(jnp.bfloat16)
		# Cast gradients to handle kvax returning fp32 gradients
		v = gu.custom_astype(v, jnp.bfloat16, cast_forward=False, cast_backward=True)
		
		# Get required parameters from kwargs
		attention_mask = kwargs.get('attention_mask')
		if attention_mask is None:
			raise ValueError("attention_mask must be provided in kwargs")
			
		query_positions = kwargs.get('query_positions', position_ids)
		kv_positions = kwargs.get('kv_positions', position_ids)
		query_segment_ids = kwargs.get('query_segment_ids', jnp.zeros((b, n), dtype=jnp.int32))
		kv_segment_ids = kwargs.get('kv_segment_ids', query_segment_ids)
		
		# Apply flash attention with proper scaling
		if self.attn_scale == 'mup':
			scale = np.float32(1.0 / self.kq_d)
		elif self.attn_scale == 'sp':
			scale = np.float32(1.0 / (self.kq_d ** 0.5))
		else:
			scale = self.attn_scale

		out = flash_attention(
			query=q_reshaped,
			key=k,
			value=v,
			query_positions=query_positions,
			query_segment_ids=query_segment_ids,
			kv_positions=kv_positions,
			kv_segment_ids=kv_segment_ids,
			mask=attention_mask,
			scale=scale,
		)

		out = out.reshape(b, n, h, i, self.kq_d).astype(jnp.bfloat16)
		# Cast output gradients to handle kvax returning fp32 gradients
		out = gu.custom_astype(out, jnp.bfloat16, cast_forward=False, cast_backward=True)
		if self.act_fn is not None:
			out = self.act_fn(out)
		y = self.o(out)
		return y

if __name__ == "__main__":
	# attn = nnx.eval_shape(lambda: SoftmaxAttention(
	# 	1024,
	# 	rngs=rng.Rngs(0),
	# ))
	# state = nnx.state(attn, nnx.Param)
	# jax.tree.map(print, state)
	test = {
		"common": jnp.ones((10, 10)),
		"exception": jnp.ones((10, 10)),
	}

	test2 = {
		"common": jnp.ones((10, 10)),
	}

	jax.tree.map(lambda x: print(x.shape), test2, test)