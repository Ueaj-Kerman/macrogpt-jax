from abc import ABC
from abc import ABC
from dataclasses import dataclass, replace
from math import ceil
from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
from transformer_engine.jax.attention import SequenceDescriptor, \
	AttnBiasType, \
	AttnMaskType, \
	QKVLayout
from transformer_engine.jax.flax.transformer import _FusedDotProductAttention, _UnfusedDotProductAttention

from ueaj.model import ueajsum as us
from ueaj.model.attention import pe
from ueaj.utils.argutils import either
from ueaj.utils.gradutils import debug_grad_flow, te_gradient_workaround


@dataclass(frozen=True)
class AttentionConfig:
	model_d: int

	kq_d: int
	v_head_d: int

	kv_heads: int
	kv_q_ratio: int

	rope_theta: float | None
	window_size: int | None = None

	param_config: us.ParamConfig = us.ParamConfig("", group=nnx.Param)

	act_fn: Callable[[jax.Array], jax.Array] | None = None

	_k_config: us.ParamConfig | None = None
	_v_config: us.ParamConfig | None = None
	_q_config: us.ParamConfig | None = None
	_o_config: us.ParamConfig | None = None

	_fused: bool | None = None

	dropout: float = 0.

	@property
	def k_config(self):
		return either(self._k_config, self.param_config)

	@property
	def v_config(self):
		return either(self._v_config, self.param_config)

	@property
	def q_config(self):
		return either(self._q_config, self.param_config)

	@property
	def o_config(self):
		return either(self._o_config, self.param_config.with_initializer(nnx.initializers.zeros))

	@property
	def fused(self) -> bool:
		"""Whether to use fused KV projection. Auto-enables when K and V dims match."""
		if self._fused is not None:
			return self._fused
		# Auto-fuse when K and V dimensions are the same
		return self.kq_d == self.v_head_d

	def with_down(self, config: us.ParamConfig):
		# Check for redundancy - if config is same as param_config, raise error
		if config == self.param_config:
			raise ValueError("Redundant specification: k_config is same as param_config")
		return replace(self, _k_config=config)

	def with_up(self, config: us.ParamConfig):
		# Check for redundancy - if config is same as param_config, raise error
		if config == self.param_config:
			raise ValueError("Redundant specification: v_config is same as param_config")
		return replace(self, _v_config=config)

	def with_q(self, config: us.ParamConfig):
		# Check for redundancy - if config is same as param_config, raise error
		if config == self.param_config:
			raise ValueError("Redundant specification: q_config is same as param_config")
		return replace(self, _q_config=config)

	def with_o(self, config: us.ParamConfig):
		# Check for redundancy - if config is same as param_config with zeros init, raise error
		if config == self.param_config.with_initializer(nnx.initializers.zeros):
			raise ValueError("Redundant specification: o_config is same as param_config with zeros initializer")
		return replace(self, _o_config=config)

	def with_fused(self, fused: bool = True):
		assert not fused or self.kq_d == self.v_head_d, "Fused kv must have same kq_d and v_head_d"
		return replace(self, _fused=fused)

	def window_tuple(self) -> tuple[int, int] | None:
		return (0, self.window_size) if self.window_size else None


class SoftmaxAttention(nnx.Module, ABC):
	def __init__(self, config: AttentionConfig, rngs: rng.Rngs):
		super().__init__()
		self.config = config

		size_dict = {
			'd': config.model_d,
			'k': config.kq_d,
			'v': config.v_head_d,
			'h': config.kv_heads,
			'i': config.kv_q_ratio,
			'f': 2
		}

		make_ueajsum = lambda c: us.Ueajsum(c, size_dict, rngs=rngs)
		if config.fused:
			assert config.kq_d == config.v_head_d, "Fused kv must have same kq_d and v_head_d"
			self.kv = make_ueajsum(
				us.parse("bnd,*fdhk->bnfhk").param(config.k_config).in_axes({1: (1,)}).batch_axes({1: (0,)}),
			)
			self.fused_kv = True
		else:
			self.k = make_ueajsum(us.parse("bnd,*dhk->bnhk").param(config.k_config).in_axes_zero())
			self.v = make_ueajsum(us.parse("bnd,*dhv->bnhv").param(config.v_config).in_axes_zero())
			self.fused_kv = False

		self.q = make_ueajsum(us.parse("bnd,*dhik->bnhik").param(config.q_config).in_axes_zero())
		self.o = make_ueajsum(
			us.parse("bnhiv,*hivd->bnd").param(config.o_config).in_axes({1: (0, 1, 2)}).group_map(
				lambda x: x.with_initializer(nnx.initializers.zeros),
				nnx.Param
			)
		)

		if config.rope_theta is not None:
			# Use the same dtype as the attention mechanism
			rope_dtype = config.q_config.dtype if config.q_config.dtype is not None else jnp.float32
			self.rope = pe.RoPE(pe.RoPEConfig(config.rope_theta, config.kq_d, rope_dtype))
		else:
			self.rope = None

	def __call__(self, x, **kwargs):
		position_ids = kwargs.get('position_ids', None)
		if position_ids is not None:
			assert position_ids.shape[:2] == x.shape[:2], "Position IDs must match input shape"

		if self.fused_kv:
			kv = self.kv(x)
			k = kv[:, :, 0, :, :]
		else:
			k = self.k(x)
			v = self.v(x)
		q = self.q(x)

		if self.rope:
			if position_ids is not None:
				# Use provided position IDs for RoPE
				rope = self.rope.compute_freqs(position_ids)
				k = self.rope.invoke(k, rope)
				q = self.rope.invoke(q, rope)
			elif 'rope' in kwargs:
				rope = kwargs['rope']
				sin_part, _ = rope
				assert sin_part.shape[:2] == q.shape[:2] or sin_part.shape[0] == q.shape[
					1], f"RoPE frequencies must match input shape, got {sin_part.shape[:2]} and {q.shape[:2]}"
				k = self.rope.invoke(k, rope)
				q = self.rope.invoke(q, rope)
			else:
				k = self.rope(k)
				q = self.rope(q)

		# No caching - just prepare tensors
		if self.fused_kv:
			kv = kv.at[:, :, 0, :, :].set(k)
			kv = kv.astype(jnp.bfloat16)
		else:
			k = k.astype(jnp.bfloat16)
			v = v.astype(jnp.bfloat16)

		# Reshape tensors for attention
		b, n, h, i, k_dim = q.shape
		q_reshaped = q.reshape(b, n, h * i, k_dim).astype(jnp.bfloat16)

		attn_kwargs = {k: v for k, v in kwargs.items() if k not in ['position_ids', 'rope']}

		fused = self.fused_kv
		if jax.default_backend() == 'gpu':
			model = _FusedDotProductAttention(
				attention_dropout=self.config.dropout,
				attn_mask_type=AttnMaskType.PADDING_CAUSAL_MASK,
				attn_bias_type=AttnBiasType.NO_BIAS,
				qkv_layout=QKVLayout.BSHD_BS2HD if self.fused_kv else QKVLayout.BSHD_BSHD_BSHD,
				window_size=self.config.window_tuple(),
				max_segments_per_seq=ceil(n / 1024),
			)
		else:  # TODO TPU attention, maybe just ring attn repo?
			if fused:
				k, v = kv[:, :, 0, :, :], kv[:, :, 1, :, :]
				fused = False
			model = _UnfusedDotProductAttention(
				attention_dropout=self.config.dropout,
				attn_bias_type=AttnBiasType.NO_BIAS,
				attn_mask_type=AttnMaskType.PADDING_CAUSAL_MASK,
				transpose_batch_sequence=False,
				window_size=self.config.window_tuple(),
			)
		if fused:
			out = model.apply({}, query=q_reshaped, key=kv, value=None, **attn_kwargs)
		else:
			out = model.apply({}, query=q_reshaped, key=k, value=v, **attn_kwargs)

		out = out.reshape(b, n, h, i, self.config.v_head_d).astype(jnp.bfloat16)
		if self.config.act_fn is not None:
			out = self.config.act_fn(out)
		y = self.o(out)
		return y


if __name__ == '__main__':
	import os

	os.environ["NVTE_FUSED_ATTN"] = os.environ.get("NVTE_FUSED_ATTN", "1")

	input_embeds = jnp.zeros((1, 1024, 512), dtype=jnp.float16)
	# attention_kwargs = dict(
	# 	sequence_descriptor=SequenceDescriptor.from_segment_ids_and_pos(
	# 		jnp.array([[1, 1, 1, 1, 1, 1, 0, 0] * 16])
	# 	),
	# )
	attn = SoftmaxAttention(
		AttentionConfig(model_d=512, kq_d=64, v_head_d=64, kv_heads=8, kv_q_ratio=1, rope_theta=0.1),
		rngs=rng.Rngs(0)
	)
	# Test with float16 (should use fused)
	print("Testing with float16...")
	# output = attn(input_embeds, **attention_kwargs)
	print(f"Success! Input shape: {input_embeds.shape}, Output shape: {output.shape}")
