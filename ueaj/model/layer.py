import functools
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Callable, Literal

import jax
from flax import nnx
from flax.nnx import rnglib as rng
import jax.numpy as jnp
from transformer_engine.jax.attention import SequenceDescriptor

from ueaj.model.attention.soft_attn import SoftmaxAttention, AttentionConfig
from ueaj.model.mlp import MLP, GMLP, BMLP, MLPConfig
from ueaj.model.rmsnorm import RMSNorm, RMSNormConfig
from ueaj.model.ueajsum import ParamConfig
from ueaj.utils.argutils import either


@dataclass(frozen=True)
class TransformerLayerConfig:
	"""Configuration for a transformer layer combining attention and MLP blocks."""
	model_d: int
	param_config: ParamConfig

	# MLP type: "gated", "nongated", or "bayesian"
	mlp_type: Literal["gated", "nongated", "bayesian"] = "gated"

	# Direct normalization options
	norm_scale: str = "centered"  # "uncentered", "centered", "none"

	# Direct attention configuration options
	kq_d: int | None = None
	v_head_d: int | None = None
	kv_heads: int | None = None
	kv_q_ratio: int | None = None
	rope_theta: float | None = None
	window_size: int | None = None
	dropout: float = 0.
	attn_activation_fn: Callable[[jax.Array], jax.Array] | None = None

	# Direct MLP configuration options
	hidden_d: int | None = None
	activation_fn: Callable[[jax.Array], jax.Array] | None = None

	# Private overrides
	_norm_config: RMSNormConfig | None = None
	_attention_config: AttentionConfig | None = None
	_mlp_config: MLPConfig | None = None
	_attention_norm_config: Optional[RMSNormConfig] = None
	_mlp_norm_config: Optional[RMSNormConfig] = None

	@property
	def norm_config(self) -> RMSNormConfig:
		"""Get normalization configuration."""
		if self._norm_config is not None:
			return self._norm_config
		return RMSNormConfig(
			model_d=self.model_d,
			scale=self.norm_scale,
			_scale_dtype=self.param_config.dtype
		)

	@property
	def attention_config(self) -> AttentionConfig:
		"""Get attention configuration, constructing from individual params if needed."""
		if self._attention_config is not None:
			# Check for redundant specifications
			if (self.kq_d is not None or self.v_head_d is not None or self.kv_heads is not None or 
			    self.attn_activation_fn is not None):
				raise ValueError("Cannot specify both attention_config and individual attention parameters")
			return self._attention_config
		
		return AttentionConfig(
			model_d=self.model_d,
			kq_d=self.kq_d or 128,
			v_head_d=self.v_head_d or self.kq_d or 128,
			kv_heads=self.kv_heads or (self.model_d // 128),
			kv_q_ratio=self.kv_q_ratio or 1,
			rope_theta=self.rope_theta,
			window_size=self.window_size,
			dropout=self.dropout,
			act_fn=self.attn_activation_fn,
			param_config=self.param_config
		)

	@property
	def mlp_config(self) -> MLPConfig:
		"""Get MLP configuration, constructing from individual params if needed."""
		if self._mlp_config is not None:
			# Check for redundant specifications
			if self.hidden_d is not None or self.activation_fn is not None:
				raise ValueError("Cannot specify both mlp_config and individual MLP parameters")
			return self._mlp_config
		
		return MLPConfig(
			model_d=self.model_d,
			hidden_d=self.hidden_d or (self.model_d * 4),
			activation_fn=self.activation_fn or nnx.swish,
			param_config=self.param_config
		)

	@property
	def attn_norm_config(self) -> RMSNormConfig:
		"""Get attention normalization config, defaulting to standard RMSNorm."""
		return either(
			self._attention_norm_config,
			self.norm_config
		)

	@property
	def mlp_norm_config(self) -> RMSNormConfig:
		"""Get MLP normalization config, defaulting to standard RMSNorm."""
		return either(
			self._mlp_norm_config,
			self.norm_config
		)

	def with_attention_config(self, config: AttentionConfig):
		"""Update attention configuration."""
		return replace(self, _attention_config=config)

	def with_mlp_config(self, config: MLPConfig):
		"""Update MLP configuration."""
		return replace(self, _mlp_config=config)

	def with_attention_norm(self, config: RMSNormConfig):
		"""Update attention normalization configuration."""
		return replace(self, _attention_norm_config=config)

	def with_mlp_norm(self, config: RMSNormConfig):
		"""Update MLP normalization configuration."""
		return replace(self, _mlp_norm_config=config)

	def with_mlp_type(self, mlp_type: Literal["gated", "nongated", "bayesian"]):
		"""Set the MLP type."""
		return replace(self, mlp_type=mlp_type)

	def validate(self):
		"""Validate that all dimensions are consistent."""
		assert self.attention_config.model_d == self.model_d, \
			f"Attention model_d {self.attention_config.model_d} != {self.model_d}"
		assert self.mlp_config.model_d == self.model_d, \
			f"MLP model_d {self.mlp_config.model_d} != {self.model_d}"
		if self._attention_norm_config:
			assert self._attention_norm_config.model_d == self.model_d, \
				f"Attention norm model_d {self._attention_norm_config.model_d} != {self.model_d}"
		if self._mlp_norm_config:
			assert self._mlp_norm_config.model_d == self.model_d, \
				f"MLP norm model_d {self._mlp_norm_config.model_d} != {self.model_d}"
		assert self.norm_config.model_d == self.model_d, \
			f"Norm model_d {self.norm_config.model_d} != {self.model_d}"


class TransformerLayer(nnx.Module):
	"""A transformer layer with attention and MLP blocks."""

	def __init__(self, config: TransformerLayerConfig, rngs: rng.Rngs):
		super().__init__()

		# Validate configuration
		config.validate()

		# Initialize attention components
		self.attn = SoftmaxAttention(config.attention_config, rngs)
		self.attn_norm = RMSNorm(config.attn_norm_config)

		# Initialize MLP components
		if config.mlp_type == "gated":
			mlp_class = GMLP
		elif config.mlp_type == "nongated":
			mlp_class = MLP
		elif config.mlp_type == "bayesian":
			mlp_class = BMLP
		else:
			raise ValueError(f"Unknown MLP type: {config.mlp_type}")
		
		self.mlp = mlp_class(config.mlp_config, rngs)
		self.mlp_norm = RMSNorm(config.mlp_norm_config)

		self.config = config

	def __call__(self, x, **kwargs):
		"""
		Forward pass through the transformer layer.

		Args:
			x: Input tensor of shape (batch, sequence, model_d)
			**kwargs: Additional arguments passed to attention (e.g., rope, sequence_descriptor)

		Returns:
			Output tensor of same shape as input
		"""
		# Attention block with residual connection
		# Use TransformerEngine workaround for attention normalization
		x += self.attn(self.attn_norm(x), **kwargs)
		x += self.mlp(self.mlp_norm(x))
		return x

if __name__ == "__main__":
	"""Test TransformerLayer.bwd."""
	# Create layer config with all necessary settings
	model_d = 2048
	tensor_config = ParamConfig("", group=nnx.Param).with_dtype(jnp.bfloat16)# .with_grad_dtype(jnp.float32)

	# Load config using transformers
	attention_config = AttentionConfig(
		model_d=model_d,
		kq_d=64,
		v_head_d=64,
		kv_heads=8,
		kv_q_ratio=1,
		rope_theta=500_000.,
		param_config=tensor_config
	)

	layer_config = TransformerLayerConfig(
		model_d=model_d,
		use_gated_mlp=True,
		attention_config=attention_config, # .with_o(attention_config.o_config.with_initializer(nnx.initializers.lecun_normal())),
		mlp_config=MLPConfig(
			model_d=model_d,
			hidden_d=1024,
			param_config=tensor_config
		),
		norm_config=RMSNormConfig(
			model_d=model_d,
			_scale_dtype=tensor_config.dtype,
			scale="centered"
		),
	)

	kwargs = {
		# 'mask': jnp.ones((8192, 8192), dtype=jnp.bool),
		'sequence_descriptor': SequenceDescriptor.from_seqlens(jnp.array([8192]*4)),
	}

	@nnx.jit
	@nnx.value_and_grad(argnums=(0, 1, 3))
	def layers(layer1, layer2, x, y):
		x = layer1(x, **kwargs)
		x = layer2(x, **kwargs)
		return jnp.sum((x - y)**2) / 2

	# def layers(layer, x, y):
	# 	gdef, params, etc = nnx.split(layer, nnx.Param, ...)
	# 	@jax.jit
	# 	@functools.partial(jax.value_and_grad, argnums=(0, 1))
	# 	def inner(params, x, y):
	# 		layer = nnx.merge(gdef, params, etc)(x, **kwargs)
	# 		return jnp.sum((layer - y)**2) / 2
	# 	return inner(params, x, y)

	layer1 = TransformerLayer(layer_config, rngs=rng.Rngs(0))
	layer2 = TransformerLayer(layer_config, rngs=rng.Rngs(0))

	x = jax.random.normal(jax.random.PRNGKey(0), (4, 8192, model_d)).astype(jnp.bfloat16)
	dh = jax.random.normal(jax.random.PRNGKey(0), (4, 8192, model_d)).astype(jnp.bfloat16)
	output, (dlayer1, dlayer2, dx) = layers(layer1, layer2, x, x+dh)

	print(output - x)
	print(dlayer1)
	print(dlayer2)
	print(dx - dh)