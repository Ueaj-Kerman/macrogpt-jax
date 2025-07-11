"""
Llama model implementation for loading and running Llama models from HuggingFace.
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
from safetensors import safe_open
from transformer_engine.jax.attention import SequenceDescriptor

from ueaj.model.layer import TransformerLayer, TransformerLayerConfig
from ueaj.model.attention.soft_attn import AttentionConfig
from ueaj.model.mlp import MLPConfig
from ueaj.model.rmsnorm import RMSNorm, RMSNormConfig
from ueaj.model import ueajsum as us
from ueaj.model.llama.weight_loader import load_weights_from_safetensors
from ueaj.model.ueajsum import ParamConfig
from ueaj.utils import either


@dataclass
class LlamaConfig:
	"""Configuration for Llama model."""
	tensor_config: ParamConfig
	
	# Core model dimensions
	vocab_size: int = 128256  # Llama 3 vocab size
	model_d: int = 4096  # Hidden dimension
	num_layers: int = 32
	
	# Sequence length
	max_position_embeddings: int = 131072
	
	# Optional: for RMSNorm epsilon (model-wide, not per-layer)
	rms_norm_eps: float = 1e-5
	
	# Model configurations
	tie_word_embeddings: bool = False  # Whether to tie input/output embeddings
	
	# Direct layer configuration options
	mlp_type: Literal["gated", "nongated", "bayesian"] = "gated"
	
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
	activation_fn: Callable[[jax.Array], jax.Array] | None = None  # Default to nnx.swish
	
	# Direct normalization options
	norm_scale: Literal["uncentered", "centered", "none", "single"] = "centered"

	# Whether to recenter hidden states before lm_head
	head_recenter: Literal["recenter", "none"] = "recenter"
	head_cap: Literal["none", "tanh"] = "tanh"
	
	# Private overrides
	_layer_config: TransformerLayerConfig | None = None
	_embed_config: us.ParamConfig | None = None
	_norm_config: RMSNormConfig | None = None

	@classmethod
	def from_pretrained(cls, model_id: str) -> "LlamaConfig":
		"""Load config from HuggingFace model."""
		from transformers import LlamaConfig as HFLlamaConfig

		# Load config using transformers
		hf_config = HFLlamaConfig.from_pretrained(model_id)

		kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
		model_d = hf_config.hidden_size
		tensor_config = ParamConfig("", group=nnx.Param).with_dtype(
			jax.dtypes.canonicalize_dtype(hf_config.torch_dtype)
		)

		# Convert HuggingFace config to our LlamaConfig
		return cls(
			vocab_size=hf_config.vocab_size,
			model_d=model_d,
			num_layers=hf_config.num_hidden_layers,
			max_position_embeddings=hf_config.max_position_embeddings,
			rms_norm_eps=getattr(hf_config, "rms_norm_eps", 1e-5),
			tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
			tensor_config=tensor_config,
			use_gated_mlp=True,
			# Attention parameters
			kq_d=hf_config.head_dim,
			v_head_d=hf_config.head_dim,
			kv_heads=kv_heads,
			kv_q_ratio=hf_config.num_attention_heads // kv_heads,
			rope_theta=getattr(hf_config, "rope_theta", 10000.0),
			# MLP parameters
			hidden_d=hf_config.intermediate_size,
			mlp_type="gated",
			# Normalization
			norm_scale="uncentered",
			head_recenter="none",
			head_cap="none",
		)

	@property
	def layer_config(self) -> TransformerLayerConfig:
		"""Get layer configuration, constructing from individual params if needed."""
		if self._layer_config is not None:
			# Check for redundant specifications
			if (self.kq_d is not None or self.v_head_d is not None or self.kv_heads is not None or 
			    self.attn_activation_fn is not None):
				raise ValueError("Cannot specify both layer_config and individual attention parameters")
			if self.hidden_d is not None or self.activation_fn is not None:
				raise ValueError("Cannot specify both layer_config and individual MLP parameters")
			return self._layer_config
		
		return TransformerLayerConfig(
			model_d=self.model_d,
			param_config=self.tensor_config,
			mlp_type=self.mlp_type,
			kq_d=self.kq_d,
			v_head_d=self.v_head_d,
			kv_heads=self.kv_heads,
			kv_q_ratio=self.kv_q_ratio,
			rope_theta=self.rope_theta,
			window_size=self.window_size,
			dropout=self.dropout,
			attn_activation_fn=self.attn_activation_fn,
			hidden_d=self.hidden_d,
			activation_fn=self.activation_fn,
			norm_scale=self.norm_scale
		)

	@property
	def embed_config(self) -> us.ParamConfig:
		return either(self._embed_config, self.tensor_config.with_initializer(nnx.initializers.normal(stddev=1.)))

	@property
	def norm_config(self) -> RMSNormConfig:
		return either(self._norm_config, RMSNormConfig(
			model_d=self.model_d,
			_scale_dtype=self.tensor_config.dtype,
			scale=self.norm_scale
		))

	def with_tied(self, tie_word_embeddings: bool) -> "LlamaConfig":
		"""Set whether to tie input/output word embeddings."""
		return replace(self, tie_word_embeddings=tie_word_embeddings)

	def with_layer_config(self, layer_config: TransformerLayerConfig) -> "LlamaConfig":
		"""Set the layer configuration."""
		return replace(self, _layer_config=layer_config)

	def with_embed_config(self, embed_config: us.ParamConfig) -> "LlamaConfig":
		"""Set the embedding configuration."""
		return replace(self, _embed_config=embed_config)

	def with_attention_config(self, attention_config: AttentionConfig) -> "LlamaConfig":
		"""Set the attention configuration, updating the layer config."""
		return self.with_layer_config(self.layer_config.with_attention_config(attention_config))

	def with_mlp_config(self, mlp_config: MLPConfig) -> "LlamaConfig":
		"""Set the MLP configuration, updating the layer config."""
		return self.with_layer_config(self.layer_config.with_mlp_config(mlp_config))

	def with_attention_norm_config(self, attention_norm_config: RMSNormConfig) -> "LlamaConfig":
		"""Set the attention normalization configuration, updating the layer config."""
		layer_config = self.layer_config
		layer_config = replace(layer_config, _attention_norm_config=attention_norm_config)
		return self.with_layer_config(layer_config.with_attention_norm(attention_norm_config))

	def with_mlp_norm_config(self, mlp_norm_config: RMSNormConfig) -> "LlamaConfig":
		"""Set the MLP normalization configuration, updating the layer config."""
		return self.with_layer_config(self.layer_config.with_mlp_norm(mlp_norm_config))


class LlamaModel(nnx.Module):
	"""Llama model for inference."""

	def __init__(self, config: LlamaConfig, rngs: rng.Rngs):
		super().__init__()
		self.config = config

		# Token embeddings
		self.embed_tokens = nnx.Embed(
			num_embeddings=config.vocab_size,
			features=config.model_d,
			dtype=config.embed_config.dtype,
			param_dtype=config.embed_config.dtype,
			embedding_init=config.embed_config.initializer,
			rngs=rngs
		)

		# Create transformer layers
		@nnx.split_rngs(splits=config.num_layers)
		@nnx.vmap(axis_size=config.num_layers)
		def create_block(rngs: nnx.Rngs):
			return TransformerLayer(config.layer_config, rngs)
		self.layers = create_block(rngs)

		# Final layer norm
		self.norm = RMSNorm(
			config.norm_config
		)

		# Output projection (lm_head)
		# Only create if embeddings are not tied
		if not config.tie_word_embeddings:
			self.lm_head = nnx.Linear(
				in_features=config.model_d,
				out_features=config.vocab_size,
				use_bias=False,
				dtype=config.tensor_config.dtype,
				param_dtype=config.tensor_config.dtype,
				kernel_init=nnx.initializers.zeros,
				rngs=rngs
			)
		else:
			self.lm_head = None  # Will use embed_tokens.attend()

	def default_kwargs(self, batch_size: int, seq_len: int, **kwargs) -> Dict[str, Any]:
		# Create sequence descriptor if not provided
		if 'sequence_descriptor' not in kwargs:
			# Create segment IDs (all ones = all tokens in same segment)
			segment_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
			kwargs['sequence_descriptor'] = SequenceDescriptor.from_segment_ids_and_pos(segment_ids)

		if 'rope' not in kwargs and hasattr(self.layers.attn, 'rope'):
			# Create rope if not provided
			rope = self.layers.attn.rope
			kwargs['rope'] = rope.compute_freqs(jnp.arange(seq_len))

		return kwargs

	def get_activations(self, input_ids: jax.Array, **kwargs) -> jax.Array:
		"""
		Get hidden states without final norm and lm_head projection.

		Args:
			input_ids: Input token IDs of shape (batch_size, sequence_length)
			**kwargs: Additional arguments (e.g., rope positions)

		Returns:
			Hidden states of shape (batch_size, sequence_length, model_d)
		"""
		# Embed tokens
		act0 = self.embed_tokens(input_ids)

		kwargs = self.default_kwargs(*input_ids.shape, **kwargs)

		@nnx.split_rngs(splits=self.config.num_layers)
		@nnx.scan
		@nnx.remat(policy=jax.checkpoint_policies.nothing_saveable)
		def scan(act, layer):
			return layer(act, **kwargs), None

		act, _ = scan(act0, self.layers)
		# if self.config.head_recenter == "recenter":
		# 	act = act - act0

		return act

	def get_logits(self, activations: jax.Array) -> jax.Array:
		"""
		Apply final norm and lm_head to hidden states.

		Args:
			activations: Hidden states of shape (batch_size, sequence_length, model_d)

		Returns:
			Logits of shape (batch_size, sequence_length, vocab_size)
		"""
		# Final layer norm
		activations = self.norm(activations)

		# Project to vocabulary
		if self.config.tie_word_embeddings:
			# Use embedding's attend method for tied embeddings
			logits = self.embed_tokens.attend(activations)
		else:
			# Use separate lm_head
			logits = self.lm_head(activations)

		if self.config.head_cap == "tanh":
			logits = 15*jnp.tanh(logits/(15 * jnp.sqrt(self.config.model_d)))

		return logits

	def __call__(self, input_ids: jax.Array, **kwargs) -> jax.Array:
		"""
		Forward pass through the model.

		Args:
			input_ids: Input token IDs of shape (batch_size, sequence_length)
			**kwargs: Additional arguments (e.g., rope positions)

		Returns:
			Logits of shape (batch_size, sequence_length, vocab_size)
		"""
		hidden_states = self.get_activations(input_ids, **kwargs)
		return self.get_logits(hidden_states)

	@classmethod
	def from_pretrained(
		cls,
		model_path: str,
		rngs: Optional[rng.Rngs] = None,
		dtype: Optional[jax.typing.DTypeLike] = None,
		abstract: bool = False,
	) -> "LlamaModel":
		"""
		Load a pretrained Llama model from safetensors files.

		Args:
			model_path: Path to directory containing safetensors files
			rngs: Random number generators
			dtype: Data type for model parameters
			abstract: If True, create abstract model without allocating weights

		Returns:
			Loaded LlamaModel instance
		"""
		if rngs is None:
			rngs = rng.Rngs(0)

		# Load config

		config = LlamaConfig.from_pretrained(model_path)

		if dtype is not None:
			config.tensor_config = config.tensor_config.with_dtype(dtype)

		# Create model - use eval_shape for abstract initialization
		if abstract:
			model = nnx.eval_shape(lambda: cls(config, rng.Rngs(0)))
		else:
			model = cls(config, rngs)

		# Load weights from safetensors
		model_dir = Path(model_path)
		if not model_dir.exists():
			raise ValueError(f"Model directory {model_path} does not exist")

		# Find all safetensors files
		safetensor_files = sorted(model_dir.glob("*.safetensors"))
		if not safetensor_files:
			raise ValueError(f"No safetensors files found in {model_path}")

		# Load weights using the cleaner weight loader
		loaded_keys, skipped_keys = load_weights_from_safetensors(model, safetensor_files)

		if skipped_keys:
			print(f"Warning: Skipped {len(skipped_keys)} unrecognized keys during weight loading")
			if len(skipped_keys) < 10:
				print(f"Skipped keys: {skipped_keys}")

		return model

	@classmethod
	def create_abstract(cls, config: LlamaConfig) -> "LlamaModel":
		"""
		Create an abstract Llama model with only shape information (no weight allocation).

		This is useful for:
		- Inspecting model structure without using memory
		- Loading weights from checkpoints more efficiently
		- Distributed model initialization

		Args:
			config: Model configuration

		Returns:
			Abstract LlamaModel with ShapeDtypeStruct instead of actual arrays
		"""

		# Create a function that initializes the model with a new Rngs instance
		def init_fn():
			return cls(config, rng.Rngs(0))

		# Create model using eval_shape to avoid allocating actual weights
		abstract_model = nnx.eval_shape(init_fn)
		return abstract_model
