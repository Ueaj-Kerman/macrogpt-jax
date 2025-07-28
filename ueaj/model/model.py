"""
Llama model implementation for loading and running Llama models from HuggingFace.
"""

from ueaj.model.layer import *
from ueaj.llama.weight_loader import *
from ueaj.model.rmsnorm import *
from ueaj.utils.configurator import *


@config
class LlamaModel(nnx.Module):
	"""Llama model with minimal configuration.
	
	Only necessary parameters:
	- vocab_size: Vocabulary size
	- model_d: Hidden dimension
	- num_layers: Number of transformer layers
	- rngs: Random number generators
	
	Component creation functions:
	- transformer_layer: Creates transformer layers
	- norm: Creates normalization layers
	- embed: Creates embedding layer
	
	All other configuration passed through kwargs to components.
	"""

	def __init__(self, 
		vocab_size: int,
		model_d: int, 
		num_layers: int,
		rngs: rng.Rngs,
		tie_word_embeddings: bool = False,
		transformer_layer: Callable = TransformerLayer,
		norm: Callable = RMSNorm,
		embed: Callable = nnx.Embed,
		**kwargs  # Pass through to components
	):
		super().__init__()
		# Store necessary config
		self.vocab_size = vocab_size
		self.model_d = model_d
		self.num_layers = num_layers
		self.tie_word_embeddings = tie_word_embeddings
		
		# Extract some kwargs for local use
		self.max_position_embeddings = kwargs.get('max_position_embeddings', 131072)
		self.head_cap = kwargs.get('head_cap', 'tanh')

		# Token embeddings - invoke embed callable with defaults
		embed_dtype = kwargs.get('param_dtype', jnp.bfloat16)
		self.embed_tokens = embed(
			num_embeddings=vocab_size,
			features=model_d,
			dtype=embed_dtype,
			param_dtype=embed_dtype,
			embedding_init=nnx.initializers.normal(stddev=1.0),
			rngs=rngs
		)

		# Create transformer layers
		@nnx.split_rngs(splits=num_layers)
		@nnx.vmap(axis_size=num_layers)
		def create_block(rngs: nnx.Rngs):
			return transformer_layer(
				model_d=model_d,
				rngs=rngs
			)
		self.layers = create_block(rngs)

		# Final layer norm - invoke norm callable
		self.norm = norm(model_d=model_d, rngs=rngs)

		# Output projection (lm_head) - only if not tied
		if not tie_word_embeddings:
			self.lm_head = nnx.Linear(
				in_features=model_d,
				out_features=vocab_size,
				use_bias=False,
				dtype=embed_dtype,
				param_dtype=embed_dtype,
				kernel_init=nnx.initializers.zeros,
				rngs=rngs
			)
		else:
			self.lm_head = None
	
	@property
	def config(self):
		"""For backwards compatibility."""
		return self

	def default_kwargs(self, batch_size: int, seq_len: int, **kwargs) -> Dict[str, Any]:
		# Get or create segment IDs
		if 'query_segment_ids' not in kwargs:
			# Default: all tokens in same segment
			kwargs['query_segment_ids'] = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)
		if 'kv_segment_ids' not in kwargs:
			kwargs['kv_segment_ids'] = kwargs['query_segment_ids']

		if 'rope' not in kwargs:
			# Check if first layer has rope (layers is vmapped)
			first_layer = jax.tree.leaves(self.layers)[0]
			if hasattr(first_layer, 'attn') and hasattr(first_layer.attn, 'rope') and first_layer.attn.rope is not None:
				# Create rope if not provided
				rope = first_layer.attn.rope
				# Cast position IDs to float to ensure correct dtype propagation
				pos_ids = jnp.arange(seq_len, dtype=jnp.float32)
				kwargs['rope'] = rope.compute_freqs(pos_ids)
		
		# Create attention mask and position IDs for all layers
		if 'position_ids' not in kwargs:
			# Default to sequential positions
			kwargs['position_ids'] = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
			kwargs['position_ids'] = jnp.broadcast_to(kwargs['position_ids'], (batch_size, seq_len))
		
		query_positions = kwargs.get('query_positions', kwargs['position_ids'])
		kv_positions = kwargs.get('kv_positions', kwargs['position_ids'])
		
		# Import kvax components
		from kvax.ops import create_attention_mask
		from kvax.utils.common import FlashAttentionParamsConfig
		from ueaj.utils.kvax_context import get_kvax_context
		
		# Get kvax context
		kvax_ctx = get_kvax_context()
		
		# Create attention mask once for all layers
		with kvax_ctx():
			# Create flash attention params
			fwd_params = FlashAttentionParamsConfig(
				query_block_size=64,
				kv_block_size=64,
				num_warps=4,
				num_stages=3
			)
			bwd_params = FlashAttentionParamsConfig(
				query_block_size=64,
				kv_block_size=64,
				num_warps=4,
				num_stages=3
			)
			
			# Create attention mask
			attention_mask = create_attention_mask(
				query_positions=query_positions,
				query_segment_ids=kwargs['query_segment_ids'],
				kv_positions=kv_positions,
				kv_segment_ids=kwargs['kv_segment_ids'],
				calc_bwd_mask=True,  # Required for gradient computation
				fwd_params=fwd_params,
				bwd_params=bwd_params,
			)
		
		kwargs['attention_mask'] = attention_mask
		kwargs['query_positions'] = query_positions
		kwargs['kv_positions'] = kv_positions

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

		# Import kvax context
		from ueaj.utils.kvax_context import get_kvax_context
		kvax_ctx = get_kvax_context()
		
		# Run all layers within kvax context
		with kvax_ctx():
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
		return from_pretrained(cls, model_path, rngs, dtype, abstract)
