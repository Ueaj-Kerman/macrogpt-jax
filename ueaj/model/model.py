"""
Llama model implementation for loading and running Llama models from HuggingFace.
"""
import jax

from ueaj.model.layer import *
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
		tie_word_embeddings: bool = False,
		transformer_layer: Callable = TransformerLayer,
		norm: Callable = RMSNorm,
		embed: Callable = nnx.Embed,
		lm_head: Callable = Einsum,
		head_cap: str = 'tanh',
		*,
		rngs: rng.Rngs,
		mesh: Optional[jax.sharding.Mesh] = None,
	):
		super().__init__()
		# Store necessary config
		self.vocab_size = vocab_size
		self.model_d = model_d
		self.num_layers = num_layers
		self.tie_word_embeddings = tie_word_embeddings
		self.mesh = mesh
		
		# Extract some kwargs for local use
		self.head_cap = head_cap

		# Token embeddings - invoke embed callable with defaults
		self.embed_tokens = embed(
			num_embeddings=vocab_size,
			features=model_d,
			dtype=jnp.bfloat16,
			param_dtype=jnp.bfloat16,
			embedding_init=nnx.initializers.normal(stddev=1.0),
			rngs=rngs
		)
		
		# Apply sharding to embeddings if mesh is provided
		if mesh is not None:
			partition_spec = jax.sharding.PartitionSpec('tensor', None)
			named_sharding = jax.NamedSharding(mesh, partition_spec)
			self.embed_tokens.embedding.value = jax.lax.with_sharding_constraint(
				self.embed_tokens.embedding.value, named_sharding
			)

		# Create transformer layers
		@nnx.split_rngs(splits=num_layers)
		@nnx.vmap(axis_size=num_layers)
		def create_block(rngs: nnx.Rngs):
			return transformer_layer(
				model_d=model_d,
				rngs=rngs,
				mesh=mesh
			)
		self.layers = create_block(rngs)

		# Final layer norm - invoke norm callable
		self.norm = norm(model_d=model_d, rngs=rngs, mesh=mesh)

		# Output projection (lm_head) - only if not tied
		if not tie_word_embeddings:
			# Use lm_head callable for output projection
			self.lm_head = lm_head(
				"bnd,dv->bnv",
				size_dict={'d': model_d, 'v': vocab_size},
				initializer=zeros_init,
				rngs=rngs,
				dtype=jnp.bfloat16,
				mesh=mesh,
				sharding=(None, 'tensor')
			)
		else:
			self.lm_head = None
	

	def get_activations(self, input_ids: jax.Array, mesh: Optional[jax.sharding.Mesh] = None, **kwargs) -> jax.Array:
		"""
		Get hidden states without final norm and lm_head projection.

		Args:
			input_ids: Input token IDs of shape (batch_size, sequence_length)
			**kwargs: Additional arguments (e.g., rope positions)

		Returns:
			Hidden states of shape (batch_size, sequence_length, model_d)
			:param input_ids:
			:param mesh:
		"""
		# Embed tokens
		act0: jax.Array = self.embed_tokens(input_ids)

		kwargs = self.default_kwargs(*input_ids.shape, **kwargs)

		# Run all layers
		@nnx.split_rngs(splits=self.num_layers)
		@nnx.scan
		@nnx.remat(policy=jax.checkpoint_policies.nothing_saveable)
		def scan(act, layer):
			# Ensure output dtype matches input dtype
			out = layer(act, mesh=mesh, **kwargs)
			return out.astype(act.dtype), None

		if mesh is not None:
			act0 = jax.lax.with_sharding_constraint(act0, jax.NamedSharding(mesh, TENSOR_SHARDING))

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
		if self.tie_word_embeddings:
			# Use embedding's attend method for tied embeddings
			logits = self.embed_tokens.attend(activations)
		else:
			# Use separate lm_head
			logits = self.lm_head(activations)

		if self.head_cap == "tanh":
			logits = 15*jnp.tanh(logits/15)

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

		# Import kvax components using the new clean API
		from kvax.ops.flash_attention_clean import create_attention_mask
		from kvax.utils.common import FlashAttentionParamsConfig

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

		# Create attention mask using the new API - no context manager needed
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
	
	def reshard(self, mesh: jax.sharding.Mesh) -> None:
		"""Reshard the embedding tokens along the vocabulary dimension.
		
		Args:
			mesh: JAX mesh to shard on
		"""
		# Reshard embeddings - shard along vocabulary dimension
		if hasattr(self.embed_tokens, 'embedding'):
			# Standard nnx.Embed
			partition_spec = jax.sharding.PartitionSpec('tensor', None)
			named_sharding = jax.NamedSharding(mesh, partition_spec)
			self.embed_tokens.embedding.value = jax.lax.with_sharding_constraint(
				self.embed_tokens.embedding.value, named_sharding
			)

