"""
Next token prediction loss for language models with chunked computation.
"""
import functools
from typing import Optional, Tuple, Callable

import jax
import jax.numpy as jnp
from flax import nnx
from optax import softmax_cross_entropy_with_integer_labels
import ueaj.utils as us

def ntp_loss_mask(
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None,
	loss_mask: Optional[jax.Array] = None
) -> jax.Array:
	"""Create a loss mask for next-token prediction tasks.
	
	This function generates a binary mask indicating which positions should be
	included in the loss computation. It handles multiple masking scenarios:
	- Always masks the last position (no target for final token)
	- Masks positions after padding tokens
	- Masks positions at document boundaries
	- Applies any additional custom mask
	
	Args:
		inputs: Input token IDs of shape (batch_size, sequence_length).
		document_ids: Optional array of document IDs for each token, same shape as inputs.
			Loss is masked at document boundaries (where document ID changes).
		pad_token_id: Optional token ID to ignore in loss computation (typically padding).
			Positions after a padding token are masked.
		loss_mask: Optional custom loss mask to apply, same shape as inputs[:, :].
			Combined with other masks via element-wise multiplication.
	
	Returns:
		Binary mask of shape (batch_size, sequence_length) where 1 indicates
		positions to include in loss and 0 indicates positions to mask.
	
	Example:
		>>> inputs = jnp.array([[1, 2, 3, 0, 0]])  # 0 is padding
		>>> mask = ntp_loss_mask(inputs, pad_token_id=0)
		>>> # Result: [[1, 1, 0, 0, 0]]  # Masks padding and last position
	"""
	mask = jnp.ones(inputs.shape[:2], dtype=jnp.int4)
	mask = mask.at[:, -1].set(0)

	if loss_mask is not None:
		mask *= loss_mask

	if pad_token_id is not None:
		mask = mask.at[:, :-1].mul(inputs[:, 1:] != pad_token_id)

	if document_ids is not None:
		mask = mask.at[:, :-1].mul(document_ids[:, :-1] == document_ids[:, 1:])

	return mask

def ntp_args(
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None,
	loss_mask: Optional[jax.Array] = None
) -> Tuple[jax.Array, jax.Array, jax.Array] | Tuple[jax.Array, jax.Array]:
	"""Prepare targets and mask for next-token prediction loss.
	
	This function creates the target sequence by shifting inputs and generates
	the corresponding loss mask. It's a convenience function that combines
	target creation with mask generation.
	
	Args:
		inputs: Input token IDs of shape (batch_size, sequence_length).
		document_ids: Optional array of document IDs for each token.
			See ntp_loss_mask for details.
		pad_token_id: Optional padding token ID to mask.
			See ntp_loss_mask for details.
		loss_mask: Optional custom loss mask.
			See ntp_loss_mask for details.
	
	Returns:
		Tuple of (targets, mask) where:
		- targets: Shifted input tokens where each position contains the next token,
		  shape (batch_size, sequence_length). The last position is set to the
		  original last token (will be masked anyway).
		- mask: Loss mask from ntp_loss_mask.
	
	Example:
		>>> inputs = jnp.array([[1, 2, 3, 4, 5]])
		>>> targets, mask = ntp_args(inputs)
		>>> # targets: [[2, 3, 4, 5, 5]]  # Each position has next token
		>>> # mask:    [[1, 1, 1, 1, 0]]  # Last position masked
	"""
	mask = ntp_loss_mask(inputs, document_ids, pad_token_id, loss_mask)
	targets = inputs.at[:, :-1].set(inputs[:, 1:])
	return targets, mask

@jax.named_scope("chunked_loss")
def chunked_softmax_cross_entropy(
	inputs: jax.Array,
	activations: jax.Array,
	logit_projection: Callable[[jax.Array, ...], jax.Array],
	chunk_size: int = 1024,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None,
	loss_mask: Optional[jax.Array] = None,
	return_loss_mask: bool = False,
	**logit_proj_kwargs
) -> jax.Array | Tuple[jax.Array, jax.Array]:
	"""Compute cross-entropy loss for language modeling with chunked processing.
	
	This function efficiently computes the next-token prediction loss by processing
	the sequence in chunks. This approach reduces memory usage compared to
	materializing all logits at once, which is crucial for large vocabulary sizes.
	
	The function:
	1. Prepares targets and mask using ntp_args
	2. Processes activations in chunks through the logit projection
	3. Computes softmax cross-entropy loss for each chunk
	4. Returns per-token loss values
	
	Args:
		inputs: Input token IDs of shape (batch_size, sequence_length).
		activations: Hidden states from model of shape 
			(batch_size, sequence_length, hidden_dim).
		logit_projection: Function that projects hidden states to vocabulary logits.
			Should map (batch_size, chunk_size, hidden_dim) -> 
			(batch_size, chunk_size, vocab_size).
		chunk_size: Number of tokens to process at once. Larger values use more
			memory but may be faster. Default: 1024.
		document_ids: Optional document IDs for masking at boundaries.
			See ntp_loss_mask for details.
		pad_token_id: Optional padding token ID to mask.
			See ntp_loss_mask for details.
		loss_mask: Optional custom loss mask.
			See ntp_loss_mask for details.
		return_loss_mask: Whether to return the loss mask.
	
	Returns:
		Per-token loss values of shape (batch_size, sequence_length).
		Masked positions will have loss value of 0.
	
	Example:
		>>> from ueaj.model import LlamaModel
		>>> from ueaj.llama.weight_loader import from_pretrained
		>>> model = from_pretrained(LlamaModel, "meta-llama/Llama-3.2-1B")
		>>> inputs = jnp.array([[1, 2, 3, 4, 5]])
		>>> activations = model.learn_associative(inputs)
		>>> loss = chunked_softmax_cross_entropy(
		...     inputs=inputs,
		...     activations=activations,
		...     logit_projection=model.get_logits,
		...     chunk_size=512
		... )
		>>> avg_loss = jnp.mean(loss)  # Average over all valid tokens
	
	Note:
		The chunked processing is implemented using ueaj.utils.chunked_scan,
		which efficiently processes the sequence without materializing all
		intermediate results at once.
	"""
	targets, mask = ntp_args(inputs, loss_mask=loss_mask, document_ids=document_ids, pad_token_id=pad_token_id)

	@jax.named_scope("chunked_loss")
	def loss_fn(_, x):
		activations, targets, mask = x
		logits = logit_projection(activations, **logit_proj_kwargs)
		loss = softmax_cross_entropy_with_integer_labels(logits, targets)
		return None, loss * mask.astype(loss.dtype)

	_, output = us.chunked_scan(loss_fn, None, (activations, targets, mask), chunk_size=chunk_size, axis=1, out_axis=1, use_checkpointing=True)
	if return_loss_mask:
		return output, mask
	return output

if __name__ == "__main__":
	tok = jnp.array([[1, 2, 3, 4, 5, 0]])
	arr = jnp.array([[1, 1, 1, 2, 2, 2]])
	args = ntp_args(tok, arr, 0)
	print("Inputs: \t", tok)
	print("Doc Ids:\t", arr)
	print("Targets:\t", args[1])
	print("Mask:   \t", args[0])
