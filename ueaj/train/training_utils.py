"""Training utilities for model training and evaluation."""

import functools
from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import safetensors.flax as st

from kvax.utils import PADDING_SEGMENT_ID
from ueaj.model import LlamaModel
from ueaj.opt.next_token_loss import chunked_softmax_cross_entropy
from ueaj.utils import tensor_stats
from ueaj.utils.compile import compile_function


def test(
		g_def: nnx.GraphDef[LlamaModel],
		params: nnx.State,
		etc: nnx.State,
		inputs: jax.Array,
		document_ids: Optional[jax.Array] = None,
		pad_token_id: Optional[int] = None,
) -> Tuple[jax.Array, Any]:
	"""Evaluate model and compute loss.

	Args:
		g_def: Model graph definition
		params: Model parameters
		etc: Other model state
		inputs: Input token IDs
		document_ids: Document IDs for each token
		pad_token_id: Token ID for padding

	Returns:
		Tuple of (loss_value, (mean_loss, std_loss))
	"""
	model = nnx.merge(g_def, params, etc)
	kwargs: dict[str, Any] = {}
	if document_ids is not None:
		segment_ids = document_ids
	else:
		segment_ids = jnp.zeros_like(inputs, dtype=jnp.int32)
	# Mark padding tokens with PADDING_SEGMENT_ID
	if pad_token_id is not None:
		segment_ids = jnp.where(inputs == pad_token_id, PADDING_SEGMENT_ID, segment_ids)
	kwargs['query_segment_ids'] = segment_ids
	kwargs['kv_segment_ids'] = segment_ids

	# Note: The model.get_activations already handles kvax context internally
	activations = model.get_activations(inputs, **kwargs)

	def logit_projection(hidden_states: jax.Array, g_def, params, etc) -> jax.Array:
		"""Reconstruct a fresh module to keep RNG state from drifting under remat."""
		model = nnx.merge(g_def, params, etc)
		return model.get_logits(hidden_states)

	token_loss, loss_mask = chunked_softmax_cross_entropy(
		inputs,
		activations,
		logit_projection,
		document_ids=document_ids,
		pad_token_id=pad_token_id,
		return_loss_mask=True,
		g_def=g_def,
		params=params,
		etc=etc,
	)
	count = loss_mask.sum(dtype=jnp.float32)

	loss_val = token_loss.sum() / jnp.sqrt(count)
	mean_loss = token_loss.sum() / count
	std_loss = jnp.sqrt(jnp.square(token_loss - mean_loss).sum() / count)
	return loss_val, (mean_loss, std_loss)


def collect_statistics(
		params: nnx.State,
		dmodel: nnx.State,
		delta: nnx.State,
		opt_state: optax.OptState,
		mean_loss: jax.Array,
		std_loss: jax.Array,
		stats_to_collect: Tuple[str, ...],
) -> Dict[str, Any]:
	"""Collect training statistics based on requested metrics.

	Args:
		params: Model parameters
		dmodel: Gradients
		delta: Parameter updates
		opt_state: Optimizer state
		mean_loss: Mean loss value
		std_loss: Standard deviation of loss
		stats_to_collect: Tuple of statistic names to collect

	Returns:
		Dictionary of collected statistics
	"""
	stats = {}

	if 'mean_loss' in stats_to_collect:
		stats['mean_loss'] = mean_loss
	if 'std_loss' in stats_to_collect:
		stats['std_loss'] = std_loss
	if 'grad_norm' in stats_to_collect:
		stats['grad_norm'] = jax.tree.map(lambda dt: jnp.sqrt(jnp.mean(dt ** 2, dtype=jnp.float32)), dmodel)
	if 'update_norm' in stats_to_collect:
		stats['update_norm'] = jax.tree.map(lambda dt: jnp.sqrt(jnp.mean(dt ** 2, dtype=jnp.float32)), delta)

	# Parameter statistics - compute once if any param stat is needed
	# JAX will eliminate the computation if none are requested
	param_stats = jax.tree.map(tensor_stats, params)

	v = next(iter(nnx.to_flat_state(param_stats)))[1].value
	for key in v.keys():
		if f"param_{key}" in stats_to_collect:
			stats[f'param_{key}'] = jax.tree.map(lambda s: s[key], param_stats,
												 is_leaf=lambda x: isinstance(x, dict) and key in x)

	if {'mom_l1_norm', 'mom_l2_norm', 'mom_k_eff'}.intersection(stats_to_collect):
		mom_stats = jax.tree.map(tensor_stats, opt_state[0].mu.layers)

		if 'mom_l1_norm' in stats_to_collect:
			stats['mom_l1_norm'] = jax.tree.map(lambda s: s['l1_norm'], mom_stats,
												is_leaf=lambda x: isinstance(x, dict) and 'l1_norm' in x)
		if 'mom_l2_norm' in stats_to_collect:
			stats['mom_l2_norm'] = jax.tree.map(lambda s: s['l2_norm'], mom_stats,
												is_leaf=lambda x: isinstance(x, dict) and 'l2_norm' in x)
		if 'mom_k_eff' in stats_to_collect:
			stats['mom_k_eff'] = jax.tree.map(lambda s: s.get('k_eff', jnp.nan), mom_stats,
											  is_leaf=lambda x: isinstance(x, dict) and 'k_eff' in x)

	return stats


@functools.partial(jax.jit, donate_argnums=(1, 4), static_argnums=(2, 7, 8))
def train_step(
		g_def: nnx.GraphDef[LlamaModel],
		state: nnx.State,
		opt: Callable,
		opt_args: Dict[str, Any],
		opt_state: optax.OptState,
		inputs: jax.Array,
		document_ids: Optional[jax.Array] = None,
		pad_token_id: Optional[int] = None,
		stats_to_collect: Tuple[str, ...] = (),  # Static arg: tuple of stat names to collect
):
	"""Single training step with gradient computation and parameter update.

	Args:
		g_def: Model graph definition
		state: Model state (parameters + other)
		opt: Optimizer factory function
		opt_args: Arguments for optimizer
		opt_state: Optimizer state
		inputs: Input token IDs
		document_ids: Document IDs for each token
		pad_token_id: Token ID for padding
		stats_to_collect: Tuple of statistic names to collect

	Returns:
		Tuple of (updated_state, updated_opt_state, statistics_dict)
	"""
	params, etc = nnx.split_state(state, nnx.Param, nnx.Not(nnx.Param))
	# Cast params if needed for computation (e.g., to bfloat16)
	casted_params = jax.tree.map(lambda p: p.astype(jnp.bfloat16) if p.dtype != jnp.bfloat16 else p, params)

	dmodel, (mean_loss, std_loss) = jax.grad(test, has_aux=True, argnums=1)(
		g_def,
		casted_params,
		etc,
		inputs,
		document_ids,
		pad_token_id
	)

	with jax.profiler.TraceAnnotation("update"):
		# Update parameters
		dmodel_updates, opt_state = opt(model=nnx.merge(g_def, params, etc), **opt_args).update(dmodel, opt_state,
																								params)
		new_params = optax.apply_updates(params, dmodel_updates)

		# Rescale embeddings
		embed_scale = jnp.sqrt(jnp.mean(jnp.square(params.embed_tokens.embedding.value), axis=1))
		embed_scale = jnp.where(embed_scale > 1.25, embed_scale, 1.)  # reset magnitude if too high
		new_params.embed_tokens.embedding.value = new_params.embed_tokens.embedding.value / embed_scale[:, None]

		delta = jax.tree.map(lambda p, up: p.astype(jnp.float32) - up, params, new_params)
		params = new_params

		state = nnx.merge_state(params, etc)

	# Collect statistics
	stats = collect_statistics(
		params, dmodel, delta, opt_state, mean_loss, std_loss, stats_to_collect
	)

	return state, opt_state, stats


@functools.partial(jax.jit, static_argnums=(4,))
def run_test(graph_def, state, tokens_struct, document_ids_struct, pad_token):
	"""JIT-compiled test function that runs evaluation with bfloat16 parameters.

	Args:
		graph_def: Model graph definition
		state: Model state
		tokens_struct: Input tokens structure
		document_ids_struct: Document IDs structure
		pad_token: Padding token ID (static)

	Returns:
		Tuple of (mean_loss, std_loss)
	"""
	params, etc = nnx.split_state(state, nnx.Param, nnx.Not(nnx.Param))
	params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
	return test(graph_def, params, etc, tokens_struct, document_ids_struct, pad_token)[1]


def compile_training_functions(
		graph_def: nnx.GraphDef[LlamaModel],
		state: nnx.State,
		make_optimizer: Callable,
		opt_arg_0: Dict[str, Any],
		opt_state: optax.OptState,
		tokens_struct: jax.ShapeDtypeStruct,
		document_ids_struct: jax.ShapeDtypeStruct,
		pad_token: int,
) -> Tuple[Callable, Callable, Callable]:
	"""Compile all training and evaluation functions.

	Args:
		graph_def: Model graph definition
		state: Model state
		make_optimizer: Optimizer factory function
		opt_arg_0: Initial optimizer arguments
		opt_state: Optimizer state
		tokens_struct: Structure descriptor for tokens
		document_ids_struct: Structure descriptor for document IDs
		pad_token: Padding token ID

	Returns:
		Tuple of (test_compiled, train_step_fast, train_step_stats):
			- test_compiled: Compiled test function
			- train_step_fast: Fast training step (minimal stats)
			- train_step_stats: Training step with detailed stats
	"""
	print("\nCompiling test step...")
	test_compiled = compile_function(
		run_test,
		sample_args=(graph_def, state, tokens_struct, document_ids_struct, pad_token),
		name="Test Step"
	)

	# Compile two versions of train_step - minimal and detailed statistics
	print("\nCompiling train step with minimal statistics...")
	train_step_fast = compile_function(
		train_step,
		sample_args=(graph_def, state, make_optimizer, opt_arg_0, opt_state, tokens_struct),
		sample_kwargs={
			'document_ids': document_ids_struct,
			'pad_token_id': pad_token,
			'stats_to_collect': ('mean_loss', 'std_loss'),
		},
		name="Train Step (Fast)"
	)

	print("\nCompiling train step with detailed statistics...")
	train_step_stats = compile_function(
		train_step,
		sample_args=(graph_def, state, make_optimizer, opt_arg_0, opt_state, tokens_struct),
		sample_kwargs={
			'document_ids': document_ids_struct,
			'pad_token_id': pad_token,
			'stats_to_collect': (
				'mean_loss', 'std_loss', 'grad_norm', 'update_norm', 'param_l2_norm'
			),
		},
		name="Train Step (Stats)"
	)

	return test_compiled, train_step_fast, train_step_stats


def save_model(state: nnx.State, model_path: str) -> None:
	"""Save model state to a safetensors file.

	Args:
		state: Model state to save
		model_path: Path to save the model file
	"""
	leaves = jax.tree.leaves_with_path(nnx.to_pure_dict(state))
	entries = {}
	for path, leaf in leaves:
		key = ".".join([p.key for p in path])
		entries[key] = leaf
		print(key, leaf.shape)

	st.save_file(entries, model_path)
