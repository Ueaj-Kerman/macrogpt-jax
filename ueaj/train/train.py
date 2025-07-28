import itertools
import os
import math
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"

import functools
from typing import Optional, Tuple, Any, Callable, Dict, Literal

import jax

import wandb
import uuid

import numpy as np
from flax import nnx
from flax.nnx import rnglib as rng
from transformer_engine.jax.attention import SequenceDescriptor
import time

from ueaj import data, model
from ueaj.opt import chunked_softmax_cross_entropy
from ueaj.utils.stats import k_eff, tensor_stats
from ueaj.utils.compile import compile_function
from ueaj.utils.tensorutil import precision_aware_update
from ueaj.opt.self_pred import learn_associative
from ueaj.opt import OptimizerConfig
from ueaj.opt import multiscale as ms
from ueaj.model import configs
import gc
from jax import numpy as jnp
import optax


def test(
	g_def: nnx.GraphDef[model.LlamaModel],
	params: nnx.State,
	etc: nnx.State,
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None
) -> Tuple[jax.Array, Any]:
	model = nnx.merge(g_def, params, etc)
	kwargs: dict[str, Any] = {}  # {'implementation': 'cudnn'}
	kwargs['sequence_descriptor'] = SequenceDescriptor.from_seqlens((inputs != pad_token_id).sum(axis=1))
	activations = model.get_activations(inputs, **kwargs)

	token_loss, loss_mask = chunked_softmax_cross_entropy(
		inputs,
		activations,
		model.get_logits,
		document_ids=document_ids,
		pad_token_id=pad_token_id,
		return_loss_mask=True
	)
	count = loss_mask.sum(dtype=jnp.float32)

	loss_val = token_loss.sum() / jnp.sqrt(count)
	mean_loss = token_loss.sum() / count
	std_loss = jnp.sqrt(jnp.square(token_loss - mean_loss).sum() / count)
	return loss_val, (mean_loss, std_loss)

@functools.partial(jax.jit, donate_argnums=(1, 4), static_argnums=(2, 7, 8, 9))
def train_step(
	g_def: nnx.GraphDef[model.LlamaModel],
	state: nnx.State,
	opt: Callable[[Dict[str, jax.Array]], optax.GradientTransformation],
	opt_args: Dict[str, Any],
	opt_state: optax.OptState,
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None,
	stats_to_collect: Tuple[str, ...] = (),  # Static arg: tuple of stat names to collect
	technique: Literal['backprop', 'associative'] = 'backprop'
):
	params, etc = nnx.split_state(state, nnx.Param, nnx.Not(nnx.Param))
	# Cast params if needed for computation (e.g., to bfloat16)
	casted_params = jax.tree.map(lambda p: p.astype(jnp.bfloat16), params)
	if technique == 'backprop':
		dmodel, (mean_loss, std_loss) = jax.grad(test, has_aux=True, argnums=1)(
			g_def,
			casted_params,
			etc,
			inputs,
			document_ids,
			pad_token_id
		)
	elif technique == 'associative':
		model = nnx.merge(g_def, casted_params, etc)
		dmodel, (mean_loss, std_loss) = learn_associative(
			model,
			inputs,
			document_ids,
			pad_token_id
		)
	else:
		raise ValueError(f"Unknown technique: {technique}")

	# Update parameters
	dmodel_updates, opt_state = opt(**opt_args).update(dmodel, opt_state, params)
	up_params = optax.apply_updates(params, dmodel_updates)

	delta = jax.tree.map(lambda p, up: p.astype(jnp.float32) - up, params, up_params)
	params = up_params

	state = nnx.merge_state(params, etc)

	# Always return a stats dict
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
			stats[f'param_{key}'] = jax.tree.map(lambda s: s[key], param_stats, is_leaf=lambda x: isinstance(x, dict) and key in x)

	if {'mom_l1_norm', 'mom_l2_norm', 'mom_k_eff'}.intersection(stats_to_collect):
		mom_stats = jax.tree.map(tensor_stats, opt_state[0].mu.layers)

		if 'mom_l1_norm' in stats_to_collect:
			stats['mom_l1_norm'] = jax.tree.map(lambda s: s['l1_norm'], mom_stats, is_leaf=lambda x: isinstance(x, dict) and 'l1_norm' in x)
		if 'mom_l2_norm' in stats_to_collect:
			stats['mom_l2_norm'] = jax.tree.map(lambda s: s['l2_norm'], mom_stats, is_leaf=lambda x: isinstance(x, dict) and 'l2_norm' in x)
		if 'mom_k_eff' in stats_to_collect:
			stats['mom_k_eff'] = jax.tree.map(lambda s: s.get('k_eff', jnp.nan), mom_stats, is_leaf=lambda x: isinstance(x, dict) and 'k_eff' in x)

	return state, opt_state, stats


import datasets
import transformers

# batch_size, seq_len = 5, 4096
batch_size, seq_len = 6, 8192
pad_token = 50431

print("Loading tokenizer...")
tokenizer = transformers.GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
# Set model_max_length properly - this is the attribute that actually controls the warning
tokenizer.model_max_length = seq_len  # Set to 4096
print("Vocab Size:", tokenizer.vocab_size)

print("Loading dataset...")
dataset = datasets.load_dataset(
	"HuggingFaceFW/fineweb-edu",
	name="sample-10BT",
	split="train",
	streaming=True,
)

print("Setting up train iterator...")

# Use the new prepare_dataset function
dataset, (tokens_struct, document_ids_struct) = data.prepare_dataset(
	dataset,
	tokenizer,
	batch_size=batch_size,
	seq_len=seq_len,
	pad_token_id=pad_token,
	buffer_size=32
)

test_tokens, test_doc_ids = next(dataset)
dataset.send(None)

print("Loading model...")
# Use the default UEAJ configuration
config = configs.UEAJ_150M
model = model.LlamaModel(config, rngs=rng.Rngs(0))

graph_def, state = nnx.split(model, nnx.Param)
state = jax.tree.map(lambda x: x.astype(jnp.float32), state)

import ueaj.opt.multiscale as ms

# opt_fn = lambda lr: optax.adamw(learning_rate=lr, b1=.95, b2=.999, weight_decay=1e-2, mu_dtype=jnp.float32)
def make_optimizer(lr, cooldown_frac):
	opt = OptimizerConfig(model)
	c_lr = lr * (1-cooldown_frac)
	# norm optimizer
	norm = optax.lion(learning_rate=0.015625 * c_lr, b1=.95, b2=.95, weight_decay=1e-2)

	# lm_head optimizer
	lm_head = optax.adamw(learning_rate=0.5 * c_lr, b1=.95, b2=.999, weight_decay=1e-3)

	# embed optimizer (no wd)
	embed = optax.adam(learning_rate=c_lr, b1=.95, b2=.999)

	# tensor = ms.muon(lr=c_lr / 8, wd=1e-3)
	tensor = ms.multiscale_muon(lr=lr / 8, wd=1e-3, cd_frac=cooldown_frac)
	# tensor = optax.adamw(learning_rate=0.5 * c_lr, b1=0., b2=0., weight_decay=1e-3)

	opt[...] = norm
	opt['embed_tokens'] = embed
	opt['lm_head'] = lm_head
	opt['layers', ['mlp', 'attn']] = tensor

	return opt.create_optimizer()

opt_fn = lambda lr, cf: make_optimizer(lr=lr, cooldown_frac=cf)
opt_arg_0 = {'lr': jnp.array(0.0625), 'cf': jnp.array(0.)}
opt_state = opt_fn(**opt_arg_0).init(state)

print(jax.tree.map(lambda x: (x.shape, x.dtype), opt_state))

minimal_stats = ('mean_loss', 'std_loss')  # Just mean_loss and std_loss
detailed_stats = (
	'mean_loss', 'std_loss', 'grad_norm', 'param_l1_norm', 'update_norm'
)

# Compile two versions of train_step - minimal and detailed statistics
print("\nCompiling train step with minimal statistics...")
train_step_fast = compile_function(
	train_step,
	sample_args=(graph_def, state, opt_fn, opt_arg_0, opt_state, tokens_struct),
	sample_kwargs={
		'document_ids': document_ids_struct,
		'pad_token_id': pad_token,
		'stats_to_collect': minimal_stats,
		'technique': 'associative'
	},
	name="Train Step (Fast)"
)

# TODO: no momentum
# TODO: reguarlize embeds
# TODO: lower lr?
# TODO: remove cooldown

print("\nCompiling train step with detailed statistics...")
train_step_stats = compile_function(
	train_step,
	sample_args=(graph_def, state, opt_fn, opt_arg_0, opt_state, tokens_struct),
	sample_kwargs={
		'document_ids': document_ids_struct,
		'pad_token_id': pad_token,
		'stats_to_collect': detailed_stats,
		'technique': 'associative'
	},
	name="Train Step (Stats)"
)

print("\nCompiling test step...")
@functools.partial(jax.jit, static_argnums=(4,))
def run_test(graph_def, state, tokens_struct, document_ids_struct, pad_token):
	params, etc = nnx.split_state(state, nnx.Param, nnx.Not(nnx.Param))
	params = jax.tree.map(lambda x: x.astype(jnp.bfloat16), params)
	return test(graph_def, params, etc, tokens_struct, document_ids_struct, pad_token)[1]

test_compiled = compile_function(
	run_test,
	sample_args=(graph_def, state, tokens_struct, document_ids_struct, pad_token),
	name="Test Step"
)

# 315
baseline_tokens_per_iter = 49152
warmup_tokens = 500*baseline_tokens_per_iter # 2,457,600
stable_tokens = warmup_tokens + 50000*baseline_tokens_per_iter # 245,760,000
cooldown_tokens = stable_tokens + 4000*baseline_tokens_per_iter # 98,304,000

print("Total tokens: ", cooldown_tokens)

print("Starting training...")
trained_tokens = 0
for i, batch in enumerate(dataset):
	# opt_arg = {'lr': min(i / warmup_iters, math.cos(math.pi * i / (2 * cooldown))) * opt_arg_0['lr']}
	if trained_tokens < warmup_tokens:
		cf = 0.
		lr = trained_tokens / warmup_tokens
	elif trained_tokens < stable_tokens:
		cf = 0.
		lr = 1.
	elif trained_tokens < cooldown_tokens:
		cf = (trained_tokens - stable_tokens) / (cooldown_tokens - stable_tokens)
		lr = 1. # adjusted internally
	else:
		break

	opt_arg = {'lr': lr * opt_arg_0['lr'], 'cf': cf}
	tokens, doc_ids = batch
	if i == 0:
		gc.collect()

	start_train = time.time()

	# Use statistics version every 10 iterations
	if i % 10 == 0:
		state, opt_state, stats = train_step_stats(
			graph_def, state, opt_arg, opt_state, tokens, document_ids=doc_ids
		)
	else:
		state, opt_state, stats = train_step_fast(
			graph_def, state, opt_arg, opt_state, tokens, document_ids=doc_ids
		)

	# Extract loss values from stats
	mean_loss = stats['mean_loss']
	std_loss = stats['std_loss']

	dataset.send(None)
	gc.collect()

	start_wait = time.time()
	mean_loss.block_until_ready()
	end_wait = time.time()

	state.embed_tokens.embedding.value = nnx.RMSNorm(config.model_d, use_scale=False, rngs=nnx.Rngs(0))(state.embed_tokens.embedding.value)

	train_time, wait_time = end_wait - start_train, end_wait - start_wait

	# Calculate tokens per second
	tokens_per_second = (batch_size * seq_len) / train_time
	trained_tokens += (batch_size * seq_len)

	if wait_time < .01:
		print("[Warn] Training is outpacing data loading!")

	wandb_dict = {
		"step": i,
		"tokens": trained_tokens,
		"lr": opt_arg['lr'],
		"train_time": train_time,
		"wait_time": wait_time,
		"tokens_per_second": tokens_per_second,
	}

	# Run test every 100 iterations
	if i % 100 == 0:
		test_mean, test_std = test_compiled(graph_def, state, test_tokens, test_doc_ids)
		wandb_dict["test_loss"] = float(test_mean)
		wandb_dict["test_loss_std"] = float(test_std)
		print(f"Test loss: {float(test_mean):.2f}, Train time: {train_time:.2f}s, Wait time: {wait_time:.2f}s, Tokens/s: {tokens_per_second:.0f}")

	# Log all statistics from the stats dict
	for stat_name, stat_value in stats.items():
		if isinstance(stat_value, dict) or hasattr(stat_value, 'items'):
			for key, value in nnx.to_flat_state(stat_value):
				if hasattr(value, 'value'):
					value = value.value
				wandb_dict[f"{stat_name}-" + ".".join(key)] = float(jnp.mean(value))
		else:
			# Handle scalar stats
			wandb_dict[stat_name] = float(stat_value)

	if i == 0:
		run_id = uuid.uuid4()
		wandb.init(project="nanogpt-euclidian", name=f"run-{run_id}")
	wandb.log(wandb_dict)

	# Print basic info always, detailed stats when available
	base_msg = f"[{i}] Train loss: {float(mean_loss):.2f}, Std loss: {float(std_loss):.2f}, Tokens/s: {tokens_per_second:.0f}, Tokens: {trained_tokens}"
	print(base_msg)

	if np.isnan(mean_loss):
		print("Loss is NaN, stopping training...")
		break

wandb.finish()

print("Done!")