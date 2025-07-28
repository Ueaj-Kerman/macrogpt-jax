import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"  # Required for kvax

import functools
from typing import Optional, Tuple, Any, Callable, Dict

import jax

import wandb
import uuid

import numpy as np
from flax import nnx
from flax.nnx import rnglib as rng
from kvax.utils import PADDING_SEGMENT_ID
from ueaj.utils.kvax_context import *
import time

from ueaj import data, model
from ueaj.opt import *
from ueaj.utils.compile import *
from ueaj.utils.tensorutil import *
from ueaj.model import *
import gc
from jax import numpy as jnp
import optax


# Initialize kvax context once
_kvax_ctx = get_kvax_context()

def test(
	g_def: nnx.GraphDef[model.LlamaModel],
	params: nnx.State,
	etc: nnx.State,
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None
) -> Tuple[jax.Array, Any]:
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

@functools.partial(jax.jit, donate_argnums=(1, 4), static_argnums=(2, 7, 8))
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
):
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

	# Update parameters
	dmodel_updates, opt_state = opt(**opt_args).update(dmodel, opt_state, params)
	new_params = optax.apply_updates(params, dmodel_updates)
	# rescale embeddings
	embed_scale = jnp.sqrt(jnp.mean(jnp.square(params.embed_tokens.embedding.value), axis=1))
	embed_scale = jnp.where(embed_scale > 1.25, embed_scale, 1.) # reset magnitude if too high
	new_params.embed_tokens.embedding.value = new_params.embed_tokens.embedding.value / embed_scale[:, None]

	delta = jax.tree.map(lambda p, up: p.astype(jnp.float32) - up, params, new_params)
	params = new_params

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
batch_size, seq_len = 1, 6*8192
pad_token = 50431

print("Loading tokenizer...")
tokenizer = transformers.GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
# Set model_max_length properly - this is the attribute that actually controls the warning
tokenizer.model_max_length = seq_len  # Set to 4096
print("Vocab Size:", tokenizer.vocab_size)

# Create structure descriptors for compilation
tokens_struct = jax.ShapeDtypeStruct((batch_size, seq_len), jax.numpy.int32)
document_ids_struct = jax.ShapeDtypeStruct((batch_size, seq_len), jax.numpy.int32)

print("Loading model...")
# Use the default UEAJ configuration
model = configs.UEAJ_150M(rngs=rng.Rngs(0))

graph_def, state = nnx.split(model, nnx.Param)
state = jax.tree.map(lambda x: x.astype(jnp.float32), state)

import ueaj.opt.multiscale as ms

opt_name = os.environ.get('OPTIMIZER', 'multiscale')
def make_optimizer(lr, warmup):
	lr *= warmup

	opt = OptimizerConfig(model)
	# norm optimizer
	norm = optax.lion(learning_rate=0.015625 * lr, b1=.95, b2=.95, weight_decay=1e-2)

	# lm_head optimizer
	lm_head = optax.adamw(learning_rate=0.5 * lr, b1=.95, b2=.999, weight_decay=1e-3)

	# embed optimizer (no wd)
	embed = optax.adam(learning_rate=lr, b1=.95, b2=.999)

	# Select optimizer with env_var
	if opt_name == 'multiscale':
		tensor = ms.multiscale_muon(lr=lr / 8, wd=1e-3, warmup_frac=warmup**2)
	elif opt_name == 'muon':
		tensor = ms.muon(lr=lr / 8, wd=1e-3)
	elif opt_name == 'adamw':
		tensor = optax.adamw(learning_rate=0.5 * lr, b1=.95, b2=0.999, weight_decay=1e-3)
	else:
		raise ValueError(f'Unrecognized optimizer name: {opt_name}')

	opt[...] = norm
	opt['embed_tokens'] = embed
	opt['lm_head'] = lm_head
	opt['layers', ['mlp', 'attn']] = tensor

	return opt.create_optimizer()

print(f"Initializing optimizer {opt_name}...")
opt_arg_0 = {'lr': jnp.array(0.0625), 'warmup': jnp.array(1.)}
opt_state = make_optimizer(**opt_arg_0).init(state)

print(jax.tree.map(lambda x: (x.shape, x.dtype), opt_state))

minimal_stats = ('mean_loss', 'std_loss')  # Just mean_loss and std_loss
detailed_stats = (
	'mean_loss', 'std_loss', 'grad_norm', 'param_l1_norm', 'update_norm'
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

# Compile two versions of train_step - minimal and detailed statistics
print("\nCompiling train step with minimal statistics...")
train_step_fast = compile_function(
	train_step,
	sample_args=(graph_def, state, make_optimizer, opt_arg_0, opt_state, tokens_struct),
	sample_kwargs={
		'document_ids': document_ids_struct,
		'pad_token_id': pad_token,
		'stats_to_collect': minimal_stats,
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
		'stats_to_collect': detailed_stats,
	},
	name="Train Step (Stats)"
)


run_name = os.environ.get("RUN_NAME")
if run_name is None:
	run_name = input("Enter a run ID: ")
else:
	print(f"Using run name: {run_name}")

print("Loading dataset...")
dataset = datasets.load_dataset(
	"HuggingFaceFW/fineweb-edu",
	name="sample-10BT",
	split="train",
	streaming=True,
)

# Use the new prepare_dataset function
print("Setting up train iterator...")
dataset, (_, _) = data.prepare_dataset(
	dataset,
	tokenizer,
	batch_size=batch_size,
	seq_len=seq_len,
	pad_token_id=pad_token,
	buffer_size=32
)

print("Fetching test set...")
test_tokens, test_doc_ids = next(dataset)
dataset.send(None)

warmup_tokens 		=    10_000_000
max_train_tokens	= 1_000_000_000

print("Starting training...")
model_path = os.environ.get("MODEL_PATH")
if model_path is not None:
	print(f"Will write model to {model_path}")

trained_tokens = 0
for i, batch in enumerate(dataset):
	if trained_tokens >= max_train_tokens:
		break
	# opt_arg = {'lr': min(i / warmup_iters, math.cos(math.pi * i / (2 * cooldown))) * opt_arg_0['lr']}
	warmup = min(trained_tokens / warmup_tokens, 1.)

	opt_arg = {**opt_arg_0, 'warmup': warmup}
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
		run_id = str(uuid.uuid4())[:5]
		wandb.init(project="nanogpt-euclidian", name=f"{run_name}-{run_id}")
	wandb.log(wandb_dict)

	# Print basic info always, detailed stats when available
	base_msg_parts = [
		f"Train loss: {float(mean_loss):.2f}",
		f"Std loss: {float(std_loss):.2f}",
		f"Tokens/s: {tokens_per_second:.0f}",
		f"Tokens: {trained_tokens}",
	]
	base_msg = f"[{i}] " + ", ".join(base_msg_parts)
	print(base_msg)

	if np.isnan(mean_loss):
		print("Loss is NaN, stopping training...")
		break

if model_path is not None:
	import safetensors.flax as st

	leaves = jax.tree.leaves_with_path(nnx.to_pure_dict(state))
	entries = {}
	for path, leaf in leaves:
		entries[".".join([p.key for p in path])] = leaf
		print(".".join([p.key for p in path]), leaf.shape)

	st.save_file(entries, model_path)
wandb.finish()

print("Done!")
