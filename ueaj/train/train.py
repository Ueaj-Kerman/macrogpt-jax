import os

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"  # Required for kvax

import wandb
import gc
import time

import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
import datasets
import transformers

from ueaj import data
from ueaj.model import configs
from ueaj.train import (training_utils, optimizer_setup, logging_utils)

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

print(f"Initializing optimizer {optimizer_setup.get_optimizer_name()}...")
base_lr = float(os.environ.get("BASE_LR", 0.025))
print(f"Using base learning rate: {base_lr}")
opt_arg_0 = {'lr': jnp.array(base_lr), 'warmup': jnp.array(1.)}
# Convert params to fp32 for optimizer (master weights)
state = jax.tree.map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, state)

opt_state = optimizer_setup.make_optimizer(**opt_arg_0, model=model).init(state)

print(jax.tree.map(lambda x: (x.shape, x.dtype), opt_state))

# Compile all training functions
test_compiled, train_step_fast, train_step_stats = training_utils.compile_training_functions(
	graph_def=graph_def,
	state=state,
	make_optimizer=optimizer_setup.make_optimizer,
	opt_arg_0=opt_arg_0,
	opt_state=opt_state,
	tokens_struct=tokens_struct,
	document_ids_struct=document_ids_struct,
	pad_token=pad_token,
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

warmup_tokens 		=   	10_000_000
cooldown_tokens 	=	   100_000_000
max_train_tokens	=	10_000_000_000

print("Starting training...")
model_path = os.environ.get("MODEL_PATH")
if model_path is not None:
	print(f"Will write model to {model_path}")

trained_tokens = 0
for i, batch in enumerate(dataset):
	if trained_tokens >= max_train_tokens:
		break

	start_train = time.time()

	# Run training
	train_fn = train_step_stats if i % 10 == 0 else train_step_fast
	warmup = min(trained_tokens / warmup_tokens, 1.)

	opt_arg = {'lr': jnp.array(base_lr * warmup), 'warmup': warmup}
	tokens, doc_ids = batch

	state, opt_state, stats = train_fn(
		graph_def, state, opt_arg, opt_state, tokens,
		document_ids=doc_ids
	)

	dataset.send(None)
	gc.collect()

	start_wait = time.time()
	stats['mean_loss'].block_until_ready()
	end_wait = time.time()

	train_time, wait_time = end_wait - start_train, end_wait - start_wait

	if wait_time < .01:
		print("[Warn] Training is outpacing data loading!")

	# Run test every 100 iterations
	test_stats = None
	if i % 100 == 0:
		test_stats = test_compiled(graph_def, state, test_tokens, test_doc_ids)
		print(f"Test loss: {test_stats[0]:.2f}")

	# Log metrics and get values for printing
	log_values = logging_utils.log_training_metrics(
		stats=stats,
		step=i,
		trained_tokens=trained_tokens,
		opt_arg=opt_arg,
		train_time=train_time,
		wait_time=wait_time,
		batch_size=batch_size,
		seq_len=seq_len,
		run_name=run_name,
		test_loss=test_stats,
	)

	# Update trained_tokens from the logging function
	trained_tokens = log_values['trained_tokens']

	# Print basic info always
	print(f"[{i}] Train loss: {log_values['mean_loss']:.2f}, Std loss: {log_values['std_loss']:.2f}, Tokens/s: {log_values['tokens_per_second']:.0f}, Tokens: {trained_tokens}")

	if np.isnan(log_values['mean_loss']):
		print("Loss is NaN, stopping training...")
		break

if model_path is not None:
	training_utils.save_model(state, model_path)

wandb.finish()
print("Done!")
