# from transformer_engine.jax.attention import SequenceDescriptor

from test_memory import memory_report
from ueaj.model.layer import TransformerLayer
from ueaj.model.llama import LlamaConfig


def compile_layer_bwd():
	import functools
	import jax
	import jax.numpy as jnp
	from flax import nnx
	from flax.nnx import rnglib as rng
	from ueaj.model.mlp import GMLP, MLPConfig
	from ueaj.model.rmsnorm import RMSNormConfig, RMSNorm
	from ueaj.model.ueajsum import ParamConfig
	print("Initialized JAX")

	config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
	print("Loaded config:")
	print(config)

	layer = nnx.eval_shape(
		lambda: TransformerLayer(
			config=config.layer_config,
			rngs=rng.Rngs(0)
		)
	)

	sample_x = jax.ShapeDtypeStruct(shape=(16, 4096, 2048), dtype=jnp.bfloat16)
	segment_ids = jax.ShapeDtypeStruct(shape=(16, 4096), dtype=jnp.int32)

	def layer_diff(graph_def, params, etc, x, segment_ids):
		# Use segment IDs directly instead of SequenceDescriptor
		return nnx.merge(graph_def, params, etc)(x, query_segment_ids=segment_ids, kv_segment_ids=segment_ids)

	# return jnp.square(x-y).sum(dtype=jnp.bfloat16)

	# compile function and let memory report print memory consumption of pass
	graph, params, etc = nnx.split(layer, nnx.Param, ...)

	def function(graph_def, params, etc, x, segment_ids, y):
		output, callback = jax.vjp(functools.partial(layer_diff, graph_def, etc=etc, x=x, segment_ids=segment_ids), params)
		return callback((y - output))

	# Create sample inputs
	x = sample_x
	y = sample_x  # Same shape as x for the loss computation

	# AOT compile the function
	lowered = jax.jit(function).lower(graph, params, etc, x, segment_ids, y)

	print("Lowered")
	print(lowered.as_text())
	print(lowered.cost_analysis())

	compiled = lowered.compile()
	print("Compiled")
	print(compiled.as_text())

	print("Other Analysis:")
	import pprint
	pprint.pprint(compiled.cost_analysis())
	mem_analysis = compiled.memory_analysis()
	print(mem_analysis)

	print("\n============= Memory Report ==============")
	print(f"Total memory usage: {mem_analysis.temp_size_in_bytes / 1024 ** 3:.2f} GB")


if __name__ == "__main__":
	import shutil

	shutil.rmtree("/tmp/jax_cache", ignore_errors=True)
	# .5GB for the input
	# 2GB for hidden state checkpoint
	# .5GB for output state checkpoint
	# ~= 3.5GB total
	print("Compiling mlp bwd pass")
	with memory_report():
		compile_layer_bwd()
