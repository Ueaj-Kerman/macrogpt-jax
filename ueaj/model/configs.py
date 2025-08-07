"""Default model configurations and parameter counting utilities."""
import operator
from functools import reduce

import jax
from flax import nnx

from ueaj import model
from ueaj.model.nn import *


def count_parameters(config) -> int:
	"""Count the number of parameters in a model configuration.

	Returns:
		Total number of parameters
	"""

	# Use nnx.eval_shape to create the model without allocating memory
	def create_model():
		return config(rngs=nnx.Rngs(0))

	# Create model shape without allocating
	model_shape = nnx.eval_shape(create_model)

	# Get all parameters
	params = nnx.state(model_shape, nnx.Param)

	counts = jax.tree.map(lambda x: reduce(operator.mul, x.shape, 1), params)
	total = jax.tree.reduce(lambda x, y: x + y, counts)

	return total


def format_param_count(count: int) -> str:
	"""Format parameter count in human-readable form."""
	if count >= 1e9:
		return f"{count / 1e9:.1f}B"
	elif count >= 1e6:
		return f"{count / 1e6:.1f}M"
	elif count >= 1e3:
		return f"{count / 1e3:.1f}K"
	else:
		return str(count)



# UEAJ configuration
def ueaj_model(vocab_size: int, model_d: int, num_layers: int, kq_ratio: int = 1):
	"""Create UEAJ model configuration using override pattern."""
	from ueaj.model import SoftmaxAttention
	from ueaj.model import MLP
	from ueaj.model import TransformerLayer
	from ueaj.model import LlamaModel

	# Return the overridden model class
	return LlamaModel.override(
		vocab_size=vocab_size,
		model_d=model_d,
		num_layers=num_layers,
		transformer_layer=TransformerLayer.override(
			attn=SoftmaxAttention.override(
				kq_d=64,
				kv_heads=model_d // 128,
				kv_q_ratio=kq_ratio,
				rope_theta=2_000.0,
				act_fn=leaky_relu_squared,
			),
			mlp=MLP.override(
				act_fn=leaky_relu_squared,
			),
		)
	)


# Type annotations to help IDEs
UEAJ_NH = ueaj_model(50432, 768, 1)
UEAJ_150M = ueaj_model(50432, 768, 12)
UEAJ_1B = ueaj_model(50432, 1536, 32, kq_ratio=2)

if __name__ == "__main__":
	print(f"UEAJ-150M has {format_param_count(count_parameters(UEAJ_150M))} parameters")
	print(f"UEAJ-1B has {format_param_count(count_parameters(UEAJ_1B))} parameters")
