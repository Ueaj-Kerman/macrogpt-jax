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


def llama3_model(target_params_b: float, vocab_size: int):
	"""Create LLaMA 3-style model configuration using scaling laws.

	Uses empirically-derived scaling formulas from Meta's LLaMA 3 family
	(1B, 3B, 8B, 70B, 405B) to predict optimal architectural hyperparameters
	for any target model size.

	Args:
		target_params_b: Target parameter count in billions (e.g., 1.0 for 1B)
		vocab_size: Vocabulary size (e.g., 50432 for GPT-2, 128256 for LLaMA 3)

	Returns:
		Model configuration class following LLaMA 3 architectural patterns

	Examples:
		>>> LLAMA3_1B = llama3_model(1.24, 128256)
		>>> LLAMA3_150M = llama3_model(0.15, 50432)
	"""
	import numpy as np
	from ueaj.model import SoftmaxAttention, TransformerLayer, LlamaModel, RMSNorm, GMLP

	# Known LLaMA 3 configs: (total_params_b, hidden_size, num_layers, intermediate_size)
	known_configs = [
		(1.24, 2048, 16, 8192),
		(3.21, 3072, 28, 8192),
		(8.03, 4096, 32, 14336),
		(70.6, 8192, 80, 28672),
		(405.0, 16384, 126, 53248),
	]

	known_params = np.array([cfg[0] for cfg in known_configs])
	known_hidden = np.array([cfg[1] for cfg in known_configs])
	known_layers = np.array([cfg[2] for cfg in known_configs])
	known_intermediate = np.array([cfg[3] for cfg in known_configs])

	# Fit power laws: param = a * size^b (in log space: log(param) = log(a) + b*log(size))
	log_params = np.log(known_params)

	# Fit hidden_size: h = a * P^b
	hidden_poly = np.polyfit(log_params, np.log(known_hidden), 1)
	hidden_size_raw = np.exp(hidden_poly[1]) * (target_params_b ** hidden_poly[0])
	# Round to nearest power-of-2 or 1.5×power-of-2
	powers_of_2 = np.array([2**i for i in range(8, 16)])  # 256 to 32768
	powers_plus = np.concatenate([powers_of_2, powers_of_2 * 1.5])
	powers_plus = np.sort(powers_plus)
	hidden_size = int(powers_plus[np.argmin(np.abs(powers_plus - hidden_size_raw))])

	# Fit num_layers: l = a * P^b
	layers_poly = np.polyfit(log_params, np.log(known_layers), 1)
	layers_raw = np.exp(layers_poly[1]) * (target_params_b ** layers_poly[0])
	num_layers = int(round(layers_raw / 4) * 4)  # Round to multiple of 4
	num_layers = max(4, num_layers)  # At least 4 layers

	# Fit intermediate_size: i = a * P^b
	intermediate_poly = np.polyfit(log_params, np.log(known_intermediate), 1)
	intermediate_raw = np.exp(intermediate_poly[1]) * (target_params_b ** intermediate_poly[0])
	intermediate_size = int(round(intermediate_raw / 256) * 256)  # Round to multiple of 256
	intermediate_size = max(256, intermediate_size)  # At least 256

	# Head configuration
	head_dim = 128 if hidden_size >= 3072 else 64
	num_attention_heads = hidden_size // head_dim
	num_kv_heads = min(8, num_attention_heads)  # GQA with up to 8 KV heads

	# Embedding tying: LLaMA 3 pattern is tied for 1B/3B, untied for 8B+
	# Use tied if target is in the small range
	tie_word_embeddings = target_params_b < 4.0

	return LlamaModel.override(
		vocab_size=vocab_size,
		model_d=hidden_size,
		num_layers=num_layers,
		tie_word_embeddings=tie_word_embeddings,
		transformer_layer=TransformerLayer.override(
			attn=SoftmaxAttention.override(
				kq_d=head_dim,
				kv_heads=num_kv_heads,
				kv_q_ratio=num_attention_heads // num_kv_heads,
				rope_theta=500_000.0,
			),
			mlp=GMLP.override(
				hidden_d=intermediate_size,
			),
		)
	)


# LLaMA 3-style configurations (Meta architecture)
LLAMA3_150M = llama3_model(0.15, vocab_size=50432)
LLAMA3_500M = llama3_model(0.5, vocab_size=50432)
LLAMA3_1B = llama3_model(1.24, vocab_size=128256)
LLAMA3_3B = llama3_model(3.21, vocab_size=128256)
LLAMA3_8B = llama3_model(8.03, vocab_size=128256)
LLAMA3_70B = llama3_model(70.6, vocab_size=128256)
LLAMA3_405B = llama3_model(405.0, vocab_size=128256)

# Backwards compatibility alias
UEAJ_150M = LLAMA3_150M

if __name__ == "__main__":
	print("="*80)
	print("LLaMA 3-style Model Configurations")
	print("="*80)
	print(f"LLAMA3-150M has {format_param_count(count_parameters(LLAMA3_150M))} parameters")
	print(f"LLAMA3-500M has {format_param_count(count_parameters(LLAMA3_500M))} parameters")
	print(f"LLAMA3-1B has {format_param_count(count_parameters(LLAMA3_1B))} parameters")
	print(f"LLAMA3-3B has {format_param_count(count_parameters(LLAMA3_3B))} parameters")
	print(f"LLAMA3-8B has {format_param_count(count_parameters(LLAMA3_8B))} parameters")
	print(f"LLAMA3-70B has {format_param_count(count_parameters(LLAMA3_70B))} parameters")

	print("\n" + "="*80)
	print("Custom Interpolation Examples")
	print("="*80)
	LLAMA3_7B = llama3_model(7.0, vocab_size=128256)
	LLAMA3_13B = llama3_model(13.0, vocab_size=128256)

	print(f"LLAMA3-7B has {format_param_count(count_parameters(LLAMA3_7B))} parameters")
	print(f"LLAMA3-13B has {format_param_count(count_parameters(LLAMA3_13B))} parameters")
