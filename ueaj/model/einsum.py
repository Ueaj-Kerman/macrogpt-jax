"""Simplified einsum implementation with 2-argument support.

Weight matrix format: (...batch_dims, reducing_dims, non_reducing_dims)
"""
import operator
from functools import reduce
from typing import *
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
from ueaj.utils.configurator import *


def parse_einsum_expr(expr: str) -> Tuple[str, str, str]:
	"""Parse einsum expression to extract input, weight, and output shapes.

	Example: "bnd,dh->bnh" returns ("bnd", "dh", "bnh")
	"""
	parts = expr.split("->")
	if len(parts) != 2:
		raise ValueError(f"Invalid einsum expression: {expr}")

	inputs = parts[0].split(",")
	if len(inputs) != 2:
		raise ValueError(f"Expected exactly 2 inputs, got {len(inputs)}")

	return inputs[0].strip(), inputs[1].strip(), parts[1].strip()


def compute_weight_shape_and_order(
	weight_expr: str,
	output_expr: str,
	batch_dims: str,
	size_dict: dict[str, int]
):
	weight_set = set(weight_expr)
	batch_set = set(batch_dims)
	output_set = set(output_expr)

	reducing_dims = weight_set - output_set
	all_non_reducing_dims = weight_set - reducing_dims

	non_reducing_dims = all_non_reducing_dims - batch_set
	batch_dims = all_non_reducing_dims.intersection(batch_set)

	batch_dims = tuple(batch_dims)
	non_reducing_dims = tuple(non_reducing_dims)
	reducing_dims = tuple(reducing_dims)
	reordered = batch_dims + reducing_dims + non_reducing_dims

	batch_shape = tuple([size_dict[d] for d in batch_dims])
	non_reducing_shape = tuple([size_dict[d] for d in non_reducing_dims])
	reducing_shape = tuple([size_dict[d] for d in reducing_dims])

	non_reducing = reduce(operator.mul, non_reducing_shape, 1)
	reducing = reduce(operator.mul, reducing_shape, 1)

	shape = batch_shape + (reducing,) + (non_reducing,)

	reshape = batch_shape + reducing_shape + non_reducing_shape
	transpose = [reordered.index(d) for d in weight_expr]
	return shape, reshape, transpose


@config
class Einsum(nnx.Module):
	"""Simple einsum layer with 2-argument support.

	Weight matrix format: (...batch_dims, reducing_dims, non_reducing_dims)

	TODO: ensure dimension relative order doesn't change, i.e. dhi shouldn't become dih
	"""

	def __init__(
		self,
		expr: str,
		initializer: nnx.initializers.Initializer,
		size_dict: dict[str, int],
		batch_dims: str = "",
		rngs: rng.Rngs = None,
		dtype: jnp.dtype = jnp.bfloat16
	):
		"""Initialize einsum layer.

		Args:
			expr: Einsum expression (e.g., "bnd,fdh->fbnh")
			initializer: Weight initializer
			size_dict: Mapping from dimension names to sizes
			batch_dims: Batch dimensions in weight matrix (e.g., "f")
			rngs: Random number generators
			dtype: Parameter dtype (default: bfloat16)
		"""
		super().__init__()

		# Parse expression
		self.input_expr, self.weight_expr, self.output_expr = parse_einsum_expr(expr)
		self.batch_dims = batch_dims
		self.size_dict = size_dict

		# Compute weight shape and dimension order
		w_shape, reshape, transpose = compute_weight_shape_and_order(
			self.weight_expr, self.output_expr, batch_dims, size_dict
		)

		# Initialize weight with proper shape
		if rngs is None:
			raise ValueError("rngs must be provided")

		# Initialize with the rearranged shape
		w = initializer(rngs.params(), w_shape, dtype)

		self.reshape = reshape
		self.transpose = transpose
		self.w = nnx.Param(w)

		# Build einsum expression for forward pass
		self.einsum_expr = f"{self.input_expr},{self.weight_expr}->{self.output_expr}"

	def __call__(self, x: jax.Array) -> jax.Array:
		"""Apply einsum transformation."""
		w = self.w.value
		w = w.reshape(self.reshape)
		w = jnp.transpose(w, self.transpose)
		return jnp.einsum(self.einsum_expr, x, w)


if __name__ == "__main__":
	# Test the implementation
	rngs = rng.Rngs(0)

	# Example: "bnd,fdh->fbnh" with batch_dims="f"
	size_dict = {"d": 8, "h": 16, "f": 2}

	einsum = Einsum(
		"bnd,fdh->fbnh",
		nnx.initializers.glorot_uniform(),
		size_dict,
		batch_dims="f",
		rngs=rngs
	)

	x = jnp.ones((10, 20, 8))
	y = einsum(x)
	print(f"Test: {x.shape} -> {y.shape}")
	print(f"Weight shape: {einsum.w.value.shape}")
	print(f"Expected output shape: (2, 8, 16)")
