"""Simplified einsum implementation with 2-argument support.

Weight matrix format: (...batch_dims, reducing_dims, non_reducing_dims)
"""
import operator
import functools
from functools import reduce
from typing import *
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
from ueaj.utils.configurator import *
from dataclasses import dataclass


@dataclass
class EinsumMetadata:
	"""Metadata for einsum parameters to enable optimizer transformations."""
	canonical_shape: Tuple[int, ...]
	reduced_shape: Tuple[int, ...]
	transpose_axes: List[int]


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


def compute_weight_metadata(
	weight_expr: str,
	output_expr: str,
	batch_dims: str,
	size_dict: dict[str, int]
) -> EinsumMetadata:
	"""Compute metadata for transforming between canonical and reduced forms."""
	weight_set = set(weight_expr)
	batch_set = set(batch_dims)
	output_set = set(output_expr)

	reducing_dims = weight_set - output_set
	all_non_reducing_dims = weight_set - reducing_dims

	non_reducing_dims = all_non_reducing_dims - batch_set
	batch_dims_set = all_non_reducing_dims.intersection(batch_set)

	# Keep original order from weight_expr
	batch_dims_tuple = tuple(d for d in weight_expr if d in batch_dims_set)
	reducing_dims_tuple = tuple(d for d in weight_expr if d in reducing_dims)
	non_reducing_dims_tuple = tuple(d for d in weight_expr if d in non_reducing_dims)
	
	# Canonical shape (original einsum shape)
	canonical_shape = tuple(size_dict[d] for d in weight_expr)
	
	# Reduced form dimensions
	batch_shape = tuple(size_dict[d] for d in batch_dims_tuple)
	reducing_shape = tuple(size_dict[d] for d in reducing_dims_tuple)
	non_reducing_shape = tuple(size_dict[d] for d in non_reducing_dims_tuple)
	
	# Reduced shape (for optimizer)
	reducing_size = reduce(operator.mul, reducing_shape, 1)
	non_reducing_size = reduce(operator.mul, non_reducing_shape, 1)
	reduced_shape = batch_shape + (reducing_size, non_reducing_size)
	
	# Compute transformation from canonical to reduced
	# First reshape to separate batch/reducing/non-reducing
	reordered = batch_dims_tuple + reducing_dims_tuple + non_reducing_dims_tuple
	transpose_axes = [weight_expr.index(d) for d in reordered]

	return EinsumMetadata(
		canonical_shape=canonical_shape,
		reduced_shape=reduced_shape,
		transpose_axes=transpose_axes
	)


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
		dtype: jnp.dtype = jnp.bfloat16,
		*,
		rngs: rng.Rngs,
		mesh: Optional[jax.sharding.Mesh] = None,
		sharding: Optional[tuple[None | str, ...]] = None
	):
		"""Initialize einsum layer.

		Args:
			expr: Einsum expression (e.g., "bnd,fdh->fbnh")
			initializer: Weight initializer
			size_dict: Mapping from dimension names to sizes
			batch_dims: Batch dimensions in weight matrix (e.g., "f")
			rngs: Random number generators
			dtype: Parameter dtype (default: bfloat16)
			mesh: JAX mesh
			sharding: JAX sharding
		"""
		super().__init__()

		# Parse expression
		self.input_expr, self.weight_expr, self.output_expr = parse_einsum_expr(expr)
		self.batch_dims = batch_dims
		self.size_dict = size_dict

		# Initialize weight with canonical shape

		key = rngs.params()
		
		# Compute metadata to get canonical shape
		temp_metadata = compute_weight_metadata(
			self.weight_expr, self.output_expr, batch_dims, size_dict
		)
		canonical_shape = temp_metadata.canonical_shape
		
		w = initializer(key, canonical_shape, dtype)
		
		if mesh is not None and sharding is not None:
			partition_spec = jax.sharding.PartitionSpec(*sharding)
			w = jax.lax.with_sharding_constraint(w, jax.NamedSharding(mesh, partition_spec))

		self.sharding = sharding
		self.w = nnx.Param(w)
		
		# Store metadata for optimizer transformations
		self.metadata = compute_weight_metadata(
			self.weight_expr, self.output_expr, batch_dims, size_dict
		)

		# Build einsum expression for forward pass
		self.einsum_expr = f"{self.input_expr},{self.weight_expr}->{self.output_expr}"

	def __call__(self, x: jax.Array) -> jax.Array:
		"""Apply einsum transformation."""
		# Weights are now stored in canonical form, use directly
		return jnp.einsum(self.einsum_expr, x, self.w.value)
	
	def get_einsum_metadata(self) -> Dict[str, EinsumMetadata]:
		"""Returns metadata for einsum-aware optimizers."""
		return {'w': self.metadata}
	
	def reshard(self, mesh: jax.sharding.Mesh) -> None:
		"""Reshard the weight parameter on the given mesh.
		Args:
			mesh: JAX mesh to shard on
		"""
		sharding = self.sharding
		if sharding is None:
			# Default to no sharding
			sharding = (None,) * len(self.metadata.canonical_shape)
		
		if len(sharding) != len(self.metadata.canonical_shape):
			raise ValueError(
				f"Sharding spec length {len(sharding)} doesn't match "
				f"weight dimensions {len(self.metadata.canonical_shape)}"
			)
		
		# Create PartitionSpec from sharding
		partition_spec = jax.sharding.PartitionSpec(*sharding)
		
		# Apply sharding constraint
		named_sharding = jax.NamedSharding(mesh, partition_spec)
		self.w.value = jax.lax.with_sharding_constraint(self.w.value, named_sharding)
