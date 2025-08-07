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
from ueaj.utils.gradutils import custom_scale
from dataclasses import dataclass


# Initializer type for Einsum
InitializerFn = Callable[[int, int], nnx.initializers.Initializer]


# Default initializer functions
def lecun_normal_init(in_dims: int, out_dims: int) -> nnx.initializers.Initializer:
	"""LeCun normal initializer (default for Einsum).
	
	Args:
		in_dims: Number of input dimensions (reducing dims)
		out_dims: Number of output dimensions (non-reducing dims)
	
	Returns:
		Initializer with stddev = sqrt(1/in_dims)
	"""
	# For Einsum weights, the reducing dimensions are the input
	# LeCun normal uses variance = 1/fan_in
	# Since we're given the actual fan_in, we don't need to specify axes
	# We use the default axes which work for 2D weights
	return nnx.initializers.lecun_normal()


def zeros_init(in_dims: int, out_dims: int) -> nnx.initializers.Initializer:
	"""Zero initializer utility.
	
	Args:
		in_dims: Number of input dimensions (reducing dims)
		out_dims: Number of output dimensions (non-reducing dims)
	
	Returns:
		Zero initializer
	"""
	return nnx.initializers.zeros_init()


def scaled_normal_init(scale: float = 0.75) -> InitializerFn:
	"""Creates a scaled normal initializer function.
	
	Args:
		scale: Scale factor for the normal distribution
	
	Returns:
		An initializer function that creates a normal initializer with the given scale
	"""
	def init_fn(in_dims: int, out_dims: int) -> nnx.initializers.Initializer:
		return nnx.initializers.normal(stddev=scale)
	return init_fn


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
		initializer: InitializerFn = lecun_normal_init,
		size_dict: dict[str, int] = None,
		batch_dims: str = "",
		dtype: jnp.dtype = jnp.bfloat16,
		static_scale: Optional[float] = None,
		*,
		rngs: rng.Rngs,
		mesh: Optional[jax.sharding.Mesh] = None,
		sharding: Optional[tuple[None | str, ...]] = None
	):
		"""Initialize einsum layer.

		Args:
			expr: Einsum expression (e.g., "bnd,fdh->fbnh")
			initializer: Initializer function that takes (in_dims, out_dims) and returns an initializer
			size_dict: Mapping from dimension names to sizes
			batch_dims: Batch dimensions in weight matrix (e.g., "f")
			dtype: Parameter dtype (default: bfloat16)
			static_scale: Optional static scaling factor to apply after matmul in forward,
			             and before matmul in backward
			rngs: Random number generators
			mesh: JAX mesh
			sharding: JAX sharding
		"""
		super().__init__()

		# Parse expression
		self.input_expr, self.weight_expr, self.output_expr = parse_einsum_expr(expr)
		self.batch_dims = batch_dims
		self.size_dict = size_dict
		self.static_scale = static_scale

		# Initialize weight with canonical shape

		key = rngs.params()

		# Compute metadata to get canonical shape and dimensions
		temp_metadata = compute_weight_metadata(
			self.weight_expr, self.output_expr, batch_dims, size_dict
		)
		canonical_shape = temp_metadata.canonical_shape
		
		# Calculate reducing and non-reducing dimensions
		weight_set = set(self.weight_expr)
		output_set = set(self.output_expr)
		reducing_dims = weight_set - output_set
		non_reducing_dims = weight_set - reducing_dims - set(batch_dims)
		
		# Calculate total size of reducing and non-reducing dimensions
		reducing_size = reduce(operator.mul, [size_dict[d] for d in reducing_dims], 1)
		non_reducing_size = reduce(operator.mul, [size_dict[d] for d in non_reducing_dims], 1)
		
		# Call the initializer function with dimensions
		init = initializer(reducing_size, non_reducing_size)
		w = init(key, canonical_shape, dtype)

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

		if self.static_scale is not None:
			x = custom_scale(
				x, self.static_scale,
				scale_forward=False, scale_backward=True
			)

		# Weights are now stored in canonical form, use directly
		x = jnp.einsum(self.einsum_expr, x, self.w.value)

		if self.static_scale is not None:
			x = custom_scale(
				x, self.static_scale,
				scale_forward=True, scale_backward=False
			)

		return x

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
