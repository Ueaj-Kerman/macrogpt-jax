import functools
from typing import Any, Callable, Union, Tuple, TypeVar, List

import jax
import jax.lax
from flax import nnx
from jax import numpy as jnp

from ueaj.utils import LOW_PRECISION


def chunked_scan(
	f: Callable,
	init: Any,
	xs: Any,
	chunk_size: int,
	axis: Union[int, Tuple[int, ...], dict, None] = 0,
	out_axis: Union[int, Tuple[int, ...], dict, None] = 0,
	use_checkpointing=False,
) -> Tuple[Any, Any]:
	def transpose(x, ax, fwd=True):
		if isinstance(ax, int):
			from_, to = (0, ax) if fwd else (ax, 0)
			return jax.tree.map(
				lambda x: jnp.moveaxis(x, from_, to),
				x
			)
		return jax.tree.map(functools.partial(transpose, ax=axis, fwd=fwd), xs)

	xs_t = transpose(xs, axis)
	leaves = jax.tree.leaves(xs_t)
	if not leaves:
		return init, xs

	scan_length = leaves[0].shape[0]
	for i, leaf in enumerate(leaves):
		assert leaf.shape[0] == scan_length, f"Scan length mismatch: {leaf.shape[0]} != {scan_length} for leaf {i}"

	# Split into chunks
	num_full_chunks = scan_length // chunk_size
	remainder = scan_length % chunk_size

	xs_scan = jax.tree.map(
		lambda t: t[:num_full_chunks * chunk_size].reshape(
			(-1, chunk_size) + t.shape[1:]
		),
		xs_t
	)

	def invoke(carry, x):
		return f(carry, transpose(x, axis, fwd=False))

	if use_checkpointing:
		carry, ys = jax.lax.scan(
			jax.remat(invoke, policy=jax.checkpoint_policies.nothing_saveable),
			init,
			xs_scan,
			unroll=2 # for some reason this fixes the loss going up bug, don't ask me why
		)
	else:
		carry, ys = jax.lax.scan(
			invoke,
			init,
			xs_scan,
		)

	if out_axis != 0:
		ys = transpose(ys, out_axis, fwd=True)

	if remainder > 0:
		xs_rem = jax.tree.map(
			lambda t: t[-remainder:],
			xs_t
		)

		carry, ys_rem = f(carry, transpose(xs_rem, axis, fwd=False))
		if out_axis != 0:
			ys_rem = transpose(ys_rem, out_axis, fwd=True)

		if ys_rem.ndim == 0:
			ys_rem = ys_rem.reshape((1,))

		ys = jax.tree.map(
			lambda y, y_rem: jnp.concatenate((y, y_rem), axis=out_axis),
			ys,
			ys_rem
		)
		return carry, ys
	return carry, ys


T = TypeVar('T', bound=nnx.Module)


class SliceModule[T]:
	def __init__(self, module: T):
		self.module = module

	def __getitem__(self, item) -> T:
		graph_def, state = nnx.split(self.module)
		state = jax.tree.map(lambda x: x[item], state)
		return nnx.merge(graph_def, state)


def slice(module: T) -> SliceModule[T]:
	return SliceModule(module)


def promote_fp8(*args) -> List[jax.Array]:
	has_fp8, gcd_type = False, None
	for arg in args:
		if arg.dtype in LOW_PRECISION:
			has_fp8 = True
		elif gcd_type is None:
			gcd_type = arg.dtype
		else:
			gcd_type = jnp.promote_types(gcd_type, arg.dtype)

	if has_fp8 and gcd_type is not None:
		return [
			(
				jax.lax.convert_element_type(a, jnp.dtype(gcd_type))
				if a.dtype in LOW_PRECISION
				else a
			) for a in args
		]
	else:
		return list(args)


def tensor_stats(W: jax.Array) -> dict:
	"""Compute various statistics for a tensor.

	Args:
		W: Input tensor

	Returns:
		Dictionary containing:
		- l1_norm: L1 norm (for precision monitoring)
		- l2_norm: L2 norm
		- log_l1_norm: Log of L1 norm
		- variance: Variance of the tensor
		- k_eff: Effective rank (if tensor has 2+ dimensions)
	"""
	stats = {}

	# L1 norm (useful for low precision monitoring)
	stats['l1_norm'] = jnp.mean(jnp.abs(W), dtype=jnp.float32)
	stats['log_l1_norm'] = jnp.log2(stats['l1_norm'] + 1e-10)

	# L2 norm
	stats['l2_norm'] = jnp.sqrt(jnp.mean(jnp.square(W), dtype=jnp.float32))

	# Effective rank (only for 2D+ tensors)
	if W.ndim >= 2:
		stats['k_eff'] = k_eff(W)

	return stats


def k_eff(W: jax.Array) -> jax.Array:
	"""Compute effective rank (k_eff) of a tensor.

	The effective rank is computed as:
	k_eff = m * ||W^T W||_F^2 / ||W||_F^4

	With map_state, tensors are in the format:
	(...batch_dims, input_dim, output_dim)

	Args:
		W: Input tensor with shape (..., input_dim, output_dim).
		   Should have at least 2 dimensions.

	Returns:
		Array of k_eff values with shape (...,) for batch dimensions.
	"""
	if W.ndim < 2:
		return jnp.array(1.0)

	# Get dimensions - we work with the last two dimensions
	# In the mapped format, batch/vmap dims come first, then input, then output
	in_axis = W.ndim - 2
	out_axis = W.ndim - 1

	n = W.shape[in_axis]  # input dimension
	m = W.shape[out_axis]  # output dimension

	# Cast to bfloat16 for the computation
	# (squaring numbers only requires slightly more exponent bits)
	W = W.astype(jnp.bfloat16)

	# Compute Frobenius norm squared: ||W||_F^2 = sum(W^2)
	# Use float32 for accumulation to get more mantissa bits
	f2 = jnp.sum(jnp.square(W), axis=[in_axis, out_axis], dtype=jnp.float32)

	# Compute ||W^T W||_F^2 using dot_general
	# W^T W shape: (m, m)
	# For dot_general: specify which axes to contract and which are batch
	batch_dims = tuple(range(W.ndim - 2))

	# Compute W^T @ W
	# Contract over the input dimension (axis -2)
	# Batch dimensions are all dimensions except the last 2
	WtW = jax.lax.dot_general(
		W, W,
		dimension_numbers=(
			([in_axis, ], [in_axis, ]),  # Contract over input dimension
			(batch_dims, batch_dims)  # Batch dimensions
		),
		preferred_element_type=jnp.float32  # Accumulate in float32
	)

	# Compute ||W^T W||_F^2 = sum((W^T W)^2)
	s4 = jnp.sum(jnp.square(WtW), axis=[in_axis, out_axis], dtype=jnp.float32)

	# Return k_eff = m * s4 / f2^2
	return m * s4 / (f2 ** 2)
