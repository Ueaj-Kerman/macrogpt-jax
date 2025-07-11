import functools

import jax
import jax.numpy as jnp
from jax import lax
from typing import Any, Callable, Union, Tuple, Optional, TypeVar, List
import jax.tree_util as tree
from flax import nnx
from flax.nnx import rnglib as rng


def chunked_scan(
	f: Callable,
	init: Any,
	xs: Any,
	chunk_size: int,
	axis: Union[int, Tuple[int, ...], dict, None] = 0,
	out_axis: Union[int, Tuple[int, ...], dict, None] = 0,
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

	carry, ys = jax.lax.scan(
		lambda carry, x: f(carry, transpose(x, axis, fwd=False)),
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


def sq_norm(x):
	return jnp.sum(jnp.square(x.astype(jnp.float32)))

def precision_aware_update(
	params: jax.Array,
	update: jax.Array,
	target_lr: float | jax.Array,
	rtol: float | None = 1e-2,
	max_iters: int = 32
):
	"""
	Apply update scaled to achieve target norm reduction despite precision limits.

	Args:
		params: Current parameters
		update: Normalized update direction (e.g., from muon)
		target_lr: Desired learning rate (norm reduction factor)
		rtol: Relative tolerance for achieving target norm
		max_iters: Maximum binary search iterations

	Returns:
		new_params, actual_norm_change
	"""
	# Binary search for scaling factor
	alpha_low = jnp.float32(0.0)
	alpha_high = jnp.float32(1.0)  # Assuming normalized update

	target_norm = target_lr * sq_norm(update)

	highest = None
	for i in range(max_iters):
		alpha = (alpha_low + alpha_high) / 2

		# Apply update and measure actual change
		new_params = (params.astype(update.dtype) + alpha * update).astype(params.dtype)
		if i == 0:
			highest = new_params

		actual_norm = sq_norm(new_params - params)

		# Check if we're close enough
		relative_error = jnp.abs(actual_norm - target_norm) / target_norm

		# Early exit condition (for non-JIT mode)
		if rtol is not None and relative_error < rtol:
			break

		# Adjust search bounds using JAX-compatible conditionals
		alpha_low = jnp.where(actual_norm < target_norm, alpha, alpha_low)
		alpha_high = jnp.where(actual_norm < target_norm, alpha_high, alpha)
		# prefer larger update to smaller updates
		highest = jnp.where(actual_norm > target_norm, new_params, highest)

	return highest, actual_norm, target_norm


def stochastic_rounding_update(
	params: jax.Array,
	update: jax.Array,
	target_lr: jax.Array | float,
	rngs: nnx.Rngs
):
	"""
	Apply update with stochastic rounding for better accuracy in low precision.
	
	Stochastic rounding randomly rounds to nearest representable values with
	probability proportional to proximity, preserving expected values.
	
	Args:
		params: Current parameters
		update: Update direction
		target_lr: Learning rate
		rngs: Random number generators
		
	Returns:
		new_params, actual_norm
	"""
	# Compute the ideal update in high precision
	ideal_params = params.astype(jnp.float32) + target_lr * update

	# For stochastic rounding, we need to find the two nearest representable values
	# Simple approach: add small noise before quantization

	# Get the scale of the smallest representable difference
	if params.dtype == jnp.float8_e4m3fn:
		# For FP8, use a heuristic based on the magnitude
		scale = jnp.abs(ideal_params) * 2 ** -7 + 1e-10
	else:
		scale = jnp.finfo(params.dtype).eps * jnp.abs(ideal_params) + 1e-10

	# Generate uniform random noise
	rand = jax.random.uniform(rngs.update(), shape=params.shape, minval=-0.5, maxval=0.5)

	# Add scaled noise and quantize
	new_params = (ideal_params + rand * scale).astype(params.dtype)

	actual_norm = sq_norm(new_params - params)
	return new_params, actual_norm

def quadratic_update(params: jax.Array, update: jax.Array, target_lr: float | jax.Array, n_samples=16):
	"""
	Apply update scaled to achieve target norm reduction despite precision limits.
	Uses random sampling and least squares fitting instead of binary search.

	Args:
		params: Current parameters
		update: Normalized update direction (e.g., from muon)
		target_lr: Desired learning rate (norm reduction factor)
		n_samples: Number of random samples to use

	Returns:
		new_params, actual_norm_change, target_norm
	"""
	# Sample random alpha values between 0 and 1
	key = jax.random.PRNGKey(0)  # You might want to pass this as an argument
	alphas = jax.random.uniform(key, shape=(n_samples,), minval=0.0, maxval=1.0, dtype=jnp.float32)

	# Compute actual norms for each alpha
	def compute_norm_for_alpha(alpha):
		new_params = (params.astype(update.dtype) + alpha * update).astype(params.dtype)
		return sq_norm(new_params - params)

	actual_norms = jax.vmap(compute_norm_for_alpha)(alphas)

	# Fit quadratic model: actual_norm = a * alpha^2 + b * alpha + c
	# Build design matrix [alpha^2, alpha, 1]
	A = jnp.stack([alphas**2, alphas, jnp.ones_like(alphas)], axis=1)

	# Solve least squares: A @ coeffs = actual_norms
	coeffs, _, _, _ = jnp.linalg.lstsq(A, actual_norms, rcond=None)
	a, b, c = coeffs

	# Target norm
	target_norm = target_lr * sq_norm(update)

	# Solve quadratic equation: a * alpha^2 + b * alpha + c = target_norm
	# Rearrange: a * alpha^2 + b * alpha + (c - target_norm) = 0
	discriminant = b**2 - 4*a*(c - target_norm)

	# Choose the positive root that's in [0, 1]
	alpha_optimal = (-b + jnp.sqrt(jnp.maximum(discriminant, 0))) / (2*a)
	alpha_optimal = jnp.clip(alpha_optimal, 0.0, 1.0)

	# Apply the optimal scaling
	new_params = (params.astype(update.dtype) + alpha_optimal * update).astype(params.dtype)
	actual_norm = sq_norm(new_params - params)

	return new_params, actual_norm, target_norm

def quadratic_update_2(params: jax.Array, update: jax.Array, target_lr: float | jax.Array, rtol=1e-2, max_iters=32):
	"""
	Apply update scaled to achieve target norm reduction despite precision limits.
	Args:
		params: Current parameters
		update: Normalized update direction (e.g., from muon)
		target_lr: Desired learning rate (norm reduction factor)
		rtol: Relative tolerance for achieving target norm (not used in this version)
		max_iters: Maximum binary search iterations (repurposed as N for number of samples)
	Returns:
		new_params, actual_norm_change
	"""
	N = max_iters
	key = jax.random.PRNGKey(42)  # Using a fixed seed; consider passing a dynamic key if needed

	alpha_low = jnp.float32(0.0)
	alpha_high = jnp.float32(1.0)  # Assuming normalized update

	alphas = jax.random.uniform(key, (N,)) * alpha_high

	target_norm = target_lr * sq_norm(update)

	def compute_step(alpha):
		new_params = (params.astype(update.dtype) + alpha * update).astype(params.dtype)
		return sq_norm(new_params - params)

	actual_norms = jax.vmap(compute_step)(alphas)

	actual_l2s = jnp.sqrt(actual_norms + 1e-10)  # small epsilon to avoid sqrt(0) issues

	X = jnp.column_stack([jnp.ones(N), actual_l2s])

	# Least squares solve for beta in alphas = X beta
	beta = jnp.linalg.lstsq(X, alphas, rcond=None)[0]

	b0, b1 = beta

	target_l2 = jnp.sqrt(target_norm + 1e-10)

	alpha = b0 + b1 * target_l2

	# Clip alpha to reasonable range
	alpha = jnp.clip(alpha, alpha_low, alpha_high * 1.5)  # Allow slight overshoot if needed

	# Compute final new_params and actual_norm
	new_params = (params.astype(update.dtype) + alpha * update).astype(params.dtype)
	actual_norm = sq_norm(new_params - params)

	return new_params, actual_norm, target_norm

def param_error(params: jax.Array, method_params: jax.Array, update: jax.Array):
	"""Compute statistics of (method - original) / update."""
	# Cast to float32 to avoid type promotion errors
	params_f32 = params.astype(jnp.float32)
	method_f32 = method_params.astype(jnp.float32)
	update_f32 = update.astype(jnp.float32)

	# Compute relative error
	relative_error = (method_f32 - params_f32) / (update_f32 + 1e-12)  # Add small epsilon to avoid division by zero

	# Compute statistics
	mean = relative_error.mean()
	median = jnp.median(relative_error)
	q1 = jnp.percentile(relative_error, 25)
	q3 = jnp.percentile(relative_error, 75)

	return mean, median, q1, q3


if __name__ == '__main__':
	print({"lr": 3, **{"lr": 2}})
