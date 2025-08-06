import functools
from functools import lru_cache
from typing import Sequence, Callable, Literal, Optional, Any

import optax
from optax import GradientTransformation, TransformInitFn, TransformUpdateFn
from jax import numpy as jnp
import jax
from flax import nnx


def clip_outliers(z: float | int = 2):
	def update_fn(updates, params=None):
		sigma = jax.tree.map(lambda u: jnp.sqrt(jnp.mean(jnp.square(u), dtype=jnp.float32)).astype(u.dtype), updates)
		updates = jax.tree.map(lambda u, sigma: jnp.clip(u, -z * sigma, z * sigma), updates, sigma)
		return updates

	return optax.stateless(update_fn)


def cast_to(dtype: jnp.dtype | None) -> GradientTransformation:
	return optax.stateless(
		lambda updates, params: jax.tree.map(
			lambda u, p: u.astype(dtype if dtype is not None else p.dtype),
			updates,
			params
		)
	)


def lerp(a, b, alpha):
	return a + alpha * (b - a)


def leading_multiply(array, target):
	return array.reshape((-1,) + (1,) * (len(target.shape) - 1)) * target


def multiscale_momentum(
	mom_coeffs: Sequence[float],
	accumulate: float | Sequence[float] | None = None,
	preconditioner: Callable[[jax.Array], jax.Array] = lambda x: jnp.sign(x),
	dtype=jnp.float32,
	cooldown_frac: float = 0.
) -> GradientTransformation:
	mom_coeffs = jnp.array(mom_coeffs)

	if accumulate is None:
		accumulate = jnp.ones_like(mom_coeffs)
	else:
		accumulate = jnp.array(accumulate)

	base_scales = ((1 - mom_coeffs) / (1 - jnp.min(mom_coeffs)))
	base_scales = base_scales / jnp.sqrt(base_scales.sum() + 1e-6)

	cooldown = (1 - jnp.max(mom_coeffs)) / (1 - mom_coeffs)
	scales = jnp.where(cooldown_frac > cooldown, 0, base_scales)

	def init_fn(params):
		return jax.tree.map(lambda x: jnp.zeros((len(mom_coeffs),) + x.shape, dtype=dtype), params)

	def update_fn(grad, state, params=None):
		state = jax.tree.map(
			lambda grad, mom: (
					leading_multiply(accumulate * (1 - mom_coeffs), grad[None, ...])
					+ leading_multiply(mom_coeffs, mom)
			).astype(dtype),
			grad,
			state
		)

		update = jax.tree.map(
			lambda grad, mom: jnp.sum(
				leading_multiply(
					scales,
					preconditioner(
						leading_multiply(1 - mom_coeffs, grad[None, ...]) + leading_multiply(mom_coeffs, mom)
					)
				),
				axis=0
			),
			grad,
			state
		)
		return update, state

	return GradientTransformation(init_fn, update_fn)


def scale_by_muonP(base_lr: float):
	"""
	Scale lr by muP, assumes
	:param base_lr:
	:param wd:
	:return:
	"""
	return optax.stateless(
		lambda updates, params: jax.tree.map(
			lambda u, p: -max(u.shape[-1] / u.shape[-2], 1) * base_lr * u,
			updates,
			params
		)
	)


# -3 (base)
# -4
def multiscale_muon(lr=.125, warmup_frac=1., wd: Optional[float] = None, method: Literal['muon', 'optimal'] = 'muon', dtype=jnp.float32):
	return optax.chain(
		clip_outliers(2),
		multiscale_momentum(
			[0.96875, 0.9921875, 0.998046875],
			preconditioner=functools.partial(orthogonalize, method=method),
			accumulate=[1., warmup_frac, warmup_frac**2],
			dtype=dtype
		),
		optax.add_decayed_weights(wd) if wd else optax.identity(),
		scale_by_muonP(lr),
		cast_to(None)
	)


def muon(lr=.125, wd: Optional[float] = None, method: Literal['muon', 'optimal'] = 'muon'):
	return optax.chain(
		clip_outliers(2),
		multiscale_momentum(
			[0.96875],
			preconditioner=functools.partial(orthogonalize, method=method)
		),
		optax.add_decayed_weights(wd) if wd else optax.identity(),
		scale_by_muonP(lr),
		cast_to(None)
	)


def orthogonalize(
	x,
	ns_steps=5,
	eps=1e-8,
	method: Literal['muon', 'optimal'] = 'muon',
	convergence_speed: Optional[float] = 1e-3
):
	if method == 'muon':
		coeffs = (3.4445, -4.7750, 2.0315)
	elif method == 'optimal':
		with jax.ensure_compile_time_eval():
			coeffs = optimal_composition(convergence_speed, ns_steps)
	else:
		raise ValueError(f'Unknown value for `method`: {method}')
	coeffs = jnp.array(coeffs)

	x_shape = x.shape
	if x.ndim <= 1:
		raise ValueError(f'Input must have shape (m, n), got {x.shape}')
	elif x.ndim == 2:
		x = x[None, ...]
	elif x.ndim > 3:
		x = x.reshape((-1,) + x.shape[-2:])
	return jax.vmap(
		functools.partial(
			optax.contrib._muon.orthogonalize_via_newton_schulz,
			ns_steps=ns_steps,
			eps=eps,
			ns_coeffs=coeffs
		),
		in_axes=0
	)(x).reshape(x_shape)


def optimal_quintic(l, u):
	assert 0 <= l <= u
	if 1 - 5e-6 <= l / u:
		# Above this threshold, the equoscillating polynomials
		# is numerically equal to...
		return (15 / 8) / u, (-10 / 8) / (u ** 3), (3 / 8) / (u ** 5)
	# This initialization becomes exact as l -> u
	q = (3 * l + 1) / 4
	r = (l + 3) / 4
	E, old_E = jnp.inf, None
	count = 0
	while not old_E or abs(old_E - E) > 1e-15:
		old_E = E
		LHS = jnp.array(
			[
				[l, l ** 3, l ** 5, 1],
				[q, q ** 3, q ** 5, -1],
				[r, r ** 3, r ** 5, 1],
				[u, u ** 3, u ** 5, -1],
			]
		)
		a, b, c, E = jnp.linalg.solve(LHS, jnp.ones(4))
		q, r = jnp.sqrt(
			(-3 * b + jnp.array([-1, 1]) * jnp.sqrt(9 * b ** 2 - 20 * a * c)) /
			(10 * c)
		)
		count += 1
		if count > 1000:
			raise ValueError(f'Failed to converge after {count} iterations.')
	return float(a), float(b), float(c)


@lru_cache
def optimal_composition(min_orth, num_iters, cushion=0.02407327424182761):
	u = 1
	coefficients = []
	for _ in range(num_iters):
		a, b, c = optimal_quintic(max(min_orth, cushion * u), u)
		# Due to cushioning , this may be centered around 1 with
		# respect to 0.024*u, u. Recenter it around 1 with respect
		# to l, u, meaning find c so that 1 - c*p(l) = c*p(u) - 1:
		pl = a * min_orth + b * min_orth ** 3 + c * min_orth ** 5
		pu = a * u + b * u ** 3 + c * u ** 5
		rescalar = 2 / (pl + pu)
		a *= rescalar
		b *= rescalar
		c *= rescalar
		# Optionally incorporate safety factor here :
		# a /= 1.01; b /= 1.01**3; c /= 1.01**5
		coefficients.append((a, b, c))
		min_orth = a * min_orth + b * min_orth ** 3 + c * min_orth ** 5
		u = 2 - min_orth
	return coefficients


def einsum_aware(optimizer: GradientTransformation, model: nnx.Module) -> GradientTransformation:
	"""Wrap an optimizer to handle einsum parameters in reduced form.
	
	This wrapper automatically detects einsum modules in the model and transforms
	their parameters to reduced form before the optimizer update, then transforms
	them back to canonical form afterward.
	
	Args:
		optimizer: The base optimizer (e.g., muon, multiscale_muon)
		model: The model containing potential einsum layers
		
	Returns:
		A wrapped optimizer that handles einsum transformations
	"""
	# Build a registry mapping parameter paths to their einsum metadata
	path_to_metadata = {}
	
	# Use iter_modules to find all einsum modules
	for path, module in model.iter_modules():
		if hasattr(module, 'get_einsum_metadata'):
			metadata = module.get_einsum_metadata()
			for param_name, meta in metadata.items():
				# path is already a tuple, append param name and 'value'
				# nnx.state returns paths with 'value' at the end for Param objects
				full_path = path + (param_name, 'value')
				path_to_metadata[full_path] = meta
	
	if not path_to_metadata:
		# No einsum layers found, return original optimizer
		return optimizer
	
	def transform_tree(tree, to_reduced=True):
		"""Transform einsum parameters between canonical and reduced forms."""
		def transform_param(path, param):
			# Handle both raw arrays and nnx.Param objects
			if hasattr(param, 'value'):
				value = param.value
			elif isinstance(param, jax.Array):
				value = param
			else:
				return param
			
			# Convert path to tuple (handle both string keys and indices)
			# Special handling for GetAttrKey - extract the name
			path_parts = []
			for p in path:
				if hasattr(p, 'key'):
					path_parts.append(p.key)
				elif hasattr(p, 'name'):  # GetAttrKey
					path_parts.append(p.name)
				else:
					path_parts.append(p)
			path_tuple = tuple(path_parts)
			
			# Check if this path matches any einsum parameter
			# Look for suffixes to handle vmapped cases
			meta = None
			for i in range(len(path_tuple)):
				suffix = path_tuple[i:]
				if suffix in path_to_metadata:
					meta = path_to_metadata[suffix]
					break
			
			if meta is None:
				return param
			
			# Calculate number of batch dimensions
			expected_shape = meta.canonical_shape if to_reduced else meta.reduced_shape
			batch_ndim = len(value.shape) - len(expected_shape)
			
			if batch_ndim < 0 or value.shape[batch_ndim:] != expected_shape:
				# Shape doesn't match
				return param
			
			if to_reduced:
				# Transform canonical -> reduced
				# Offset transpose axes by batch dimensions
				axes = list(range(batch_ndim)) + [ax + batch_ndim for ax in meta.transpose_axes]
				transposed = jnp.transpose(value, axes)
				
				# Reshape keeping batch dims intact
				batch_shape = value.shape[:batch_ndim]
				reshaped = transposed.reshape(batch_shape + meta.reduced_shape)
				
				# Return appropriately based on input type
				if hasattr(param, 'replace'):
					return param.replace(value=reshaped)
				else:
					return reshaped
			else:
				# Transform reduced -> canonical
				# Compute intermediate shape from reduced shape
				batch_shape = value.shape[:batch_ndim]
				reducing_size, non_reducing_size = meta.reduced_shape
				
				# Infer intermediate shape from canonical shape and transpose axes
				canonical_dims = len(meta.canonical_shape)
				intermediate_shape = [0] * canonical_dims
				for i, ax in enumerate(meta.transpose_axes):
					intermediate_shape[i] = meta.canonical_shape[ax]
				
				# Reshape to intermediate
				reshaped = value.reshape(batch_shape + tuple(intermediate_shape))
				
				# Inverse transpose (offset by batch dims)
				inv_transpose = [0] * canonical_dims
				for i, ax in enumerate(meta.transpose_axes):
					inv_transpose[ax] = i
				axes = list(range(batch_ndim)) + [ax + batch_ndim for ax in inv_transpose]
				transposed = jnp.transpose(reshaped, axes)
				
				# Return appropriately based on input type
				if hasattr(param, 'replace'):
					return param.replace(value=transposed)
				else:
					return transposed
		
		return jax.tree_util.tree_map_with_path(transform_param, tree)
	
	# Create wrapped optimizer
	def init_fn(params):
		# Transform to reduced form before init
		reduced_params = transform_tree(params, to_reduced=True)
		return optimizer.init(reduced_params)
	
	def update_fn(updates, state, params=None):
		# Transform gradients and params to reduced form
		reduced_updates = transform_tree(updates, to_reduced=True)
		reduced_params = transform_tree(params, to_reduced=True) if params is not None else None
		
		# Apply optimizer update
		new_updates, new_state = optimizer.update(reduced_updates, state, reduced_params)
		
		# Transform updates back to canonical form
		canonical_updates = transform_tree(new_updates, to_reduced=False)
		
		return canonical_updates, new_state
	
	return GradientTransformation(init_fn, update_fn)


if __name__ == '__main__':
	for fn, (a, b, c) in zip("fghijklmno", optimal_composition(1e-3, 5)):
		print(f"{fn}(x) = {a}x + {b}x^3 + {c}x^5")