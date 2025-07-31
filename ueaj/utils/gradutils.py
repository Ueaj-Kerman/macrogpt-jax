import functools
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from jax import lax, dtypes
from typing import Any, NamedTuple, Callable
from flax import nnx
from flax.nnx import rnglib as rng
from orbax.checkpoint.checkpoint_utils import PLACEHOLDER


def _normalize_dtype(dtype: Any):
	"""Convert various dtype formats to a JAX-compatible dtype"""
	return dtypes.canonicalize_dtype(dtype)


def custom_astype(x: jax.Array, dtype: Any, cast_forward: bool = True, cast_backward: bool = True) -> jax.Array:
	"""
	Custom astype with control over forward and backward pass casting.

	Args:
		x: Input array
		dtype: Target dtype
		cast_forward: If True, cast to dtype in forward pass
		cast_backward: If True, cast gradients to dtype in backward pass

	Returns:
		Array potentially casted to dtype based on flags

	Examples:
		# Old astype_fwd_noop_bwd:
		custom_astype(x, dtype, cast_forward=True, cast_backward=False)
		
		# Old noop_fwd_astype_bwd:
		custom_astype(x, dtype, cast_forward=False, cast_backward=True)
	"""
	# Convert dtype outside the custom_vjp to avoid argument issues
	jax_dtype = _normalize_dtype(dtype)

	@jax.custom_vjp
	def _custom_astype(x_inner):
		if cast_forward:
			return lax.convert_element_type(x_inner, jax_dtype)
		else:
			return x_inner

	def _fwd(x_inner):
		# Forward: conditionally convert based on cast_forward
		if cast_forward:
			y = lax.convert_element_type(x_inner, jax_dtype)
		else:
			y = x_inner
		return y, None

	def _bwd(_, g):
		# Backward: conditionally cast gradient based on cast_backward
		if cast_backward:
			return (lax.convert_element_type(g, jax_dtype),)
		else:
			return (g,)

	_custom_astype.defvjp(_fwd, _bwd)
	return _custom_astype(x)


# Compatibility aliases
def astype_fwd_noop_bwd(x: jax.Array, dtype: Any) -> jax.Array:
	"""Legacy function - use custom_astype(x, dtype, cast_forward=True, cast_backward=False) instead."""
	return custom_astype(x, dtype, cast_forward=True, cast_backward=False)


def noop_fwd_astype_bwd(x: jax.Array, dtype: Any) -> jax.Array:
	"""Legacy function - use custom_astype(x, dtype, cast_forward=False, cast_backward=True) instead."""
	return custom_astype(x, dtype, cast_forward=False, cast_backward=True)


def identity_grad(x: jax.Array, lambda_: float | jax.Array = .1):
	@jax.custom_vjp
	def _igrad(x: jax.Array):
		return x

	def _igrad_fwd(x: jax.Array):
		return x, x

	def _igrad_bwd(resid: jax.Array, grad: jax.Array):
		return (grad + lambda_ * resid,)

	_igrad.defvjp(_igrad_fwd, _igrad_bwd)
	return _igrad(x)


def debug_tensor(x: jax.Array, name: str = "tensor", show_values: bool = False) -> jax.Array:
	"""
	Debug function that prints tensor information in forward and backward passes.

	Args:
		x: Input array
		name: Name for the tensor being tracked (for clearer debug output)
		show_values: If True, also print RMS values of the tensor/gradient

	Returns:
		Input array unchanged (identity function with debug prints)

	Examples:
		# Old debug_dtype:
		debug_tensor(x, name="my_tensor", show_values=False)
		
		# Old debug_grad_flow:
		debug_tensor(x, name="my_tensor", show_values=True)
	"""

	@jax.custom_vjp
	def _debug_tensor(x_inner):
		return x_inner

	def _fwd(x_inner):
		# Forward: print debug info
		print(f"[FWD] {name}: dtype = {x_inner.dtype}")
		if show_values:
			jax.debug.print("[FWD] {}: {}", name, jnp.sqrt(jnp.square(x_inner).mean()))
		return x_inner, None

	def _bwd(_, g):
		# Backward: print gradient info
		print(f"[BWD] {name}: dtype = {g.dtype}")
		if show_values:
			jax.debug.print("[BWD] {}: {}", name, jnp.sqrt(jnp.square(g).mean()))
		return (g,)

	_debug_tensor.defvjp(_fwd, _bwd)
	return _debug_tensor(x)


# Compatibility aliases
def debug_dtype(x: jax.Array, name: str = "tensor") -> jax.Array:
	"""Legacy function - use debug_tensor(x, name, show_values=False) instead."""
	return debug_tensor(x, name, show_values=False)


def debug_grad_flow(x: jax.Array, name: str = "tensor") -> jax.Array:
	"""Legacy function - use debug_tensor(x, name, show_values=True) instead."""
	return debug_tensor(x, name, show_values=True)


def te_gradient_workaround(x: jax.Array) -> jax.Array:
	"""
	Workaround for TransformerEngine CUDA illegal memory access error.

	This error occurs when normalized tensors are passed to TE's fused attention
	during JIT compilation. The custom VJP breaks the problematic gradient pattern
	while preserving correct gradients.
	"""

	@jax.custom_vjp
	def _identity_with_grad(x):
		return x

	def _fwd(x):
		return x, x.sum()

	def _bwd(_, g):
		# Pass gradient through normally
		return (g,)

	_identity_with_grad.defvjp(_fwd, _bwd)
	return _identity_with_grad(x)


_PLACEHOLDER = FrozenDict({FrozenDict(): FrozenDict()})

class Checkpoint(nnx.Variable):
	pass

class WrappedVJP(nnx.Module):
	def __init__(self, vjp_fn: Callable, wrapper: Callable):
		super().__init__()
		state, fn_def = jax.tree.flatten(vjp_fn)

		self.state = {i: Checkpoint(s) for i, s in enumerate(state)}
		self.fn_def = fn_def
		self.wrapper = wrapper

	def __call__(self, *args):
		state = (self.state[i] for i in range(len(self.state)))
		vjp_fn = jax.tree.unflatten(self.fn_def, state)
		if self.wrapper is None:
			return vjp_fn(*args)
		else:
			return self.wrapper(vjp_fn(*args))

def nnx_vjp(fun, *args, has_aux=False, wrt=nnx.Param, **kwargs):
	"""
	Compute the VJP of a function using nnx.vjp.

	Args:
		fun: Function to differentiate
		*args: Positional arguments to fun
		has_aux: Whether fun returns auxiliary data
		wrt: Filter for which module parameters to differentiate (default: nnx.Param)
		**kwargs: Keyword arguments to fun

	Returns:
		(output, vjp_fn) if has_aux=False
		((output, aux), vjp_fn) if has_aux=True
	"""

	# Combine args and kwargs into a single tree
	def is_module(x):
		return isinstance(x, nnx.Module)

	# Apply splitting to the entire tree
	vals, tree_def = jax.tree.flatten((args, kwargs), is_leaf=is_module)

	diff_states, diff_vals, non_diff_vals = set(), [], []
	for i, val in enumerate(vals):
		if isinstance(val, nnx.Module):
			graph_def, params, other_state = nnx.split(val, wrt, ...)
			diff_vals.append(params)
			non_diff_vals.append((graph_def, other_state))
			diff_states.add(i)
		else:
			diff_vals.append(val)
			non_diff_vals.append(None)

	def wrapper(diff):
		vals = []
		for i, (diff_val, non_diff_val) in enumerate(zip(diff, non_diff_vals)):
			if i in diff_states:
				params = diff_val
				graph_def, other_state = non_diff_val
				vals.append(nnx.merge(graph_def, params, other_state))
			else:
				vals.append(diff_val)
		args, kwargs = jax.tree.unflatten(tree_def, vals)
		return fun(*args, **kwargs)

	vjp_out = jax.vjp(wrapper, diff_vals, has_aux=has_aux)

	def vjp_wrap(d_diff):
		args, kwargs = jax.tree.unflatten(tree_def, d_diff)
		if not kwargs:
			return args
		else:
			return args, kwargs

	_vjp_fn = vjp_out[1]

	# vjp_fn = WrappedVJP(_vjp_fn, lambda result: vjp_wrap(result[0]))
	vjp_fn = lambda x: vjp_wrap(_vjp_fn(x)[0])

	if has_aux:
		return vjp_out[0], vjp_fn, vjp_out[2]
	else:
		return vjp_out[0], vjp_fn

@functools.partial(jax.jit, static_argnums=(0,))
def run_vjp(fn_def, state):
	fn = jax.tree.unflatten(fn_def, state)
	return fn(jnp.ones((3, 20)))

def custom_grad_vectors(vec1: jax.Array, vec2: jax.Array, var_lr: float = 1.0) -> jax.Array:
	"""
	Returns the first vector, but with custom gradient behavior:
	- Gradient of vec1 is just the gradient itself
	- Gradient of vec2 is the error between variance estimate and actual gradient variance
	
	Args:
		vec1: First input vector (mean)
		vec2: Second input vector (variance estimate)
		var_lr: Learning rate multiplier for variance gradient
		
	Returns:
		vec1 unchanged
	"""
	@jax.custom_vjp
	def _custom_grad(v1, v2):
		return v1
	
	def _fwd(v1, v2):
		return v1, v2
	
	def _bwd(v2_saved, g):
		# Gradient for vec1 is just g
		# Gradient for vec2 is the negative error: -(g^2 - variance_estimate)
		# This trains the variance to predict the squared gradient
		# Negated so SGD moves in the correct direction
		grad_var = var_lr * (v2_saved - g * g)
		return (g, grad_var)
	
	_custom_grad.defvjp(_fwd, _bwd)
	return _custom_grad(vec1, vec2)


if __name__ == "__main__":
	linear = nnx.LoRA(10, 5, 10, rngs=rng.Rngs(0))
	x = jnp.ones((3, 10))

	output, vjp_fn = nnx_vjp(lambda linear, x: linear(x), linear, x)

	graph_def, state = nnx.split(vjp_fn)
	print(graph_def)
	print(state)

	print(vjp_fn(jnp.ones((3, 10))))