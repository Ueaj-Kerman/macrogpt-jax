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


def astype_fwd_noop_bwd(x: jax.Array, dtype: Any) -> jax.Array:
	"""
	Custom astype that preserves gradient dtype in backwards pass.

	Forward pass: converts x from its original dtype to target dtype
	Backward pass: keeps gradients in whatever dtype they come in (no recasting)

	This avoids issues where fp8->fp16 casting in forward pass would
	try to cast fp16 gradients back to fp8 in backward pass.
	"""
	# Convert dtype outside the custom_vjp to avoid argument issues
	jax_dtype = _normalize_dtype(dtype)

	@jax.custom_vjp
	def _astype_preserve_grad(x_inner):
		return lax.convert_element_type(x_inner, jax_dtype)

	def _fwd(x_inner):
		# Forward: convert to target dtype, don't save anything (we won't need it)
		y = lax.convert_element_type(x_inner, jax_dtype)
		return y, None

	def _bwd(_, g):
		# Backward: return gradient as-is, don't cast back to original dtype
		return (g,)

	_astype_preserve_grad.defvjp(_fwd, _bwd)
	return _astype_preserve_grad(x)


def noop_fwd_astype_bwd(x: jax.Array, dtype: Any) -> jax.Array:
	"""
	Custom astype that does nothing in forward pass but casts in backward pass.

	Forward pass: returns x unchanged (no dtype conversion)
	Backward pass: casts gradients to the specified dtype

	This is the opposite of astype_fwd_noop_bwd.
	"""
	# Convert dtype outside the custom_vjp to avoid argument issues
	jax_dtype = _normalize_dtype(dtype)

	@jax.custom_vjp
	def _noop_fwd_astype_bwd(x_inner):
		return x_inner

	def _fwd(x_inner):
		# Forward: return as-is, save original dtype for backward
		return x_inner, None

	def _bwd(orig_dtype, g):
		# Backward: cast gradient to target dtype
		return (lax.convert_element_type(g, jax_dtype),)

	_noop_fwd_astype_bwd.defvjp(_fwd, _bwd)
	return _noop_fwd_astype_bwd(x)


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


def debug_dtype(x: jax.Array, name: str = "tensor") -> jax.Array:
	"""
	Debug version of astype that prints dtype information in forward and backward passes.

	Forward pass: converts x from its original dtype to target dtype and prints info
	Backward pass: keeps gradients in whatever dtype they come in and prints info

	Args:
		x: Input array
		dtype: Target dtype for forward pass
		name: Optional name for the tensor being tracked (for clearer debug output)

	Returns:
		Array with converted dtype
	"""

	@jax.custom_vjp
	def _debug_astype(x_inner):
		return x_inner

	def _fwd(x):
		# Forward: convert to target dtype
		print(f"[FWD] {name}: dtype = {x.dtype}")
		return x, None

	def _bwd(_, g):
		# Backward: return gradient as-is and print debug info
		print(f"[BWD] {name}: dtype = {g.dtype}")
		return (g,)

	_debug_astype.defvjp(_fwd, _bwd)
	return _debug_astype(x)


def debug_grad_flow(x: jax.Array, name: str = "tensor") -> jax.Array:
	@jax.custom_vjp
	def _debug_astype(x_inner):
		return x_inner

	def _fwd(x):
		# Forward: convert to target dtype
		return x, None

	def _bwd(_, g):
		# Backward: return gradient as-is and print debug info
		jax.debug.print("[BWD] {}: {}", name, jnp.sqrt(jnp.square(g).mean()))
		return (g,)

	_debug_astype.defvjp(_fwd, _bwd)
	return _debug_astype(x)


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