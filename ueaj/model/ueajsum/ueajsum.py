import itertools
import operator
from functools import reduce
from typing import Sequence, Mapping, Any

import jax
# todo ueajsum to einsum translation
#	1. grab arguments and configs in order
#	2. add dummies for missing shapes, or reshape output
#	3. update mixsum and varsum

from flax import nnx
from flax.core import FrozenDict
from flax.nnx import rnglib as rng
from jax import numpy as jnp

from ueaj.model.ueajsum import config as cfg
from ueaj.model.ueajsum import parser
from ueaj.model.ueajsum.mixsum import mixsum
from ueaj.utils import either


class Ueajsum(nnx.Module):
	def __init__(self, config: cfg.UeajsumConfig, shape_map: Mapping[str, int], rngs: rng.Rngs):
		super().__init__()

		self.config = config
		self.shape_map = FrozenDict(shape_map)

		for k, v in config.kwarg_configs.items():
			if isinstance(v, cfg.ParamConfig):
				shape = tuple(map(lambda x: shape_map[x], v.shape))
				tensor = v.group(v.initializer(key=rngs.params(), shape=shape).astype(v.dtype))
				setattr(self, k, tensor)

		for i, v in enumerate(config.arg_configs):
			if isinstance(v, cfg.ParamConfig):
				shape = tuple(map(lambda x: shape_map[x], v.shape))
				tensor = v.group(v.initializer(key=rngs.params(), shape=shape).astype(v.dtype))
				setattr(self, f"w_{i}", tensor)

	def __call__(self, *args, **kwargs):
		return self.invoke(self.config, False, *args, **kwargs)

	def parse(self, expr: str) -> cfg.UeajsumConfig:
		kwargs = self._default_arg_dict(pairs=False)
		args = [None] * (max([i for i in kwargs.keys() if isinstance(i, int)])+1)
		for k, v in dict(kwargs).items():
			if isinstance(k, int):
				args[k] = v
				del kwargs[k]
		return parser.parse(expr, True, *args, **kwargs)

	def parse_and_call(self, expr: str, *args, **kwargs):
		return self.invoke(self.parse(expr), False, *args, **kwargs)

	def invoke(self, terms: cfg.UeajsumConfig, override: bool, *args: jax.Array, **kwargs: jax.Array) -> jax.Array:
		arg_dict = self._build_args(terms, override, args, kwargs)

		accumulator = None
		for expr in terms.sums:
			dims = set()
			tensors = []
			configs = []

			for arg in expr:
				tensor, config = arg_dict[arg]
				tensors.append(tensor)
				configs.append(config)
				dims.update(config.shape)

			output_shape = "".join([v for v in terms.result_config.shape if v in dims])

			output = mixsum(configs, terms.result_config.with_shape(output_shape), *tensors)

			old_shape = output.shape
			new_shape = []
			idx = 0
			for v in terms.result_config.shape:
				if v in output_shape:
					new_shape.append(old_shape[idx])
					idx += 1
				else:
					new_shape.append(1)

			output = output.reshape(new_shape)

			if accumulator is None:
				accumulator = output
			else:
				accumulator += output

		return accumulator

	def _default_arg_dict(self, pairs: bool = True):
		arg_dict = {}

		for k, v in enumerate(self.config.arg_configs):
			if isinstance(v, cfg.ParamConfig):
				param = getattr(self, f"w_{k}")
				# Extract value from nnx.Param if needed
				param_value = param.value if hasattr(param, 'value') else param
				arg_dict[k] = (param_value, v) if pairs else v

		for k, v in self.config.kwarg_configs.items():
			if isinstance(v, cfg.ParamConfig):
				param = getattr(self, k)
				# Extract value from nnx.Param if needed
				param_value = param.value if hasattr(param, 'value') else param
				arg_dict[k] = (param_value, v) if pairs else v

		return arg_dict

	def _build_args(self, terms: cfg.UeajsumConfig, override: bool, args: Sequence[jax.Array], kwargs: Mapping[str, jax.Array]):
		arg_dict = self._default_arg_dict()

		c = 0
		for arg in args:
			while c in arg_dict:
				c += 1
			arg_dict[c] = (arg, None)

		for k, v in kwargs.items():
			arg_dict[k] = (v, None)

		for k, v in itertools.chain(enumerate(terms.arg_configs), terms.kwarg_configs.items()):
			if v is None:
				continue
			if k not in arg_dict:
				raise ValueError(f"Missing argument {k}")
			entry = arg_dict[k]
			if override or entry[1] is None:
				arg_dict[k] = (entry[0], v)

		return arg_dict

	def map_state(self, state: nnx.State | Mapping[str, Any], from_optimizer: bool = False):
		"""
		Maps parameter tensors to/from optimizer format by reshaping:
		- If from_optimizer=False: reshape to (reducing_dims, batch_dims)
		- If from_optimizer=True: reshape back to original shape
		
		Uses simple transpose + reshape operations with argsort for inverse.
		"""
		import numpy as np
		
		state = nnx.State(state) if not isinstance(state, nnx.State) else state
		
		# Process each parameter
		for k, v in itertools.chain(enumerate(self.config.arg_configs), self.config.kwarg_configs.items()):
			if not isinstance(v, cfg.ParamConfig):
				continue
			if v.in_axes is None:
				continue

			# Determine the key in state
			state_key = f"w_{k}" if isinstance(k, int) else k
			if state_key not in state:
				continue

			# Get the parameter value
			param = state[state_key]
			
			# Handle both nnx.Param and raw arrays
			if hasattr(param, 'value'):
				param_value = param.value
			else:
				param_value = param

			batch_axes = tuple(sorted(either(v.batch_axes, ())))
			assert not set(v.in_axes) - set(range(len(v.shape))), "In axes must be within shape"
			assert not set(batch_axes) - set(range(len(v.shape))), "Batch axes must be within shape"
			assert not set(batch_axes).intersection(v.in_axes), "Batch axes cannot be in in_axes"

			# calculate number of vdims for from_optimizer=True
			if from_optimizer:
				v_dims = len(param_value.shape) - (2 + len(batch_axes))
			else:
				v_dims = len(param_value.shape) - len(v.shape)

			shift = lambda x: x + v_dims

			in_axes = tuple(sorted(map(shift, v.in_axes)))
			batch_axes = tuple(range(v_dims)) + tuple(map(shift, batch_axes))

			combined = set(in_axes + batch_axes)
			out_axes = tuple(v for v in range(v_dims + len(v.shape)) if v not in combined)

			transpose = batch_axes + in_axes + out_axes
			
			# Fix: For batch_axes, we need to handle v_dims correctly
			def length(x):
				if x < v_dims:
					# This is a vmapped dimension - get its size from the param
					return param_value.shape[x]
				else:
					# This is an original dimension - look it up in shape_map
					return self.shape_map[v.shape[x-v_dims]]

			batch_lens = tuple(map(length, batch_axes)) # keep
			in_lens = tuple(map(length, in_axes)) # lookup unshifted in self state
			out_lens = tuple(map(length, out_axes)) # lookup unshifted in self state

			reshape = batch_lens + (reduce(operator.mul, in_lens, 1),) + (reduce(operator.mul, out_lens, 1),)

			if not from_optimizer:
				# Reshape to mapped shape
				new_value = param_value.transpose(*transpose).reshape(*reshape)
				if hasattr(param, 'replace'):
					# Preserve the original wrapper type (VariableState, Param, etc)
					state[state_key] = param.replace(value=new_value)
				elif hasattr(param, 'value'):
					# Fallback: create Param if replace not available
					state[state_key] = nnx.Param(new_value)
				else:
					state[state_key] = new_value
			else:
				# Reshape from mapped shape back to original
				def argsort(seq):
					# http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
					return sorted(range(len(seq)), key=seq.__getitem__)
				new_value = param_value.reshape(*batch_lens, *in_lens, *out_lens).transpose(argsort(transpose))
				if hasattr(param, 'replace'):
					# Preserve the original wrapper type (VariableState, Param, etc)
					state[state_key] = param.replace(value=new_value)
				elif hasattr(param, 'value'):
					# Fallback: create Param if replace not available
					state[state_key] = nnx.Param(new_value)
				else:
					state[state_key] = new_value

		return state

if __name__ == "__main__":
	size_dict = {
		'd': 16,
		'k': 4,
		'v': 4,
		'h': 3,
		'i': 4,
		'f': 2
	}

	make_ueajsum = lambda c: Ueajsum(c, size_dict, rngs=nnx.Rngs(0))

	# Test with explicit in_axes configuration
	kv = make_ueajsum(
		parser.parse("bnd,*fdhik->bnfhik").in_axes({1: (1,)}).batch_axes({1: (0,)})
	)
	params = nnx.state(kv, nnx.Param)
	print("Original shapes:", jax.tree.map(lambda x: x.shape, params))

	# Map to optimizer format
	mapped = kv.map_state(params, from_optimizer=False)
	print("Mapped shapes:", jax.tree.map(lambda x: x.shape, mapped))

	# Map back from optimizer format
	unmapped = kv.map_state(mapped, from_optimizer=True)
	print("Unmapped shapes:", jax.tree.map(lambda x: x.shape, unmapped))
	print("All close:", jax.tree.map(jax.numpy.allclose, params, unmapped))