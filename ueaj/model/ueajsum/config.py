import itertools
import operator
from dataclasses import dataclass, replace
from functools import reduce
from typing import Tuple, Callable, Sequence, Type, Mapping

import jax
from flax import nnx
from flax.core import FrozenDict
from flax.nnx.nn import initializers as inits
from jax import lax, numpy as jnp


@dataclass(frozen=True)
class ArgumentConfig:
	"""
	Statistics for an argument in an einsum

	:param variance: The expected variance of the argument, if non-null, the outputs and gradients will be scaled. For
	parameters, this also determines the initializer. For outputs, this sets the expected variance of the gradient.
	:param precision: The precision with which to compute the output, or if specified for an input, the precision with
	which to compute the gradient.
	:param _dtype: For parameters, the dtype of the parameter, for inputs it is an optional assertion.
	:param _grad_dtype: For parameters, the dtype of the gradient, though it can be applied to inputs too, but is not
	recommended.
	:param in_axes: Specify the input/reducing dimensions for initialization and optimizers (like muon)
	:param batch_axes: Specify the batch/fused dimensions that should be treated as batch dims for optimizers
	"""
	shape: str

	variance: float | None = None
	precision: lax.PrecisionLike = None

	_dtype: jax.typing.DTypeLike | None = None
	_grad_dtype: jax.typing.DTypeLike | None = None

	in_axes: Tuple[int, ...] = None
	batch_axes: Tuple[int, ...] = None

	@property
	def grad_dtype(self) -> jax.typing.DTypeLike | None:
		if self._grad_dtype is None:
			return self._dtype
		else:
			return self._grad_dtype

	@property
	def dtype(self) -> jax.typing.DTypeLike:
		return self._dtype

	def with_shape(self, output_shape: str) -> "ArgumentConfig":
		"""Return a config with the same attributes but a different shape."""
		return replace(self, shape=output_shape)

	def with_precision(self, precision: lax.PrecisionLike) -> "ArgumentConfig":
		"""Return a config with the same attributes but a different precision."""
		return replace(self, precision=precision)

	def with_dtype(self, dtype: jax.typing.DTypeLike) -> "ArgumentConfig":
		"""Return a config with the same attributes but a different dtype."""
		return replace(self, _dtype=dtype)

	def with_grad_dtype(self, grad_dtype: jax.typing.DTypeLike) -> "ArgumentConfig":
		"""Return a config with the same attributes but a different grad_dtype."""
		return replace(self, _grad_dtype=grad_dtype)

	def with_variance(self, variance: float) -> "ArgumentConfig":
		"""Return a config with the same attributes but a different variance."""
		return replace(self, variance=variance)

	def with_reduce_axes(self, in_axes: Tuple[int, ...]) -> "ArgumentConfig":
		"""Return a config with the same attributes but a different reduce_dims_optimizer_heuristic."""
		return replace(self, in_axes=in_axes)

	def with_batch_axes(self, batch_axes: Tuple[int, ...]) -> "ArgumentConfig":
		"""Return a config with the same attributes but different batch_axes."""
		return replace(self, batch_axes=batch_axes)


@dataclass(frozen=True)
class ParamConfig(ArgumentConfig):
	# todo sharding
	group: Type[nnx.Variable] = nnx.Param
	_initializer: inits.Initializer | None = None

	def with_group(self, group: Type[nnx.Variable]) -> "ParamConfig":
		"""Return a param config with the same attributes but a different group."""
		return replace(self, group=group)

	def with_initializer(self, initializer: inits.Initializer | None) -> "ParamConfig":
		"""Return a param config with the same attributes but a different initializer."""
		return replace(self, _initializer=initializer)

	@property
	def dtype(self) -> jax.typing.DTypeLike:
		if self._dtype is None:
			return jnp.bfloat16
		else:
			return self._dtype

	@property
	def initializer(self) -> inits.Initializer | None:
		if self._initializer is None:
			if self.variance is not None:
				return inits.normal(stddev=jnp.sqrt(self.variance))
			if self.in_axes is not None:
				return inits.lecun_normal(in_axis=self.in_axes)
			else:
				return inits.lecun_normal(in_axis=0)
		return self._initializer


def group_filter(group: nnx.Variable | Sequence[nnx.Variable] | None | Callable[..., bool]):
	if callable(group) and not issubclass(group, nnx.Variable):
		predicate = group
	elif isinstance(group, Sequence):
		predicate = lambda arg: isinstance(arg, ParamConfig) and arg.group in group
	elif group is None:
		predicate = lambda arg: isinstance(arg, ArgumentConfig)
	else:
		predicate = lambda arg: isinstance(arg, ParamConfig) and arg.group == group

	return predicate


def dtype(dtype: jax.typing.DTypeLike):
	return lambda arg: replace(arg, _dtype=dtype)


def grad_dtype(dtype: jax.typing.DTypeLike):
	return lambda arg: replace(arg, _grad_dtype=dtype)


SUMMATIONS = Tuple[Tuple[str | int, ...], ...]


@dataclass(frozen=True)
class UeajsumConfig:
	no_instantiation: bool

	arg_configs: Tuple[ArgumentConfig | ParamConfig, ...]
	kwarg_configs: FrozenDict[str, ArgumentConfig | ParamConfig]

	result_config: ArgumentConfig

	sums: SUMMATIONS

	def map_arg(
		self,
		arg: str | int | Sequence[str | int],
		fn: Callable[[ArgumentConfig | ParamConfig], ArgumentConfig | ParamConfig]
	):
		if isinstance(arg, str) and arg in self.kwarg_configs:
			current = dict(self.kwarg_configs)
			current[arg] = fn(current[arg])
			return replace(
				self,
				kwarg_configs=FrozenDict(current),
			)
		elif isinstance(arg, int):
			current = list(self.arg_configs)
			current[arg] = fn(current[arg])
			return replace(
				self,
				arg_configs=tuple(current),
			)
		elif isinstance(arg, Sequence):
			cfg = self
			for a in arg:
				cfg = cfg.map_arg(a, fn)
			return cfg
		else:
			raise ValueError(f"Unknown argument {arg}")

	def map_args(self, fn: Callable[[ArgumentConfig | ParamConfig], ArgumentConfig | ParamConfig]):
		return replace(
			self,
			arg_configs=tuple(map(fn, self.arg_configs)),
			kwarg_configs=FrozenDict({k: fn(v) for k, v in self.kwarg_configs.items()}),
		)

	def map_result(self, fn: Callable[[ArgumentConfig], ArgumentConfig]):
		return replace(
			self,
			result_config=fn(self.result_config),
		)

	def unit(self):
		return replace(
			self.map_args(lambda arg: arg.with_variance(1.0)),
			result_config=self.result_config.with_variance(1.0),
		)

	def filt_map(self, fn, predicate: Callable[..., bool]):
		return self.map_args(lambda arg: fn(arg) if predicate(arg) else arg)

	def group_map(self, fn, group: nnx.Variable | Sequence[nnx.Variable] | None | Callable[..., bool]):
		return self.filt_map(fn, group_filter(group))

	def param(self, config: ParamConfig):
		return self.group_map(lambda a: config.with_shape(a.shape), nnx.Param)

	def bf16(self):
		return self.map_args(dtype(jnp.bfloat16))

	def fp8(self):
		return self.map_args(dtype(jnp.float8_e4m3fn))

	def bf16_params(self):
		return self.group_map(dtype(jnp.bfloat16), nnx.Param)

	def fp8_params(self):
		return self.group_map(dtype(jnp.float8_e4m3fn), nnx.Param)

	def bf16_grads(self):
		return self.group_map(grad_dtype(jnp.bfloat16), nnx.Param)

	def fp32_grads(self):
		return self.group_map(grad_dtype(jnp.float32), nnx.Param)

	def in_axes(self, mapping: Mapping[str | int, Tuple[int, ...]]) -> 'UeajsumConfig':
		config = self
		for k, v in mapping.items():
			config = config.map_arg(k, lambda arg: arg.with_reduce_axes(v))
		return config

	def batch_axes(self, mapping: Mapping[str | int, Tuple[int, ...]]) -> 'UeajsumConfig':
		config = self
		for k, v in mapping.items():
			config = config.map_arg(k, lambda arg: arg.with_batch_axes(v))
		return config

	def in_axes_zero(self):
		param_axes = map(
			lambda kv: kv[0],
			filter(
				lambda kv: isinstance(kv[1], ParamConfig),
				itertools.chain(enumerate(self.arg_configs), self.kwarg_configs.items())
			)
		)
		return self.in_axes({k: (0,) for k in param_axes})

	def get_arg(self, key: int | str):
		if isinstance(key, int):
			return self.arg_configs[key]
		elif isinstance(key, str):
			return self.kwarg_configs[key]
		else:
			raise ValueError(f"Unknown key {key}")


if __name__ == "__main__":
	a = sorted([1, 3, 1], reverse=True)
	b = sorted([1, 3, 2], reverse=True)
	print(a)
	print(b)
	print(a < b)
