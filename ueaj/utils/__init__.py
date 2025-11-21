import os
import typing

def either(a, b):
	return a if a is not None else b

import jax
import jax.numpy as jnp

LOW_PRECISION = set(map(jax.dtypes.canonicalize_dtype, (
	jnp.float8_e4m3fn,
	jnp.float8_e5m2,
	jnp.float8_e4m3fnuz,
	jnp.float8_e5m2fnuz,
	jnp.float8_e4m3b11fnuz,
)))

T = typing.TypeVar('T')

class Value(typing.Generic[T]):
	__slots__ = ("_name", "value")

	_name: str
	value: T

	def __init__(self, name: str, default: T):
		self._name = name
		self._set(default)

	def _set(self, value: T) -> None:
		self.value = value


def make_config(name: str, default: T, *args, **kwargs) -> Value[T]:
	holder = Value(name, default)
	jax.config.add_option(
		name,
		holder,
		str,
		args,
		kwargs
	)
	return holder

backend = jax.default_backend()
DEFAULT_ACCUM_TYPE = make_config(
	'default_accumulation_type',
	jax.dtypes.canonicalize_dtype(os.environ.get('DEFAULT_ACCUM_TYPE', jnp.bfloat16 if backend == 'tpu' else jnp.float32)),
	help='Set the default accumulation type for rmsnorm among other things.'
)

from ueaj.utils.gradutils import *
from ueaj.utils.tensorutil import *
from ueaj.utils.profiling import *
from ueaj.utils.profile_config import maybe_profile, get_profiling_config