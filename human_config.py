import functools
import inspect
from dataclasses import dataclass
from typing import Callable

from flax.core import FrozenDict

@dataclass(frozen=True)
class FunctionOverride:
	fn: Callable | None
	args: FrozenDict

def configurable(fn):
	signature = inspect.signature(fn)
	@functools.wraps(fn)
	def wrapper(*args, **kwargs):
		items = signature.parameters.items()

		new_args = []
		for arg in args:
			if isinstance(arg, FunctionOverride):
				if arg.fn is not None:
					f = arg.fn
				else:
					f = items.get

			else:
				new_args.append(arg)
		args = tuple(new_args)

		return fn(*args, **kwargs)
	return wrapper
