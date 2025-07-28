"""
Simple configuration system for hierarchical function and class composition.

The @config decorator adds an .override() method to functions and classes.
- For functions: Returns the function with an added .override() method
- For classes: Returns the class with an added .override() class method

Example:
    @config
    def create_embed(vocab_size, d, init="normal"):
        return Embed(vocab_size, d, init)

    @config
    def create_fp8_embed(vocab_size, d, init="normal"):
        return ...

    @config
    class Llama:
        def __init__(self, model_d, vocab_size=32000, embed=create_embed):
            self.embed = embed(vocab_size, model_d)

    # Create configured factory function
    llama_3 = Llama.override(
        vocab_size=128256,
        # Override args of default function
        embed=override(init="kaiming"),
    )

    llama_fp8 = Llama.override(
        vocab_size=128256,
        # Replace the default function entirely
        embed=create_fp8_embed.override(init="kaiming"),
    )

    model = llama_3(4096)  # llama_3 is a function, not a class

Key points:
- @config adds .override() to functions and classes mutably!
- func.override(**kwargs) returns a new function with baked-in overrides
- Class.override(**kwargs) returns a factory function (not a class)
- override(**kwargs) is a special marker to override args of a default function
- Type system recognizes decorated objects through Annotated[T, HasOverride]

@Author: @_ueaj
"""
import functools
import inspect
from functools import wraps, partial
from typing import Callable, TypeVar, Protocol, Any, ParamSpec, Type, overload, Annotated

P = ParamSpec('P')
T = TypeVar('T')
C = TypeVar('C')

# Simple protocol for the override method
class HasOverride(Protocol):
	"""Protocol for types that have the override method."""
	@classmethod
	def override(cls, **overrides: Any) -> Callable[..., Any]: ...

@overload
def config(obj: Type[C]) -> Annotated[Type[C], HasOverride]:
	...

@overload
def config(obj: Callable[P, T]) -> Annotated[Callable[P, T], HasOverride]:
	...

def config(obj: C) -> Annotated[C, HasOverride]:
	"""Add .override() method to a function or class.
	
	The returned object is annotated to indicate it has both:
	- Its original type/interface (C)
	- The HasOverride protocol (providing the .override() method)
	"""
	if inspect.isclass(obj):
		return _config_class(obj)
	if callable(obj):
		return _config_function(obj)
	raise TypeError("@config can only be used on callables (functions or classes)")


def override(**kwargs):
	"""Special marker for overriding just the arguments of a default."""
	return ('override', kwargs)


def _apply_and_call(target, sig, overrides, args, kwargs):
	"""Helper to bind arguments, apply overrides, and call the target."""
	bound = sig.bind_partial(*args, **kwargs)
	bound.apply_defaults()

	varkw_name = next((p.name for p in sig.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD), None)

	for name, value in overrides.items():
		if name in sig.parameters:
			if isinstance(value, tuple) and value[0] == 'override':
				_, override_kwargs = value
				current = bound.arguments.get(name, sig.parameters[name].default)

				if hasattr(current, 'override'):
					bound.arguments[name] = current.override(**override_kwargs)
				elif callable(current) and current != inspect.Parameter.empty:
					bound.arguments[name] = partial(current, **override_kwargs)
				else:
					raise TypeError(
						f"Argument '{name}' is not a configurable or callable, so its arguments cannot be overridden."
					)
			else:
				bound.arguments[name] = value
		elif varkw_name:
			if varkw_name not in bound.arguments:
				bound.arguments[varkw_name] = {}
			bound.arguments[varkw_name][name] = value

	return target(*bound.args, **bound.kwargs)

def _validate_overrides(target_name, sig, overrides):
	"""Checks if override keys are valid for the given signature."""
	accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
	for name in overrides:
		if name not in sig.parameters and not accepts_kwargs:
			available = [k for k in sig.parameters.keys() if k != 'self']
			raise TypeError(
				f"Invalid override '{name}' for '{target_name}'. "
				f"Available parameters: {available}"
			)

def _override_method(func, **overrides: Any):
	if hasattr(func, 'overrides'):
		overrides = {**func.overrides, **overrides}
		func = func.__wrapped__

	sig = inspect.signature(func)
	_validate_overrides(func.__name__, sig, overrides)

	@wraps(func)
	def overridden(*args: P.args, **kwargs: P.kwargs) -> T:
		return _apply_and_call(func, sig, overrides, args, kwargs)

	overridden.overrides = overrides
	return config(overridden)


def _config_function(func: Callable[P, T]):
	"""Adds .override() to a function."""
	func.override = functools.partial(_override_method, func)
	return func

def _config_class(cls):
	"""Adds .override() to a class that returns a factory function.

	The class itself is modified in-place to add the override method.
	The type system sees this through the Annotated return type.
	"""
	def init(*args, **kwargs) -> C:
		return cls(*args, **kwargs)

	# Add the override method
	cls.override = functools.partial(_override_method, init)
	return cls