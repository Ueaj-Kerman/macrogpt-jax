
from flax import nnx
from jax import numpy as jnp

class A(nnx.Module):
	def __init__(self, rngs: nnx.Rngs):
		super().__init__()
		self.w = nnx.Param(nnx.initializers.ones(rngs.params(), (1, 2), jnp.float32))

class B(nnx.Module):
	def __init__(self, rngs: nnx.Rngs):
		super().__init__()
		self.c = {str(i): A(rngs) for i in range(3)}

class C(nnx.Module):
	def __init__(self, rngs: nnx.Rngs):
		super().__init__()
		self.a = [B(rngs) for _ in range(3)]


c = C(nnx.Rngs(0))

params = nnx.state(c, nnx.Param)

print(params)

print(" == Test modification == ")
for path, module in c.iter_modules():
	if not isinstance(module, A):
		continue
	value = params
	for k in path:
		value = value[k]
	print(path, value)
