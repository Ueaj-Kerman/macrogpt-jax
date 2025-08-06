import jax

from ueaj.model.soft_attn import *
from ueaj.model.mlp import *
from ueaj.model.rmsnorm import *
from ueaj.utils.configurator import *

TENSOR_SHARDING = jax.P(("batch", "sequence", "tensor"))

@config
class TransformerLayer(nnx.Module):
	"""Transformer layer with minimal configuration.
	
	Takes only necessary arguments and component creation functions.
	All configuration is passed through to component constructors.
	"""

	def __init__(self, 
		model_d: int,
		attn: Callable = SoftmaxAttention,
		mlp: Callable = GMLP,
		norm: Callable = RMSNorm,
		*,
		rngs: rng.Rngs,
		mesh: Optional[jax.sharding.Mesh] = None,
	):
		super().__init__()
		
		# Initialize components directly - they should be pre-configured with override
		self.attn = attn(model_d=model_d, rngs=rngs, mesh=mesh)
		self.mlp = mlp(model_d=model_d, rngs=rngs, mesh=mesh)
		self.attn_norm = norm(model_d=model_d, rngs=rngs, mesh=mesh)
		self.mlp_norm = norm(model_d=model_d, rngs=rngs, mesh=mesh)
		
		# Store model dimension
		self.model_d = model_d

	def __call__(self, x, mesh: Optional[jax.sharding.Mesh] = None, **kwargs):
		"""
		Forward pass through the transformer layer.

		Args:
			x: Input tensor of shape (batch, sequence, model_d)
			**kwargs: Additional arguments passed to attention (e.g., rope, sequence_descriptor)

		Returns:
			Output tensor of same shape as input
		"""
		# Attention block with residual connection
		# Use TransformerEngine workaround for attention normalization
		x += self.attn(self.attn_norm(x), **kwargs)
		x += self.mlp(self.mlp_norm(x))
		if mesh is not None:
			x = jax.lax.with_sharding_constraint(x, jax.NamedSharding(mesh, TENSOR_SHARDING))
		return x
