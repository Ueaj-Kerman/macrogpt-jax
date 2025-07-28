from ueaj.model.soft_attn import *
from ueaj.model.mlp import *
from ueaj.model.rmsnorm import *
from ueaj.utils.configurator import *


@config
class TransformerLayer(nnx.Module):
	"""Transformer layer with minimal configuration.
	
	Takes only necessary arguments and component creation functions.
	All configuration is passed through to component constructors.
	"""

	def __init__(self, 
		model_d: int,
		rngs: rng.Rngs,
		attn: Callable = SoftmaxAttention,
		mlp: Callable = GMLP,
		norm: Callable = RMSNorm,
	):
		super().__init__()
		
		# Initialize components directly - they should be pre-configured with override
		self.attn = attn(model_d=model_d, rngs=rngs)
		self.mlp = mlp(model_d=model_d, rngs=rngs)
		self.attn_norm = norm(model_d=model_d, rngs=rngs)
		self.mlp_norm = norm(model_d=model_d, rngs=rngs)
		
		# Store model dimension
		self.model_d = model_d

	def __call__(self, x, **kwargs):
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
		return x

if __name__ == "__main__":
	"""Test TransformerLayer.bwd."""
	# Test transformer layer
	model_d = 2048
	tensor_config = ParamConfig("", group=nnx.Param).with_dtype(jnp.bfloat16)# .with_grad_dtype(jnp.float32)


	layer_config = {
		'model_d': model_d,
		'kq_d': 64,
		'v_head_d': 64,
		'kv_heads': 8,
		'kv_q_ratio': 1,
		'rope_theta': 500_000.,
		'mlp_type': 'gated',
		'hidden_d': 1024,
		'param_config': tensor_config,
		'norm_scale': 'centered',
		'norm_scale_dtype': tensor_config.dtype
	}

	kwargs = {
		# Create segment IDs for 4 sequences of length 8192 each
		'query_segment_ids': jnp.zeros((4, 8192), dtype=jnp.int32),
		'kv_segment_ids': jnp.zeros((4, 8192), dtype=jnp.int32),
	}

	@nnx.jit
	@nnx.value_and_grad(argnums=(0, 1, 3))
	def layers(layer1, layer2, x, y):
		x = layer1(x, **kwargs)
		x = layer2(x, **kwargs)
		return jnp.sum((x - y)**2) / 2


	# Simple test configuration
	layer1 = TransformerLayer(model_d=model_d, rngs=rng.Rngs(0))
	layer2 = TransformerLayer(model_d=model_d, rngs=rng.Rngs(0))

	x = jax.random.normal(jax.random.PRNGKey(0), (4, 8192, model_d)).astype(jnp.bfloat16)
	dh = jax.random.normal(jax.random.PRNGKey(0), (4, 8192, model_d)).astype(jnp.bfloat16)
	output, (dlayer1, dlayer2, dx) = layers(layer1, layer2, x, x+dh)

	print(output)
	print("dlayer1 keys:", list(dlayer1.keys()) if hasattr(dlayer1, 'keys') else type(dlayer1))
	print("dlayer2 keys:", list(dlayer2.keys()) if hasattr(dlayer2, 'keys') else type(dlayer2))
	print("dx shape:", dx.shape)