import jax
import jax.numpy as jnp
import optax
from optax import GradientTransformation


def lerp(a, b, alpha):
	return a + alpha * (b - a)


def cast_to(dtype: jnp.dtype | None) -> GradientTransformation:
	return optax.stateless(
		lambda updates, params: jax.tree.map(
			lambda u, p: u.astype(dtype if dtype is not None else p.dtype),
			updates,
			params
		)
	)


def scale_by_muonP(base_lr: float):
	"""Scale lr by muP, assumes 2D+ weight matrices."""
	return optax.stateless(
		lambda updates, params: jax.tree.map(
			lambda u, p: -max(u.shape[-1] / u.shape[-2], 1) * base_lr * u,
			updates,
			params
		)
	)


def project_rms_rows(max_rms: float = 1.25) -> GradientTransformation:
	"""Project matrix rows to RMS=1 when they exceed a threshold.

	For embeddings of shape [vocab_size, embed_dim], ensures each token's
	embedding vector doesn't grow unbounded. When a row's RMS exceeds
	max_rms, it's normalized back to RMS=1.

	Should be chained at the END of an optimizer (after learning rate scaling).
	"""
	def project_one(u, p):
		new_val = p + u
		row_rms = jnp.sqrt(jnp.mean(jnp.square(new_val), axis=-1, keepdims=True))
		scale = jnp.where(row_rms > max_rms, row_rms, 1.0)
		target = new_val / scale
		return target - p

	return optax.stateless(
		lambda updates, params: jax.tree.map(project_one, updates, params)
	)
