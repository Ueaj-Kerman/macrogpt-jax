from typing import Optional, Tuple

from flax import nnx
import jax
from ueaj import model as m
from ueaj.utils.gradutils import nnx_vjp
from ueaj.utils.tensorutil import slice
from ueaj.opt import chunked_softmax_cross_entropy
from jax import numpy as jnp
from ueaj.model.configs import UEAJ_150M

def learn_local(prev_vjp, prev_act, curr_act, forward_model, **kwargs):
	"""
	Compute self-prediction loss for local attention.
	"""
	next_act, next_vjp = nnx_vjp(lambda model: model(curr_act, **kwargs), forward_model)
	act_grad = (next_act - curr_act) # ask prev layer to learn from current layer
	if prev_act is not None:
		act_grad += .75*(prev_act - curr_act) # ask prev layer to unlearn what it already learnt
	act_grad = act_grad / jnp.sqrt(jnp.mean(jnp.square(act_grad))+1e-3)
	(dprev, ) = prev_vjp(-act_grad) # negative gradient
	dprev = jax.tree.map(lambda x: x / (act_grad.shape[0] * act_grad.shape[1]), dprev)
	return dprev, next_act, next_vjp

@nnx.jit
def learn_associative(
	model,
	inputs: jax.Array,
	document_ids: Optional[jax.Array] = None,
	pad_token_id: Optional[int] = None,
	**kwargs
) -> Tuple[nnx.State, Tuple[jax.Array, jax.Array]]:
	kwargs = model.default_kwargs(*inputs.shape, **kwargs)

	# Embed tokens
	act0, embed_vjp = nnx_vjp(lambda model: model(inputs), model.embed_tokens)

	dembed, act, prev_vjp = learn_local(embed_vjp, None, act0, slice(model.layers)[0], **kwargs)

	dlayers = []
	for i in range(1, model.config.num_layers):
		layer = slice(model.layers)[i]
		dlayer, next_act, curr_vjp = learn_local(prev_vjp, act0, act, layer, **kwargs)
		prev_vjp = curr_vjp
		act0 = act
		act = next_act

		dlayers.append(dlayer)
		# if i % 3 == 0:
		# 	jax.debug.print("layer l2 norm {} {}", i, jnp.square(dlayer.mlp.down_proj.w_1.value).mean(dtype=jnp.float32))
		# 	jax.debug.print("layer 2 act norm {} {}", i, jnp.square(act).mean(dtype=jnp.float32))

	@nnx.grad(has_aux=True, argnums=(0, 1, 2))
	def head(act, norm, lm_head):
		token_loss, loss_mask = chunked_softmax_cross_entropy(
			inputs,
			act,
			lambda x: lm_head(norm(x)),
			document_ids=document_ids,
			pad_token_id=pad_token_id,
			return_loss_mask=True
		)
		count = loss_mask.sum(dtype=jnp.float32)

		loss_val = token_loss.sum() / count

		mean_loss = token_loss.sum() / count
		std_loss = jnp.sqrt(jnp.square(token_loss - mean_loss).sum() / count)
		return loss_val, (mean_loss, std_loss)

	(dact, dnorm, dlm_head), (mean_loss, std_loss) = head(act, model.norm, model.lm_head)

	# embed skip connection for faster convergence
	dembed = jax.tree.map(lambda x, y: x + y, dembed, embed_vjp(dact)[0])

	# dact += (act0 - act)
	dlayers.append(prev_vjp(dact)[0])

	dlayers = jax.tree.map(lambda *x: jnp.stack(x, axis=0), *dlayers)

	dmodel = nnx.State({})
	dmodel.embed_tokens = dembed
	dmodel.layers = dlayers
	dmodel.norm = dnorm
	dmodel.lm_head = dlm_head

	return dmodel, (mean_loss, std_loss)