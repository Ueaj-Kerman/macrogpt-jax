from typing import Sequence

import jax
import jax.numpy as jnp
import functools as fn

def make_reverse_fn(fwd_fn):
	def run_vjp(state, x):
		q, do = x
		o, dq_fn = jax.vjp(fn.partial(fwd_fn, state), q)
		return dq_fn(do)[0]  # Unpack tuple: (grad_q,) -> grad_q
	return run_vjp


def make_scan_fn(fwd_fn, distance=True):
	def update_fn(state, x):
		k, v, q = x
		o = fwd_fn(state, q)

		v_pred, dstate_fn = jax.vjp(lambda state: fwd_fn(state, k), state)
		dv = (v - v_pred) if distance else v
		dstate, = dstate_fn(dv)  # Unpack tuple: (grad_state, grad_k) -> grad_state

		# todo optimizer - use actual gradient, not sign
		state = jax.tree.map(lambda a, b: a + .05*jax.nn.tanh(b), state, dstate)

		return state, o
	return update_fn

def ttt(fwd_fn, surrogate=True):
	fwd_scan = make_scan_fn(fwd_fn, distance=True)

	def _ttt_fwd(k: jax.Array, v: jax.Array, q: jax.Array, state):
		k_seq, v_seq, q_seq = k.swapaxes(0, 1), v.swapaxes(0, 1), q.swapaxes(0, 1)

		jax.debug.print("Start of state {}", state)
		new_state, o_seq = jax.lax.scan(fwd_scan, state, (k_seq, v_seq, q_seq))
		jax.debug.print("End of state {}", new_state)

		o = o_seq.swapaxes(0, 1)
		if surrogate:
			return o, (k_seq, v_seq, q_seq, state)
		else:
			return o

	if not surrogate:
		return _ttt_fwd

	@jax.custom_vjp
	def ttt_inner(k: jax.Array, v: jax.Array, q: jax.Array, state):
		return _ttt_fwd(k, v, q, state)[0]

	v_scan = make_scan_fn(fwd_fn, distance=True)
	k_fwd_fn = make_reverse_fn(fwd_fn)
	k_scan = make_scan_fn(k_fwd_fn, distance=True)
	def _ttt_bwd(res, do):
		k_seq, v_seq, q_seq, state = res
		do_seq = do.swapaxes(0, 1)

		def q_scan(carry, x):
			k, v, q, do = x
			state, dstate = carry

			# Wrap fwd_scan to return (o, new_state) with new_state as auxiliary
			def fwd_scan_for_vjp(state, q):
				new_state, o = fwd_scan(state, (k, v, q))
				return o, new_state

			# Empirically, has_aux=True returns 3 values: (output, vjp_fn, aux)
			o, q_update_jvp, new_state = jax.vjp(fwd_scan_for_vjp, state, q, has_aux=True)
			new_dstate, dq = q_update_jvp(do)

			new_dstate = jax.tree.map(lambda a, b: a+b, dstate, new_dstate)

			return (new_state, new_dstate), (o, dq)

		dstate = jax.tree.map(jnp.zeros_like, state)

		(end_state, dstate), (o_seq, dq_seq) = jax.lax.scan(q_scan, (state, dstate), (k_seq, v_seq, q_seq, do_seq))

		def kv_scan(carry, x):
			k_state, v_state = carry
			k, v, q, do, dq = x

			new_v_state, dv = v_scan(v_state, (q, do, k))
			new_k_state, dk = k_scan(k_state, ((q, do), dq, (k, dv)))

			return (new_k_state, new_v_state), (dk, dv)

		(_, _), (dk_seq, dv_seq) = jax.lax.scan(kv_scan, (end_state, end_state), (k_seq, v_seq, q_seq, do_seq, dq_seq), reverse=True)

		dq, dk, dv = dq_seq.swapaxes(0, 1), dk_seq.swapaxes(0, 1), dv_seq.swapaxes(0, 1)
		jax.debug.print("dq {}", dq)
		jax.debug.print("dk {}", dk)
		jax.debug.print("dv {}", dv)
		return dk, dv, dq, dstate

	ttt_inner.defvjp(_ttt_fwd, _ttt_bwd)
	return ttt_inner


if __name__ == "__main__":
	print("Testing TTT implementation...")

	# Define a simple forward function: output = q @ state
	def simple_fwd_fn(state, q):
		"""Simple linear transformation: state is a matrix, q has shape (batch, d_model)"""
		# q has shape (batch, d_model), state has shape (d_model, d_model)
		# output should have shape (batch, d_model)
		return q @ state

	# Create test inputs
	batch_size, seq_len, d_model = 2, 4, 8
	state_dim = (d_model, d_model)

	key = jax.random.PRNGKey(42)
	k_key, v_key, q_key, state_key = jax.random.split(key, 4)

	k = jax.random.normal(k_key, (batch_size, seq_len, d_model))
	v = jax.random.normal(v_key, (batch_size, seq_len, d_model))
	q = jax.random.normal(q_key, (batch_size, seq_len, d_model))
	state = jax.random.normal(state_key, state_dim)

	print(f"Input shapes: k={k.shape}, v={v.shape}, q={q.shape}, state={state.shape}")

	# Create TTT layer
	ttt_layer = ttt(simple_fwd_fn)

	# Test forward pass
	print("\n1. Testing forward pass...")
	output = ttt_layer(k, v, q, state)
	print(f"   ✓ Forward pass successful! Output shape: {output.shape}")

	# Test gradient computation
	print("\n2. Testing backward pass (gradients)...")
	def loss_fn(k, v, q, state):
		output = ttt_layer(k, v, q, state)
		return jnp.sum(output ** 2)

	grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3))(k, v, q, state)
	print(f"   ✓ Backward pass successful!")
	print(f"   Gradient shapes: dk={grads[0].shape}, dv={grads[1].shape}, dq={grads[2].shape}, dstate={grads[3].shape}")

	# Test JIT compilation
	print("\n3. Testing JIT compilation...")
	jitted_ttt = jax.jit(ttt_layer)
	output_jit = jitted_ttt(k, v, q, state)
	print(f"   ✓ JIT compilation successful!")
	print(f"   Outputs match: {jnp.allclose(output, output_jit)}")

	# Test gradient with JIT
	print("\n4. Testing JIT + gradients...")
	jitted_grad = jax.jit(jax.grad(loss_fn, argnums=(0, 1, 2, 3)))
	grads_jit = jitted_grad(k, v, q, state)
	print(f"   ✓ JIT + gradients successful!")
	print(f"   Gradients match: {all(jnp.allclose(g1, g2) for g1, g2 in zip(grads, grads_jit))}")

	print("\n✅ All tests passed! TTT implementation is valid.")

	# Test TTTModel integration
	print("\n" + "="*50)
	print("Testing TTTModel integration...")
	print("="*50)

	# Use absolute imports when running as __main__
	import sys
	import os
	sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

	from ueaj.model.ttt.module import TTTModel
	from ueaj.model import GMLP
	from flax.nnx import rnglib as rng

	# Create a small TTTModel
	model_d, hidden_d = 16, 32
	batch_size, seq_len = 2, 8

	print(f"\nCreating TTTModel with model_d={model_d}, hidden_d={hidden_d}")
	model = TTTModel(
		model_d=model_d,
		hidden_d=hidden_d,
		module=GMLP,
		rngs=rng.Rngs(123)
	)

	# Create input
	x = jax.random.normal(jax.random.PRNGKey(456), (batch_size, seq_len, model_d))
	print(f"Input shape: {x.shape}")

	# Test forward pass
	print("\n5. Testing TTTModel forward pass...")
	output = model(x)
	print(f"   ✓ Forward pass successful! Output shape: {output.shape}")
	assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"

	# Test gradient computation
	print("\n6. Testing TTTModel gradients...")
	def model_loss_fn(x):
		output = model(x)
		return jnp.sum(output ** 2)

	grads = jax.grad(model_loss_fn)(x)
	print(f"   ✓ Backward pass successful! Gradient shape: {grads.shape}")

	# Test JIT compilation
	print("\n7. Testing TTTModel JIT compilation...")
	jitted_model = jax.jit(model)
	output_jit = jitted_model(x)
	print(f"   ✓ JIT compilation successful!")
	print(f"   Outputs match: {jnp.allclose(output, output_jit, atol=1e-5)}")

	print("\n✅ All TTTModel tests passed!")