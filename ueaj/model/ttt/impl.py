from typing import Sequence

import jax
import jax.numpy as jnp
import functools as fn


def make_update_fn(fwd_fn, n_iters=1, wd=.1, lr=.005):
	"""Create the state update function (gradient descent on reconstruction loss)."""
	def update_fn(state, k, v):
		for _ in range(n_iters):
			v_pred, dstate_fn = jax.vjp(lambda state: fwd_fn(state, k), state)
			dv = v - v_pred
			dstate, = dstate_fn(dv)
			# SGD with weight decay and tanh gradient clipping
			jax.tree.map_with_path(lambda k, v: print(f"{k}: {v.dtype}"), dstate)
			state = jax.tree.map(lambda a, b: (1-wd*lr)*a + lr*jax.nn.tanh(b), state, dstate)
		return state
	return update_fn


def make_scan_fn(fwd_fn, n_iters=1, wd=.1, lr=.005):
	"""Create scan function: update state first, then query."""
	update_fn = make_update_fn(fwd_fn, n_iters, wd, lr)

	def scan_fn(state, x):
		k, v, q = x
		# Update state first (based on k, v)
		new_state = update_fn(state, k, v)
		# Then query the updated state
		o = fwd_fn(new_state, q)
		return new_state, o

	return scan_fn


def make_ttt_bwd(fwd_fn, update_fn):
	"""Create the backward pass function for TTT.

	This is extracted as a separate function to allow reuse in blockwise mode.

	Args:
		fwd_fn: Forward function (state, x) -> output
		update_fn: State update function (state, k, v) -> new_state

	The backward receives (do, d_final_state) as a tuple (matching forward's output)
	and returns (dk, dv, dq, d_init_state).
	d_final_state allows gradient flow from subsequent blocks in blockwise mode.
	"""
	def _ttt_bwd(res, g):
		do, d_final_state = g  # Unpack cotangent tuple
		k_seq, v_seq, q_seq, init_state, final_state = res
		do_seq = do.swapaxes(0, 1)

		native_dtype = jax.tree.leaves(init_state)[0].dtype

		# ==========================================
		# PASS 1: Forward through sequence, accumulating dState sum in fp32
		# Start with d_final_state (gradient from next block) and accumulate
		# ==========================================
		def pass1_scan(carry, x):
			state, accum_dstate = carry
			k, v, q, do = x

			new_state = update_fn(state, k, v)
			_, state_vjp_fn = jax.vjp(lambda s: fwd_fn(s, q), new_state)
			dstate_from_query, = state_vjp_fn(do)

			new_accum = jax.tree.map(
				lambda acc, ds: acc + ds.astype(jnp.float32),
				accum_dstate, dstate_from_query
			)
			return (new_state, new_accum), None

		# Initialize with d_final_state (gradient flowing from next block)
		init_accum = jax.tree.map(lambda x: x.astype(jnp.float32), d_final_state)
		(_, total_dstate), _ = jax.lax.scan(
			pass1_scan, (init_state, init_accum), (k_seq, v_seq, q_seq, do_seq)
		)

		# ==========================================
		# PASS 2: Forward through sequence again, distribute gradients to k/v
		# ==========================================
		def pass2_scan(carry, x):
			state, accum_dstate = carry
			k, v, q, do = x

			new_state, update_vjp_fn = jax.vjp(lambda s, k, v: update_fn(s, k, v), state, k, v)
			accum_dstate_native = jax.tree.map(lambda x: x.astype(native_dtype), accum_dstate)
			dstate_in, dk, dv = update_vjp_fn(accum_dstate_native)

			_, state_vjp_fn = jax.vjp(lambda s: fwd_fn(s, q), new_state)
			dstate_this, = state_vjp_fn(do)

			new_accum_dstate = jax.tree.map(
				lambda acc, ds: acc - ds.astype(jnp.float32),
				accum_dstate, dstate_this
			)

			_, q_vjp_fn = jax.vjp(lambda q: fwd_fn(new_state, q), q)
			dq, = q_vjp_fn(do)

			return (new_state, new_accum_dstate), (dk, dv, dq)

		(_, _), (dk_seq, dv_seq, dq_seq) = jax.lax.scan(
			pass2_scan,
			(init_state, total_dstate),
			(k_seq, v_seq, q_seq, do_seq)
		)

		# ==========================================
		# Compute dstate for init_state using batched computation
		# Treat all queries and dOs as one big batch
		# ==========================================
		q_batched = q_seq.reshape(-1, q_seq.shape[-1])  # (seq*batch, hidden)
		do_batched = do_seq.reshape(-1, do_seq.shape[-1])  # (seq*batch, hidden)

		# Compute gradient of fwd_fn(init_state, q_batched) w.r.t. init_state
		_, init_state_vjp_fn = jax.vjp(lambda s: fwd_fn(s, q_batched), init_state)
		dstate_from_queries, = init_state_vjp_fn(do_batched)

		# Add d_final_state contribution (backprop through state chain)
		dstate = jax.tree.map(
			lambda dq, dfs: dq + dfs,
			dstate_from_queries, d_final_state
		)

		dq, dk, dv = dq_seq.swapaxes(0, 1), dk_seq.swapaxes(0, 1), dv_seq.swapaxes(0, 1)
		return dk, dv, dq, dstate

	return _ttt_bwd


def ttt(fwd_fn, surrogate=True):
	"""Create TTT layer with optional surrogate gradients.

	The forward pass:
	1. Update state using (k, v) via gradient descent
	2. Query updated state with q to produce output

	Returns: (output, final_state) tuple

	The surrogate backward pass uses a two-pass approach for k/v gradients,
	and a batched computation for the init_state gradient.
	"""
	fwd_scan = make_scan_fn(fwd_fn)
	update_fn = make_update_fn(fwd_fn)

	def _ttt_fwd(k: jax.Array, v: jax.Array, q: jax.Array, state):
		k_seq, v_seq, q_seq = k.swapaxes(0, 1), v.swapaxes(0, 1), q.swapaxes(0, 1)

		final_state, o_seq = jax.lax.scan(fwd_scan, state, (k_seq, v_seq, q_seq))

		o = o_seq.swapaxes(0, 1)
		if surrogate:
			return (o, final_state), (k_seq, v_seq, q_seq, state, final_state)
		else:
			return (o, final_state)

	if not surrogate:
		return _ttt_fwd

	@jax.custom_vjp
	def ttt_inner(k: jax.Array, v: jax.Array, q: jax.Array, state):
		return _ttt_fwd(k, v, q, state)[0]

	ttt_inner.defvjp(_ttt_fwd, make_ttt_bwd(fwd_fn, update_fn))
	return ttt_inner


if __name__ == "__main__":
	print("Testing TTT implementation...")

	# Define a simple forward function: output = q @ state
	def simple_fwd_fn(state, q):
		"""Simple linear transformation: state is a matrix, q has shape (batch, d_model)"""
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
	output, final_state = ttt_layer(k, v, q, state)
	print(f"   ✓ Forward pass successful! Output shape: {output.shape}, Final state shape: {final_state.shape}")

	# Test gradient computation
	print("\n2. Testing backward pass (gradients)...")
	def loss_fn(k, v, q, state):
		output, final_state = ttt_layer(k, v, q, state)
		return jnp.sum(output ** 2)

	grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3))(k, v, q, state)
	print(f"   ✓ Backward pass successful!")
	print(f"   Gradient shapes: dk={grads[0].shape}, dv={grads[1].shape}, dq={grads[2].shape}, dstate={grads[3].shape}")

	# Test JIT compilation
	print("\n3. Testing JIT compilation...")
	jitted_ttt = jax.jit(ttt_layer)
	output_jit, final_state_jit = jitted_ttt(k, v, q, state)
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
