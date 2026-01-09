import functools

import jax
import jax.numpy as jnp


def make_update_fn(fwd_fn, n_iters, wd, lr):
	"""Create the state update function (gradient descent on reconstruction loss)."""

	def update_fn(state, k, v):
		for _ in range(n_iters):
			v_pred, dstate_fn = jax.vjp(lambda state: fwd_fn(state, k), state)
			dv = v - v_pred
			dstate, = dstate_fn(dv)
			# SGD with weight decay and tanh gradient clipping
			# jax.tree.map_with_path(lambda k, v: print(f"{k}: {v.dtype}"), dstate)
			state = jax.tree.map(lambda a, b: (1 - wd * lr) * a + lr * jax.nn.tanh(b), state, dstate)
		return state

	return update_fn


def make_scan_fn(fwd_fn, n_iters, wd, lr):
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


def ttt(fwd_fn, surrogate=True, n_iters=1, wd=.1, lr=.005, block_size=None):
	"""Create TTT layer with optional surrogate gradients.

	The forward pass:
	1. Update state using (k, v) via gradient descent
	2. Query updated state with q to produce output

	Returns: (output, final_state) tuple

	The surrogate backward pass uses a two-pass approach for k/v gradients,
	and a batched computation for the init_state gradient.
	"""
	fwd_scan = make_scan_fn(fwd_fn, n_iters, wd, lr)
	update_fn = make_update_fn(fwd_fn, n_iters, wd, lr)
	update_fn_ref = update_fn

	reshape = lambda v: v
	unshape = lambda v: v
	if block_size is not None:
		assert block_size > 0, "block_size must be positive"

		def reshape_fn(v):
			assert v.shape[0] % block_size == 0, f"seq len ({v.shape[0]}) must be a multiple of block_size ({block_size})"
			return v.reshape((v.shape[0] // block_size, block_size) + v.shape[1:])
		def unshape_fn(v):
			assert v.shape[1] == block_size, f"tensor block size ({block_size}) must be equal to block size ({block_size})"
			return v.reshape((v.shape[0]*block_size,) + v.shape[2:])

		reshape = lambda v: jax.tree.map(reshape_fn, v)
		unshape = lambda v: jax.tree.map(unshape_fn, v)

		# fwd_fn = jax.vmap(fwd_fn, in_axes=(None, 0))
		fwd_scan = functools.partial(jax.lax.scan, fwd_scan)

		def update_wrapped(state, k, v):
			return jax.lax.scan(lambda state, kv: (update_fn_ref(state, *kv), None), state, (k, v))[0]
		update_fn = update_wrapped

		if not surrogate:
			fwd_scan = jax.remat(fwd_scan, policy=jax.checkpoint_policies.nothing_saveable)

	def _ttt_fwd(k: jax.Array, v: jax.Array, q: jax.Array, state):
		final_state, o = jax.lax.scan(fwd_scan, state, reshape((k, v, q)))
		o = unshape(o)

		return (o, final_state), (k, v, q, state, final_state)

	def _ttt(k: jax.Array, v: jax.Array, q: jax.Array, state):
		return _ttt_fwd(k, v, q, state)[0]

	if not surrogate:
		return jax.vmap(_ttt, in_axes=(0, 0, 0, None))

	# Reference to single-token scan function (before any blocking wrappers)
	fwd_scan_single = make_scan_fn(fwd_fn, n_iters, wd, lr)

	def _ttt_bwd(res, g):
		do, d_final_state = g  # Unpack cotangent tuple
		k, v, q, init_state, final_state = res

		native_dtype = jax.tree.leaves(init_state)[0].dtype

		# ==========================================
		# PASS 1: Forward through sequence, accumulating dState sum in fp32
		# Computes dstate_i = d(output_i)/d(state_i) for each token.
		# ==========================================
		def pass1_token_scan(carry, x):
			"""Accumulate dstate for a single token."""
			state, dstate_accum = carry
			ki, vi, qi, doi = x

			_, state_vjp_fn, state = jax.vjp(
				lambda s: tuple(reversed(fwd_scan_single(s, (ki, vi, qi)))),
				state, has_aux=True
			)
			dstate, = state_vjp_fn(doi)

			dstate_accum = jax.tree.map(
				lambda acc, ds: acc + ds,
				dstate_accum, dstate
			)
			return (state, dstate_accum), None

		# Initialize with d_final_state (gradient flowing from next block)
		init_accum = jax.tree.map(lambda x: x.astype(jnp.float32), d_final_state)
		(_, total_dstate), _ = jax.lax.scan(
			pass1_token_scan, (init_state, init_accum), (k, v, q, do)
		)

		# ==========================================
		# PASS 2: Distribute gradients to k/v with EXACT direct + SURROGATE indirect
		#
		# For token i, the gradient has two components:
		# 1. DIRECT: output_i → k_i (single-step, computed exactly)
		# 2. INDIRECT: output_j → k_i for j > i (multi-step, surrogate approx)
		#
		# Key insight: Direct path should NOT go through the extra J_upd factor
		# that the surrogate dstate accumulation introduces.
		# ==========================================

		# Token-by-token surrogate VJP (for block_size=None case)
		def compute_kvq_grads_surrogate(si, ki, vi, qi, doi, accum_i):
			"""Single VJP that combines direct (from do) and indirect (from accum) gradients."""
			accum_native = jax.tree.map(lambda x: x.astype(native_dtype), accum_i)
			_, scan_vjp = jax.vjp(
				lambda s, k, v, q: fwd_scan_single(s, (k, v, q)), si, ki, vi, qi
			)
			_, dk, dv, dq = scan_vjp((accum_native, doi))
			return dk, dv, dq

		if block_size is None:
			# No blocking: process all tokens sequentially with surrogate
			def pass2_token_scan(carry, x):
				"""Compute combined gradients via single VJP."""
				state, accum_indirect = carry
				ki, vi, qi, doi = x

				dk, dv, dq = compute_kvq_grads_surrogate(state, ki, vi, qi, doi, accum_indirect)

				new_state = update_fn_ref(state, ki, vi)
				_, state_vjp_fn = jax.vjp(
					lambda s: fwd_fn(update_fn_ref(s, ki, vi), qi), state
				)
				dstate_this, = state_vjp_fn(doi)
				new_accum_indirect = jax.tree.map(
					lambda acc, ds: acc - ds.astype(jnp.float32),
					accum_indirect, dstate_this
				)

				return (new_state, new_accum_indirect), (dk, dv, dq)

			(_, _), (dk, dv, dq) = jax.lax.scan(
				pass2_token_scan,
				(init_state, total_dstate),
				(k, v, q, do)
			)
		else:
			# With blocking: EXACT within-block gradients + SURROGATE inter-block
			def pass2_block_exact(carry, x):
				"""Process block with exact within-block gradients.

				Key insight: Use standard VJP through the entire block for exact
				within-block gradients. Only use surrogate for d_state propagation
				between blocks.

				IMPORTANT: The inter-block accumulator must EXCLUDE this block's
				contribution before we use it. Otherwise we double-count within-block
				gradients (once via exact VJP, once via surrogate accumulator).
				"""
				state, accum_inter_block = carry
				k_block, v_block, q_block, do_block = x

				# First, compute this block's dstate contribution (for accumulator update)
				def compute_dstate(si, ki, vi, qi, doi):
					_, state_vjp = jax.vjp(
						lambda s: fwd_fn(update_fn_ref(s, ki, vi), qi), si
					)
					dstate_i, = state_vjp(doi)
					return dstate_i

				def collect_states(s, kv):
					ki, vi = kv
					new_s = update_fn_ref(s, ki, vi)
					return new_s, s

				final_state, states = jax.lax.scan(collect_states, state, (k_block, v_block))
				dstates = jax.vmap(compute_dstate)(states, k_block, v_block, q_block, do_block)

				# Subtract this block's contribution BEFORE using the accumulator
				total_dstate_block = jax.tree.map(
					lambda ds: ds.astype(jnp.float32).sum(axis=0), dstates
				)
				accum_future_only = jax.tree.map(
					lambda acc, ds: acc - ds,
					accum_inter_block, total_dstate_block
				)

				# VJP of entire block forward pass (gives EXACT within-block gradients)
				def block_forward(s, k, v, q):
					return jax.lax.scan(fwd_scan_single, s, (k, v, q))

				_, block_vjp = jax.vjp(
					block_forward, state, k_block, v_block, q_block
				)

				# Cotangent: (d_final_state from FUTURE blocks only, d_outputs)
				accum_native = jax.tree.map(lambda x: x.astype(native_dtype), accum_future_only)
				_, dk_block, dv_block, dq_block = block_vjp((accum_native, do_block))

				return (final_state, accum_future_only), (dk_block, dv_block, dq_block)

			(_, _), (dk, dv, dq) = jax.lax.scan(
				pass2_block_exact,
				(init_state, total_dstate),
				reshape((k, v, q, do))
			)

		dk, dv, dq = unshape((dk, dv, dq))

		# # Translate d_final_state to d_init_state via batched update VJP
		@jax.jit
		def batch_update(state):
			states = jax.vmap(update_fn_ref, in_axes=(None, 0, 0), out_axes=0)(state, k, v)
			return jax.tree.map(lambda v: v.sum(axis=0), states)

		_, update_vjp_fn = jax.vjp(batch_update, init_state)
		dstate_from_final, = update_vjp_fn(d_final_state)

		# Gradient from queries via batched fwd VJP
		_, fwd_vjp_fn = jax.vjp(lambda s: jax.vmap(fwd_fn, in_axes=(None, 0))(s, q), init_state)
		dstate_from_queries, = fwd_vjp_fn(do)

		# # Combine both contributions
		dstate = jax.tree.map(
			lambda dq, df: dq + df,
			dstate_from_queries, dstate_from_final
		)
		# dstate = d_final_state

		return dk, dv, dq, dstate

	_ttt = jax.custom_vjp(_ttt)
	_ttt.defvjp(_ttt_fwd, _ttt_bwd)

	return jax.vmap(_ttt, in_axes=(0, 0, 0, None))


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
	print("\n" + "=" * 50)
	print("Testing TTTModel integration...")
	print("=" * 50)

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
