#!/usr/bin/env python3
"""Simple test of model forward pass."""

import os
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.model import configs

print("Creating model...")
model = configs.LLAMA3_2B(rngs=rng.Rngs(0))

# Create test input with variety of tokens
batch_size, seq_len = 1, 512
key = jax.random.key(42)
test_tokens = jax.random.randint(key, (batch_size, seq_len), 0, 1000)

print(f"\n=== Forward Pass ===")
logits = model(test_tokens)
print(f"Logits shape: {logits.shape}")
print(f"Logits dtype: {logits.dtype}")
print(f"Logits |mean|: {float(jnp.mean(jnp.abs(logits))):.4f}")
print(f"Has NaN: {bool(jnp.any(jnp.isnan(logits)))}")

if jnp.any(jnp.isnan(logits)):
    print("❌ Logits are NaN!")
    exit(1)

print(f"\n=== Computing Loss ===")
labels = test_tokens
log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
one_hot = jax.nn.one_hot(labels, model.vocab_size)
loss_per_token = -jnp.sum(one_hot * log_probs, axis=-1)
loss = jnp.mean(loss_per_token)

print(f"Loss: {float(loss):.4f}")
print(f"Has NaN: {bool(jnp.isnan(loss))}")

if jnp.isnan(loss):
    print("❌ Loss is NaN!")
    # Debug: check intermediate values
    print("\nDebug info:")
    print(f"  log_probs has NaN: {bool(jnp.any(jnp.isnan(log_probs)))}")
    print(f"  log_probs has -inf: {bool(jnp.any(jnp.isneginf(log_probs)))}")
    print(f"  loss_per_token has NaN: {bool(jnp.any(jnp.isnan(loss_per_token)))}")
    print(f"  Min log_prob: {float(jnp.min(log_probs))}")
    print(f"  Max log_prob: {float(jnp.max(log_probs))}")
    exit(1)

print(f"\n✅ Success! Loss = {float(loss):.4f}")
