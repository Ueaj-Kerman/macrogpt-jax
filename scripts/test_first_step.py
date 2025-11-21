#!/usr/bin/env python3
"""Test first training step to debug NaN."""

import os
os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
import transformers

from ueaj.model import configs

print("Creating model...")
model = configs.LLAMA3_2B(rngs=rng.Rngs(0))

print(f"Model config:")
print(f"  tie_word_embeddings: {model.tie_word_embeddings}")
print(f"  vocab_size: {model.vocab_size}")
print(f"  model_d: {model.model_d}")

# Create realistic test input (all 1s is probably bad!)
batch_size, seq_len = 1, 512
print(f"\nCreating test data...")

# Use a variety of token IDs like real data would have
key = jax.random.key(42)
test_tokens = jax.random.randint(key, (batch_size, seq_len), 0, min(1000, model.vocab_size))
print(f"  Tokens shape: {test_tokens.shape}")
print(f"  Tokens range: [{jnp.min(test_tokens)}, {jnp.max(test_tokens)}]")
print(f"  Tokens dtype: {test_tokens.dtype}")

print(f"\nRunning forward pass (getting activations before final projection)...")
# Call model to get final hidden states
logits = model(test_tokens)
print(f"  Full forward pass produces logits directly")
print(f"  Logits shape: {logits.shape}")

# For testing, let's just use the logits directly
activations = None  # We'll skip this step
print(f"  Activations shape: {activations.shape}")
print(f"  Activations dtype: {activations.dtype}")
print(f"  Activations |mean|: {float(jnp.mean(jnp.abs(activations))):.6f}")
print(f"  Activations |max|: {float(jnp.max(jnp.abs(activations))):.6f}")
print(f"  Has NaN: {bool(jnp.any(jnp.isnan(activations)))}")
print(f"  Has Inf: {bool(jnp.any(jnp.isinf(activations)))}")

if jnp.any(jnp.isnan(activations)):
    print("❌ Activations have NaN!")
    exit(1)

print(f"\nGetting logits...")
logits = model.get_logits(activations)
print(f"  Logits shape: {logits.shape}")
print(f"  Logits dtype: {logits.dtype}")
print(f"  Logits |mean|: {float(jnp.mean(jnp.abs(logits))):.6f}")
print(f"  Logits |max|: {float(jnp.max(jnp.abs(logits))):.6f}")
print(f"  Has NaN: {bool(jnp.any(jnp.isnan(logits)))}")
print(f"  Has Inf: {bool(jnp.any(jnp.isinf(logits)))}")

if jnp.any(jnp.isnan(logits)):
    print("❌ Logits have NaN!")
    exit(1)

print(f"\nComputing loss (simple cross-entropy)...")
labels = test_tokens  # Use same tokens as labels for testing
log_probs = jax.nn.log_softmax(logits, axis=-1)
print(f"  log_probs |mean|: {float(jnp.mean(jnp.abs(log_probs))):.6f}")
print(f"  log_probs |min|: {float(jnp.min(log_probs)):.6f}")
print(f"  log_probs has NaN: {bool(jnp.any(jnp.isnan(log_probs)))}")
print(f"  log_probs has Inf: {bool(jnp.any(jnp.isinf(log_probs)))}")

if jnp.any(jnp.isnan(log_probs)) or jnp.any(jnp.isinf(log_probs)):
    print("❌ log_probs have NaN/Inf!")
    exit(1)

# Compute cross-entropy
one_hot = jax.nn.one_hot(labels, model.vocab_size)
loss_per_token = -jnp.sum(one_hot * log_probs, axis=-1)
loss = jnp.mean(loss_per_token)

print(f"\nLoss: {float(loss):.6f}")
print(f"  Has NaN: {bool(jnp.isnan(loss))}")
print(f"  Has Inf: {bool(jnp.isinf(loss))}")

if jnp.isnan(loss):
    print("❌ Loss is NaN!")
    exit(1)

print(f"\n✅ All checks passed! Loss = {float(loss):.4f}")
