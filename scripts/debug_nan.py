#!/usr/bin/env python3
"""Debug NaN loss issue."""

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

print("\n=== Checking Model Weights ===")
state = nnx.state(model)

def check_params(path, param):
    if hasattr(param, 'value'):
        arr = param.value
        has_nan = jnp.any(jnp.isnan(arr))
        has_inf = jnp.any(jnp.isinf(arr))
        mean = jnp.mean(jnp.abs(arr))
        max_val = jnp.max(jnp.abs(arr))

        status = "✓" if not (has_nan or has_inf) else "✗"
        print(f"{status} {path}: shape={arr.shape}, dtype={arr.dtype}, |mean|={float(mean):.6f}, |max|={float(max_val):.6f}, nan={has_nan}, inf={has_inf}")

        if has_nan or has_inf:
            print(f"  ERROR: Found {'NaN' if has_nan else 'Inf'} in {path}!")
            return False
    return True

all_good = True
for path, param in jax.tree_util.tree_leaves_with_path(state):
    path_str = '/'.join(str(k) for k in path)
    if not check_params(path_str, param):
        all_good = False

if not all_good:
    print("\n❌ Model has NaN/Inf values in initialization!")
    exit(1)

print("\n=== Creating Test Input ===")
batch_size, seq_len = 1, 512
test_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
print(f"Input shape: {test_input.shape}, dtype: {test_input.dtype}")
print(f"Input range: [{jnp.min(test_input)}, {jnp.max(test_input)}]")

print("\n=== Running Forward Pass ===")
try:
    logits = model(test_input)
    print(f"✓ Forward pass succeeded")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits dtype: {logits.dtype}")
    print(f"  Logits |mean|: {float(jnp.mean(jnp.abs(logits))):.6f}")
    print(f"  Logits |max|: {float(jnp.max(jnp.abs(logits))):.6f}")
    print(f"  Has NaN: {jnp.any(jnp.isnan(logits))}")
    print(f"  Has Inf: {jnp.any(jnp.isinf(logits))}")

    if jnp.any(jnp.isnan(logits)):
        print("\n❌ Forward pass produces NaN!")
        exit(1)

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== Testing Loss Computation ===")
from ueaj.opt.next_token_loss import chunked_softmax_cross_entropy

# Create dummy labels
labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

try:
    # Check if we need lm_head projection
    if hasattr(model, 'lm_head'):
        print("Using lm_head projection...")
        loss = chunked_softmax_cross_entropy(
            logits,
            labels,
            logit_projection=lambda x: model.lm_head(x)
        )
    else:
        print("No lm_head, using logits directly...")
        # Assuming logits are already projected to vocab size
        loss_per_token = -jnp.sum(
            jax.nn.one_hot(labels, logits.shape[-1]) * jax.nn.log_softmax(logits, axis=-1),
            axis=-1
        )
        loss = jnp.mean(loss_per_token)

    print(f"✓ Loss computation succeeded")
    print(f"  Loss value: {float(loss):.6f}")
    print(f"  Has NaN: {jnp.isnan(loss)}")
    print(f"  Has Inf: {jnp.isinf(loss)}")

    if jnp.isnan(loss):
        print("\n❌ Loss is NaN!")
        exit(1)

except Exception as e:
    print(f"✗ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n✅ All checks passed!")
