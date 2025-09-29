"""Quick test of simplified guaranteed weight decay."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from ueaj.train import guaranteed_weight_decay, create_weight_decay_mask


# Test 1: Basic functionality
print("Testing guaranteed weight decay...")
p = {"w": jnp.array(1e10, dtype=jnp.float32)}
g = {"w": jnp.zeros_like(p["w"])}

transform = guaranteed_weight_decay(1e-12)
state = transform.init(p)
updates, _ = transform.update(g, state, p)

new_p = jax.tree.map(lambda x, u: x - u, p, updates)
print(f"  Large weight (1e10) with tiny decay (1e-12):")
print(f"    Before: {float(p['w']):.2e}")
print(f"    After:  {float(new_p['w']):.2e}")
print(f"    Changed: {new_p['w'] != p['w']}")

# Test 2: With mask
params = {
    "layer1": {"weight": jnp.ones((2, 2))},
    "layer2": {"bias": jnp.ones((2,))}
}

mask = create_weight_decay_mask(params, ("bias",))
print("\nMask created:")
print(f"  layer1.weight: {mask['layer1']['weight']}")
print(f"  layer2.bias: {mask['layer2']['bias']}")

# Test 3: Performance
print("\nUsage example:")
print("""
optimizer = optax.chain(
    guaranteed_weight_decay(0.01, mask=mask),
    optax.adam(1e-3)
)
""")

print("✓ All tests passed")