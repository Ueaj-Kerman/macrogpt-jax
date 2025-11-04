"""
Simple test to see if parameterization approach works at all.
"""

import jax
import jax.numpy as jnp
from functools import partial
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ueaj.model.parallel_scan import parallel_scan as jacobi_scan

print("Starting benchmark...")

# Small test case
hidden_size = 128
seq_len = 256
input_size = 64

print(f"Config: hidden={hidden_size}, seq={seq_len}, input={input_size}")

key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 5)

print("Creating parameters...")
params = {
    'W_hh': jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1,
    'W_xh': jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1,
    'b_h': jax.random.normal(subkeys[2], (hidden_size,)) * 0.01,
}

print("Defining cell function...")
def elman_cell_fn(params, h, x):
    h_next = jnp.tanh(params['W_hh'] @ h + params['W_xh'] @ x + params['b_h'])
    return h_next, h_next

# Use partial to bind params
elman_cell = partial(elman_cell_fn, params)

print("Creating inputs...")
inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
h0 = jnp.zeros(hidden_size)

print("Creating JIT function...")
jacobi_fn = jax.jit(lambda: jacobi_scan(
    elman_cell, h0, inputs, num_iterations=5
))

print("First call (compilation)...")
start = time.time()
result = jacobi_fn()
jax.block_until_ready(result)
compile_time = time.time() - start
print(f"✓ Compiled in {compile_time:.2f}s")

print("Second call (should be fast)...")
start = time.time()
result = jacobi_fn()
jax.block_until_ready(result)
run_time = time.time() - start
print(f"✓ Runtime: {run_time*1000:.2f}ms")

print("\nSUCCESS! Parameterization approach works.")
