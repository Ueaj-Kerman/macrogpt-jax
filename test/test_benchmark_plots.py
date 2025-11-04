"""
Comprehensive benchmark with plots comparing Jacobi vs Newton.

Newton uses smaller scales to avoid compilation issues.
"""

import jax
import jax.numpy as jnp
from functools import partial
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ueaj.model.parallel_scan import parallel_scan as jacobi_scan
from ueaj.model.parallel_scan_newton import parallel_scan_newton as newton_scan
from jax import lax
import matplotlib.pyplot as plt
import numpy as np

# Custom color scheme
HIGHLIGHT = '#A58BFC'  # Purple highlight
COLOR_1 = '#62A2E9'    # Pastel blue
COLOR_2 = '#58BA58'    # Pastel green
COLOR_3 = '#F78281'    # Pastel red/pink

# Set style for transparent background with white text
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['text.color'] = 'white'
plt.rcParams['axes.labelcolor'] = 'white'
plt.rcParams['xtick.color'] = 'white'
plt.rcParams['ytick.color'] = 'white'
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['grid.alpha'] = 0.2

print("Starting comprehensive benchmark...")

def create_elman_rnn(hidden_size, input_size, key):
    """Create Elman RNN parameters and cell function."""
    key, *subkeys = jax.random.split(key, 4)
    params = {
        'W_hh': jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1,
        'W_xh': jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1,
        'b_h': jax.random.normal(subkeys[2], (hidden_size,)) * 0.01,
    }

    def elman_cell_fn(params, h, x):
        h_next = jnp.tanh(params['W_hh'] @ h + params['W_xh'] @ x + params['b_h'])
        return h_next, h_next

    return partial(elman_cell_fn, params)


def benchmark_method(method_fn, cell, h0, inputs, num_iterations, warmup=2, runs=10):
    """Benchmark a method with warmup and multiple runs."""
    # JIT compile
    fn = jax.jit(lambda: method_fn(cell, h0, inputs, num_iterations=num_iterations))

    # Warmup
    for _ in range(warmup):
        result = fn()
        jax.block_until_ready(result)

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.time()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.time() - start)

    return np.mean(times), np.std(times), result


def sequential_reference(cell, h0, inputs):
    """Sequential reference implementation."""
    return lax.scan(cell, h0, inputs)


print("\n" + "="*70)
print("1. CONVERGENCE RATE COMPARISON")
print("="*70)

# Small problem for convergence analysis
hidden = 64
seq_len = 128
input_size = 32

key = jax.random.PRNGKey(42)
cell = create_elman_rnn(hidden, input_size, key)
key, subkey = jax.random.split(key)
inputs = jax.random.normal(subkey, (seq_len, input_size))
h0 = jnp.zeros(hidden)

# Get reference solution
ref_carry, ref_outputs = sequential_reference(cell, h0, inputs)

# Test different iteration counts
jacobi_iterations = [1, 2, 3, 5, 7, 10, 15, 20]
newton_iterations = [1, 2, 3, 4, 5]

jacobi_errors = []
newton_errors = []

print("\nJacobi convergence:")
for num_iter in jacobi_iterations:
    final_carry, outputs = jacobi_scan(cell, h0, inputs, num_iterations=num_iter)
    error = jnp.max(jnp.abs(outputs - ref_outputs))
    jacobi_errors.append(float(error))
    print(f"  {num_iter:2d} iterations: error = {error:.2e}")

print("\nNewton convergence:")
for num_iter in newton_iterations:
    final_carry, outputs = newton_scan(cell, h0, inputs, num_iterations=num_iter)
    error = jnp.max(jnp.abs(outputs - ref_outputs))
    newton_errors.append(float(error))
    print(f"  {num_iter:2d} iterations: error = {error:.2e}")

# Plot convergence
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.semilogy(jacobi_iterations, jacobi_errors, 'o-', color=HIGHLIGHT, label='Jacobi (Linear)',
            linewidth=2.5, markersize=8)
ax.semilogy(newton_iterations, newton_errors, 's-', color=COLOR_1, label='Newton (Quadratic)',
            linewidth=2.5, markersize=8)
ax.axhline(1e-3, color=COLOR_3, linestyle='--', alpha=0.6, linewidth=2, label='ML-sufficient (1e-3)')
ax.set_xlabel('Number of Iterations', fontsize=12)
ax.set_ylabel('Max Error', fontsize=12)
ax.set_title('Convergence Rate: Jacobi vs Newton', fontsize=14, fontweight='bold', color='white')
ax.legend(fontsize=11, facecolor='none', edgecolor='white')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('docs/convergence_rate.png', dpi=150, transparent=True)
print("\n✓ Saved: docs/convergence_rate.png")


print("\n" + "="*70)
print("2. SCALING WITH HIDDEN DIMENSION")
print("="*70)

seq_len = 256
hidden_dims = [32, 64, 128, 256, 512]  # Newton can handle up to 512

jacobi_times_h = []
newton_times_h = []

print(f"\nSequence length: {seq_len}")
for hidden in hidden_dims:
    print(f"\nHidden dim: {hidden}")

    key = jax.random.PRNGKey(42)
    cell = create_elman_rnn(hidden, input_size, key)
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (seq_len, input_size))
    h0 = jnp.zeros(hidden)

    # Jacobi (15 iterations)
    print("  Jacobi (15 iter)...", end='', flush=True)
    mean_time, _, _ = benchmark_method(jacobi_scan, cell, h0, inputs, num_iterations=15, runs=20)
    jacobi_times_h.append(mean_time * 1000)  # Convert to ms
    print(f" {mean_time*1000:.2f} ms")

    # Newton (3 iterations) - only for smaller sizes
    if hidden <= 256:
        print("  Newton (3 iter)...", end='', flush=True)
        mean_time, _, _ = benchmark_method(newton_scan, cell, h0, inputs, num_iterations=3, runs=20)
        newton_times_h.append(mean_time * 1000)
        print(f" {mean_time*1000:.2f} ms")
    else:
        newton_times_h.append(None)
        print("  Newton: skipped (compilation too slow)")

# Plot hidden dimension scaling
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(hidden_dims, jacobi_times_h, 'o-', color=HIGHLIGHT, label='Jacobi (15 iter)',
        linewidth=2.5, markersize=8)
newton_hidden = [h for h, t in zip(hidden_dims, newton_times_h) if t is not None]
newton_valid = [t for t in newton_times_h if t is not None]
ax.plot(newton_hidden, newton_valid, 's-', color=COLOR_1, label='Newton (3 iter)',
        linewidth=2.5, markersize=8)
ax.set_xlabel('Hidden Dimension', fontsize=12)
ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title(f'Scaling with Hidden Dimension (seq_len={seq_len})', fontsize=14, fontweight='bold', color='white')
ax.legend(fontsize=11, facecolor='none', edgecolor='white')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('docs/scaling_hidden_dim.png', dpi=150, transparent=True)
print("\n✓ Saved: docs/scaling_hidden_dim.png")


print("\n" + "="*70)
print("3. SCALING WITH SEQUENCE LENGTH")
print("="*70)

hidden = 128
seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]

jacobi_times_s = []
newton_times_s = []

print(f"\nHidden dim: {hidden}")
for seq_len in seq_lengths:
    print(f"\nSeq length: {seq_len}")

    key = jax.random.PRNGKey(42)
    cell = create_elman_rnn(hidden, input_size, key)
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, (seq_len, input_size))
    h0 = jnp.zeros(hidden)

    # Jacobi (15 iterations)
    print("  Jacobi (15 iter)...", end='', flush=True)
    mean_time, _, _ = benchmark_method(jacobi_scan, cell, h0, inputs, num_iterations=15, runs=20)
    jacobi_times_s.append(mean_time * 1000)
    print(f" {mean_time*1000:.2f} ms")

    # Newton (3 iterations) - only for shorter sequences
    if seq_len <= 512:
        print("  Newton (3 iter)...", end='', flush=True)
        mean_time, _, _ = benchmark_method(newton_scan, cell, h0, inputs, num_iterations=3, runs=20)
        newton_times_s.append(mean_time * 1000)
        print(f" {mean_time*1000:.2f} ms")
    else:
        newton_times_s.append(None)
        print("  Newton: skipped (compilation too slow)")

# Plot sequence length scaling
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(seq_lengths, jacobi_times_s, 'o-', color=HIGHLIGHT, label='Jacobi (15 iter)',
        linewidth=2.5, markersize=8)
newton_seqs = [s for s, t in zip(seq_lengths, newton_times_s) if t is not None]
newton_valid_s = [t for t in newton_times_s if t is not None]
ax.plot(newton_seqs, newton_valid_s, 's-', color=COLOR_1, label='Newton (3 iter)',
        linewidth=2.5, markersize=8)
ax.set_xlabel('Sequence Length', fontsize=12)
ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title(f'Scaling with Sequence Length (hidden={hidden})', fontsize=14, fontweight='bold', color='white')
ax.legend(fontsize=11, facecolor='none', edgecolor='white')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('docs/scaling_seq_length.png', dpi=150, transparent=True)
print("\n✓ Saved: docs/scaling_seq_length.png")


print("\n" + "="*70)
print("4. MEMORY USAGE COMPARISON")
print("="*70)

seq_len = 1024
hidden_dims_mem = [32, 64, 128, 256, 512, 1024, 2048]

jacobi_mem = []
newton_mem = []

for hidden in hidden_dims_mem:
    # Memory in MB
    j_mem = seq_len * hidden * 4 / 1e6  # float32
    n_mem = seq_len * hidden * hidden * 4 / 1e6  # Jacobian storage
    jacobi_mem.append(j_mem)
    newton_mem.append(n_mem)
    print(f"Hidden {hidden:4d}: Jacobi {j_mem:6.1f} MB, Newton {n_mem:8.1f} MB, Ratio: {n_mem/j_mem:.0f}x")

# Plot memory comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Linear scale
ax1.plot(hidden_dims_mem, jacobi_mem, 'o-', color=HIGHLIGHT, label='Jacobi O(T×d)',
         linewidth=2.5, markersize=8)
ax1.plot(hidden_dims_mem, newton_mem, 's-', color=COLOR_1, label='Newton O(T×d²)',
         linewidth=2.5, markersize=8)
ax1.set_xlabel('Hidden Dimension', fontsize=12)
ax1.set_ylabel('Memory (MB)', fontsize=12)
ax1.set_title(f'Memory Usage (seq_len={seq_len})', fontsize=14, fontweight='bold', color='white')
ax1.legend(fontsize=11, facecolor='none', edgecolor='white')
ax1.grid(True, alpha=0.2)

# Log scale
ax2.semilogy(hidden_dims_mem, jacobi_mem, 'o-', color=HIGHLIGHT, label='Jacobi O(T×d)',
             linewidth=2.5, markersize=8)
ax2.semilogy(hidden_dims_mem, newton_mem, 's-', color=COLOR_1, label='Newton O(T×d²)',
             linewidth=2.5, markersize=8)
ax2.set_xlabel('Hidden Dimension', fontsize=12)
ax2.set_ylabel('Memory (MB, log scale)', fontsize=12)
ax2.set_title(f'Memory Usage - Log Scale', fontsize=14, fontweight='bold', color='white')
ax2.legend(fontsize=11, facecolor='none', edgecolor='white')
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('docs/memory_comparison.png', dpi=150, transparent=True)
print("\n✓ Saved: docs/memory_comparison.png")


print("\n" + "="*70)
print("5. TIME PER ITERATION")
print("="*70)

# Compare time per iteration at small scale
hidden = 128
seq_len = 256

key = jax.random.PRNGKey(42)
cell = create_elman_rnn(hidden, input_size, key)
key, subkey = jax.random.split(key)
inputs = jax.random.normal(subkey, (seq_len, input_size))
h0 = jnp.zeros(hidden)

iter_counts = [1, 2, 3, 5, 10, 15]
jacobi_time_per_iter = []
newton_time_per_iter = []

print(f"\nConfig: hidden={hidden}, seq_len={seq_len}")
for num_iter in iter_counts:
    # Jacobi
    mean_time, _, _ = benchmark_method(jacobi_scan, cell, h0, inputs, num_iterations=num_iter, runs=20)
    time_per_iter = (mean_time * 1000) / num_iter
    jacobi_time_per_iter.append(time_per_iter)
    print(f"Jacobi {num_iter:2d} iter: {mean_time*1000:.2f} ms total, {time_per_iter:.3f} ms/iter")

    # Newton (only up to 5 iterations)
    if num_iter <= 5:
        mean_time, _, _ = benchmark_method(newton_scan, cell, h0, inputs, num_iterations=num_iter, runs=20)
        time_per_iter = (mean_time * 1000) / num_iter
        newton_time_per_iter.append(time_per_iter)
        print(f"Newton {num_iter:2d} iter: {mean_time*1000:.2f} ms total, {time_per_iter:.3f} ms/iter")

# Plot time per iteration
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(iter_counts, jacobi_time_per_iter, 'o-', color=HIGHLIGHT, label='Jacobi',
        linewidth=2.5, markersize=8)
ax.plot(iter_counts[:len(newton_time_per_iter)], newton_time_per_iter, 's-',
        color=COLOR_1, label='Newton', linewidth=2.5, markersize=8)
ax.set_xlabel('Number of Iterations', fontsize=12)
ax.set_ylabel('Time per Iteration (ms)', fontsize=12)
ax.set_title(f'Cost per Iteration (hidden={hidden}, seq={seq_len})', fontsize=14, fontweight='bold', color='white')
ax.legend(fontsize=11, facecolor='none', edgecolor='white')
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('docs/time_per_iteration.png', dpi=150, transparent=True)
print("\n✓ Saved: docs/time_per_iteration.png")


print("\n" + "="*70)
print("BENCHMARK COMPLETE!")
print("="*70)
print("\nGenerated plots:")
print("  1. docs/convergence_rate.png       - Convergence comparison")
print("  2. docs/scaling_hidden_dim.png     - Scaling with hidden dimension")
print("  3. docs/scaling_seq_length.png     - Scaling with sequence length")
print("  4. docs/memory_comparison.png      - Memory usage comparison")
print("  5. docs/time_per_iteration.png     - Time cost per iteration")
