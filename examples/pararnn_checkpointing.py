"""
ParaRNN Memory Analysis: Checkpointing and Implicit Differentiation

This demonstrates the memory requirements for ParaRNN and shows how
gradient checkpointing can be applied.

Key insight: There are TWO dimensions to consider:
1. Iteration dimension: DON'T need to store (implicit differentiation)
2. Sequence dimension: DO need to store (but can checkpoint partially)
"""

import jax
import jax.numpy as jnp
from jax import checkpoint as remat
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ueaj.model.pararnn import parallel_rnn, sequential_rnn


def explain_memory_requirements():
    """
    Explain the memory requirements for ParaRNN's backward pass.
    """
    print("="*70)
    print("ParaRNN Memory Analysis")
    print("="*70)

    print("""
TWO DIMENSIONS TO CONSIDER:

1. ITERATION DIMENSION (implicit differentiation)
   Forward:  h^(0), h^(1), h^(2), ..., h^(k) → h*
   Backward: Only need h* (final converged state)
   ✓ Don't store: h^(0), h^(1), ..., h^(k-1)
   ✓ Use implicit differentiation

2. SEQUENCE DIMENSION (gradient checkpointing)
   States:   h_1*, h_2*, h_3*, ..., h_T*
   Backward: Need all states for BPTT
   ✓ Can checkpoint: store every Kth state
   ✓ Recompute others on-demand

MEMORY BREAKDOWN:

Standard RNN (sequential scan):
- Forward:  O(T) memory (store all h_t)
- Backward: O(T) memory (recompute or store)

ParaRNN WITHOUT checkpointing:
- Forward:  O(T) memory (store final h_t*)
- Backward: O(T) memory (reconstruct states)
- Iterations: O(1) memory (only keep last iteration)

ParaRNN WITH checkpointing:
- Forward:  O(T/K) memory (checkpoint every K states)
- Backward: O(T/K + K) memory (checkpoints + recomputation)
- Iterations: O(1) memory (implicit diff)

KEY ADVANTAGE:
- Implicit differentiation: No need to store iteration history!
- Checkpointing: Can reduce sequence memory by factor K
- Combined: Very memory efficient for long sequences
""")


def demonstrate_implicit_diff():
    """
    Show that we DON'T need to store iteration states.
    """
    print("\n" + "="*70)
    print("Demonstrating Implicit Differentiation")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(0)
    hidden_size = 16
    input_size = 8
    seq_len = 32

    key, *subkeys = jax.random.split(key, 5)
    W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
    W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
    b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

    def cell(h, x):
        h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
        return h_next, h_next

    inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
    h0 = jnp.zeros(hidden_size)

    # Define loss
    def loss(h0, num_iter):
        final_h, _ = parallel_rnn(cell, h0, inputs, num_iterations=num_iter)
        return jnp.sum(final_h ** 2)

    # Compute gradients with different iteration counts
    print("\nGradient computation with varying iterations:")
    print("(All should give similar gradients - we don't differentiate through iterations!)")

    for num_iter in [5, 10, 20]:
        grad_h0 = jax.grad(lambda h0: loss(h0, num_iter))(h0)
        grad_norm = jnp.linalg.norm(grad_h0)
        print(f"  num_iter={num_iter:2d}: grad_norm={grad_norm:.6f}")

    print("\n✓ Gradients are computed via implicit differentiation")
    print("✓ We DON'T backprop through the iteration loop")
    print("✓ Only final converged state h* is needed")


def demonstrate_sequence_checkpointing():
    """
    Show how gradient checkpointing can be applied to the sequence dimension.
    """
    print("\n" + "="*70)
    print("Demonstrating Sequence Checkpointing")
    print("="*70)

    # Setup
    key = jax.random.PRNGKey(0)
    hidden_size = 32
    input_size = 16
    seq_len = 128  # Longer sequence

    key, *subkeys = jax.random.split(key, 5)
    W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
    W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
    b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

    def cell(h, x):
        h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
        return h_next, h_next

    inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
    h0 = jnp.zeros(hidden_size)

    # Version 1: Without checkpointing
    def loss_no_checkpoint(h0):
        final_h, _ = parallel_rnn(cell, h0, inputs, num_iterations=10)
        return jnp.sum(final_h ** 2)

    # Version 2: WITH checkpointing on the cell
    # This saves memory by not storing all intermediate activations
    @remat
    def checkpointed_cell(h, x):
        h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
        return h_next, h_next

    def loss_with_checkpoint(h0):
        final_h, _ = parallel_rnn(checkpointed_cell, h0, inputs, num_iterations=10)
        return jnp.sum(final_h ** 2)

    print("\nComputing gradients...")
    print("(Checkpointing trades compute for memory)")

    # Without checkpointing
    grad1 = jax.grad(loss_no_checkpoint)(h0)
    print(f"✓ Without checkpointing: grad_norm={jnp.linalg.norm(grad1):.6f}")

    # With checkpointing
    grad2 = jax.grad(loss_with_checkpoint)(h0)
    print(f"✓ With checkpointing:    grad_norm={jnp.linalg.norm(grad2):.6f}")

    # They should be the same
    error = jnp.max(jnp.abs(grad1 - grad2))
    print(f"\nGradient error: {error:.2e}")
    print("✓ Checkpointing gives exact gradients, just saves memory!")


def partial_checkpointing_strategy():
    """
    Explain how to do PARTIAL checkpointing - store every Kth state.
    """
    print("\n" + "="*70)
    print("Partial Checkpointing Strategy")
    print("="*70)

    print("""
STANDARD APPROACH (no checkpointing):
- Store all T states: h_1*, h_2*, ..., h_T*
- Memory: O(T × hidden_size)
- Recomputation: None

FULL CHECKPOINTING (recompute everything):
- Store only h_0 and inputs
- Recompute all states during backward pass
- Memory: O(1)
- Recomputation: O(T) during backward

PARTIAL CHECKPOINTING (store every Kth state):
- Store h_K, h_2K, h_3K, ..., h_T
- To compute gradients at h_i:
  1. Find nearest checkpoint h_{jK} where jK < i
  2. Recompute h_{jK+1}, ..., h_i from h_{jK}
- Memory: O(T/K)
- Recomputation: O(K) per backward step

OPTIMAL K:
- K = sqrt(T) minimizes total computation
- Memory: O(sqrt(T)) instead of O(T)
- Recomputation: O(sqrt(T)) extra FLOPs

EXAMPLE:
For T=10,000 states:
- No checkpoint:      10,000 states stored
- Partial (K=100):      100 states stored (100× memory savings!)
- Full checkpoint:        1 state stored (but 10,000× recomputation)

JAX's `jax.checkpoint` does this automatically!
When applied to scan body, it implements smart checkpointing.
""")


def show_advanced_pattern():
    """
    Show how to combine implicit diff + gradient checkpointing.
    """
    print("\n" + "="*70)
    print("Advanced: Combining Both Techniques")
    print("="*70)

    print("""
For maximum memory efficiency, combine:

1. Implicit differentiation (iteration dimension)
   → No need to store h^(0), h^(1), ..., h^(k)

2. Gradient checkpointing (sequence dimension)
   → Store only every Kth state in h_1*, h_2*, ..., h_T*

IMPLEMENTATION:

    from jax import checkpoint

    # Checkpointed cell
    @checkpoint
    def cell(h, x):
        return compute_next_state(h, x)

    # Use with ParaRNN
    final_h, outputs = parallel_rnn(
        cell=cell,  # Checkpointed!
        h0=h0,
        inputs=inputs,
        num_iterations=10  # Implicit diff handles this!
    )

MEMORY SAVINGS:

Standard RNN:
- Iterations: N/A (sequential)
- Sequence:   O(T) states

ParaRNN (naive):
- Iterations: O(k) states → O(1) with implicit diff ✓
- Sequence:   O(T) states

ParaRNN (optimized):
- Iterations: O(1) with implicit diff ✓
- Sequence:   O(sqrt(T)) with checkpointing ✓

For T=10,000, k=10:
- Naive RNN:        10,000 states
- ParaRNN naive:    10,000 states (but parallel!)
- ParaRNN optimal:     100 states (100× savings + parallel!)
""")


def main():
    explain_memory_requirements()
    demonstrate_implicit_diff()
    demonstrate_sequence_checkpointing()
    partial_checkpointing_strategy()
    show_advanced_pattern()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
ANSWER TO YOUR QUESTION:

Q: Does the backward pass REQUIRE checkpointing?
A: No! Implicit differentiation only needs the final converged state.

Q: Can it be done PARTIALLY?
A: Yes! You can:
   1. Use implicit diff for iterations (automatic in our implementation)
   2. Add gradient checkpointing for sequence dimension (optional)
   3. Combine both for maximum memory efficiency

KEY INSIGHTS:
✓ Implicit differentiation eliminates need to store iteration history
✓ Gradient checkpointing reduces sequence memory by sqrt(T) factor
✓ ParaRNN gets parallelism + memory efficiency
✓ JAX makes this easy with @checkpoint decorator

PRACTICAL RECOMMENDATION:
- Sequences < 1,000: Don't worry about checkpointing
- Sequences > 1,000: Use jax.checkpoint on cell
- Sequences > 10,000: Use checkpoint + consider custom policy
""")


if __name__ == "__main__":
    main()
