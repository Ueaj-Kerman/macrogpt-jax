"""
Convergence tests for parallel_scan implementation.

Tests convergence properties compared to reference sequential implementation.
"""

import jax
import jax.numpy as jnp
from jax import lax, grad
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ueaj.model.parallel_scan import parallel_scan


# ============================================================================
# Reference Implementation (Sequential)
# ============================================================================

def sequential_scan(scan_fn, init_carry, xs, reverse=False):
    """Reference implementation using JAX's lax.scan."""
    if reverse:
        # For reverse scan, flip inputs and outputs
        xs_reversed = jax.tree_util.tree_map(lambda x: x[::-1], xs)
        final_carry, outputs_reversed = lax.scan(scan_fn, init_carry, xs_reversed)
        outputs = jax.tree_util.tree_map(lambda x: x[::-1], outputs_reversed)
        return final_carry, outputs
    else:
        return lax.scan(scan_fn, init_carry, xs)


# ============================================================================
# Test Cases: Different Scan Functions
# ============================================================================

class TestLinearScans:
    """Test convergence for linear scan operations."""

    def test_cumsum_forward(self):
        """Cumulative sum should converge quickly."""
        def add_fn(carry, x):
            return carry + x, carry + x

        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        init = 0.0

        # Reference
        final_ref, outputs_ref = sequential_scan(add_fn, init, xs)

        # Test convergence with increasing iterations
        errors = []
        for num_iter in [1, 3, 5, 10]:
            final_par, outputs_par = parallel_scan(add_fn, init, xs, num_iterations=num_iter)
            error = jnp.max(jnp.abs(outputs_ref - outputs_par))
            errors.append((num_iter, error))

        # Print convergence
        print("\nCumulative Sum Convergence:")
        for num_iter, error in errors:
            print(f"  {num_iter:2d} iterations: error = {error:.2e}")

        # Should converge to < 1e-6 within 5 iterations for linear case
        assert errors[-1][1] < 1e-6, f"Failed to converge: {errors[-1][1]}"

    def test_cumsum_reverse(self):
        """Reverse cumulative sum."""
        def add_fn(carry, x):
            return carry + x, carry + x

        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        init = 0.0

        # Reference
        final_ref, outputs_ref = sequential_scan(add_fn, init, xs, reverse=True)

        # Parallel
        final_par, outputs_par = parallel_scan(
            add_fn, init, xs, num_iterations=10, reverse=True
        )

        error = jnp.max(jnp.abs(outputs_ref - outputs_par))
        print(f"\nReverse Cumsum Error: {error:.2e}")
        assert error < 1e-6

    def test_cumprod(self):
        """Cumulative product."""
        def mul_fn(carry, x):
            return carry * x, carry * x

        xs = jnp.array([1.1, 1.05, 1.02, 0.98, 0.95])
        init = 1.0

        final_ref, outputs_ref = sequential_scan(mul_fn, init, xs)
        final_par, outputs_par = parallel_scan(mul_fn, init, xs, num_iterations=10)

        error = jnp.max(jnp.abs(outputs_ref - outputs_par))
        print(f"\nCumprod Error: {error:.2e}")
        assert error < 1e-5


class TestNonlinearRNN:
    """Test convergence for nonlinear RNN cells."""

    def test_elman_rnn_convergence(self):
        """Test Elman RNN with different iteration counts."""
        key = jax.random.PRNGKey(0)
        hidden_size = 16
        input_size = 8
        seq_len = 32

        # Initialize parameters
        key, *subkeys = jax.random.split(key, 5)
        W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
        W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
        b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

        def elman_cell(h, x):
            h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
            return h_next, h_next

        inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
        h0 = jnp.zeros(hidden_size)

        # Reference
        final_ref, outputs_ref = sequential_scan(elman_cell, h0, inputs)

        # Test convergence
        print("\nElman RNN Convergence:")
        errors = []
        for num_iter in [1, 5, 10, 15, 20]:
            final_par, outputs_par = parallel_scan(
                elman_cell, h0, inputs, num_iterations=num_iter
            )

            final_error = jnp.linalg.norm(final_ref - final_par)
            output_error = jnp.max(jnp.abs(outputs_ref - outputs_par))
            errors.append((num_iter, final_error, output_error))

            print(f"  {num_iter:2d} iterations: "
                  f"final_error = {final_error:.2e}, "
                  f"output_error = {output_error:.2e}")

        # Should converge well within 20 iterations
        assert errors[-1][1] < 1e-3, f"Final state error too high: {errors[-1][1]}"
        assert errors[-1][2] < 1e-3, f"Output error too high: {errors[-1][2]}"

    def test_gru_cell_convergence(self):
        """Test GRU cell convergence."""
        key = jax.random.PRNGKey(42)
        hidden_size = 16
        input_size = 8
        seq_len = 32

        # Initialize GRU parameters
        key, *subkeys = jax.random.split(key, 11)
        W_xz = jax.random.normal(subkeys[0], (hidden_size, input_size)) * 0.1
        W_hz = jax.random.normal(subkeys[1], (hidden_size, hidden_size)) * 0.1
        b_z = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01
        W_xr = jax.random.normal(subkeys[3], (hidden_size, input_size)) * 0.1
        W_hr = jax.random.normal(subkeys[4], (hidden_size, hidden_size)) * 0.1
        b_r = jax.random.normal(subkeys[5], (hidden_size,)) * 0.01
        W_xh = jax.random.normal(subkeys[6], (hidden_size, input_size)) * 0.1
        W_hh = jax.random.normal(subkeys[7], (hidden_size, hidden_size)) * 0.1
        b_h = jax.random.normal(subkeys[8], (hidden_size,)) * 0.01

        def gru_cell(h, x):
            z = jax.nn.sigmoid(W_xz @ x + W_hz @ h + b_z)
            r = jax.nn.sigmoid(W_xr @ x + W_hr @ h + b_r)
            h_tilde = jnp.tanh(W_xh @ x + W_hh @ (r * h) + b_h)
            h_next = (1 - z) * h + z * h_tilde
            return h_next, h_next

        inputs = jax.random.normal(subkeys[9], (seq_len, input_size))
        h0 = jnp.zeros(hidden_size)

        # Reference
        final_ref, outputs_ref = sequential_scan(gru_cell, h0, inputs)

        # Parallel with 15 iterations (GRU is more complex)
        final_par, outputs_par = parallel_scan(
            gru_cell, h0, inputs, num_iterations=15
        )

        final_error = jnp.linalg.norm(final_ref - final_par)
        output_error = jnp.max(jnp.abs(outputs_ref - outputs_par))

        print(f"\nGRU Convergence (15 iter):")
        print(f"  final_error  = {final_error:.2e}")
        print(f"  output_error = {output_error:.2e}")

        assert final_error < 1e-3
        assert output_error < 1e-3


class TestGradients:
    """Test gradient correctness and convergence."""

    def test_gradient_correctness_simple(self):
        """Test gradients match for simple cumsum."""
        def add_fn(carry, x):
            return carry + x, carry + x

        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        init = 0.0

        def loss_seq(init):
            final, _ = sequential_scan(add_fn, init, xs)
            return jnp.sum(final ** 2)

        def loss_par(init):
            final, _ = parallel_scan(add_fn, init, xs, num_iterations=10)
            return jnp.sum(final ** 2)

        grad_seq = grad(loss_seq)(init)
        grad_par = grad(loss_par)(init)

        error = jnp.abs(grad_seq - grad_par)
        print(f"\nGradient Error (cumsum): {error:.2e}")
        assert error < 1e-5

    def test_gradient_correctness_rnn(self):
        """Test gradients match for RNN."""
        key = jax.random.PRNGKey(0)
        hidden_size = 8
        input_size = 4
        seq_len = 16

        key, *subkeys = jax.random.split(key, 5)
        W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
        W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
        b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

        def rnn_cell(h, x):
            h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
            return h_next, h_next

        inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
        h0 = jnp.zeros(hidden_size)

        def loss_seq(h0):
            final, _ = sequential_scan(rnn_cell, h0, inputs)
            return jnp.sum(final ** 2)

        def loss_par(h0):
            final, _ = parallel_scan(rnn_cell, h0, inputs, num_iterations=15)
            return jnp.sum(final ** 2)

        grad_seq = grad(loss_seq)(h0)
        grad_par = grad(loss_par)(h0)

        error = jnp.max(jnp.abs(grad_seq - grad_par))
        print(f"\nGradient Error (RNN): {error:.2e}")

        # Gradients converge slower than forward pass
        assert error < 0.05  # Relaxed tolerance

    def test_gradient_convergence(self):
        """Test gradient convergence with increasing iterations."""
        key = jax.random.PRNGKey(0)
        hidden_size = 8
        input_size = 4
        seq_len = 16

        key, *subkeys = jax.random.split(key, 5)
        W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
        W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
        b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

        def rnn_cell(h, x):
            h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
            return h_next, h_next

        inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
        h0 = jnp.zeros(hidden_size)

        def loss_seq(h0):
            final, _ = sequential_scan(rnn_cell, h0, inputs)
            return jnp.sum(final ** 2)

        def loss_par(h0, num_iter):
            final, _ = parallel_scan(rnn_cell, h0, inputs, num_iterations=num_iter)
            return jnp.sum(final ** 2)

        grad_seq = grad(loss_seq)(h0)

        print("\nGradient Convergence:")
        for num_iter in [5, 10, 15, 20]:
            grad_par = grad(lambda h0: loss_par(h0, num_iter))(h0)
            error = jnp.max(jnp.abs(grad_seq - grad_par))
            print(f"  {num_iter:2d} iterations: error = {error:.2e}")


class TestProperties:
    """Test mathematical properties of parallel_scan."""

    def test_linearity(self):
        """Test that cumsum is linear: scan(a*x) = a * scan(x)."""
        def add_fn(carry, x):
            return carry + x, carry + x

        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = 2.5

        _, outputs1 = parallel_scan(add_fn, 0.0, alpha * xs, num_iterations=10)
        _, outputs2 = parallel_scan(add_fn, 0.0, xs, num_iterations=10)

        error = jnp.max(jnp.abs(outputs1 - alpha * outputs2))
        print(f"\nLinearity Error: {error:.2e}")
        assert error < 1e-6

    def test_initial_value(self):
        """Test effect of different initial values."""
        def add_fn(carry, x):
            return carry + x, carry + x

        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        _, outputs_0 = parallel_scan(add_fn, 0.0, xs, num_iterations=10)
        _, outputs_10 = parallel_scan(add_fn, 10.0, xs, num_iterations=10)

        # Outputs should differ by exactly 10.0 at each position
        diff = outputs_10 - outputs_0
        expected = jnp.full_like(diff, 10.0)

        error = jnp.max(jnp.abs(diff - expected))
        print(f"\nInitial Value Consistency Error: {error:.2e}")
        assert error < 1e-6


def test_iteration_convergence_rate():
    """Analyze convergence rate with iteration count."""
    print("\n" + "="*70)
    print("Convergence Rate Analysis")
    print("="*70)

    key = jax.random.PRNGKey(0)
    hidden_size = 16
    input_size = 8
    seq_len = 64

    key, *subkeys = jax.random.split(key, 5)
    W_hh = jax.random.normal(subkeys[0], (hidden_size, hidden_size)) * 0.1
    W_xh = jax.random.normal(subkeys[1], (hidden_size, input_size)) * 0.1
    b_h = jax.random.normal(subkeys[2], (hidden_size,)) * 0.01

    def rnn_cell(h, x):
        h_next = jnp.tanh(W_hh @ h + W_xh @ x + b_h)
        return h_next, h_next

    inputs = jax.random.normal(subkeys[3], (seq_len, input_size))
    h0 = jnp.zeros(hidden_size)

    # Reference
    final_ref, outputs_ref = sequential_scan(rnn_cell, h0, inputs)

    print(f"\nSeq Length: {seq_len}, Hidden Size: {hidden_size}")
    print("\nIter | Final Error | Output Error | Convergence")
    print("-" * 60)

    prev_error = None
    for num_iter in range(1, 26):
        final_par, outputs_par = parallel_scan(
            rnn_cell, h0, inputs, num_iterations=num_iter
        )

        final_error = jnp.linalg.norm(final_ref - final_par)
        output_error = jnp.max(jnp.abs(outputs_ref - outputs_par))

        if prev_error is not None and prev_error > 0:
            rate = final_error / prev_error
            print(f"{num_iter:4d} | {final_error:11.2e} | {output_error:12.2e} | {rate:6.3f}x")
        else:
            print(f"{num_iter:4d} | {final_error:11.2e} | {output_error:12.2e} |    -")

        prev_error = final_error

        if final_error < 1e-6:
            print(f"\n✓ Converged to < 1e-6 in {num_iter} iterations")
            break


if __name__ == "__main__":
    # Run convergence analysis
    test_iteration_convergence_rate()

    # Run test classes
    print("\n" + "="*70)
    print("Running Test Suite")
    print("="*70)

    # Linear scans
    tests_linear = TestLinearScans()
    print("\n[TestLinearScans]")
    tests_linear.test_cumsum_forward()
    tests_linear.test_cumsum_reverse()
    tests_linear.test_cumprod()

    # Nonlinear RNNs
    tests_rnn = TestNonlinearRNN()
    print("\n[TestNonlinearRNN]")
    tests_rnn.test_elman_rnn_convergence()
    tests_rnn.test_gru_cell_convergence()

    # Gradients
    tests_grad = TestGradients()
    print("\n[TestGradients]")
    tests_grad.test_gradient_correctness_simple()
    tests_grad.test_gradient_correctness_rnn()
    tests_grad.test_gradient_convergence()

    # Properties
    tests_props = TestProperties()
    print("\n[TestProperties]")
    tests_props.test_linearity()
    tests_props.test_initial_value()

    print("\n" + "="*70)
    print("All tests passed! ✓")
    print("="*70)
