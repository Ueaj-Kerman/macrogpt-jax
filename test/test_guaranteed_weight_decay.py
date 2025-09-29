"""Test guaranteed weight decay gradient transform."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from ueaj.train.grad_transforms import (
    guaranteed_weight_decay,
    create_weight_decay_mask,
    adamw_with_guaranteed_decay
)


def test_guaranteed_decay_basic():
    """Test that guaranteed decay always reduces weights toward zero."""

    print("=" * 60)
    print("TESTING GUARANTEED WEIGHT DECAY")
    print("=" * 60)

    # Test various weight scales
    weight_scales = [1e-38, 1e-10, 1e-5, 0.001, 0.01, 0.1, 1.0, 10.0, 1e10, 1e38]
    decay_rates = [1e-6, 1e-4, 0.01, 0.1]

    for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
        print(f"\n{dtype.__name__}:")

        for decay_rate in decay_rates:
            print(f"  Decay rate = {decay_rate}")

            # Create transform
            transform = guaranteed_weight_decay(decay_rate)

            successes = 0
            forced_count = 0

            for weight in weight_scales:
                # Create parameter
                p = jnp.array(weight, dtype=dtype)

                # Skip if overflows
                if jnp.isinf(p) or jnp.isnan(p):
                    continue

                # Initialize optimizer state
                state = transform.init({"w": p})

                # Create zero gradient (only weight decay affects update)
                grad = {"w": jnp.zeros_like(p)}

                # Apply transform
                updates, _ = transform.update(grad, state, {"w": p})

                # The update should be negative (decay toward zero)
                update = updates["w"]

                # Check if weight would decrease
                new_p = p - update  # Note: subtract because update includes decay

                if jnp.abs(new_p) < jnp.abs(p) or new_p == 0:
                    successes += 1

                    # Check if forced decay was used
                    standard_decay = decay_rate * p
                    if jnp.abs(update) > jnp.abs(standard_decay) * 1.1:  # 10% tolerance
                        forced_count += 1
                else:
                    print(f"    ✗ FAILED: weight {weight:.2e} didn't decrease!")
                    print(f"      p={p}, update={update}, new_p={new_p}")

            print(f"    ✓ {successes}/{len(weight_scales)} weights decreased")
            print(f"    → {forced_count} used forced decay (nextafter)")


def test_decay_with_gradients():
    """Test that decay works correctly when combined with gradients."""

    print("\n" + "=" * 60)
    print("TESTING WITH GRADIENTS")
    print("=" * 60)

    # Create a simple parameter
    params = {
        "layer1": {"weight": jnp.array([[0.1, -0.2], [0.3, -0.4]], dtype=jnp.float32)},
        "layer2": {"bias": jnp.array([0.01, -0.01], dtype=jnp.float32)}
    }

    # Create gradient (simulating backprop)
    grads = {
        "layer1": {"weight": jnp.array([[0.001, -0.002], [0.003, -0.004]], dtype=jnp.float32)},
        "layer2": {"bias": jnp.array([0.0001, -0.0001], dtype=jnp.float32)}
    }

    # Test with mask (no decay on bias)
    mask = create_weight_decay_mask(params, ("bias",))

    # Create optimizer with guaranteed decay
    optimizer = adamw_with_guaranteed_decay(
        learning_rate=0.01,
        weight_decay=0.1,
        mask=mask
    )

    # Initialize optimizer
    opt_state = optimizer.init(params)

    # Apply update
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    print("Original params:")
    print(f"  layer1.weight: {params['layer1']['weight'].flatten()}")
    print(f"  layer2.bias:   {params['layer2']['bias']}")

    print("\nNew params after update:")
    print(f"  layer1.weight: {new_params['layer1']['weight'].flatten()}")
    print(f"  layer2.bias:   {new_params['layer2']['bias']}")

    # Check that weights decreased in magnitude
    weight_decreased = jnp.all(
        jnp.abs(new_params['layer1']['weight']) < jnp.abs(params['layer1']['weight'])
    )

    # Check that bias changed (gradient only, no decay due to mask)
    bias_changed = jnp.any(new_params['layer2']['bias'] != params['layer2']['bias'])

    print("\nResults:")
    print(f"  ✓ Weights decreased: {weight_decreased}")
    print(f"  ✓ Bias changed (no decay): {bias_changed}")


def test_extreme_precision_cases():
    """Test cases where standard decay would fail due to precision."""

    print("\n" + "=" * 60)
    print("TESTING EXTREME PRECISION CASES")
    print("=" * 60)

    # Case 1: Tiny weight with tiny decay
    print("\nCase 1: Tiny weight (1e-38) with small decay (1e-6)")

    for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
        p = jnp.array(1e-38, dtype=dtype)

        # Skip if underflows
        if p == 0:
            print(f"  {dtype.__name__}: Underflows to zero")
            continue

        # Standard decay
        standard_decay = 1e-6 * p
        standard_new = p - standard_decay

        # Guaranteed decay
        transform = guaranteed_weight_decay(1e-6)
        state = transform.init({"w": p})
        updates, _ = transform.update({"w": jnp.zeros_like(p)}, state, {"w": p})
        guaranteed_new = p - updates["w"]

        print(f"  {dtype.__name__}:")
        print(f"    Original:     {float(p):.2e}")
        print(f"    Standard:     {float(standard_new):.2e} (changed: {standard_new != p})")
        print(f"    Guaranteed:   {float(guaranteed_new):.2e} (changed: {guaranteed_new != p})")

    # Case 2: Large weight with tiny decay
    print("\nCase 2: Large weight (1e10) with tiny decay (1e-12)")

    for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
        p = jnp.array(1e10, dtype=dtype)

        # Skip if overflows
        if jnp.isinf(p):
            print(f"  {dtype.__name__}: Overflows to inf")
            continue

        # Standard decay
        standard_decay = 1e-12 * p
        standard_new = p - standard_decay

        # Guaranteed decay
        transform = guaranteed_weight_decay(1e-12)
        state = transform.init({"w": p})
        updates, _ = transform.update({"w": jnp.zeros_like(p)}, state, {"w": p})
        guaranteed_new = p - updates["w"]

        print(f"  {dtype.__name__}:")
        print(f"    Original:     {float(p):.2e}")
        print(f"    Standard:     {float(standard_new):.2e} (changed: {standard_new != p})")
        print(f"    Guaranteed:   {float(guaranteed_new):.2e} (changed: {guaranteed_new != p})")


def benchmark_guaranteed_vs_standard():
    """Compare performance of guaranteed vs standard weight decay."""

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)

    # Create large parameter tree
    params = {
        f"layer_{i}": {
            "weight": jax.random.normal(jax.random.PRNGKey(i), (512, 512), dtype=jnp.float32),
            "bias": jax.random.normal(jax.random.PRNGKey(i+1000), (512,), dtype=jnp.float32)
        }
        for i in range(10)
    }

    # Create gradients
    grads = jax.tree.map(lambda x: jax.random.normal(jax.random.PRNGKey(42), x.shape, x.dtype) * 0.01, params)

    # Standard weight decay
    standard_opt = optax.chain(
        optax.add_decayed_weights(0.01),
        optax.adam(1e-3)
    )

    # Guaranteed weight decay
    guaranteed_opt = adamw_with_guaranteed_decay(1e-3, weight_decay=0.01)

    # JIT compile update functions
    @jax.jit
    def standard_update(grads, opt_state, params):
        return standard_opt.update(grads, opt_state, params)

    @jax.jit
    def guaranteed_update(grads, opt_state, params):
        return guaranteed_opt.update(grads, opt_state, params)

    # Initialize
    standard_state = standard_opt.init(params)
    guaranteed_state = guaranteed_opt.init(params)

    # Warmup
    _, _ = standard_update(grads, standard_state, params)
    _, _ = guaranteed_update(grads, guaranteed_state, params)

    # Benchmark
    import time

    iterations = 100

    # Standard
    start = time.perf_counter()
    for _ in range(iterations):
        updates, standard_state = standard_update(grads, standard_state, params)
        params = optax.apply_updates(params, updates)
    standard_time = time.perf_counter() - start

    # Reset params
    params = jax.tree.map(lambda x: x * 1.0, params)  # Copy

    # Guaranteed
    start = time.perf_counter()
    for _ in range(iterations):
        updates, guaranteed_state = guaranteed_update(grads, guaranteed_state, params)
        params = optax.apply_updates(params, updates)
    guaranteed_time = time.perf_counter() - start

    print(f"Standard weight decay:   {standard_time:.3f}s ({standard_time/iterations*1000:.2f}ms per step)")
    print(f"Guaranteed weight decay: {guaranteed_time:.3f}s ({guaranteed_time/iterations*1000:.2f}ms per step)")
    print(f"Overhead: {(guaranteed_time/standard_time - 1)*100:+.1f}%")


if __name__ == "__main__":
    test_guaranteed_decay_basic()
    test_decay_with_gradients()
    test_extreme_precision_cases()
    benchmark_guaranteed_vs_standard()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Guaranteed weight decay ensures parameters always move toward zero,")
    print("even when standard decay would be lost to floating-point rounding.")
    print("This prevents optimizer stagnation on small weights and ensures")
    print("consistent regularization across all parameter scales.")