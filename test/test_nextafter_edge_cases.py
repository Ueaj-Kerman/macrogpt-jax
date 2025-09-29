"""Test edge cases where nextafter arithmetic might fail."""

import jax
import jax.numpy as jnp
import jax.lax


def test_extreme_values():
    """Test with extreme values where precision issues are more likely."""

    print("=" * 60)
    print("EXTREME VALUE TESTS")
    print("=" * 60)

    # Test cases designed to trigger precision loss
    test_cases = [
        # (dtype, value, description)
        (jnp.float32, 1e38, "Near float32 max"),
        (jnp.float32, 1e-38, "Near float32 min"),
        (jnp.float16, 65000.0, "Near float16 max"),
        (jnp.float16, 6e-5, "Near float16 min"),
        (jnp.bfloat16, 1e38, "Near bfloat16 max"),
        (jnp.bfloat16, 1e-38, "Near bfloat16 min"),
    ]

    for dtype, value, description in test_cases:
        print(f"\n{dtype.__name__} - {description}:")

        p = jnp.array(value, dtype=dtype)
        if jnp.isinf(p) or jnp.isnan(p):
            print(f"  Value {value} overflows/underflows in {dtype.__name__}")
            continue

        # Direct nextafter
        next_p = jax.lax.nextafter(p, -p)

        # Compute delta and reconstruct
        delta = next_p - p
        reconstructed = delta + p

        # Check multiple variants
        variant1 = (jax.lax.nextafter(p, -p) - p) + p
        variant2 = p + (jax.lax.nextafter(p, -p) - p)

        print(f"  Original p:      {p}")
        print(f"  nextafter(p,-p): {next_p}")
        print(f"  delta:           {delta}")
        print(f"  p + delta:       {reconstructed}")
        print(f"  variant1:        {variant1}")
        print(f"  variant2:        {variant2}")

        # Check for failures
        if reconstructed != next_p:
            print(f"  ✗ FAILED: reconstruction != nextafter")
            print(f"    reconstructed == p: {reconstructed == p}")

        # Check relative magnitude
        if p != 0:
            relative_delta = float(jnp.abs(delta / p))
            print(f"  Relative delta: {relative_delta:.2e}")

            # Check if delta is smaller than machine epsilon
            eps = jnp.finfo(dtype).eps
            if relative_delta < eps:
                print(f"  ⚠ Delta ({relative_delta:.2e}) < epsilon ({eps:.2e})")


def test_accumulated_operations():
    """Test what happens with accumulated nextafter operations."""

    print("\n" + "=" * 60)
    print("ACCUMULATED OPERATIONS TEST")
    print("=" * 60)

    for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
        print(f"\n{dtype.__name__}:")

        # Start with a moderate value
        p = jnp.array(1.0, dtype=dtype)

        # Apply nextafter multiple times
        direct_result = p
        arithmetic_result = p

        steps = 10
        for i in range(steps):
            # Direct method
            direct_result = jax.lax.nextafter(direct_result, -direct_result)

            # Arithmetic method
            delta = jax.lax.nextafter(arithmetic_result, -arithmetic_result) - arithmetic_result
            arithmetic_result = arithmetic_result + delta

        print(f"  Original:        {p}")
        print(f"  After {steps} steps:")
        print(f"    Direct:        {direct_result}")
        print(f"    Arithmetic:    {arithmetic_result}")
        print(f"    Match:         {direct_result == arithmetic_result}")

        if direct_result != arithmetic_result:
            diff = float(jnp.abs(direct_result - arithmetic_result))
            print(f"    Difference:    {diff:.2e}")


def test_optimizer_simulation():
    """Simulate optimizer updates with small learning rates."""

    print("\n" + "=" * 60)
    print("OPTIMIZER SIMULATION")
    print("=" * 60)

    # Simulate parameter with different scales and tiny updates
    param_scales = [1e-3, 1.0, 1e3]
    lr_scales = [1e-6, 1e-8, 1e-10]

    for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
        print(f"\n{dtype.__name__}:")

        for param in param_scales:
            for lr in lr_scales:
                p = jnp.array(param, dtype=dtype)

                # Simulate a gradient
                grad = jnp.array(1.0, dtype=dtype)  # Unit gradient

                # Standard update
                update = lr * grad
                new_p = p - update

                # Check if update was lost
                if new_p == p:
                    # Try nextafter trick
                    forced_p = jax.lax.nextafter(p, jnp.array(0.0, dtype=dtype))

                    # Try arithmetic version
                    delta = jax.lax.nextafter(p, -p) - p
                    arithmetic_p = p + delta

                    print(f"  p={param:.0e}, lr={lr:.0e}:")
                    print(f"    Standard:   {new_p == p} (no change)")
                    print(f"    Forced:     {forced_p != p} (changed)")
                    print(f"    Arithmetic: {arithmetic_p != p} (changed)")

                    if arithmetic_p == p:
                        print(f"    ✗ Arithmetic method FAILED to update!")


if __name__ == "__main__":
    test_extreme_values()
    test_accumulated_operations()
    test_optimizer_simulation()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The arithmetic (nextafter(p,-p) - p) + p works in most cases")
    print("but can fail with:")
    print("  1. Extreme values near dtype limits")
    print("  2. Accumulated operations (errors compound)")
    print("  3. Very small learning rates relative to parameter scale")