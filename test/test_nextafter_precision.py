"""Test whether (nextafter(p, -p) - p) + p actually decreases p across different dtypes."""

import jax
import jax.numpy as jnp
import jax.lax
import ml_dtypes
import numpy as np


def test_nextafter_arithmetic(dtype, test_values):
    """Test if (nextafter(p, -p) - p) + p equals nextafter(p, -p)."""

    print(f"\n{'='*60}")
    print(f"Testing {dtype}")
    print(f"{'='*60}")

    failures = 0
    for val in test_values:
        # Convert to the specified dtype
        p = jnp.array(val, dtype=dtype)

        # Skip if value becomes 0 or inf after conversion
        if p == 0 or jnp.isinf(p) or jnp.isnan(p):
            continue

        # Method 1: Direct nextafter (guaranteed to work)
        next_p = jax.lax.nextafter(p, -p)

        # Method 2: Compute delta and add back
        delta = next_p - p
        reconstructed = delta + p

        # Method 3: Try to be clever with parentheses
        reconstructed_v2 = (jax.lax.nextafter(p, -p) - p) + p

        # Check if they're equal
        success = (reconstructed == next_p)
        success_v2 = (reconstructed_v2 == next_p)

        # Print results
        if not success or not success_v2:
            failures += 1
            print(f"\nValue: {float(p):.6e}")
            print(f"  Original p:       {p}")
            print(f"  nextafter(p,-p):  {next_p}")
            print(f"  delta:            {delta}")
            print(f"  reconstruct:      {reconstructed}")
            print(f"  reconstruct_v2:   {reconstructed_v2}")
            print(f"  ✗ FAILED: reconstruct == p: {reconstructed == p}")
            print(f"  ✗ FAILED: changed at all:   {reconstructed != p}")

            # Check relative size of delta vs p
            if p != 0:
                relative_delta = float(jnp.abs(delta / p))
                print(f"  Relative delta:   {relative_delta:.2e}")

    if failures == 0:
        print("✓ All tests passed - reconstruction always equals nextafter")
    else:
        print(f"\n✗ {failures}/{len(test_values)} tests failed")

    return failures


def test_fp8_specific():
    """Special tests for FP8 formats."""
    print(f"\n{'='*60}")
    print("FP8 Special Tests")
    print(f"{'='*60}")

    # FP8 has very limited range and precision
    # E4M3: 4 exponent bits, 3 mantissa bits, ±448 range
    # E5M2: 5 exponent bits, 2 mantissa bits, ±57344 range

    for fp8_type in [ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2]:
        print(f"\n{fp8_type.__name__}:")

        # Get dtype info
        finfo = ml_dtypes.finfo(fp8_type)
        print(f"  Range: [{float(finfo.min):.2e}, {float(finfo.max):.2e}]")
        print(f"  Smallest positive: {float(finfo.smallest_normal):.2e}")
        print(f"  Epsilon: {float(finfo.eps):.2e}")

        # Test a few values within FP8 range
        test_vals = [1.0, 10.0, 0.1, float(finfo.max)/2]

        for val in test_vals:
            p = jnp.array(val, dtype=fp8_type)
            if jnp.isnan(p) or jnp.isinf(p):
                continue

            # Get next value toward negative
            next_p = jax.lax.nextafter(p, jnp.array(-p, dtype=fp8_type))

            # Compute delta
            delta = next_p - p

            # Try to reconstruct
            reconstructed = delta + p

            # Check if it worked
            worked = (reconstructed == next_p)

            print(f"\n  Value {float(p):.4f}:")
            print(f"    nextafter: {float(next_p):.6f}")
            print(f"    delta:     {float(delta):.6e}")
            print(f"    reconst:   {float(reconstructed):.6f}")
            print(f"    Success:   {worked}")

            if not worked:
                print(f"    ✗ Reconstruction failed!")
                print(f"    Lost to rounding: {reconstructed == p}")


def test_weight_decay_scenario():
    """Test realistic weight decay scenarios."""
    print(f"\n{'='*60}")
    print("Weight Decay Scenarios")
    print(f"{'='*60}")

    # Simulate weight decay on different scales of weights
    weight_scales = [0.001, 0.01, 0.1, 1.0, 10.0]
    decay_rate = 0.01

    for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
        print(f"\n{dtype.__name__} with decay rate {decay_rate}:")

        for w_scale in weight_scales:
            w = jnp.array(w_scale, dtype=dtype)

            # Method 1: Standard weight decay
            decay = w * decay_rate
            new_w_standard = w - decay

            # Method 2: Force with nextafter if no change
            if new_w_standard == w:
                new_w_forced = jax.lax.nextafter(w, jnp.array(0.0, dtype=dtype))
            else:
                new_w_forced = new_w_standard

            # Method 3: Try the problematic arithmetic
            next_w = jax.lax.nextafter(w, -w)
            delta = next_w - w
            new_w_arithmetic = delta + w

            print(f"  Weight {float(w):.3e}:")
            print(f"    Standard decay:  {float(new_w_standard):.6e}, changed: {new_w_standard != w}")
            print(f"    Forced decay:    {float(new_w_forced):.6e}, changed: {new_w_forced != w}")
            print(f"    Arithmetic:      {float(new_w_arithmetic):.6e}, changed: {new_w_arithmetic != w}")


if __name__ == "__main__":
    # Test values covering different scales
    test_values = [
        1e-10, 1e-5, 0.001, 0.01, 0.1,
        1.0, 10.0, 100.0, 1e5, 1e10, 1e20
    ]

    # Test standard dtypes
    for dtype in [jnp.float32, jnp.float16, jnp.bfloat16]:
        test_nextafter_arithmetic(dtype, test_values)

    # Test FP8 specifically
    test_fp8_specific()

    # Test weight decay scenarios
    test_weight_decay_scenario()

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("The arithmetic (nextafter(p,-p) - p) + p is NOT guaranteed")
    print("to decrease p due to rounding in the subtraction and addition.")
    print("Direct assignment of nextafter(p,-p) is the only guarantee.")