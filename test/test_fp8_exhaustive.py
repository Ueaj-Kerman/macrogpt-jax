"""Exhaustive test of all FP8 values for nextafter arithmetic precision."""

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np


def test_all_fp8_values_vectorized(fp8_dtype):
    """Test all 256 possible FP8 values in parallel using vectorized operations."""

    print(f"\n{'='*60}")
    print(f"EXHAUSTIVE TEST: {fp8_dtype.__name__}")
    print(f"{'='*60}")

    # Generate all possible 8-bit patterns (0-255)
    all_bytes = np.arange(256, dtype=np.uint8)

    # Reinterpret as FP8 values
    # This gives us all possible FP8 values including special values (NaN, Inf)
    all_fp8_values = all_bytes.view(fp8_dtype)

    # Convert to JAX arrays
    p_values = jnp.array(all_fp8_values, dtype=fp8_dtype)

    # Filter out NaN and Inf values
    valid_mask = ~(jnp.isnan(p_values) | jnp.isinf(p_values))
    valid_p = p_values[valid_mask]

    print(f"Total values: 256")
    print(f"Valid values (not NaN/Inf): {len(valid_p)}")

    # Vectorized operations on all valid values at once
    # Method 1: Direct nextafter
    next_p_direct = jax.lax.nextafter(valid_p, -valid_p)

    # Method 2: Arithmetic reconstruction
    delta = next_p_direct - valid_p
    reconstructed = delta + valid_p

    # Method 3: All in one expression
    variant = (jax.lax.nextafter(valid_p, -valid_p) - valid_p) + valid_p

    # Check for failures
    failures_reconstruct = reconstructed != next_p_direct
    failures_variant = variant != next_p_direct

    # Count failures
    num_failures_reconstruct = jnp.sum(failures_reconstruct)
    num_failures_variant = jnp.sum(failures_variant)

    print(f"\nReconstruction failures: {num_failures_reconstruct}/{len(valid_p)}")
    print(f"Variant failures: {num_failures_variant}/{len(valid_p)}")

    if num_failures_reconstruct > 0:
        print(f"\n✗ FAILURES DETECTED in reconstruction!")
        # Show some examples of failures
        failed_p = valid_p[failures_reconstruct][:10]  # Show first 10 failures
        for p in failed_p:
            next_p = jax.lax.nextafter(p, -p)
            delta = next_p - p
            reconst = delta + p
            print(f"  p={float(p):.4e}, nextafter={float(next_p):.4e}, "
                  f"delta={float(delta):.4e}, reconst={float(reconst):.4e}")
            print(f"    reconst == p: {reconst == p}, reconst == nextafter: {reconst == next_p}")

    # Also test where values round back to original
    rounds_to_original = reconstructed == valid_p
    num_rounds_to_original = jnp.sum(rounds_to_original)

    if num_rounds_to_original > 0:
        print(f"\n⚠ {num_rounds_to_original} values rounded back to original!")
        rounded_p = valid_p[rounds_to_original][:5]  # Show first 5
        for p in rounded_p:
            next_p = jax.lax.nextafter(p, -p)
            delta = next_p - p
            reconst = delta + p
            print(f"  p={float(p):.4e}, delta={float(delta):.4e}, "
                  f"nextafter changed: {next_p != p}, reconst changed: {reconst != p}")

    # Analyze delta magnitudes
    relative_deltas = jnp.abs(delta / jnp.where(valid_p != 0, valid_p, 1.0))
    min_rel_delta = jnp.min(relative_deltas[valid_p != 0])
    max_rel_delta = jnp.max(relative_deltas[valid_p != 0])
    mean_rel_delta = jnp.mean(relative_deltas[valid_p != 0])

    print(f"\nDelta statistics (relative to value):")
    print(f"  Min:  {float(min_rel_delta):.2e}")
    print(f"  Max:  {float(max_rel_delta):.2e}")
    print(f"  Mean: {float(mean_rel_delta):.2e}")

    # Check special value behavior
    print(f"\nSpecial values:")
    # Zero
    zero = jnp.array(0.0, dtype=fp8_dtype)
    zero_next = jax.lax.nextafter(zero, jnp.array(-1.0, dtype=fp8_dtype))
    print(f"  nextafter(0, -1) = {float(zero_next):.4e}")

    # Smallest positive
    finfo = ml_dtypes.finfo(fp8_dtype)
    smallest = jnp.array(finfo.smallest_normal, dtype=fp8_dtype)
    smallest_next = jax.lax.nextafter(smallest, zero)
    print(f"  nextafter(smallest_positive, 0) = {float(smallest_next):.4e}")

    return num_failures_reconstruct == 0


def analyze_fp8_precision_boundaries():
    """Analyze where precision issues occur in FP8."""

    print(f"\n{'='*60}")
    print("FP8 PRECISION BOUNDARY ANALYSIS")
    print(f"{'='*60}")

    for fp8_dtype in [ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2]:
        print(f"\n{fp8_dtype.__name__}:")

        # Generate all values
        all_bytes = np.arange(256, dtype=np.uint8)
        all_fp8_values = all_bytes.view(fp8_dtype)
        p_values = jnp.array(all_fp8_values, dtype=fp8_dtype)

        # Filter valid values and sort
        valid_mask = ~(jnp.isnan(p_values) | jnp.isinf(p_values))
        valid_p = jnp.sort(p_values[valid_mask])

        # Compute gaps between consecutive values
        gaps = jnp.diff(valid_p)
        unique_gaps = jnp.unique(gaps[gaps > 0])

        print(f"  Unique gap sizes: {len(unique_gaps)}")
        print(f"  Smallest gap: {float(jnp.min(unique_gaps)):.4e}")
        print(f"  Largest gap: {float(jnp.max(unique_gaps)):.4e}")

        # Check where arithmetic fails
        next_p = jax.lax.nextafter(valid_p, -valid_p)
        delta = next_p - valid_p
        reconstructed = delta + valid_p

        failures = reconstructed != next_p

        if jnp.any(failures):
            failed_values = valid_p[failures]
            print(f"  Arithmetic fails at {len(failed_values)} values")

            # Analyze pattern of failures
            failed_magnitudes = jnp.abs(failed_values)
            print(f"  Failure magnitude range: [{float(jnp.min(failed_magnitudes)):.2e}, "
                  f"{float(jnp.max(failed_magnitudes)):.2e}]")


def benchmark_methods():
    """Compare performance of direct vs arithmetic methods."""

    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")

    for fp8_dtype in [ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2]:
        # Generate all valid FP8 values
        all_bytes = np.arange(256, dtype=np.uint8)
        all_fp8_values = all_bytes.view(fp8_dtype)
        p_values = jnp.array(all_fp8_values, dtype=fp8_dtype)
        valid_mask = ~(jnp.isnan(p_values) | jnp.isinf(p_values))
        valid_p = p_values[valid_mask]

        # JIT compile both methods
        @jax.jit
        def direct_method(p):
            return jax.lax.nextafter(p, -p)

        @jax.jit
        def arithmetic_method(p):
            return (jax.lax.nextafter(p, -p) - p) + p

        # Warm up
        _ = direct_method(valid_p).block_until_ready()
        _ = arithmetic_method(valid_p).block_until_ready()

        # Time both methods
        import time

        # Direct method
        start = time.perf_counter()
        for _ in range(1000):
            result = direct_method(valid_p).block_until_ready()
        direct_time = time.perf_counter() - start

        # Arithmetic method
        start = time.perf_counter()
        for _ in range(1000):
            result = arithmetic_method(valid_p).block_until_ready()
        arithmetic_time = time.perf_counter() - start

        print(f"\n{fp8_dtype.__name__} (1000 iterations on {len(valid_p)} values):")
        print(f"  Direct method:     {direct_time:.4f}s")
        print(f"  Arithmetic method: {arithmetic_time:.4f}s")
        print(f"  Overhead:          {(arithmetic_time/direct_time - 1)*100:.1f}%")


if __name__ == "__main__":
    # Test both FP8 types exhaustively
    e4m3_success = test_all_fp8_values_vectorized(ml_dtypes.float8_e4m3fn)
    e5m2_success = test_all_fp8_values_vectorized(ml_dtypes.float8_e5m2)

    # Analyze precision boundaries
    analyze_fp8_precision_boundaries()

    # Performance comparison
    benchmark_methods()

    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")

    if e4m3_success and e5m2_success:
        print("✓ Arithmetic (nextafter(p,-p) - p) + p works for ALL FP8 values!")
        print("  This is because FP8's coarse precision means deltas are large")
        print("  relative to the values, avoiding rounding issues.")
    else:
        print("✗ Arithmetic method fails for some FP8 values.")
        print("  Even with coarse precision, some edge cases break.")

    print("\nKey insight: The arithmetic works when:")
    print("  |delta| > eps * |p|")
    print("where eps is the machine epsilon for the dtype.")