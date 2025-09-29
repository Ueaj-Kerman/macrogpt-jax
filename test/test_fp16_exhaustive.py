"""Exhaustive test of ALL FP16 and BF16 values for nextafter arithmetic precision."""

import jax
import jax.numpy as jnp
import numpy as np
import ml_dtypes


def test_all_16bit_values(dtype, dtype_name):
    """Test all 65,536 possible 16-bit float values in parallel."""

    print(f"\n{'='*70}")
    print(f"EXHAUSTIVE TEST: {dtype_name} (ALL 65,536 values)")
    print(f"{'='*70}")

    # Generate all possible 16-bit patterns (0-65535)
    all_bytes = np.arange(65536, dtype=np.uint16)

    # Reinterpret as float16/bfloat16 values
    if dtype == jnp.float16:
        all_float_values = all_bytes.view(np.float16)
    else:  # bfloat16
        # For bfloat16, we need to use ml_dtypes
        all_float_values = all_bytes.view(ml_dtypes.bfloat16)

    # Convert to JAX arrays
    p_values = jnp.array(all_float_values, dtype=dtype)

    # Filter out NaN and Inf values
    valid_mask = ~(jnp.isnan(p_values) | jnp.isinf(p_values))
    valid_p = p_values[valid_mask]

    # Also exclude zeros (nextafter(0, -0) doesn't change)
    nonzero_mask = valid_p != 0.0
    test_p = valid_p[nonzero_mask]

    print(f"Total bit patterns: 65,536")
    print(f"Valid values (not NaN/Inf): {len(valid_p):,}")
    print(f"Non-zero test values: {len(test_p):,}")

    # Vectorized operations on all values at once
    print("\nPerforming vectorized nextafter operations...")

    # Method 1: Direct nextafter (ground truth)
    next_p_direct = jax.lax.nextafter(test_p, -test_p)

    # Method 2: Arithmetic reconstruction
    delta = next_p_direct - test_p
    reconstructed = delta + test_p

    # Method 3: All in one expression (what compiler might optimize)
    variant = (jax.lax.nextafter(test_p, -test_p) - test_p) + test_p

    # Check for failures
    print("Checking for failures...")
    failures_reconstruct = reconstructed != next_p_direct
    failures_variant = variant != next_p_direct

    # Count failures
    num_failures_reconstruct = int(jnp.sum(failures_reconstruct))
    num_failures_variant = int(jnp.sum(failures_variant))

    print(f"\n📊 RESULTS:")
    print(f"  Reconstruction failures: {num_failures_reconstruct:,}/{len(test_p):,} "
          f"({num_failures_reconstruct/len(test_p)*100:.4f}%)")
    print(f"  Variant failures: {num_failures_variant:,}/{len(test_p):,} "
          f"({num_failures_variant/len(test_p)*100:.4f}%)")

    # Analyze failures if any
    if num_failures_reconstruct > 0:
        print(f"\n✗ FAILURES DETECTED in arithmetic reconstruction!")

        # Get all failed values
        failed_p = test_p[failures_reconstruct]
        failed_next = next_p_direct[failures_reconstruct]
        failed_delta = delta[failures_reconstruct]
        failed_reconst = reconstructed[failures_reconstruct]

        # Show statistics about failures
        failed_magnitudes = jnp.abs(failed_p)
        print(f"\n  Failure statistics:")
        print(f"    Total failures: {len(failed_p):,}")
        print(f"    Magnitude range: [{float(jnp.min(failed_magnitudes)):.2e}, "
              f"{float(jnp.max(failed_magnitudes)):.2e}]")

        # Categorize failures
        rounded_to_original = failed_reconst == failed_p
        rounded_to_wrong = ~rounded_to_original

        print(f"    Rounded back to original: {int(jnp.sum(rounded_to_original)):,}")
        print(f"    Rounded to wrong value: {int(jnp.sum(rounded_to_wrong)):,}")

        # Show some examples
        print(f"\n  Example failures (first 10):")
        for i in range(min(10, len(failed_p))):
            p = failed_p[i]
            next_p = failed_next[i]
            d = failed_delta[i]
            r = failed_reconst[i]
            print(f"    p={float(p):+.4e}, nextafter={float(next_p):+.4e}, "
                  f"delta={float(d):+.4e}, reconst={float(r):+.4e}")
            if r == p:
                print(f"      → Rounded back to original!")
            else:
                print(f"      → Rounded to wrong value!")

    else:
        print(f"\n✓ SUCCESS: Arithmetic works for ALL {len(test_p):,} test values!")

    # Analyze where values DON'T change
    no_change = test_p == next_p_direct
    num_no_change = int(jnp.sum(no_change))
    if num_no_change > 0:
        print(f"\n⚠ {num_no_change} values where nextafter doesn't change the value")

    # Analyze delta magnitudes
    print(f"\n📈 Delta magnitude analysis:")
    nonzero_p = test_p[test_p != 0]
    nonzero_delta = delta[test_p != 0]
    relative_deltas = jnp.abs(nonzero_delta / nonzero_p)

    print(f"  Relative delta (|delta/p|):")
    print(f"    Min:    {float(jnp.min(relative_deltas)):.2e}")
    print(f"    Max:    {float(jnp.max(relative_deltas)):.2e}")
    print(f"    Mean:   {float(jnp.mean(relative_deltas)):.2e}")
    print(f"    Median: {float(jnp.median(relative_deltas)):.2e}")

    # Check distribution of relative deltas
    eps = jnp.finfo(dtype).eps
    below_eps = relative_deltas < eps
    print(f"  Values with relative delta < machine epsilon ({eps:.2e}): "
          f"{int(jnp.sum(below_eps)):,} ({float(jnp.mean(below_eps))*100:.2f}%)")

    return num_failures_reconstruct == 0


def analyze_failure_patterns():
    """Analyze patterns in where arithmetic fails."""

    print(f"\n{'='*70}")
    print("FAILURE PATTERN ANALYSIS")
    print(f"{'='*70}")

    for dtype, dtype_name in [(jnp.float16, "float16"), (jnp.bfloat16, "bfloat16")]:
        print(f"\n{dtype_name}:")

        # Generate all values
        if dtype == jnp.float16:
            all_values = np.arange(65536, dtype=np.uint16).view(np.float16)
        else:
            all_values = np.arange(65536, dtype=np.uint16).view(ml_dtypes.bfloat16)

        p_values = jnp.array(all_values, dtype=dtype)
        valid_mask = ~(jnp.isnan(p_values) | jnp.isinf(p_values)) & (p_values != 0)
        test_p = p_values[valid_mask]

        # Test arithmetic
        next_p = jax.lax.nextafter(test_p, -test_p)
        delta = next_p - test_p
        reconstructed = delta + test_p
        failures = reconstructed != next_p

        if jnp.any(failures):
            failed_p = test_p[failures]

            # Analyze by magnitude bins
            log_magnitudes = jnp.log10(jnp.abs(failed_p))
            bins = jnp.arange(-40, 40, 2)
            hist, _ = jnp.histogram(log_magnitudes, bins=bins)

            # Find bins with failures
            nonzero_bins = hist > 0
            if jnp.any(nonzero_bins):
                print("  Failures by magnitude (log10):")
                bin_indices = jnp.where(nonzero_bins)[0]
                for idx in bin_indices:
                    count = hist[idx]
                    low = bins[idx]
                    high = bins[idx + 1]
                    print(f"    10^{float(low):.0f} to 10^{float(high):.0f}: {int(count):,} failures")


def benchmark_all_methods():
    """Benchmark performance on full value range."""

    print(f"\n{'='*70}")
    print("PERFORMANCE ON FULL VALUE RANGE")
    print(f"{'='*70}")

    for dtype, dtype_name in [(jnp.float16, "float16"), (jnp.bfloat16, "bfloat16")]:
        # Generate all valid values
        if dtype == jnp.float16:
            all_values = np.arange(65536, dtype=np.uint16).view(np.float16)
        else:
            all_values = np.arange(65536, dtype=np.uint16).view(ml_dtypes.bfloat16)

        p_values = jnp.array(all_values, dtype=dtype)
        valid_mask = ~(jnp.isnan(p_values) | jnp.isinf(p_values)) & (p_values != 0)
        test_p = p_values[valid_mask]

        print(f"\n{dtype_name} ({len(test_p):,} values):")

        # JIT compile methods
        @jax.jit
        def direct_method(p):
            return jax.lax.nextafter(p, -p)

        @jax.jit
        def arithmetic_method(p):
            return (jax.lax.nextafter(p, -p) - p) + p

        # Warm up
        _ = direct_method(test_p).block_until_ready()
        _ = arithmetic_method(test_p).block_until_ready()

        # Time
        import time

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _ = direct_method(test_p).block_until_ready()
        direct_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(iterations):
            _ = arithmetic_method(test_p).block_until_ready()
        arith_time = time.perf_counter() - start

        print(f"  Direct nextafter:  {direct_time:.3f}s ({direct_time/iterations*1000:.2f}ms per iter)")
        print(f"  Arithmetic method: {arith_time:.3f}s ({arith_time/iterations*1000:.2f}ms per iter)")
        print(f"  Overhead: {(arith_time/direct_time - 1)*100:+.1f}%")


if __name__ == "__main__":
    # Test both 16-bit formats exhaustively
    fp16_success = test_all_16bit_values(jnp.float16, "float16")
    bf16_success = test_all_16bit_values(jnp.bfloat16, "bfloat16")

    # Analyze patterns
    analyze_failure_patterns()

    # Benchmark
    benchmark_all_methods()

    print(f"\n{'='*70}")
    print("FINAL CONCLUSION")
    print(f"{'='*70}")

    if fp16_success and bf16_success:
        print("✓ The arithmetic (nextafter(p,-p) - p) + p works for ALL values!")
        print("  Surprising result: No precision loss in practice!")
    else:
        print("✗ The arithmetic method has failures.")
        print("  As predicted, floating-point arithmetic isn't perfectly associative.")

    print("\nThis exhaustive test provides definitive proof of behavior across")
    print("the entire representable range of 16-bit floating-point formats.")