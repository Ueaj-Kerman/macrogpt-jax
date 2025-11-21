"""
JAX profiling utilities for performance analysis.

Provides context managers and decorators for profiling JAX computations
using both Perfetto and TensorBoard/XProf.
"""

import jax
import jax.numpy as jnp
from contextlib import contextmanager
from typing import Optional, Callable, Any
import os
import time
from functools import wraps


@contextmanager
def profile_trace(
    log_dir: str,
    name: str = "trace",
    create_perfetto_link: bool = False,
    create_perfetto_trace: bool = False,
    tensorboard: bool = True
):
    """
    Context manager for profiling JAX computations.

    Args:
        log_dir: Directory to save profiling data
        name: Name for this profiling session (creates subdirectory)
        create_perfetto_link: If True, creates interactive Perfetto link (blocks until clicked)
        create_perfetto_trace: If True, saves Perfetto trace files
        tensorboard: If True, saves TensorBoard-compatible traces

    Example:
        with profile_trace("./profiles", name="forward_pass"):
            output = model(inputs)
            output.block_until_ready()
    """
    trace_dir = os.path.join(log_dir, name)
    os.makedirs(trace_dir, exist_ok=True)

    print(f"📊 Profiling to: {trace_dir}")

    if tensorboard and not (create_perfetto_link or create_perfetto_trace):
        # TensorBoard mode (default)
        create_perfetto_trace = True

    with jax.profiler.trace(
        trace_dir,
        create_perfetto_link=create_perfetto_link,
        create_perfetto_trace=create_perfetto_trace
    ):
        yield

    if tensorboard:
        print(f"✓ TensorBoard: tensorboard --logdir={log_dir}")
    if create_perfetto_trace and not create_perfetto_link:
        trace_file = f"{trace_dir}/plugins/profile/*/trace.json.gz"
        print(f"✓ Perfetto: Upload {trace_file} to https://ui.perfetto.dev")


def profile_function(
    log_dir: str,
    name: Optional[str] = None,
    warmup_steps: int = 3,
    profile_steps: int = 1
):
    """
    Decorator for profiling JAX functions.

    Args:
        log_dir: Directory to save profiling data
        name: Name for profiling session (defaults to function name)
        warmup_steps: Number of warmup iterations before profiling
        profile_steps: Number of iterations to profile

    Example:
        @profile_function("./profiles", warmup_steps=5)
        def train_step(state, batch):
            return updated_state, metrics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = name or func.__name__

            print(f"🔥 Warming up {func_name} ({warmup_steps} steps)...")
            for _ in range(warmup_steps):
                result = func(*args, **kwargs)
                # Handle both single values and tuples
                if isinstance(result, tuple):
                    jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
                else:
                    if hasattr(result, 'block_until_ready'):
                        result.block_until_ready()

            print(f"📊 Profiling {func_name} ({profile_steps} steps)...")
            with profile_trace(log_dir, name=func_name):
                for _ in range(profile_steps):
                    result = func(*args, **kwargs)
                    if isinstance(result, tuple):
                        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
                    else:
                        if hasattr(result, 'block_until_ready'):
                            result.block_until_ready()

            return result

        return wrapper
    return decorator


@contextmanager
def profile_scope(name: str, enabled: bool = True):
    """
    Lightweight profiling scope using JAX's TraceAnnotation.

    This adds named scopes to profiler traces without creating separate trace files.
    Useful for marking different sections within a single profiling session.

    Args:
        name: Name of the scope (appears in profiler UI)
        enabled: Whether profiling is enabled (allows conditional profiling)

    Example:
        with profile_trace("./profiles"):
            with profile_scope("attention"):
                attn_output = attention_layer(x)
            with profile_scope("mlp"):
                mlp_output = mlp_layer(attn_output)
    """
    if enabled:
        with jax.profiler.TraceAnnotation(name):
            yield
    else:
        yield


def time_jax_function(func: Callable, *args, **kwargs) -> tuple[Any, float]:
    """
    Accurately time a JAX function accounting for async dispatch.

    Args:
        func: JAX function to time
        *args, **kwargs: Arguments to pass to function

    Returns:
        (result, elapsed_time_ms): Function result and elapsed time in milliseconds

    Example:
        result, time_ms = time_jax_function(jitted_forward, model, inputs)
        print(f"Forward pass: {time_ms:.2f}ms")
    """
    # Warmup
    _ = func(*args, **kwargs)

    # Time with proper blocking
    start = time.perf_counter()
    result = func(*args, **kwargs)

    # Block on all outputs
    if isinstance(result, tuple):
        jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
    else:
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    elapsed_ms = (time.perf_counter() - start) * 1000

    return result, elapsed_ms


def benchmark_function(
    func: Callable,
    *args,
    num_iterations: int = 100,
    warmup: int = 10,
    **kwargs
) -> dict[str, float]:
    """
    Benchmark a JAX function with statistical analysis.

    Args:
        func: Function to benchmark
        *args, **kwargs: Arguments to pass to function
        num_iterations: Number of timing iterations
        warmup: Number of warmup iterations

    Returns:
        Dictionary with mean, std, min, max times in milliseconds

    Example:
        stats = benchmark_function(train_step, state, batch, num_iterations=50)
        print(f"Mean: {stats['mean']:.2f}ms ± {stats['std']:.2f}ms")
    """
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    # Collect timings
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        result = func(*args, **kwargs)

        # Block on all outputs
        if isinstance(result, tuple):
            jax.tree.map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
        else:
            if hasattr(result, 'block_until_ready'):
                result.block_until_ready()

        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    times_array = jnp.array(times)
    return {
        "mean": float(jnp.mean(times_array)),
        "std": float(jnp.std(times_array)),
        "min": float(jnp.min(times_array)),
        "max": float(jnp.max(times_array)),
        "median": float(jnp.median(times_array)),
    }
