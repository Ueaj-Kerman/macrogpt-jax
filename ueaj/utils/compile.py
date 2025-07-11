"""JAX compilation utilities with logging and statistics."""

import time
from typing import Callable, Any, Optional, Dict
import jax


def compile_function(
    func: Callable,
    sample_args: tuple,
    sample_kwargs: Optional[Dict[str, Any]] = None,
    name: str = "Function"
) -> Callable:
    """Compile a JAX function with detailed logging and statistics.
    
    Args:
        func: The function to compile (should be jit-decorated)
        sample_args: Sample arguments for tracing
        sample_kwargs: Sample keyword arguments for tracing
        name: Name of the function for logging
        
    Returns:
        Compiled function
    """
    if sample_kwargs is None:
        sample_kwargs = {}
    
    # Box drawing characters
    top_left = "╭"
    top_right = "╮"
    bottom_left = "╰"
    bottom_right = "╯"
    horizontal = "─"
    vertical = "│"
    
    # Header
    header = f" Compiling {name} "
    box_width = 60  # Inner width (excluding the vertical bars)
    header_padding_left = (box_width - len(header)) // 2
    header_padding_right = box_width - len(header) - header_padding_left
    
    print(f"\n{top_left}{horizontal * header_padding_left}{header}{horizontal * header_padding_right}{top_right}")
    
    # Stage tracking
    def print_stage(stage: str, status: str = "..."):
        # Calculate the exact padding needed
        content = f" {stage:<15} {status} "
        padding_needed = box_width - len(content)
        stage_str = f"{vertical}{content:<{box_width}}{vertical}"
        print(stage_str)
    
    # Trace
    print_stage("Tracing", "...")
    start_time = time.time()
    traced = func.trace(*sample_args, **sample_kwargs)
    trace_time = time.time() - start_time
    print_stage("Tracing", f"✓ {trace_time:.2f}s")
    
    # Lower
    print_stage("Lowering", "...")
    start_time = time.time()
    lowered = traced.lower()
    lower_time = time.time() - start_time
    print_stage("Lowering", f"✓ {lower_time:.2f}s")
    
    # Compile
    print_stage("Compiling", "...")
    start_time = time.time()
    compiled = lowered.compile()
    compile_time = time.time() - start_time
    print_stage("Compiling", f"✓ {compile_time:.2f}s")
    
    # Separator
    print(f"{vertical}{horizontal * box_width}{vertical}")
    
    # Cost analysis
    cost = compiled.cost_analysis()
    if cost and 'flops' in cost:
        tflops = cost['flops'] * 1e-12
        print_stage("FLOPs", f"{tflops:.2f} TFLOPs")
    else:
        print_stage("FLOPs", "N/A")

    # Memory analysis
    memory = compiled.memory_analysis()
    if memory is not None:
        peak_gb = memory.temp_size_in_bytes * 1e-9
        param_gb = memory.output_size_in_bytes * 1e-9
        total_gb = peak_gb + param_gb

        print_stage("Peak VRAM", f"{peak_gb:.2f} GB")
        print_stage("Output VRAM", f"{param_gb:.2f} GB")
        print_stage("Total VRAM", f"{total_gb:.2f} GB")
    else:
        print_stage("Memory", "N/A")

    # Footer
    total_time = trace_time + lower_time + compile_time
    print(f"{vertical}{horizontal * box_width}{vertical}")
    print_stage("Total Time", f"{total_time:.2f}s")
    print(f"{bottom_left}{horizontal * box_width}{bottom_right}\n")
    
    return compiled


