"""Standalone distributed initialization module.

This module MUST be imported BEFORE any ueaj modules to ensure proper
JAX distributed initialization ordering.

Usage:
    from distributed_init import init_jax_distributed
    rank, world_size = init_jax_distributed()

    # NOW safe to import JAX and ueaj modules
    import jax
    from ueaj import ...
"""

import os


def init_jax_distributed():
    """Initialize JAX distributed if needed.

    Returns:
        Tuple of (rank, world_size)

    Environment variables:
        DIST_AUTO: If "True", use jax.distributed.initialize() auto-detection
        WORLD_SIZE: Number of processes (if not "1", manual initialization)
        RANK: Process rank for manual initialization
        HOST: Coordinator host:port for manual initialization
    """
    rank = 0
    world_size = 1

    if os.environ.get('DIST_AUTO', "False") == "True":
        # Auto-detect distributed setup
        import jax
        jax.distributed.initialize()
        rank = jax.process_index()
        world_size = jax.process_count()
    elif (world_size_str := os.environ.get("WORLD_SIZE", "1")) != "1":
        # Manual distributed setup
        import jax
        host_ip = os.environ.get('HOST', 'localhost:1234')
        rank = int(os.environ.get('RANK', '0'))
        world_size = int(world_size_str)
        jax.distributed.initialize(host_ip, world_size, rank)

    return rank, world_size
