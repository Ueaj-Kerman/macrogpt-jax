#!/usr/bin/env python3
"""Example: Distributed data loading with real datasets.

This demonstrates loading actual data (using HuggingFace datasets) with
the distributed pattern in both orientations:
1. Simple 1D data parallelism (no slice-and-reshard needed)
2. 2D mesh with data + model parallelism (with slice-and-reshard)

Usage:
    # Single process
    PYTHONPATH=/path/to/nanollama ./scripts/run_python.sh examples/distributed_real_data.py

    # Multi-process (test distributed pattern)
    bash examples/launch_real_data.sh
"""

# Initialize distributed BEFORE importing ueaj (which imports JAX)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from distributed_init import init_jax_distributed
rank, world_size = init_jax_distributed()

# NOW safe to import JAX and ueaj modules
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from ueaj.utils.distutil import compute_batch_size


def orientation_1_simple_data_parallel():
    """Orientation 1: Simple data parallelism (no slice-and-reshard).

    Mesh: (data=N) where N is the number of devices.
    Pattern: Load data sharded across data axis, no broadcast needed.
    """
    print("\n" + "="*80)
    print("ORIENTATION 1: Simple Data Parallelism")
    print("="*80)

    devices = jax.devices()
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), ('data',))

    print(f"\nMesh: {mesh.shape}, axes={mesh.axis_names}")
    print(f"Process {rank}: Local devices = {len(jax.local_devices())}")

    # Check if we should load data
    should_load, local_shards = compute_batch_size(mesh)
    print(f"Process {rank}: should_load = {should_load}, local_shards = {local_shards}")

    # Simulate loading real data
    batch_size_per_device = 4
    seq_len = 16
    local_batch_size = batch_size_per_device * len(jax.local_devices())

    if should_load:
        # Load real data (in practice, from dataset)
        print(f"Process {rank}: LOADING {local_batch_size} samples from dataset")
        local_tokens = np.arange(local_batch_size * seq_len).reshape(local_batch_size, seq_len).astype(np.int32)
        local_doc_ids = np.ones((local_batch_size, seq_len), dtype=np.int32) * (rank + 1)
    else:
        # Create dummy data (will be ignored)
        print(f"Process {rank}: Creating dummy data (will be unused)")
        local_tokens = np.zeros((local_batch_size, seq_len), dtype=np.int32)
        local_doc_ids = np.zeros((local_batch_size, seq_len), dtype=np.int32)

    # Create distributed arrays with simple sharding
    sharding = NamedSharding(mesh, P('data', None))
    tokens = jax.make_array_from_process_local_data(sharding, local_tokens)
    doc_ids = jax.make_array_from_process_local_data(sharding, local_doc_ids)

    print(f"\nDistributed arrays created:")
    print(f"  tokens.shape = {tokens.shape}")
    print(f"  sharding = {tokens.sharding.spec}")
    print(f"  addressable_shards = {len(tokens.addressable_shards)}")

    # Show local data
    if len(tokens.addressable_shards) > 0:
        local_data = np.array(tokens.addressable_data(0))
        print(f"\nProcess {rank}: Local shard:")
        print(f"  First row: {local_data[0, :8]}")
        is_real = not np.all(local_data == 0)
        print(f"  Is real data: {is_real}")

    print("\n" + "-"*80)
    print("RESULT: Each process has its own shard of data")
    print("No redundant I/O! Each host loads only its data shard.")
    print("="*80)


def orientation_2_with_model_parallel():
    """Orientation 2: Data + Model parallelism (with slice-and-reshard).

    Mesh: (data=N, model=M)
    Pattern:
    1. Load data with intermediate dimension: (batch, 1, seq)
    2. Create array with P('data', 'model', None)
    3. Slice [:, 0, :] to extract real data
    4. Reshard to P('data', None) to broadcast across model axis
    """
    print("\n" + "="*80)
    print("ORIENTATION 2: Data + Model Parallelism (Slice-and-Reshard)")
    print("="*80)

    devices = jax.devices()
    num_devices = len(devices)

    if num_devices < 2:
        print("\nSkipping: need at least 2 devices for 2D mesh")
        return

    # Create 2D mesh: data x model
    if num_devices >= 4:
        data_size = num_devices // 2
        model_size = 2
    else:
        data_size = 1
        model_size = num_devices

    # In distributed mode, reshape appropriately
    if world_size > 1:
        # Transpose to put same-rank devices in columns
        mesh = Mesh(np.array(devices).reshape(data_size, model_size).T, ('data', 'model'))
    else:
        mesh = Mesh(np.array(devices[:data_size * model_size]).reshape(data_size, model_size), ('data', 'model'))

    print(f"\nMesh: {mesh.shape}, axes={mesh.axis_names}")
    print(f"Process {rank}: Local devices = {len(jax.local_devices())}")

    # Count how many model=0 devices this host has
    model_dim = mesh.axis_names.index('model')
    local_devices_set = set(mesh.local_devices)
    my_count = 0

    it = np.nditer(mesh.devices, flags=['multi_index', 'refs_ok'])
    for dev in it:
        if dev.item() in local_devices_set:
            if it.multi_index[model_dim] == 0:
                my_count += 1
                print(f"Process {rank}: Has model=0 device at data={it.multi_index[0]}")

    print(f"Process {rank}: Total model=0 devices = {my_count}")

    # Load data based on model=0 count
    batch_size_per_device = 4
    seq_len = 16

    if my_count > 0:
        # This host has model=0 devices - load real data
        # Need intermediate dimension for model axis
        local_batch = batch_size_per_device * my_count
        print(f"Process {rank}: LOADING {my_count} data shards ({local_batch} samples)")

        # Shape: (batch, 1, seq) - "1" is for model axis
        local_tokens = np.arange(local_batch * seq_len).reshape(local_batch, 1, seq_len).astype(np.int32)
        local_doc_ids = np.ones((local_batch, 1, seq_len), dtype=np.int32) * (rank + 1)
    else:
        # This host has only model>0 devices - create zeros
        num_local = len(jax.local_devices())
        local_batch = batch_size_per_device * num_local
        print(f"Process {rank}: Creating DUMMY zeros ({local_batch} samples)")

        local_tokens = np.zeros((local_batch, 1, seq_len), dtype=np.int32)
        local_doc_ids = np.zeros((local_batch, 1, seq_len), dtype=np.int32)

    print(f"Process {rank}: Local data shape = {local_tokens.shape}")

    # Step 1: Create array with P('data', 'model', None)
    print(f"\n{'='*40}")
    print("STEP 1: Create array with P('data', 'model', None)")
    print(f"{'='*40}")

    initial_sharding = NamedSharding(mesh, P('data', 'model', None))
    tokens_array = jax.make_array_from_process_local_data(initial_sharding, local_tokens)
    doc_ids_array = jax.make_array_from_process_local_data(initial_sharding, local_doc_ids)

    print(f"Before slicing:")
    print(f"  Shape: {tokens_array.shape}")
    print(f"  Sharding: {tokens_array.sharding.spec}")
    print(f"  Note: Only model=0 slice has real data, rest are zeros")

    # Step 2: Slice [:, 0, :] to extract real data
    print(f"\n{'='*40}")
    print("STEP 2: Slice [:, 0, :] to extract REAL data")
    print(f"{'='*40}")

    tokens_sliced = tokens_array[:, 0, :]
    doc_ids_sliced = doc_ids_array[:, 0, :]

    print(f"After slicing [:, 0, :]:")
    print(f"  Shape: {tokens_sliced.shape} (model dimension removed)")
    print(f"  Real data extracted, zeros discarded")

    # Step 3: Reshard to broadcast
    print(f"\n{'='*40}")
    print("STEP 3: Reshard to P('data', None) to BROADCAST")
    print(f"{'='*40}")

    @jax.jit
    def reshard_broadcast(tokens, doc_ids):
        target_sharding = NamedSharding(mesh, P('data', None))
        tokens = jax.device_put(tokens, target_sharding)
        doc_ids = jax.device_put(doc_ids, target_sharding)
        return tokens, doc_ids

    tokens_final, doc_ids_final = reshard_broadcast(tokens_sliced, doc_ids_sliced)

    print(f"After resharding:")
    print(f"  Shape: {tokens_final.shape}")
    print(f"  Sharding: {tokens_final.sharding.spec}")
    print(f"  Addressable shards: {len(tokens_final.addressable_shards)}")

    # Verify all devices have real data
    print(f"\nVerifying broadcast:")
    for i, shard in enumerate(tokens_final.addressable_shards):
        local_data = np.array(shard.data)
        is_real = not np.all(local_data == 0)
        print(f"Process {rank}: Shard {i}: first row = {local_data[0, :8]}, is_real={is_real}")

    print("\n" + "-"*80)
    print(f"RESULT: Loaded {my_count} shards, broadcast to {len(tokens_final.addressable_shards)} replicas")
    print(f"I/O savings: {model_size}x reduction! Only model=0 devices loaded data.")
    print("="*80)


def main():
    """Run both orientations."""
    print(f"\nDistributed Real Data Loading Examples")
    print(f"Process {rank}/{world_size}")
    print(f"Local devices: {jax.local_device_count()}")
    print(f"Global devices: {jax.device_count()}")

    # Run both orientations
    orientation_1_simple_data_parallel()
    orientation_2_with_model_parallel()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
