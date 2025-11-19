"""Example: Distributed data loading without redundant I/O.

This script demonstrates the efficient distributed data loading pattern where
only hosts with devices on the first data-parallel slice actually load data.
Other hosts create dummy data and receive via JAX's automatic broadcast.

Usage:
    # Single host with multiple GPUs (1D data parallelism)
    ./scripts/run_python.sh examples/distributed_data_loading.py

    # Multi-host with data + model parallelism (2D mesh)
    WORLD_SIZE=4 RANK=0 HOST=192.168.1.100:1234 ./scripts/run_python.sh examples/distributed_data_loading.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from ueaj.utils.distutil import init_distributed, this_host_has_first

def simple_1d_example():
    """Example with 1D data parallelism (single axis)."""
    print("\n" + "="*60)
    print("1D Data Parallelism Example")
    print("="*60)

    # Create 1D mesh for data parallelism
    devices = jax.devices()
    mesh = jax.sharding.Mesh(devices, ('data',))
    print(f"Mesh shape: {mesh.shape}")
    print(f"Mesh axes: {mesh.axis_names}")

    # Check if this host should load data
    should_load = this_host_has_first(mesh, 'data')
    print(f"\nHost {jax.process_index()}: should_load={should_load}")

    # Simulate data loading
    batch_size = 4
    seq_len = 8

    if should_load:
        print("Loading real data...")
        local_data = jnp.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    else:
        print("Creating dummy data (will receive via broadcast)...")
        local_data = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    # Create distributed array with data-parallel sharding
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('data', None))
    distributed_array = jax.make_array_from_process_local_data(sharding, local_data)

    print(f"\nDistributed array shape: {distributed_array.shape}")
    print(f"Distributed array sharding: {distributed_array.sharding}")
    print(f"Addressable shards: {len(distributed_array.addressable_shards)}")

    # Verify data
    local_slice = distributed_array.addressable_data(0)
    print(f"\nLocal data slice:\n{local_slice}")


def two_d_mesh_example():
    """Example with 2D mesh (data + model/tensor parallelism).

    This demonstrates the slice-and-reshard pattern for broadcasting
    data from model/tensor=0 devices to all model/tensor replicas.
    """
    print("\n" + "="*60)
    print("2D Mesh Example (Data + Model Parallelism)")
    print("="*60)

    # Try to create a 2D mesh if we have enough devices
    devices = jax.devices()
    num_devices = len(devices)

    if num_devices < 2:
        print("Skipping 2D example: need at least 2 devices")
        return

    # Create 2D mesh (e.g., 2x2 or 4x2)
    if num_devices >= 4:
        data_size = num_devices // 2
        model_size = 2
    else:
        data_size = 1
        model_size = num_devices

    mesh_devices = np.array(devices[:data_size * model_size]).reshape(data_size, model_size)
    mesh = jax.sharding.Mesh(mesh_devices, ('data', 'model'))

    print(f"Mesh shape: {mesh.shape}")
    print(f"Mesh axes: {mesh.axis_names}")

    # Check if this host should load data
    should_load = this_host_has_first(mesh, 'data')
    print(f"\nHost {jax.process_index()}: should_load={should_load}")

    # Simulate data loading
    batch_size = 4
    seq_len = 8

    if should_load:
        print("Loading real data...")
        local_data = jnp.arange(batch_size * seq_len).reshape(batch_size, seq_len)
    else:
        print("Creating dummy data (will receive via broadcast)...")
        local_data = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    # Step 1: Create array with initial sharding (data + model)
    # This loads data only on model=0 slice
    initial_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec('data', None)  # Not sharded on model axis
    )
    data_array = jax.make_array_from_process_local_data(initial_sharding, local_data)

    print(f"\nInitial array sharding: {data_array.sharding}")
    print(f"Initial addressable shards: {len(data_array.addressable_shards)}")

    # Step 2: Reshard to broadcast across model axis (if needed)
    # In this simple case, we're already broadcasting since model axis isn't in the spec
    # But in the reference pattern, you might have P('data', 'model', None) initially
    # and then slice/reshard to P('data', None)

    print(f"\nFinal array shape: {data_array.shape}")
    print(f"Final array sharding: {data_array.sharding}")

    # Verify data
    if len(data_array.addressable_shards) > 0:
        local_slice = data_array.addressable_data(0)
        print(f"\nLocal data slice:\n{local_slice}")


def realistic_pattern_example():
    """Example matching the main3.py pattern exactly.

    This shows the exact slice-and-reshard pattern from the reference:
    1. Load data with P('data', 'tensor', None) sharding
    2. Only hosts with (data=i, tensor=0) load real data
    3. Other hosts have zeros at (data=i, tensor=j) for j>0
    4. Slice [:, 0, :] to extract ONLY the real data (discards zeros)
    5. Reshard to P('data', None) to broadcast real data to all tensor replicas
    """
    print("\n" + "="*60)
    print("Exact Slice-and-Reshard Pattern (from main3.py)")
    print("="*60)

    devices = jax.devices()
    num_devices = len(devices)

    if num_devices < 2:
        print("Skipping: need at least 2 devices")
        return

    # Create 2D mesh
    if num_devices >= 4:
        data_size = num_devices // 2
        tensor_size = 2
    else:
        data_size = 1
        tensor_size = num_devices

    mesh_devices = np.array(devices[:data_size * tensor_size]).reshape(data_size, tensor_size)
    mesh = jax.sharding.Mesh(mesh_devices, ('data', 'tensor'))

    print(f"Mesh shape: {mesh.shape}")

    # Check if this host should load data
    should_load = this_host_has_first(mesh, 'data')

    batch_size = 4
    seq_len = 8

    # Add extra dimension for tensor axis
    if should_load:
        print(f"Host {jax.process_index()}: Loading REAL data")
        # Shape: (batch_size, 1, seq_len) - the "1" is for tensor axis
        # Only hosts with tensor=0 devices will have this real data
        local_data = jnp.arange(batch_size * seq_len).reshape(batch_size, 1, seq_len)
        print(f"  Local data shape: {local_data.shape}")
        print(f"  Sample values: {local_data[0, 0, :4]}")
    else:
        print(f"Host {jax.process_index()}: Creating DUMMY zeros")
        # Hosts with only tensor>0 devices create zeros
        local_data = jnp.zeros((batch_size, 1, seq_len), dtype=jnp.int32)
        print(f"  Local data shape: {local_data.shape}")
        print(f"  Sample values: {local_data[0, 0, :4]} (all zeros)")

    # Step 1: Create array with initial sharding P('data', 'tensor', None)
    # This shards the middle dimension across the 'tensor' axis
    initial_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec('data', 'tensor', None)
    )
    data_array = jax.make_array_from_process_local_data(initial_sharding, local_data)

    print(f"\n[Before slicing]")
    print(f"  Shape: {data_array.shape}")
    print(f"  Sharding: {data_array.sharding.spec}")
    print(f"  Note: Only tensor=0 slice contains real data, rest are zeros")

    # Step 2: THE KEY SLICE [:, 0, :] - Extract only the real data!
    # This selects the first slice along the tensor axis, which has the real data
    # and discards all the dummy zeros from tensor>0 slices
    sliced_array = data_array[:, 0, :]

    print(f"\n[After slicing [:, 0, :]]")
    print(f"  Shape: {sliced_array.shape} (tensor dimension removed)")
    print(f"  Now contains only real data (zeros discarded)")

    # Step 3: Reshard to P('data', None) to broadcast across all tensor devices
    @jax.jit
    def reshard_broadcast(x):
        target_sharding = jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec('data', None)
        )
        return jax.device_put(x, target_sharding)

    final_array = reshard_broadcast(sliced_array)

    print(f"\n[After resharding to P('data', None)]")
    print(f"  Shape: {final_array.shape}")
    print(f"  Sharding: {final_array.sharding.spec}")
    print(f"  Real data now broadcast to ALL tensor devices!")

    # Verify all devices have the data
    if len(final_array.addressable_shards) > 0:
        local_slice = final_array.addressable_data(0)
        print(f"\nLocal data on this host:")
        print(f"  First row: {local_slice[0, :8]}")
        print(f"  (Should be real data, not zeros!)")


def main():
    """Run all examples."""
    # Initialize distributed runtime if needed
    rank, world_size = init_distributed()
    print(f"\nProcess {rank}/{world_size}")
    print(f"Local devices: {jax.local_device_count()}")
    print(f"Total devices: {jax.device_count()}")

    # Run examples
    simple_1d_example()
    two_d_mesh_example()
    realistic_pattern_example()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
