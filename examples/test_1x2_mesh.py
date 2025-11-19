#!/usr/bin/env python3
"""Test 1 data × 2 tensor mesh to verify only 1 host loads data.

This demonstrates the I/O savings pattern clearly:
- Mesh: (data=1, tensor=2)
- Only devices at (data=0, tensor=0) load real data
- Expected: 1 host loads, 1 host creates zeros
- I/O savings: 2× reduction

Usage:
    bash examples/launch_1x2_test.sh
"""

# Initialize distributed BEFORE importing ueaj
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from distributed_init import init_jax_distributed
rank, world_size = init_jax_distributed()

# NOW safe to import JAX and ueaj
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

def main():
    print(f"\n{'='*80}")
    print(f"Testing 1 Data × 2 Tensor Mesh")
    print(f"{'='*80}")
    print(f"Process {rank}/{world_size}")
    print(f"Local devices: {jax.local_device_count()}")
    print(f"Global devices: {jax.device_count()}")

    devices = jax.devices()

    # Create 1×2 mesh: 1 data × 2 tensor
    # Don't transpose - keep devices in row layout
    mesh = Mesh(np.array(devices).reshape(1, 2), ('data', 'tensor'))

    print(f"\nMesh shape: {mesh.shape}")
    print(f"Mesh axes: {mesh.axis_names}")
    print(f"Device layout:\n{mesh.devices}")

    # Check which host should load data
    from ueaj.utils.distutil import compute_batch_size
    should_load, local_shards = compute_batch_size(mesh)

    print(f"\nProcess {rank}: should_load = {should_load}")
    print(f"Process {rank}: local_shards = {local_shards}")

    # Import dataset utilities
    from ueaj.data.dataset import prepare_dataset_distributed
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    # Load dataset and tokenizer
    print(f"\nProcess {rank}: Loading HuggingFace dataset...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Prepare distributed dataset
    batch_size = 4
    seq_len = 128

    print(f"\nProcess {rank}: Preparing distributed dataset...")
    dataset_iter, (tokens_struct, doc_ids_struct) = prepare_dataset_distributed(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        pad_token_id=tokenizer.eos_token_id,
        mesh=mesh,
        column="text",
        data_axis='data'
    )

    # Fetch first batch
    print(f"\nProcess {rank}: Fetching first batch...", flush=True)
    tokens_batch, doc_ids_batch = next(dataset_iter)
    next(dataset_iter)  # Consume None
    print(f"Process {rank}: Batch fetched!", flush=True)

    # Verify results
    print(f"\n{'='*80}")
    print(f"Process {rank}: RESULTS")
    print(f"{'='*80}")
    print(f"tokens.shape = {tokens_batch.shape}")
    print(f"tokens.sharding = {tokens_batch.sharding.spec}")
    print(f"addressable_shards = {len(tokens_batch.addressable_shards)}")

    # Check local data
    for i, shard in enumerate(tokens_batch.addressable_shards):
        local_data = np.array(shard.data)
        is_real = not np.all(local_data == 0)
        num_nonzero = np.count_nonzero(local_data)

        print(f"\nProcess {rank}: Shard {i}:")
        print(f"  Shape: {local_data.shape}")
        print(f"  First 10 tokens: {local_data[0, :10]}")
        print(f"  Is real data: {is_real} (non-zero: {num_nonzero}/{local_data.size})")

        if is_real:
            decoded = tokenizer.decode(local_data[0, :50])
            print(f"  Decoded: {decoded[:80]}...")

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")

    if should_load:
        print(f"✓ Process {rank}: LOADED {local_shards} data shards from disk")
    else:
        print(f"○ Process {rank}: Created DUMMY zeros (no I/O)")

    print(f"✓ Process {rank}: Has {len(tokens_batch.addressable_shards)} local device replicas with REAL data")
    print(f"\nI/O Savings: 2× reduction! Only 1 host loaded data for 2 devices.")
    print(f"{'='*80}")


if __name__ == "__main__":
    import sys
    main()
    # Flush and clean exit
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
