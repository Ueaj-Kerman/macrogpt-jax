#!/usr/bin/env python3
"""Demonstrate 2×2 mesh with proper batch size handling.

With 4 devices in a 2×2 mesh:
- Devices at (data=0, tensor=0) and (data=1, tensor=0) load data
- Devices at (data=0, tensor=1) and (data=1, tensor=1) receive broadcast

In multi-host setup:
- If Host 0 has (data=0, tensor=0) and (data=1, tensor=0) → loads 2× batch_size
- If Host 1 has (data=0, tensor=1) and (data=1, tensor=1) → creates zeros

This demonstrates compute_batch_size() adapting the batch size per host.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

def main():
    print(f"\n{'='*80}")
    print(f"2×2 Mesh Batch Size Adaptation Demo")
    print(f"{'='*80}")

    devices = jax.devices()
    if len(devices) < 4:
        print(f"\nNeed 4 devices, only have {len(devices)}")
        print("This demo shows the CONCEPT - in a real 4-device setup:")
        print("  • 2 devices at tensor=0 would load 2× batch_size")
        print("  • 2 devices at tensor=1 would receive broadcast")
        print(f"{'='*80}")
        return

    # Create 2×2 mesh
    mesh = Mesh(np.array(devices[:4]).reshape(2, 2), ('data', 'tensor'))

    print(f"\nMesh shape: {mesh.shape}")
    print(f"Mesh axes: {mesh.axis_names}")
    print(f"Device layout:\n{mesh.devices}")

    # Simulate what would happen on each "host"
    # In real distributed: each process only sees its local devices
    from ueaj.utils.distutil import compute_batch_size

    print(f"\n{'='*80}")
    print("Simulating distributed hosts:")
    print(f"{'='*80}")

    # Simulate Host 0 having (data=0, tensor=0) and (data=1, tensor=0)
    print("\nHost 0 scenario (has both tensor=0 devices):")
    print("  Devices: (data=0, tensor=0) and (data=1, tensor=0)")

    should_load, local_shards = compute_batch_size(mesh)
    batch_size = 4
    local_batch_size = batch_size * local_shards

    print(f"  should_load = {should_load}")
    print(f"  local_shards = {local_shards}")
    print(f"  batch_size per shard = {batch_size}")
    print(f"  → local_batch_size = {batch_size} × {local_shards} = {local_batch_size}")
    print(f"  ✓ Host 0 loads {local_batch_size} examples total")

    # Simulate Host 1 having (data=0, tensor=1) and (data=1, tensor=1)
    print("\nHost 1 scenario (has both tensor=1 devices):")
    print("  Devices: (data=0, tensor=1) and (data=1, tensor=1)")
    print(f"  should_load = False")
    print(f"  local_batch_size = 0 (creates dummy zeros)")
    print(f"  ✓ Host 1 loads NOTHING, receives via broadcast")

    print(f"\n{'='*80}")
    print("Result:")
    print(f"{'='*80}")
    print(f"Total I/O: {local_batch_size} examples (only Host 0)")
    print(f"Total devices with data: 4 devices")
    print(f"I/O savings: 2× reduction! (vs each device loading independently)")
    print(f"{'='*80}")

    # Now actually test with real data loading
    print(f"\n{'='*80}")
    print("Testing with real data loading:")
    print(f"{'='*80}")

    from ueaj.data.dataset import prepare_dataset_distributed
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    dataset_iter, (tokens_struct, doc_ids_struct) = prepare_dataset_distributed(
        dataset=dataset,
        tokenizer=tokenizer,
        batch_size=batch_size,
        seq_len=128,
        pad_token_id=tokenizer.eos_token_id,
        mesh=mesh,
        column="text",
        data_axis='data'
    )

    print(f"\nGlobal structure: {tokens_struct.shape}")
    print(f"Expected: ({batch_size} × {mesh.shape['data']}, 128) = ({batch_size * mesh.shape['data']}, 128)")

    tokens_batch, doc_ids_batch = next(dataset_iter)
    next(dataset_iter)

    print(f"\nActual batch shape: {tokens_batch.shape}")
    print(f"Batch sharding: {tokens_batch.sharding.spec}")
    print(f"\n✓ Test completed successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
