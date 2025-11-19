#!/usr/bin/env python3
"""Complete demonstration of prepare_dataset_distributed with real data.

This shows end-to-end distributed data loading with:
- Real HuggingFace dataset (fineweb-edu)
- Real tokenizer (GPT2)
- Document packing
- Both 1×2 and 2×2 mesh configurations

Usage:
    # Single process (1 data × 2 tensor)
    PYTHONPATH=/path/to/nanollama ./scripts/run_python.sh examples/demo_distributed_complete.py --orientation 1

    # Multi-process (2 data × 2 tensor)
    bash examples/launch_demo_2host.sh
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
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np
import argparse


def demo_orientation_1():
	"""Orientation 1: Simple data parallelism (1D mesh).

	Mesh: (data=N) where N is the number of devices.
	Pattern: Load data sharded across data axis, no slice-and-reshard needed.
	"""
	print("\n" + "=" * 80)
	print("ORIENTATION 1: Simple Data Parallelism (1D Mesh)")
	print("=" * 80)

	devices = jax.devices()
	num_devices = len(devices)
	mesh = Mesh(np.array(devices), ('data',))

	print(f"\nMesh: {mesh.shape}, axes={mesh.axis_names}")
	print(f"Process {rank}: Local devices = {len(jax.local_devices())}")

	# Import here to avoid issues
	from ueaj.data.dataset import prepare_dataset_distributed
	from datasets import load_dataset
	from transformers import GPT2TokenizerFast

	# Load real dataset
	print(f"\nProcess {rank}: Loading HuggingFace dataset...")
	dataset = load_dataset(
		"HuggingFaceFW/fineweb-edu",
		name="sample-10BT",
		split="train",
		streaming=True
	)

	# Load tokenizer
	tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

	# Prepare distributed dataset
	batch_size = 4  # Per data-parallel shard
	seq_len = 128

	print(f"\nProcess {rank}: Preparing distributed dataset...")
	print(f"  batch_size={batch_size}, seq_len={seq_len}")

	dataset_iter, (tokens_struct, doc_ids_struct) = prepare_dataset_distributed(
		dataset=dataset,
		tokenizer=tokenizer,
		batch_size=batch_size,
		seq_len=seq_len,
		pad_token_id=tokenizer.eos_token_id,
		mesh=mesh,
		column="text",
		min_fill_ratio=0.99,
		buffer_size=10,
		data_axis='data'
	)

	print(f"\nProcess {rank}: Structure specs:")
	print(f"  tokens: {tokens_struct.shape} {tokens_struct.dtype}")
	print(f"  doc_ids: {doc_ids_struct.shape} {doc_ids_struct.dtype}")

	# Get first batch
	print(f"\nProcess {rank}: Fetching first batch...", flush=True)
	tokens_batch, doc_ids_batch = next(dataset_iter)
	dataset_iter.send(None)
	print(f"Process {rank}: Batch fetched!", flush=True)

	print(f"\nProcess {rank}: Batch received:")
	print(f"  tokens.shape = {tokens_batch.shape}")
	print(f"  tokens.sharding = {tokens_batch.sharding.spec}")
	print(f"  addressable_shards = {len(tokens_batch.addressable_shards)}")

	# Verify data
	for i, shard in enumerate(tokens_batch.addressable_shards):
		local_data = np.array(shard.data)
		is_real = not np.all(local_data == 0)
		num_nonzero = np.count_nonzero(local_data)
		print(f"\nProcess {rank}: Shard {i}:")
		print(f"  Shape: {local_data.shape}")
		print(f"  First 10 tokens: {local_data[0, :10]}")
		print(f"  Is real data: {is_real} (non-zero count: {num_nonzero})")

		# Decode first sequence to verify tokenization
		if is_real:
			decoded = tokenizer.decode(local_data[0, :50])
			print(f"  Decoded text (first 50 tokens): {decoded[:100]}...")

	print("\n" + "-" * 80)
	print("RESULT: Simple data parallelism - each process has unique data shard")
	print("No redundant I/O! Each host loads only its portion.")
	print("=" * 80)


def demo_orientation_2():
	"""Orientation 2: Data + Model parallelism (2D mesh).

	Mesh: (data=N, model/tensor=M)
	Pattern:
	1. Load data with intermediate dimension: (batch, 1, seq)
	2. Create array with P('data', 'model', None)
	3. Slice [:, 0, :] to extract real data
	4. Reshard to P('data', None) to broadcast across model axis
	"""
	print("\n" + "=" * 80)
	print("ORIENTATION 2: Data + Model Parallelism (2D Mesh)")
	print("=" * 80)

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

	# Import here to avoid issues
	from ueaj.data.dataset import prepare_dataset_distributed
	from datasets import load_dataset
	from transformers import GPT2TokenizerFast

	# Load real dataset
	print(f"\nProcess {rank}: Loading HuggingFace dataset...")
	dataset = load_dataset(
		"HuggingFaceFW/fineweb-edu",
		name="sample-10BT",
		split="train",
		streaming=True
	)

	# Load tokenizer
	tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

	# Prepare distributed dataset
	batch_size = 4  # Per data-parallel shard
	seq_len = 128

	print(f"\nProcess {rank}: Preparing distributed dataset...")
	print(f"  batch_size={batch_size}, seq_len={seq_len}")

	dataset_iter, (tokens_struct, doc_ids_struct) = prepare_dataset_distributed(
		dataset=dataset,
		tokenizer=tokenizer,
		batch_size=batch_size,
		seq_len=seq_len,
		pad_token_id=tokenizer.eos_token_id,
		mesh=mesh,
		column="text",
		min_fill_ratio=0.99,
		buffer_size=10,
		data_axis='data'
	)

	print(f"\nProcess {rank}: Structure specs:")
	print(f"  tokens: {tokens_struct.shape} {tokens_struct.dtype}")
	print(f"  doc_ids: {doc_ids_struct.shape} {doc_ids_struct.dtype}")

	# Get first batch
	print(f"\nProcess {rank}: Fetching first batch...")
	tokens_batch, doc_ids_batch = next(dataset_iter)
	next(dataset_iter)  # Consume None for async prefetch

	print(f"\nProcess {rank}: Batch received:")
	print(f"  tokens.shape = {tokens_batch.shape}")
	print(f"  tokens.sharding = {tokens_batch.sharding.spec}")
	print(f"  addressable_shards = {len(tokens_batch.addressable_shards)}")

	# Verify all devices have real data (broadcast worked)
	print(f"\nProcess {rank}: Verifying broadcast across model axis:")
	for i, shard in enumerate(tokens_batch.addressable_shards):
		local_data = np.array(shard.data)
		is_real = not np.all(local_data == 0)
		num_nonzero = np.count_nonzero(local_data)
		print(f"\nProcess {rank}: Shard {i}:")
		print(f"  Shape: {local_data.shape}")
		print(f"  First 10 tokens: {local_data[0, :10]}")
		print(f"  Is real data: {is_real} (non-zero count: {num_nonzero})")

		# Decode first sequence to verify tokenization
		if is_real:
			decoded = tokenizer.decode(local_data[0, :50])
			print(f"  Decoded text (first 50 tokens): {decoded[:100]}...")

	# Count how many model=0 devices this host loaded
	from ueaj.utils.distutil import compute_batch_size
	should_load, local_shards = compute_batch_size(mesh)

	print("\n" + "-" * 80)
	print(f"RESULT: This host loaded {local_shards if should_load else 0} shards")
	print(f"Broadcast to {len(tokens_batch.addressable_shards)} local device replicas")
	print(f"I/O savings: {model_size}x reduction! Only model=0 devices loaded data.")
	print("=" * 80)


def main():
	parser = argparse.ArgumentParser(description="Demo distributed data loading with real datasets")
	parser.add_argument('--orientation', type=int, choices=[1, 2], default=None,
						help='Which orientation to run (1: 1D mesh, 2: 2D mesh). Default: both')
	args = parser.parse_args()

	print(f"\nDistributed Data Loading - Complete Demo")
	print(f"Process {rank}/{world_size}")
	print(f"Local devices: {jax.local_device_count()}")
	print(f"Global devices: {jax.device_count()}")

	# Run requested orientation(s)
	if args.orientation is None or args.orientation == 1:
		demo_orientation_1()

	if args.orientation is None or args.orientation == 2:
		demo_orientation_2()

	print("\n" + "=" * 80, flush=True)
	print("Demo completed successfully!", flush=True)
	print("=" * 80, flush=True)


if __name__ == "__main__":
	import os
	import sys
	main()
	# Flush output before exit
	sys.stdout.flush()
	sys.stderr.flush()
	# Use os._exit to bypass threading cleanup issues with HuggingFace datasets
	os._exit(0)
