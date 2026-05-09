"""Distributed (multi-host, multi-axis) data loader for HuggingFace streaming datasets.

Design goal (from user spec):

    "we only want to load a number of samples proportional to the number of
     first ranks along all the non-data axis for the devices the host has
     access to. For example, if a single host has 2 / 8 GPUs and we're doing
     2x DP and 4x TP, and the single host has both GPUs of TP rank 0, then it
     would load 2 (packed) batches, and all the others would load nothing."

That is: for every device this host owns, look at its mesh coordinate. If the
device is at index 0 along *every* non-data axis, this host is the canonical
loader for that device's data slot. We count those slots, load that many
sequences from a sharded view of the streaming dataset, and let
``jax.make_array_from_process_local_data`` broadcast them to TP/PP-replicas
on other hosts.

Data-rank derivation
--------------------
There can be multiple distinct (dp_rank values) covered by a single host. We
key the HF ``IterableDataset.shard()`` call on each unique data-rank that has
at least one loader-slot device on this host. The total number of shards is
``prod(mesh.shape[ax] for ax in data_axes)``. This guarantees disjoint streams
across hosts and matches the global batch layout.

Output
------
``distributed_batch_iterator`` yields ``(tokens, doc_ids)`` jax.Arrays whose
sharding is ``PartitionSpec(data_axes, None)`` over the provided mesh — global
shape ``(global_batch, seq_len)`` — with each host contributing only the rows
for the data-ranks it owns. Hosts with zero loader slots still call
``make_array_from_process_local_data`` with empty (0, seq_len) buffers.
"""

from __future__ import annotations

from typing import Iterator, Tuple, Sequence, Callable, Optional, Generator
import itertools
import math

import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .packing import pack_documents, padding_iterator
from .batching import batch_iterator, tuple_collate
from .dataset import create_tokenize_fn, tokens_iterator


# ---------------------------------------------------------------------------
# Mesh introspection
# ---------------------------------------------------------------------------

def _device_mesh_coords(mesh: Mesh) -> dict:
	"""Return {device_id: tuple(coord_per_axis)} for every device in the mesh."""
	coords = {}
	for idx in itertools.product(*(range(s) for s in mesh.shape.values())):
		dev = mesh.devices[idx]
		coords[dev.id] = idx
	return coords


def _slots_from_coords(
	axis_names: Sequence[str],
	axis_sizes: Sequence[int],
	data_axes: Sequence[str],
	host_device_coords: Sequence[Tuple[int, ...]],
) -> Tuple[list[int], int, int]:
	"""Pure spec implementation.

	Given the mesh's axis layout and the coords of devices owned by this host,
	return ``(sorted_unique_data_ranks, num_loader_slots, total_data_world)``.

	A device is a loader slot iff its coord is 0 along every non-data axis.
	The data-rank is the row-major linearisation of its coords on ``data_axes``
	(in the order given). Slots that share a data-rank dedupe down to one rank
	(unique set), but ``num_loader_slots`` counts the underlying devices so the
	caller can size per-host scratch buffers if needed.
	"""
	axis_pos = {a: i for i, a in enumerate(axis_names)}
	non_data_axes = [a for a in axis_names if a not in data_axes]
	data_shape = [axis_sizes[axis_pos[a]] for a in data_axes]
	total = int(np.prod(data_shape)) if data_shape else 1

	ranks: list[int] = []
	for coord in host_device_coords:
		if all(coord[axis_pos[a]] == 0 for a in non_data_axes):
			rank = 0
			for a, s in zip(data_axes, data_shape):
				rank = rank * s + coord[axis_pos[a]]
			ranks.append(rank)

	return sorted(set(ranks)), len(ranks), total


def compute_host_loader_slots(
	mesh: Mesh,
	data_axes: Sequence[str],
) -> Tuple[list[int], int, int]:
	"""For the current process, determine which data-ranks it must load.

	Returns:
	    (sorted_unique_data_ranks_for_this_host,
	     num_loader_slot_devices_for_this_host,
	     total_data_world_size)
	"""
	axis_names = list(mesh.shape.keys())
	axis_sizes = [mesh.shape[a] for a in axis_names]
	device_coords = _device_mesh_coords(mesh)
	this_pid = jax.process_index()
	host_coords = [
		device_coords[d.id]
		for d in jax.local_devices()
		if d.process_index == this_pid
	]
	return _slots_from_coords(axis_names, axis_sizes, data_axes, host_coords)


# ---------------------------------------------------------------------------
# Per-data-rank streaming pipeline
# ---------------------------------------------------------------------------

def _make_shard_iterator(
	hf_dataset,
	tokenizer,
	num_shards: int,
	shard_index: int,
	seq_len: int,
	pad_token_id: int,
	column: str = "text",
	use_packing: bool = True,
):
	"""Build a single-shard tokenize -> pack -> per-sequence iterator."""
	shard = hf_dataset.shard(num_shards=num_shards, index=shard_index)
	tokenize_fn = create_tokenize_fn(tokenizer, column)
	shard = shard.map(tokenize_fn, batched=True)
	shard = shard.select_columns("tokens")
	tokens_iter = tokens_iterator(shard)
	if use_packing:
		return pack_documents(tokens_iter, max_length=seq_len, pad_token_id=pad_token_id)
	else:
		return padding_iterator(tokens_iter, max_length=seq_len, pad_token_id=pad_token_id, truncate=True)


def distributed_batch_iterator(
	hf_dataset,
	tokenizer,
	mesh: Mesh,
	data_axes: Sequence[str],
	batch_size_per_data_rank: int,
	seq_len: int,
	pad_token_id: int,
	column: str = "text",
	use_packing: bool = True,
) -> Generator[Tuple[jax.Array, jax.Array], None, None]:
	"""Yield (tokens, doc_ids) global jax.Arrays with PartitionSpec(data_axes, None).

	This generator is collective: every process in the JAX cluster must iterate
	in lockstep because ``make_array_from_process_local_data`` is a collective.
	"""
	loader_ranks, num_slots, total_data = compute_host_loader_slots(mesh, data_axes)

	# Build one shard iterator per unique data-rank we own.
	shard_iters = {
		rank: _make_shard_iterator(
			hf_dataset,
			tokenizer,
			num_shards=total_data,
			shard_index=rank,
			seq_len=seq_len,
			pad_token_id=pad_token_id,
			column=column,
			use_packing=use_packing,
		)
		for rank in loader_ranks
	}

	# Per-shard batch iterator: collates ``batch_size_per_data_rank`` sequences.
	batched = {
		rank: batch_iterator(it, batch_size_per_data_rank, drop_last=True, collate_fn=tuple_collate)
		for rank, it in shard_iters.items()
	}

	# Global batch shape across data axes. Note: sequences-per-host =
	# (#unique data ranks owned) * batch_size_per_data_rank. Devices that
	# share a data-rank (e.g. TP-rank>0) reuse the same row.
	global_batch = total_data * batch_size_per_data_rank

	tokens_sharding = NamedSharding(mesh, P(tuple(data_axes), None))
	doc_sharding = NamedSharding(mesh, P(tuple(data_axes), None))

	print(
		f"[dist_loader] pid={jax.process_index()} "
		f"loader_data_ranks={loader_ranks} num_slots={num_slots} "
		f"total_data_axis={total_data} global_batch={global_batch}"
	)

	while True:
		# Pull one batch per owned data-rank. If any is exhausted, stop.
		local_tokens = []
		local_docs = []
		try:
			for rank in loader_ranks:
				tokens, docs = next(batched[rank])
				local_tokens.append(tokens)
				local_docs.append(docs)
		except StopIteration:
			return

		if local_tokens:
			tokens_local = np.concatenate(local_tokens, axis=0)
			docs_local = np.concatenate(local_docs, axis=0)
		else:
			# Host owns no data-ranks: contribute empty buffers.
			tokens_local = np.zeros((0, seq_len), dtype=np.int32)
			docs_local = np.zeros((0, seq_len), dtype=np.int32)

		# Build the global array. Each row in tokens_local corresponds to a
		# distinct data-rank; make_array_from_process_local_data is collective
		# and stitches them into the global PartitionSpec(data_axes, None).
		global_shape = (global_batch, seq_len)
		global_tokens = jax.make_array_from_process_local_data(
			tokens_sharding, tokens_local, global_shape=global_shape
		)
		global_docs = jax.make_array_from_process_local_data(
			doc_sharding, docs_local, global_shape=global_shape
		)

		yield global_tokens, global_docs


def make_global_shape(
	mesh: Mesh,
	data_axes: Sequence[str],
	batch_size_per_data_rank: int,
	seq_len: int,
) -> Tuple[int, int]:
	total_data = int(np.prod([mesh.shape[a] for a in data_axes])) if data_axes else 1
	return (total_data * batch_size_per_data_rank, seq_len)
