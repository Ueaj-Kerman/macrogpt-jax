"""Dataset construction utilities for training."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Optional, Iterator, Tuple, Any, Generator


def create_tokenize_fn(tokenizer, column: str = "text") -> Callable:
    """Create a tokenization function for dataset.map().
    
    Args:
        tokenizer: Tokenizer instance to use
        column: Column name to tokenize from
        
    Returns:
        Function that can be passed to dataset.map()
    """
    def tokenize(examples):
        tokenized = tokenizer(examples[column], return_tensors='np')['input_ids']
        return {'tokens': tokenized}
    return tokenize


def tokens_iterator(dataset) -> Iterator[np.ndarray]:
    """Extract tokens from dataset examples.
    
    Args:
        dataset: Dataset with 'tokens' column
        
    Yields:
        Token arrays as numpy arrays
    """
    for ex in dataset:
        yield np.array(ex["tokens"], dtype=np.int32)


def prepare_dataset(
    dataset,
    tokenizer,
    batch_size: int,
    seq_len: int,
    pad_token_id: int,
    column: str = "text",
    min_fill_ratio: float = 0.99,
    buffer_size: int = 10
) -> Tuple[Generator[Tuple[jax.Array, jax.Array], None, None], Tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]]:
    """Prepare a streaming dataset for training.

    Args:
        dataset: Streaming dataset from datasets.load_dataset()
        tokenizer: Tokenizer to use
        batch_size: Batch size for training
        seq_len: Sequence length for training
        pad_token_id: Token ID to use for padding
        column: Column name to tokenize (default: "text")
        min_fill_ratio: Minimum fill ratio for document packing
        buffer_size: Buffer size for device prefetching

    Returns:
        Tuple of:
        - Iterator yielding (tokens, document_ids) batches
        - Tuple of (tokens_struct, document_ids_struct) for compilation
    """
    from . import pack_documents, batch_iterator, device_prefetch, tuple_collate

    # Tokenize the dataset
    tokenize_fn = create_tokenize_fn(tokenizer, column)
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset = dataset.select_columns("tokens")

    # Create iterator pipeline
    dataset = tokens_iterator(dataset)
    dataset = pack_documents(dataset, max_length=seq_len, min_fill_ratio=min_fill_ratio, pad_token_id=pad_token_id)
    dataset = batch_iterator(dataset, batch_size=batch_size, drop_last=True, collate_fn=tuple_collate)
    dataset = device_prefetch(dataset, buffer_size=buffer_size)

    # Create structure descriptors for compilation
    tokens_struct = jax.ShapeDtypeStruct((batch_size, seq_len), jax.numpy.int32)
    document_ids_struct = jax.ShapeDtypeStruct((batch_size, seq_len), jax.numpy.int32)

    return dataset, (tokens_struct, document_ids_struct)


def prepare_dataset_distributed(
    dataset,
    tokenizer,
    batch_size: int,
    seq_len: int,
    pad_token_id: int,
    mesh: jax.sharding.Mesh,
    column: str = "text",
    min_fill_ratio: float = 0.99,
    buffer_size: int = 10,
    data_axis: str = 'data'
) -> Tuple[Generator[Tuple[jax.Array, jax.Array], None, None], Tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]]:
    """Prepare a streaming dataset for distributed training without redundant I/O.

    This function implements efficient distributed data loading where only hosts
    with devices on the first slice of the data-parallel axis actually load data.
    For 2D meshes (e.g., data × model/tensor parallelism), only hosts with devices
    on (data=i, model/tensor=0) load data, then JAX broadcasts to other model/tensor
    slices via resharding.

    Args:
        dataset: Streaming dataset from datasets.load_dataset()
        tokenizer: Tokenizer to use
        batch_size: Batch size per data-parallel shard
        seq_len: Sequence length for training
        pad_token_id: Token ID to use for padding
        mesh: JAX device mesh for distributed training
        column: Column name to tokenize (default: "text")
        min_fill_ratio: Minimum fill ratio for document packing
        buffer_size: Buffer size for device prefetching
        data_axis: Name of the data-parallel axis in mesh (default: 'data')

    Returns:
        Tuple of:
        - Iterator yielding (tokens, document_ids) batches as distributed arrays
        - Tuple of (tokens_struct, document_ids_struct) for compilation

    Example (2D mesh with data + model parallelism):
        >>> mesh = jax.sharding.Mesh(devices.reshape(4, 2), ('data', 'model'))
        >>> # Only 4 hosts load data (one per data shard), broadcast to 2 model replicas
        >>> dataset, structs = prepare_dataset_distributed(
        ...     dataset, tokenizer, batch_size=8, seq_len=2048,
        ...     pad_token_id=0, mesh=mesh
        ... )
    """
    from . import pack_documents, batch_iterator, tuple_collate
    from ueaj.utils.distutil import compute_batch_size

    # Determine if this host should load data and how many shards
    should_load, local_shards = compute_batch_size(mesh)

    # Determine the number of non-data axes (for the intermediate dimension)
    non_data_axes = [axis for axis in mesh.axis_names if axis != data_axis]
    num_non_data_axes = len(non_data_axes)

    # Calculate total batch size for this host
    # If a host has multiple data-parallel devices (e.g., both tensor=0 slices in a 2×2 mesh),
    # it needs to load batch_size * local_shards worth of data
    local_batch_size = batch_size * local_shards

    if should_load:
        print(f"Host {jax.process_index()}: Loading data from dataset (local_shards={local_shards}, local_batch_size={local_batch_size})", flush=True)

        # Tokenize the dataset
        tokenize_fn = create_tokenize_fn(tokenizer, column)
        dataset = dataset.map(tokenize_fn, batched=True)
        dataset = dataset.select_columns("tokens")

        # Create iterator pipeline
        dataset = tokens_iterator(dataset)
        dataset = pack_documents(dataset, max_length=seq_len, min_fill_ratio=min_fill_ratio, pad_token_id=pad_token_id)
        dataset = batch_iterator(dataset, batch_size=local_batch_size, drop_last=True, collate_fn=tuple_collate)

        # Add intermediate dimension(s) for non-data axes
        # Shape: (local_batch_size, seq_len) -> (local_batch_size, 1, 1, ..., seq_len)
        if num_non_data_axes > 0:
            def reshape_iterator(base_iter):
                for tokens, doc_ids in base_iter:
                    # Insert dimensions for each non-data axis
                    new_shape_tokens = (local_batch_size,) + (1,) * num_non_data_axes + (seq_len,)
                    new_shape_doc_ids = (local_batch_size,) + (1,) * num_non_data_axes + (seq_len,)
                    yield (tokens.reshape(new_shape_tokens), doc_ids.reshape(new_shape_doc_ids))
            dataset = reshape_iterator(dataset)
    else:
        print(f"Host {jax.process_index()}: Creating dummy data iterator (will receive via broadcast)", flush=True)

        # Create dummy iterator that yields zeros with intermediate dimensions
        # Shape: (local_batch_size, 1, 1, ..., seq_len) to match the loading hosts
        def dummy_iterator():
            while True:
                shape = (local_batch_size,) + (1,) * num_non_data_axes + (seq_len,)
                tokens = np.zeros(shape, dtype=np.int32)
                doc_ids = np.zeros(shape, dtype=np.int32)
                yield (tokens, doc_ids)

        dataset = dummy_iterator()

    # Determine sharding strategy based on mesh dimensionality
    if len(mesh.axis_names) == 1:
        # Simple 1D data parallelism - no intermediate dimensions needed
        final_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(data_axis, None))
        needs_slice_and_reshard = False
    else:
        # Multi-dimensional mesh: create with all axes, then slice and reshard
        # Initial sharding: shard along ALL axes including non-data ones
        # This ensures only hosts with devices on the first slice of non-data axes load data
        initial_spec = [data_axis] + non_data_axes + [None]  # e.g., ('data', 'model', None)
        initial_sharding = jax.sharding.NamedSharding(
            mesh,
            jax.sharding.PartitionSpec(*initial_spec)
        )

        # Final sharding: only along data axis (broadcasts to all other axes)
        final_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(data_axis, None))
        needs_slice_and_reshard = True

    # Create JIT-compiled slice-and-reshard function for compute overlap
    # Uses final_sharding from closure
    @jax.jit
    def slice_and_reshard(tokens, doc_ids):
        """Slice out real data ([:, 0, ...]) and reshard to broadcast.

        This is the key step that extracts only the real data from the first
        slice of non-data axes and discards the dummy zeros, then broadcasts
        across all devices via resharding.
        """
        # Slice to extract first element along all non-data axes
        # Example: (batch, 1, seq) -> (batch, seq) via [:, 0, :]
        # Example: (batch, 1, 1, seq) -> (batch, seq) via [:, 0, 0, :]
        slice_indices = (slice(None),) + (0,) * num_non_data_axes + (slice(None),)
        tokens = tokens[slice_indices]
        doc_ids = doc_ids[slice_indices]

        # Reshard to broadcast across all non-data axes
        tokens = jax.device_put(tokens, final_sharding)
        doc_ids = jax.device_put(doc_ids, final_sharding)
        return tokens, doc_ids

    def distributed_iterator():
        """Generator that creates distributed arrays and handles slice-and-reshard broadcast."""
        for batch in dataset:
            tokens, doc_ids = batch

            if needs_slice_and_reshard:
                # Two-step process for 2D+ meshes (matching main3.py pattern):
                # Step 1: Create array with sharding across all axes
                #         Shape: (batch, 1, 1, ..., seq) with P('data', 'model', ..., None)
                #         Only hosts with (data=i, model=0, ...) actually load data
                tokens_array = jax.make_array_from_process_local_data(initial_sharding, tokens)
                doc_ids_array = jax.make_array_from_process_local_data(initial_sharding, doc_ids)

                # Step 2: Slice [:, 0, 0, ..., :] to extract real data and reshard to broadcast
                #         Shape: (batch, seq) with P('data', None)
                #         JIT enables compute overlap during resharding
                tokens_dist, doc_ids_dist = slice_and_reshard(
                    tokens_array, doc_ids_array
                )
            else:
                # Simple 1D case: directly create with final sharding
                tokens_dist = jax.make_array_from_process_local_data(final_sharding, tokens)
                doc_ids_dist = jax.make_array_from_process_local_data(final_sharding, doc_ids)

            # Support async prefetch pattern
            _ = yield (tokens_dist, doc_ids_dist)
            yield None

    # Create structure descriptors for global array shape
    # Global batch size = batch_size (per shard) × number of data-parallel shards
    data_axis_size = mesh.shape[data_axis]
    global_batch_size = batch_size * data_axis_size

    tokens_struct = jax.ShapeDtypeStruct((global_batch_size, seq_len), jnp.int32)
    document_ids_struct = jax.ShapeDtypeStruct((global_batch_size, seq_len), jnp.int32)

    return distributed_iterator(), (tokens_struct, document_ids_struct)