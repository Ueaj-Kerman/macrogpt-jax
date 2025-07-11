"""Dataset construction utilities for training."""

import jax
import numpy as np
from typing import Callable, Optional, Iterator, Tuple, Any, Generator
from . import pack_documents, batch_iterator, device_prefetch, tuple_collate


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