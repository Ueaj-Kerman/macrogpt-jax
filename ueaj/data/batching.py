"""Batching utilities for creating fixed-size batches from iterators."""

from typing import Iterator, TypeVar, Generator, Optional, Tuple
import numpy as np
import jax.numpy as jnp

T = TypeVar('T')


def batch_iterator(
    iterator: Iterator[T],
    batch_size: int,
    drop_last: bool = False,
    collate_fn: Optional[callable] = None
) -> Generator[T, None, None]:
    """
    Create fixed-size batches from an iterator.
    
    Args:
        iterator: Source iterator yielding individual items
        batch_size: Number of items per batch
        drop_last: If True, drop the last batch if it's smaller than batch_size
        collate_fn: Optional function to collate items into a batch.
                   If None, items are collected into a list.
    
    Yields:
        Batches of items. Format depends on collate_fn.
    
    Example:
        >>> data = iter(range(10))
        >>> for batch in batch_iterator(data, batch_size=3):
        ...     print(batch)
        [0, 1, 2]
        [3, 4, 5]
        [6, 7, 8]
        [9]
    """
    batch = []
    
    for item in iterator:
        batch.append(item)
        
        if len(batch) == batch_size:
            if collate_fn is not None:
                yield collate_fn(batch)
            else:
                yield batch
            batch = []
    
    # Handle remaining items
    if batch and not drop_last:
        if collate_fn is not None:
            yield collate_fn(batch)
        else:
            yield batch


def numpy_collate(batch: list) -> dict:
    """
    Collate a batch of dictionaries containing numpy arrays.
    
    Stacks arrays along a new batch dimension (axis 0).
    
    Args:
        batch: List of dictionaries with numpy array values
    
    Returns:
        Dictionary with stacked arrays
    
    Example:
        >>> batch = [
        ...     {'tokens': np.array([1, 2, 3]), 'mask': np.array([1, 1, 0])},
        ...     {'tokens': np.array([4, 5, 6]), 'mask': np.array([1, 1, 1])}
        ... ]
        >>> result = numpy_collate(batch)
        >>> result['tokens'].shape
        (2, 3)
    """
    if not batch:
        return {}
    
    # Get keys from first item
    keys = batch[0].keys()
    
    # Stack arrays for each key
    result = {}
    for key in keys:
        arrays = [item[key] for item in batch]
        result[key] = np.stack(arrays, axis=0)
    
    return result


def tuple_collate(batch: list) -> Tuple:
    """
    Collate a batch of tuples by stacking corresponding elements.
    
    Args:
        batch: List of tuples with array elements
    
    Returns:
        Tuple with stacked arrays
    
    Example:
        >>> batch = [
        ...     (np.array([1, 2, 3]), np.array([1, 1, 1])),
        ...     (np.array([4, 5, 6]), np.array([2, 2, 2]))
        ... ]
        >>> tokens, doc_ids = tuple_collate(batch)
        >>> tokens.shape
        (2, 3)
    """
    if not batch:
        return ()
    
    # Transpose list of tuples to tuple of lists
    transposed = list(zip(*batch))
    
    # Stack each element
    result = []
    for elements in transposed:
        if isinstance(elements[0], (np.ndarray, jnp.ndarray)):
            result.append(np.stack(elements, axis=0))
        else:
            result.append(elements)
    
    return tuple(result)


def padded_batch_iterator(
    iterator: Iterator[np.ndarray],
    batch_size: int,
    pad_value: int = 0,
    drop_last: bool = False
) -> Generator[np.ndarray, None, None]:
    """
    Create batches with automatic padding to the longest sequence in each batch.
    
    Args:
        iterator: Iterator yielding numpy arrays of varying lengths
        batch_size: Number of sequences per batch
        pad_value: Value to use for padding
        drop_last: If True, drop the last batch if it's smaller than batch_size
    
    Yields:
        Batched and padded arrays of shape (batch_size, max_length)
    
    Example:
        >>> sequences = [
        ...     np.array([1, 2, 3]),
        ...     np.array([4, 5]),
        ...     np.array([6, 7, 8, 9])
        ... ]
        >>> for batch in padded_batch_iterator(iter(sequences), batch_size=2):
        ...     print(batch)
        [[1 2 3]
         [4 5 0]]
        [[6 7 8 9]]
    """
    def pad_collate(batch):
        # Find maximum length in batch
        max_len = max(len(seq) for seq in batch)
        
        # Pad sequences
        padded = []
        for seq in batch:
            if len(seq) < max_len:
                pad_width = max_len - len(seq)
                padded_seq = np.pad(seq, (0, pad_width), constant_values=pad_value)
            else:
                padded_seq = seq
            padded.append(padded_seq)
        
        return np.stack(padded, axis=0)
    
    yield from batch_iterator(iterator, batch_size, drop_last, collate_fn=pad_collate)


if __name__ == "__main__":
    """Test the batching functionality with various scenarios."""
    
    print("="*60)
    print("Testing Batching Utilities")
    print("="*60)
    
    # Test 1: Basic batching with lists
    print("\n1. Basic batching")
    print("-" * 40)
    
    data = list(range(10))
    batches = list(batch_iterator(iter(data), batch_size=3))
    
    print(f"Input: {data}")
    print(f"Batch size: 3")
    print(f"Batches: {batches}")
    print(f"Last batch size: {len(batches[-1])}")
    
    # Test with drop_last
    batches_dropped = list(batch_iterator(iter(data), batch_size=3, drop_last=True))
    print(f"\nWith drop_last=True: {batches_dropped}")
    
    # Test 2: Dictionary batching
    print("\n2. Dictionary batching with numpy arrays")
    print("-" * 40)
    
    def dict_iterator():
        for i in range(5):
            yield {
                'tokens': np.array([i, i+1, i+2]),
                'labels': np.array([i*10, (i+1)*10, (i+2)*10])
            }
    
    batches = list(batch_iterator(dict_iterator(), batch_size=2, collate_fn=numpy_collate))
    
    print(f"Number of batches: {len(batches)}")
    for i, batch in enumerate(batches):
        print(f"\nBatch {i}:")
        for key, value in batch.items():
            print(f"  {key}: shape={value.shape}")
            print(f"    {value}")
    
    # Test 3: Tuple batching (like pack_documents output)
    print("\n3. Tuple batching (pack_documents style)")
    print("-" * 40)
    
    def tuple_iterator():
        for i in range(4):
            tokens = np.arange(i*3, (i+1)*3, dtype=np.int32)
            doc_ids = np.full_like(tokens, i+1)
            yield (tokens, doc_ids)
    
    batches = list(batch_iterator(tuple_iterator(), batch_size=2, collate_fn=tuple_collate))
    
    print(f"Number of batches: {len(batches)}")
    for i, (tokens, doc_ids) in enumerate(batches):
        print(f"\nBatch {i}:")
        print(f"  Tokens: {tokens}")
        print(f"  Doc IDs: {doc_ids}")
    
    # Test 4: Padded batching
    print("\n4. Padded batching with variable length sequences")
    print("-" * 40)
    
    sequences = [
        np.array([1, 2, 3], dtype=np.int32),
        np.array([4, 5], dtype=np.int32),
        np.array([6, 7, 8, 9], dtype=np.int32),
        np.array([10], dtype=np.int32),
        np.array([11, 12, 13, 14, 15], dtype=np.int32)
    ]
    
    print("Input sequences:")
    for i, seq in enumerate(sequences):
        print(f"  {i}: {seq} (length {len(seq)})")
    
    batches = list(padded_batch_iterator(iter(sequences), batch_size=2, pad_value=-1))
    
    print(f"\nBatched with padding (batch_size=2, pad_value=-1):")
    for i, batch in enumerate(batches):
        print(f"\nBatch {i}: shape={batch.shape}")
        print(batch)
    
    # Test 5: Integration with pack_documents
    print("\n5. Integration example with pack_documents")
    print("-" * 40)

    from ueaj.data.packing import pack_documents

    def doc_iterator():
        yield np.array([1, 2, 3, 4, 5], dtype=np.int32)
        yield np.array([6, 7, 8], dtype=np.int32)
        yield np.array([9, 10, 11, 12], dtype=np.int32)
        yield np.array([13, 14], dtype=np.int32)

    # Pack documents (linear packing)
    packed = pack_documents(doc_iterator(), max_length=8)

    # Batch the packed sequences
    batches = list(batch_iterator(packed, batch_size=2, collate_fn=tuple_collate))

    print(f"Packed and batched sequences:")
    for i, (tokens, doc_ids) in enumerate(batches):
        print(f"\nBatch {i}: shape={tokens.shape}")
        print(f"  Tokens: {tokens}")
        print(f"  Doc IDs: {doc_ids}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)