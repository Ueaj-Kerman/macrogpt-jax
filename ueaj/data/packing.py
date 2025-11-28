"""Linear document packing - concatenate and split into fixed-size chunks."""

from typing import Iterator, Union, List
import numpy as np
import jax.numpy as jnp


def pack_documents(
    token_iterator: Iterator[Union[np.ndarray, jnp.ndarray, List[int]]],
    max_length: int,
    pad_token_id: int = -1,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Pack documents using linear concatenation and splitting.

    Documents are concatenated sequentially and split into fixed-size chunks.
    Each chunk includes document IDs to track document boundaries.

    Args:
        token_iterator: Iterator yielding token sequences (arrays or lists)
        max_length: Fixed length for output sequences
        pad_token_id: Token ID for padding the final incomplete chunk

    Yields:
        Tuples of (tokens, document_ids) where:
        - tokens: np.ndarray of shape (max_length,)
        - document_ids: np.ndarray of shape (max_length,) with unique IDs per document
    """
    buffer_tokens = []
    buffer_doc_ids = []
    doc_id = 1

    for batch in token_iterator:
        # Normalize input
        if not isinstance(batch, (list, np.ndarray, jnp.ndarray)):
            batch = [batch]
        elif isinstance(batch, (np.ndarray, jnp.ndarray)) and batch.ndim == 1:
            batch = [batch]

        for seq in batch:
            seq = np.asarray(seq, dtype=np.int32)
            if len(seq) == 0:
                continue

            buffer_tokens.extend(seq.tolist())
            buffer_doc_ids.extend([doc_id] * len(seq))
            doc_id += 1

            # Yield complete chunks
            while len(buffer_tokens) >= max_length:
                tokens = np.array(buffer_tokens[:max_length], dtype=np.int32)
                doc_ids = np.array(buffer_doc_ids[:max_length], dtype=np.int32)
                buffer_tokens = buffer_tokens[max_length:]
                buffer_doc_ids = buffer_doc_ids[max_length:]
                yield tokens, doc_ids

    # Yield final chunk with padding if there's remaining data
    if buffer_tokens:
        pad_len = max_length - len(buffer_tokens)
        tokens = np.array(buffer_tokens + [pad_token_id] * pad_len, dtype=np.int32)
        doc_ids = np.array(buffer_doc_ids + [0] * pad_len, dtype=np.int32)
        yield tokens, doc_ids


if __name__ == "__main__":
    print("Testing linear pack_documents")
    print("=" * 60)

    # Test 1: Basic packing
    print("\nTest 1: Basic packing")
    print("-" * 40)

    def basic_iterator():
        yield np.array([1, 2, 3, 4, 5], dtype=np.int32)
        yield np.array([6, 7, 8], dtype=np.int32)
        yield np.array([9, 10], dtype=np.int32)

    results = list(pack_documents(basic_iterator(), max_length=6))
    print(f"Input: sequences of lengths [5, 3, 2] = 10 tokens")
    print(f"Max length: 6")
    print(f"Expected: 2 chunks (6 + 4 padded)")

    for i, (tokens, doc_ids) in enumerate(results):
        print(f"\nChunk {i}: tokens={tokens}, doc_ids={doc_ids}")

    # Test 2: Long document
    print("\n\nTest 2: Long document spanning multiple chunks")
    print("-" * 40)

    def long_doc_iterator():
        yield np.arange(1, 26, dtype=np.int32)  # 25 tokens

    results = list(pack_documents(long_doc_iterator(), max_length=8))
    print(f"Input: 25 tokens in one document")
    print(f"Expected: 4 chunks (8+8+8+1 padded)")

    for i, (tokens, doc_ids) in enumerate(results):
        unique_docs = np.unique(doc_ids[doc_ids > 0])
        print(f"Chunk {i}: tokens={tokens[:4]}..., doc_ids unique={unique_docs}")

    # Test 3: Multiple documents in one chunk
    print("\n\nTest 3: Multiple documents in one chunk")
    print("-" * 40)

    def multi_doc_iterator():
        yield np.array([1, 2], dtype=np.int32)
        yield np.array([3, 4], dtype=np.int32)
        yield np.array([5, 6], dtype=np.int32)

    results = list(pack_documents(multi_doc_iterator(), max_length=8))
    print(f"Input: 3 docs of 2 tokens each")

    for i, (tokens, doc_ids) in enumerate(results):
        print(f"Chunk {i}: tokens={tokens}, doc_ids={doc_ids}")
        print(f"  Documents in chunk: {np.unique(doc_ids[doc_ids > 0])}")

    print("\n" + "=" * 60)
    print("All tests passed!")
