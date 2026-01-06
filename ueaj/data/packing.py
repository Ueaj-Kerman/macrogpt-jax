"""Document packing and padding utilities for sequence batching."""

import math
from typing import Iterator, Optional, Union, List, Dict, Tuple
from collections import defaultdict
import numpy as np
import jax.numpy as jnp


def padding_iterator(
    token_iterator: Iterator[Union[np.ndarray, jnp.ndarray, List[int]]],
    max_length: int,
    pad_token_id: int = 0,
    truncate: bool = True
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Pad/truncate individual sequences to fixed length without packing.

    Each sequence becomes its own batch element, padded or truncated to max_length.
    Document IDs are 1 for actual tokens and 0 for padding (matching pack_documents format).

    Args:
        token_iterator: Iterator yielding token sequences
        max_length: Target sequence length (pad shorter, optionally truncate longer)
        pad_token_id: Token ID to use for padding
        truncate: If True, truncate sequences longer than max_length.
                  If False, split long sequences into multiple chunks.

    Yields:
        Tuples of (tokens, document_ids) where:
        - tokens: np.ndarray of shape (max_length,) with token IDs
        - document_ids: np.ndarray of shape (max_length,) where 1 = real token, 0 = padding
    """
    for seq in token_iterator:
        # Convert to numpy if needed
        if isinstance(seq, list):
            seq = np.array(seq, dtype=np.int32)
        elif isinstance(seq, jnp.ndarray):
            seq = np.array(seq, dtype=np.int32)
        elif seq.dtype != np.int32:
            seq = seq.astype(np.int32)

        if len(seq) == 0:
            continue

        if truncate:
            # Simple truncation mode: yield one sequence per input
            tokens = np.full((max_length,), pad_token_id, dtype=np.int32)
            document_ids = np.zeros((max_length,), dtype=np.int32)

            actual_len = min(len(seq), max_length)
            tokens[:actual_len] = seq[:actual_len]
            document_ids[:actual_len] = 1

            yield tokens, document_ids
        else:
            # Split mode: yield multiple chunks for long sequences
            for start in range(0, len(seq), max_length):
                chunk = seq[start:start + max_length]

                tokens = np.full((max_length,), pad_token_id, dtype=np.int32)
                document_ids = np.zeros((max_length,), dtype=np.int32)

                tokens[:len(chunk)] = chunk
                document_ids[:len(chunk)] = 1

                yield tokens, document_ids


def pack_documents(
	token_iterator: Iterator[Union[np.ndarray, jnp.ndarray, List[int]]],
	max_length: int,
	pad_token_id: Optional[int] = -1,
	min_fill_ratio: float = 0.96875,
	buffer_factor: int = 8
) -> Iterator[tuple[np.ndarray, np.ndarray]]:

	"""Pack documents into batches using power-of-2 bucketing.
	"""
	bucket_idx = lambda x: math.ceil(math.log2(x))
	buckets = range(0, bucket_idx(max_length)+1)
	buckets = {k: [] for k in buckets}
	current_size, max_size = [0], buffer_factor * max_length

	def pack_buckets():
		if pad_token_id is None:
			tokens = np.full((max_length,), -1, dtype=np.int32)
		else:
			tokens = np.full((max_length,), pad_token_id, dtype=np.int32)

		documents = np.zeros((max_length,), dtype=np.int32)
		index, doc_id = 0, 1

		# phase 1: pack whole documents from largest to smallest
		for bucket in reversed(buckets):
			sequences = buckets[bucket]
			for i, seq in reversed(list(enumerate(sequences))):
				if len(seq) + index > max_length:
					continue
				tokens[index:index + len(seq)] = seq
				documents[index:index + len(seq)] = doc_id
				index += len(seq)
				doc_id += 1
				sequences.pop(i)
				break

			if index / max_length >= min_fill_ratio:
				break

		if index / max_length >= min_fill_ratio:
			current_size[0] -= index
			if pad_token_id is not None:
				return tokens, documents
			else:
				return tokens[:index], documents[:index]

		# phase 2: prefer breaking up documents but avoid splitting too finely
		for bucket in reversed(buckets):
			sequences = buckets[bucket]
			for i, seq in reversed(list(enumerate(sequences))):
				rmin, rmax = math.ceil(max_length * min_fill_ratio) - index, max_length - index
				seq_len = len(seq)
				min_perc, max_perc = rmin / seq_len, rmax / seq_len
				if min_perc > .5 or max_perc < .5:
					continue

				splice_len = seq_len // 2
				splice = seq[:splice_len]

				tokens[index:index + len(splice)] = splice
				documents[index:index + len(splice)] = doc_id
				index += len(splice)
				doc_id += 1

				sequences.pop(i)

				sub_seq = seq[splice_len:]
				if len(sub_seq) > 0:
					buckets[bucket_idx(len(sub_seq))].append(sub_seq)
				break

			if index / max_length >= min_fill_ratio:
				break

		if index / max_length >= min_fill_ratio:
			current_size[0] -= index
			if pad_token_id is not None:
				return tokens, documents
			else:
				return tokens[:index], documents[:index]

		# phase 3: pack remaining documents
		for bucket in reversed(buckets):
			sequences = buckets[bucket]
			for i, seq in reversed(list(enumerate(sequences))):
				rmin, rmax = math.ceil(max_length * min_fill_ratio) - index, max_length - index
				seq_len = len(seq)
				min_perc, max_perc = rmin / seq_len, rmax / seq_len

				if min(min_perc, 1-min_perc) > min(max_perc, 1-max_perc):
					split_ratio = min_perc
				else:
					split_ratio = max_perc

				splice_len = min(math.ceil(seq_len * split_ratio), max_length - index)
				splice = seq[:splice_len]

				tokens[index:index + len(splice)] = splice
				documents[index:index + len(splice)] = doc_id
				index += len(splice)
				doc_id += 1

				sequences.pop(i)
				sub_seq = seq[splice_len:]
				if len(sub_seq) > 0:
					buckets[bucket_idx(len(sub_seq))].append(sub_seq)

				if index / max_length >= min_fill_ratio:
					current_size[0] -= index
					if pad_token_id is not None:
						return tokens, documents
					else:
						return tokens[:index], documents[:index]
		if index / max_length < min_fill_ratio:
			print("WARNING: min_fill_ratio not met!")

		current_size[0] -= index
		if pad_token_id is not None:
			return tokens, documents
		else:
			return tokens[:index], documents[:index]


	for batch in token_iterator:
		if not isinstance(batch[0], (list, np.ndarray, jnp.ndarray)):
			batch = [np.array(batch)]
		elif isinstance(batch[0], list):
			batch = [np.array(seq) for seq in batch]

		for seq in batch:
			if len(seq) == 0:
				continue

			splits = np.array_split(seq, math.ceil(len(seq) / max_length))
			for split in splits:
				length = len(split)
				bucket = bucket_idx(length)
				buckets[bucket].append(split)
				current_size[0] += len(split)

		while current_size[0] >= max_size:
			yield pack_buckets()
	
	# Process remaining sequences after iterator is exhausted
	while any(len(buckets[b]) > 0 for b in buckets):
		yield pack_buckets()


def test_pack_documents():
	"""Comprehensive test cases for pack_documents function."""
	
	print("Testing pack_documents with power-of-2 bucketing")
	print("=" * 60)
	
	# Test 1: Basic packing with single sequences
	print("\nTest 1: Basic single sequence packing")
	print("-" * 40)
	
	def single_seq_iterator():
		yield np.array([1, 2, 3, 4, 5], dtype=np.int32)
		yield np.array([6, 7, 8], dtype=np.int32)
		yield np.array([9, 10], dtype=np.int32)
	
	results = list(pack_documents(single_seq_iterator(), max_length=16, min_fill_ratio=0.5, pad_token_id=-1))
	print(f"Input: 3 sequences of lengths [5, 3, 2]")
	print(f"Max length: 16, min_fill_ratio: 0.5")
	print(f"Number of outputs: {len(results)}")
	
	for i, (tokens, doc_ids) in enumerate(results):
		print(f"\nOutput {i}:")
		print(f"  Tokens: {tokens}")
		print(f"  Doc IDs: {doc_ids}")
		print(f"  Fill ratio: {np.sum(doc_ids > 0) / len(tokens):.2%}")
	
	# Test 2: Long document splitting
	print("\n\nTest 2: Long document splitting")
	print("-" * 40)
	
	def long_doc_iterator():
		yield np.arange(100, 150, dtype=np.int32)  # 50 tokens
	
	results = list(pack_documents(long_doc_iterator(), max_length=16, min_fill_ratio=0.8))
	print(f"Input: 1 sequence of length 50")
	print(f"Max length: 16, min_fill_ratio: 0.8")
	print(f"Number of outputs: {len(results)}")
	
	for i, (tokens, doc_ids) in enumerate(results):
		non_pad = doc_ids > 0
		print(f"\nOutput {i}:")
		print(f"  Tokens (non-pad): {tokens[non_pad]}")
		print(f"  Unique doc IDs: {np.unique(doc_ids[non_pad])}")
		print(f"  Fill ratio: {np.sum(non_pad) / len(tokens):.2%}")
	
	# Test 3: Power-of-2 bucketing efficiency
	print("\n\nTest 3: Power-of-2 bucketing efficiency")
	print("-" * 40)
	
	def power_of_2_iterator():
		# Generate sequences with lengths near powers of 2
		lengths = [31, 32, 33, 63, 64, 65, 127, 128]
		for i, length in enumerate(lengths):
			yield np.full(length, i + 1, dtype=np.int32)
	
	results = list(pack_documents(power_of_2_iterator(), max_length=256, min_fill_ratio=0.9))
	print(f"Input: sequences with lengths near powers of 2: [31, 32, 33, 63, 64, 65, 127, 128]")
	print(f"Max length: 256, min_fill_ratio: 0.9")
	print(f"Number of outputs: {len(results)}")
	
	for i, (tokens, doc_ids) in enumerate(results):
		unique_docs = np.unique(doc_ids[doc_ids > 0])
		print(f"\nOutput {i}:")
		print(f"  Documents packed: {unique_docs}")
		print(f"  Fill ratio: {np.sum(doc_ids > 0) / len(tokens):.2%}")
	
	# Test 4: Padding behavior
	print("\n\nTest 4: Padding behavior")
	print("-" * 40)
	
	def padding_test_iterator():
		yield np.array([1, 2, 3], dtype=np.int32)
		yield np.array([4, 5], dtype=np.int32)
	
	# Test with custom pad token
	results_custom = list(pack_documents(
		padding_test_iterator(), 
		max_length=8, 
		pad_token_id=999,
		min_fill_ratio=0.5
	))
	
	# Test with default pad token (None)
	padding_test_iterator2 = lambda: (np.array([1, 2, 3], dtype=np.int32), np.array([4, 5], dtype=np.int32))
	results_default = list(pack_documents(
		(x for x in [np.array([1, 2, 3]), np.array([4, 5])]),
		max_length=8,
		pad_token_id=None,
		min_fill_ratio=0.5
	))
	
	print("With pad_token_id=999:")
	tokens, doc_ids = results_custom[0]
	print(f"  Tokens: {tokens}")
	print(f"  Padding values: {tokens[doc_ids == 0]}")
	
	print("\nWith pad_token_id=None (default -1):")
	tokens, doc_ids = results_default[0]
	print(f"  Tokens: {tokens}")
	print(f"  Length: {len(tokens)} (may be shorter than max_length)")
	
	# Test 5: Buffer factor behavior
	print("\n\nTest 5: Buffer factor behavior")
	print("-" * 40)
	
	def many_sequences_iterator():
		for i in range(20):
			yield np.full(10, i + 1, dtype=np.int32)
	
	results = list(pack_documents(
		many_sequences_iterator(),
		max_length=32,
		min_fill_ratio=0.8,
		buffer_factor=4  # Should trigger packing when buffer reaches 4 * 32 = 128 tokens
	))
	
	print(f"Input: 20 sequences of length 10 each")
	print(f"Max length: 32, buffer_factor: 4 (triggers at 128 tokens)")
	print(f"Number of outputs: {len(results)}")
	
	# Test 6: Edge cases
	print("\n\nTest 6: Edge cases")
	print("-" * 40)
	
	# Empty iterator
	results = list(pack_documents(iter([]), max_length=16))
	print(f"Empty iterator: {len(results)} outputs")
	
	# Single token sequences
	def single_token_iterator():
		for i in range(5):
			yield np.array([i], dtype=np.int32)
	
	results = list(pack_documents(single_token_iterator(), max_length=16, min_fill_ratio=0.3))
	print(f"\n5 single-token sequences: {len(results)} outputs")
	if results:
		tokens, doc_ids = results[0]
		print(f"  First output: {tokens[doc_ids > 0]}")
	
	# Exactly max_length sequence
	results = list(pack_documents(
		iter([np.arange(16, dtype=np.int32)]),
		max_length=16,
		min_fill_ratio=0.9
	))
	print(f"\nSequence exactly max_length (16): {len(results)} outputs")
	if results:
		tokens, doc_ids = results[0]
		print(f"  Fill ratio: {np.sum(doc_ids > 0) / len(tokens):.2%}")
	
	# Test 7: Document ID consistency
	print("\n\nTest 7: Document ID consistency")
	print("-" * 40)
	
	def mixed_iterator():
		yield np.array([1, 2, 3, 4, 5], dtype=np.int32)
		yield np.array([10, 20, 30], dtype=np.int32)
		yield np.array([100, 200], dtype=np.int32)
	
	results = list(pack_documents(mixed_iterator(), max_length=16, min_fill_ratio=0.6))
	
	print("Checking that document IDs are properly assigned:")
	all_tokens = []
	all_doc_ids = []
	
	for tokens, doc_ids in results:
		mask = doc_ids > 0
		all_tokens.extend(tokens[mask])
		all_doc_ids.extend(doc_ids[mask])
	
	print(f"Tokens by document:")
	for doc_id in sorted(set(all_doc_ids)):
		doc_tokens = np.array([t for t, d in zip(all_tokens, all_doc_ids) if d == doc_id])
		print(f"  Doc {doc_id}: {doc_tokens}")
	
	print("\n" + "=" * 60)
	print("All tests completed!")


if __name__ == "__main__":
	test_pack_documents()


