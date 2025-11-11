"""Example: Checkpointable Grain + HuggingFace pipeline.

Demonstrates how to build a distributed data pipeline with checkpointing support.
"""

import jax
import jax.numpy as jnp
import numpy as np
import datasets
import grain.python as grain
from transformers import AutoTokenizer

from ueaj.data.checkpoint import CheckpointableStreamSource, checkpoint

# Configuration
SEQ_LEN = 2048
BATCH_SIZE = 4
NUM_DATA_HOSTS = 4
HOST_INDEX = 0  # In real code, get from JAX's process_index


def build_checkpointable_pipeline():
    """Build a Grain pipeline with full checkpointing support."""

    # 1. Load HuggingFace streaming dataset with sharding
    hf_dataset = datasets.load_dataset(
        'HuggingFaceFW/fineweb',
        name='sample-10BT',
        split='train',
        streaming=True
    ).shard(num_shards=NUM_DATA_HOSTS, index=HOST_INDEX)

    # 2. Wrap in checkpointable source
    source = CheckpointableStreamSource(hf_dataset)

    # 3. Build Grain pipeline
    dataset = grain.MapDataset.source(source)

    # 4. Tokenize
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        tokens = tokenizer(
            example['text'],
            truncation=True,
            max_length=SEQ_LEN * 2,  # Allow some headroom
            return_attention_mask=False
        )['input_ids']
        return {
            'tokens': np.array(tokens, dtype=np.int32)
        }

    dataset = dataset.map(tokenize)

    # 5. Add document IDs
    def add_doc_ids(index, example):
        return {
            **example,
            'document_ids': np.full(len(example['tokens']), index, dtype=np.int32)
        }

    dataset = dataset.map_with_index(add_doc_ids)

    # 6. Convert to iter dataset for packing
    dataset = dataset.to_iter_dataset()

    # 7. Pack sequences
    dataset = grain.experimental.ConcatThenSplitIterDataset(
        parent=dataset,
        length_struct={'tokens': SEQ_LEN, 'document_ids': SEQ_LEN}
    )

    # 8. Batch
    dataset = dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    return dataset, source


def save_checkpoint(source: CheckpointableStreamSource, step: int, path: str):
    """Save checkpoint for data pipeline.

    Args:
        source: The CheckpointableStreamSource from the pipeline
        step: Current training step
        path: Path to save checkpoint
    """
    state = {
        'step': step,
        'source': source.state_dict(),
        # Add other pipeline state here...
    }

    checkpoint.save_to_file(state, path)
    print(f"✓ Saved checkpoint at step {step} to {path}")


def load_checkpoint(path: str) -> tuple[int, CheckpointableStreamSource]:
    """Load checkpoint and recreate pipeline.

    Args:
        path: Path to checkpoint file

    Returns:
        (step, source) tuple
    """
    state = checkpoint.load_from_file(path)

    # Recreate source from checkpoint
    source = CheckpointableStreamSource.from_state_dict(
        state['source'],
        source_factory=lambda: datasets.load_dataset(
            'HuggingFaceFW/fineweb',
            name='sample-10BT',
            split='train',
            streaming=True
        ).shard(num_shards=NUM_DATA_HOSTS, index=HOST_INDEX)
    )

    print(f"✓ Restored checkpoint from step {state['step']}")
    return state['step'], source


def train_with_checkpointing():
    """Example training loop with checkpointing."""

    # Build pipeline
    dataset, source = build_checkpointable_pipeline()
    iterator = iter(dataset)

    # Or restore from checkpoint
    # start_step, source = load_checkpoint('checkpoint.json')
    # dataset, _ = build_checkpointable_pipeline()  # Rebuild with restored source
    # iterator = iter(dataset)
    start_step = 0

    # Training loop
    for step in range(start_step, 1000):
        # Get batch
        batch = next(iterator)

        # Simulate training
        print(f"Step {step}: batch shape = {batch['tokens'].shape}")

        # Checkpoint every 100 steps
        if (step + 1) % 100 == 0:
            save_checkpoint(source, step + 1, f'checkpoint_step_{step + 1}.json')


def example_custom_serializer():
    """Example: Register custom serializer for your own classes."""

    class MyDataset:
        def __init__(self, data):
            self.data = data
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.data):
                raise StopIteration
            item = self.data[self.index]
            self.index += 1
            return item

        def to_dict(self):
            return {'data': self.data, 'index': self.index}

        @classmethod
        def from_dict(cls, state):
            obj = cls(state['data'])
            obj.index = state['index']
            return obj

    # Register serializer
    checkpoint.register(
        type_=MyDataset,
        serialize_fn=lambda obj: obj.to_dict(),
        deserialize_fn=lambda data: MyDataset.from_dict(data),
        name='my_dataset'
    )

    # Use it
    dataset = MyDataset([1, 2, 3, 4, 5])
    next(dataset)  # Advance to index 1

    # Save and restore
    state = checkpoint.save(dataset)
    restored = checkpoint.load(state)

    print(f"Original index: {dataset.index}")
    print(f"Restored index: {restored.index}")
    print(f"Next item: {next(restored)}")  # Should be 2


if __name__ == '__main__':
    # Example 1: Basic checkpointing
    print("=== Example 1: Training with checkpointing ===")
    # Uncomment to run:
    # train_with_checkpointing()

    # Example 2: Custom serializer
    print("\n=== Example 2: Custom serializer ===")
    example_custom_serializer()

    print("\n=== Example 3: Direct source checkpointing ===")
    # Build just the source
    hf_dataset = datasets.load_dataset(
        'HuggingFaceFW/fineweb',
        name='sample-10BT',
        split='train',
        streaming=True
    ).take(100)  # Small subset for demo

    source = CheckpointableStreamSource(hf_dataset)

    # Consume some items
    for i in range(5):
        item = source[i]  # Note: index ignored for streaming
        print(f"Item {i}: text length = {len(item['text'])}")

    # Checkpoint
    state = source.state_dict()
    print(f"\nCheckpoint state: {state}")

    # Note: Full restoration requires source_factory
    # because HF streaming datasets need special handling
