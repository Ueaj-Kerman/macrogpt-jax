import datasets
import numpy as np
import grain
import transformers

seq_len = 2 ** 16
tokenizer = transformers.GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
tokenizer.model_max_length = seq_len  # Set to 4096
print("Vocab Size:", tokenizer.vocab_size)

dataset = datasets.load_dataset(
	"HuggingFaceFW/fineweb-edu",
	name="sample-10BT",
	split="train",
	streaming=True,
)

dataset.from_pandas()
import sys
sys.exit(0)

print("Loaded dataset")
hf_dataset = dataset.shard(num_shards=16, index=0)

print("Sharded dataset")

# 2. Create Grain MapDataset
dataset = grain.IterDataset.source(hf_dataset)

# 3. Tokenize
dataset = dataset.map(lambda ex: {'tokens': np.array(tokenizer(ex['text'])['input_ids'])})

# 4. Add document IDs
dataset = dataset.map_with_index(
	lambda idx, ex: {**ex, 'document_ids': np.full(len(ex['tokens']), idx, dtype=np.int32)}
)

# 5. Convert to IterDataset & pack
dataset = dataset.to_iter_dataset()
dataset = grain.experimental.ConcatThenSplitIterDataset(
	parent=dataset,
	length_struct={'tokens': seq_len, 'document_ids': seq_len}
)

# 6. Batch
dataset = dataset.batch(1)

for ex in dataset:
	print(ex)
