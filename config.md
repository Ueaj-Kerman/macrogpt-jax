
```python
from configurator import config, override

@config
def llama(
    model_d,
    vocab_size,
    num_layers=12,
    embed=create_embed,
    norm=create_norm,
    ...
):
    ...

@config
def fp8_norm(...):
    ...

llama_fn = llama.override(
    # functools.partial equivalent
	vocab_size=65535,
    # override regular args
    num_layers=24,
	# override args without changing the function
    embed=override(dtype=jnp.bfloat16),
    # override args and function
    norm=fp8_norm.override(recentering="recenter"),
)

# call the function
model = llama_fn(1024)
```