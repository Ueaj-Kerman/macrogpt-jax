#!/usr/bin/env python3
"""Test and demonstrate the configurator system."""

import sys
sys.path.insert(0, '.')

from ueaj.utils.configurator import config, override

# Test functions with @config decorator
@config
def embed(vocab_size: int, d: int, init: str = "normal"):
    return f"Embed(vocab={vocab_size}, d={d}, init={init})"

@config
def norm(d: int, eps: float = 1e-6):
    return f"Norm(d={d}, eps={eps})"

@config
def fp8_norm(d: int, eps: float = 1e-6, recentering: str = "none"):
    return f"FP8Norm(d={d}, eps={eps}, recentering={recentering})"

@config
def create_attention(d: int, q_proj=norm, k_proj=norm):
    q = q_proj(d)
    k = k_proj(d)
    return {'q': q, 'k': k}

@config
def linear(d_in: int, d_out: int, bias: bool = True, init: str = "xavier"):
    return f"Linear(in={d_in}, out={d_out}, bias={bias}, init={init})"

@config
def mlp(d: int, linear=linear, factor: int = 4, activation: str = "gelu"):
    gate = linear(d, d * factor)
    up = linear(d, d * factor)
    down = linear(d * factor, d)
    return {
        'gate': gate,
        'up': up,
        'down': down,
        'activation': activation,
        'factor': factor
    }

@config
def transformer_layer(d: int, attention=create_attention, mlp=mlp, norm=norm):
    attn = attention(d)
    ff = mlp(d)
    attn_norm = norm(d)
    mlp_norm = norm(d)
    return {
        'attention': attn,
        'mlp': ff,
        'attn_norm': attn_norm,
        'mlp_norm': mlp_norm,
    }

# Configure a transformer layer with specific settings
layer = transformer_layer.override(
    attention=create_attention.override(
        q_proj=norm.override(eps=1e-15),
        k_proj=norm.override(eps=1e-7),
    ),
    mlp=mlp.override(
        factor=8,
        activation="swish",
        linear=override(bias=False, init="kaiming"),
    ),
    norm=override(eps=1e-5),
)

result = layer(768)
print("Transformer layer config:")
print(f"  Attention: {result['attention']}")
print(f"  MLP: {result['mlp']}")
print(f"  Norms: attn={result['attn_norm']}, mlp={result['mlp_norm']}")

# --- Test Classes ---
print("\n--- Testing Class Configuration ---")

@config
class SimpleModel:
    def __init__(self, d: int, vocab_size: int = 32000, eps: float = 1e-6):
        self.d = d
        self.vocab_size = vocab_size
        self.eps = eps
        print(f"SimpleModel created: d={d}, vocab_size={vocab_size}, eps={eps}")

# Override the class to create a new configured class
SimpleModel_LargeVocab = SimpleModel.override(vocab_size=128000)

# Instantiate the overridden class
model_instance = SimpleModel_LargeVocab(d=512)

# Further override the already overridden class
SimpleModel_LargeVocab_LowEps = SimpleModel_LargeVocab.override(eps=1e-8)
model_instance_2 = SimpleModel_LargeVocab_LowEps(d=256)

# --- Test Cross-Compatibility ---
print("\n--- Testing Cross-Compatibility ---")

@config
class AdvancedModel:
    def __init__(self, d: int, embed_fn=embed, norm_fn=norm):
        self.embed = embed_fn(d=d, vocab_size=32000)
        self.norm = norm_fn(d=d)
        print(f"AdvancedModel created with:\n  embed={self.embed}\n  norm={self.norm}")

# Override with a function that has its own overrides
AdvancedModel_Configured = AdvancedModel.override(
    embed_fn=embed.override(init="kaiming"),
    norm_fn=norm.override(eps=1e-8)
)

adv_model_instance = AdvancedModel_Configured(d=1024)
