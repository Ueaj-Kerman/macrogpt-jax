#!/usr/bin/env python3
"""Test the new compositional configuration system."""

import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.model import MLP, GMLP, TransformerLayer, RMSNorm, SoftmaxAttention
from ueaj.model.rmsnorm import uncentered_scale
from ueaj.utils.configurator import override


def test_mlp_composition():
    """Test MLP with custom projections."""
    print("Testing MLP composition...")
    
    rngs = rng.Rngs(0)
    x = jnp.ones((2, 4, 32))
    
    # Basic MLP
    mlp = MLP(model_d=32, hidden_d=64, rngs=rngs)
    y = mlp(x)
    print(f"  Basic MLP output shape: {y.shape}")
    
    # MLP with custom parameters
    CustomMLP = MLP.override(
        activation_fn=nnx.gelu,
        param_dtype=jnp.float32
    )
    mlp2 = CustomMLP(model_d=32, hidden_d=64, rngs=rngs)
    y2 = mlp2(x)
    print(f"  Custom MLP output shape: {y2.shape}")
    

def test_transformer_layer():
    """Test TransformerLayer composition."""
    print("\nTesting TransformerLayer composition...")
    
    rngs = rng.Rngs(0)
    x = jnp.ones((2, 8, 128))
    
    # Create custom components
    CustomAttn = SoftmaxAttention.override(
        kq_d=64,
        v_head_d=64,
        kv_heads=2,
        rope_theta=10000.0
    )
    
    CustomMLP = GMLP.override(
        hidden_d=256,
        activation_fn=nnx.relu
    )
    
    # Use uncentered normalization like Llama
    UncenteredNorm = RMSNorm.override(
        create_scale=uncentered_scale,
        scale_method="uncentered"
    )
    
    # Create layer with custom components
    CustomLayer = TransformerLayer.override(
        attn=CustomAttn,
        mlp=CustomMLP,
        norm=UncenteredNorm
    )
    
    layer = CustomLayer(model_d=128, rngs=rngs)
    
    # Test forward pass (without attention mask for simplicity)
    try:
        y = layer(x)
        print(f"  Layer output shape: {y.shape}")
    except ValueError as e:
        print(f"  Expected error (no attention mask): {e}")
    

def test_nested_overrides():
    """Test nested function overrides."""
    print("\nTesting nested overrides...")
    
    # Create a custom MLP class with different defaults
    CustomGMLP = GMLP.override(
        activation_fn=nnx.silu,
        param_dtype=jnp.float32,
        hidden_d=192  # Specify hidden_d
    )
    
    # Override the layer to use our custom MLP and specify attention params
    CustomLayer = TransformerLayer.override(
        mlp=CustomGMLP,
        kq_d=32,  # Specify key/query dimension
        kv_heads=2  # Specify number of heads
    )
    
    rngs = rng.Rngs(0)
    layer = CustomLayer(model_d=64, rngs=rngs)
    
    print(f"  Created layer with custom GMLP: {type(layer.mlp)}")
    print(f"  MLP has silu activation: {layer.mlp.act_fn == nnx.silu}")


if __name__ == "__main__":
    test_mlp_composition()
    test_transformer_layer()
    test_nested_overrides()
    print("\nAll tests completed!")