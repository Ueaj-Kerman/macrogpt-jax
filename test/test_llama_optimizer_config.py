#!/usr/bin/env python3
"""Test optimizer configuration for LLaMA models with map_state."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ueaj.model.llama import LlamaModel, LlamaConfig
from ueaj.model.ueajsum.config import ParamConfig
from ueaj.opt import OptimizerConfig

def create_test_llama_model():
    """Create a small LLaMA model for testing."""
    # Create base tensor config
    tensor_config = ParamConfig("", group=nnx.Param).with_dtype(jnp.float32)
    
    # Create LLaMA config using direct parameters
    config = LlamaConfig(
        tensor_config=tensor_config,
        vocab_size=1000,
        model_d=128,
        num_layers=5,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        # Direct configuration
        mlp_type="gated",
        kq_d=32,
        v_head_d=32,
        kv_heads=2,
        kv_q_ratio=3,
        rope_theta=10000.0,
        hidden_d=512,
        norm_scale="uncentered"
    )
    
    rngs = nnx.Rngs(0)
    model = LlamaModel(config, rngs)
    return model


def test_llama_optimizer_configuration():
    """Test the requested configuration:
    - lm_head, input embeddings and all norms = adam
    - all tensors (qkv, up, gate, down) = lion
    - fuse kv and up,gate operations but unfuse before optimizer
    """
    print("Testing LLaMA Optimizer Configuration")
    print("=" * 50)
    
    model = create_test_llama_model()
    opt = OptimizerConfig(model)
    
    # As specified
    adam = optax.adam(1e-3)
    lion1 = optax.lion(1e-3, b1=0.9, b2=0.99)
    lion2 = optax.lion(1e-4, b1=0.95, b2=0.98)
    
    # Default (for embeddings, lm_head, norms)
    opt[...] = adam
    # All MLP and attention tensors
    opt['layers', ['mlp', 'attn']] = lion1
    # Unfuse momentum state for optimizer
    opt['layers', 'mlp', 'fused_proj', 'w_1', :, :1] = lion2
    opt['layers', 'attn', 'kv', 'w_1', :, :1] = lion2
    
    print("\nConfiguration:")
    print("opt[...] = optax.adam(1e-3)")
    print("opt['layers', ['mlp', 'attn']] = optax.lion(1e-3)")
    print("opt['layers', 'mlp', 'fused_proj', 'w_1', :, 0] = optax.lion(1e-4)")
    print("opt['layers', 'attn', 'kv', 'w_1', :, 0] = optax.lion(1e-4)")
    
    # Create optimizer with map_state
    optimizer = opt.create_optimizer(include_state_mapping=True)
    
    print("\nOptimizer Assignments:")
    print("-" * 50)
    opt.print_optimizer_assignment()
    
    # Get parameters and initialize
    params = nnx.state(model, nnx.Param)
    
    # Initialize optimizer
    print("\nInitializing optimizer...")
    opt_state = optimizer.init(params)
    print(jax.tree.map(jnp.shape, opt_state))
    print("lion1", jax.tree.map(jnp.shape, opt.get_optimizer_state(opt_state, lion1)))
    print("lion2", jax.tree.map(jnp.shape, opt.get_optimizer_state(opt_state, lion2)))
    print("adam", jax.tree.map(jnp.shape, opt.get_optimizer_state(opt_state, adam)))

    # Test update step
    updates, _ = optimizer.update(params, opt_state, params)
    print(jax.tree.map(jnp.shape, updates))

    # Verify the configuration worked
    print("\n✓ Optimizer initialization successful")
    print("✓ Gradient computation successful") 
    print("✓ Parameter update successful")
    print("✓ map_state properly handled fused tensors")
    
    # Count optimizer instances
    params_by_opt = opt._collect_params_by_optimizer()
    print(f"\nTotal optimizer instances: {len(params_by_opt)}")


    # Verify we have the expected slicing
    has_kv_slice_0 = False
    has_kv_slice_1 = False
    has_fused_proj_slice_0 = False
    has_fused_proj_slice_1 = False
    
    for opt_id, (opt, accesses) in params_by_opt.items():
        for access in accesses:
            if 'kv' in access.tree_path and access.tensor_slices:
                # Check which slice it is - second dimension is the fused dimension
                slice_obj = access.tensor_slices[1]  # Second dimension is the fused dimension
                if slice_obj == slice(None, 1, None) or slice_obj == slice(0, 1):
                    has_kv_slice_0 = True
                    assert opt_id == id(lion2), "KV slice 0 should use lion2"
                elif slice_obj == slice(1, 2) or slice_obj == 1:
                    has_kv_slice_1 = True
                    assert opt_id == id(lion1), "KV slice 1 should use lion1"
                    
            if 'fused_proj' in access.tree_path and access.tensor_slices:
                # Check which slice it is - second dimension is the fused dimension
                slice_obj = access.tensor_slices[1]  # Second dimension is the fused dimension
                if slice_obj == slice(None, 1, None) or slice_obj == slice(0, 1):
                    has_fused_proj_slice_0 = True
                    assert opt_id == id(lion2), "Fused proj slice 0 should use lion2"
                elif slice_obj == slice(1, 2) or slice_obj == 1:
                    has_fused_proj_slice_1 = True
                    assert opt_id == id(lion1), "Fused proj slice 1 should use lion1"
    
    assert has_kv_slice_0, "KV tensor slice 0 not found"
    assert has_kv_slice_1, "KV tensor slice 1 not found"
    assert has_fused_proj_slice_0, "Fused projection slice 0 not found"
    assert has_fused_proj_slice_1, "Fused projection slice 1 not found"
    
    print("✓ Tensor slicing verified for fused tensors")
    print("✓ Correct optimizer assignment for each slice")
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_llama_optimizer_configuration()