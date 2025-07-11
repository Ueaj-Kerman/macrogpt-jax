#!/usr/bin/env python3
"""Test that map_state correctly transforms tensor shapes for optimizers."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import nnx

from ueaj.model import ueajsum as us


def test_map_state_kv_tensor():
    """Test map_state on a KV tensor with fused dimensions."""
    print("Testing map_state on KV tensor")
    print("=" * 50)
    
    # Create a fused KV tensor like in attention
    size_dict = {'b': 1, 'n': 16, 'f': 2, 'd': 128, 'h': 2, 'k': 32}
    kv_config = us.parse("bnd,*fdhk->bnfhk").param(us.ParamConfig("")).in_axes({1: (1,)}).batch_axes({1: (0,)})
    kv_module = us.Ueajsum(kv_config, size_dict, rngs=nnx.Rngs(0))
    
    # Get the original parameter shape
    params = nnx.state(kv_module, nnx.Param)
    original_shape = params['w_1'].value.shape
    print(f"Original KV shape: {original_shape}")
    print(f"  Config: in_axes=(1,), batch_axes=(0,)")
    
    # Apply map_state transformation directly on the module
    mapped_params = kv_module.map_state(params, from_optimizer=False)
    mapped_shape = mapped_params['w_1'].value.shape
    print(f"\nMapped KV shape: {mapped_shape}")
    
    # Verify the transformation
    assert original_shape == (2, 128, 2, 32), f"Expected original shape (2, 128, 2, 32), got {original_shape}"
    assert mapped_shape == (2, 128, 64), f"Expected mapped shape (2, 128, 64), got {mapped_shape}"
    
    # Map back
    unmapped_params = kv_module.map_state(mapped_params, from_optimizer=True)
    unmapped_shape = unmapped_params['w_1'].value.shape
    print(f"Unmapped KV shape: {unmapped_shape}")
    
    assert unmapped_shape == original_shape, f"Round-trip failed: {original_shape} -> {mapped_shape} -> {unmapped_shape}"
    print("\n✓ KV tensor map_state transformation successful")


def test_map_state_fused_proj():
    """Test map_state on a fused projection tensor."""
    print("\n\nTesting map_state on fused projection tensor")
    print("=" * 50)
    
    # Create a fused projection tensor like in GMLP
    size_dict = {'b': 1, 'n': 16, 'f': 2, 'd': 128, 'h': 512}
    proj_config = us.parse("bnd,*fdh->bnfh").param(us.ParamConfig("")).in_axes({1: (2,)}).batch_axes({1: (0,)})
    proj_module = us.Ueajsum(proj_config, size_dict, rngs=nnx.Rngs(0))
    
    # Get the original parameter shape
    params = nnx.state(proj_module, nnx.Param)
    original_shape = params['w_1'].value.shape
    print(f"Original fused_proj shape: {original_shape}")
    print(f"  Config: in_axes=(2,), batch_axes=(0,)")
    
    # Apply map_state transformation
    mapped_params = proj_module.map_state(params, from_optimizer=False)
    mapped_shape = mapped_params['w_1'].value.shape
    print(f"\nMapped fused_proj shape: {mapped_shape}")
    
    # Verify the transformation
    assert original_shape == (2, 128, 512), f"Expected original shape (2, 128, 512), got {original_shape}"
    # With in_axes=(2,) and batch_axes=(0,), the mapped shape groups in_axes together, then out_axes
    # Original: (f=2, d=128, h=512) with batch=0, in=2
    # Mapped: (batch=2, in=512, out=128)
    assert mapped_shape == (2, 512, 128), f"Expected mapped shape (2, 512, 128), got {mapped_shape}"
    
    # Map back
    unmapped_params = proj_module.map_state(mapped_params, from_optimizer=True)
    unmapped_shape = unmapped_params['w_1'].value.shape
    print(f"Unmapped fused_proj shape: {unmapped_shape}")
    
    assert unmapped_shape == original_shape, f"Round-trip failed: {original_shape} -> {mapped_shape} -> {unmapped_shape}"
    print("\n✓ Fused projection tensor map_state transformation successful")


def test_map_state_with_vmapped_layers():
    """Test map_state with vmapped layers (extra dimensions)."""
    print("\n\nTesting map_state with vmapped layers")
    print("=" * 50)
    
    # Create a tensor with vmapped dimensions
    size_dict = {'d': 128, 'h': 2, 'k': 32}
    v_dims = 3  # Simulating 3 vmapped layers
    
    # Create config for attention K projection with vmapped layers
    k_config = us.parse("nd,*dhk->nhk").param(us.ParamConfig("")).in_axes({1: (0,)})
    k_module = us.Ueajsum(k_config, size_dict, rngs=nnx.Rngs(0))
    
    # Manually add vmapped dimensions to simulate vmapped layers
    params = nnx.state(k_module, nnx.Param)
    original_value = params['w_1'].value
    # Add 3 vmapped dimensions at the front
    vmapped_value = jnp.broadcast_to(original_value, (5, 2, 4) + original_value.shape)
    params['w_1'] = nnx.Param(vmapped_value)
    
    original_shape = params['w_1'].value.shape
    print(f"Original K shape with vmap: {original_shape}")
    print(f"  v_dims=3, in_axes=(0,)")
    
    # Apply map_state transformation
    mapped_params = k_module.map_state(params, from_optimizer=False)
    mapped_shape = mapped_params['w_1'].value.shape
    print(f"\nMapped K shape: {mapped_shape}")
    
    # Expected: vmapped dims stay in front, then in_axes, then remaining
    assert original_shape == (5, 2, 4, 128, 2, 32)
    assert mapped_shape == (5, 2, 4, 128, 64)
    
    print("\n✓ Vmapped layers map_state transformation successful")


def test_map_state_preserves_values():
    """Test that map_state preserves tensor values through transformation."""
    print("\n\nTesting map_state value preservation")
    print("=" * 50)
    
    # Create a simple tensor
    size_dict = {'d': 4, 'h': 2, 'k': 3}
    config = us.parse("d,*hk->hk").param(us.ParamConfig("")).in_axes({0: (0,)})
    module = us.Ueajsum(config, size_dict, rngs=nnx.Rngs(0))
    
    # Set specific values
    params = nnx.state(module, nnx.Param)
    test_values = jnp.arange(2 * 3).reshape(2, 3).astype(jnp.float32)
    params['w_0'] = nnx.Param(test_values)
    
    print(f"Original values:\n{test_values}")
    
    # Transform and check
    mapped_params = module.map_state(params, from_optimizer=False)
    mapped_values = mapped_params['w_0'].value
    
    print(f"\nMapped shape: {test_values.shape} -> {mapped_values.shape}")
    print(f"Mapped values:\n{mapped_values}")
    
    # With in_axes=(0,), the tensor is reshaped to (in_dims, out_dims) = (2, 3)
    # Since there are no batch_axes, it stays the same shape
    assert mapped_values.shape == (2, 3)
    assert jnp.allclose(mapped_values, test_values)
    
    # Transform back
    unmapped_params = module.map_state(mapped_params, from_optimizer=True)
    unmapped_values = unmapped_params['w_0'].value
    
    print(f"\nUnmapped values:\n{unmapped_values}")
    assert jnp.allclose(unmapped_values, test_values)
    
    print("\n✓ Values preserved through map_state transformation")


if __name__ == "__main__":
    test_map_state_kv_tensor()
    test_map_state_fused_proj()
    test_map_state_with_vmapped_layers()
    test_map_state_preserves_values()
    print("\n\nAll map_state tests passed! ✓")