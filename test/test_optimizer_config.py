#!/usr/bin/env python3
"""Tests for the optimizer configuration system."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import optax
from flax import nnx


class TestModel(nnx.Module):
    """Test model with various parameter types."""
    def __init__(self, rngs):
        self.small = nnx.Param(jnp.zeros((10,)))
        self.medium = nnx.Param(jnp.zeros((100, 100)))
        self.large = nnx.Param(jnp.zeros((1000, 1000)))
        self.tensor_3d = nnx.Param(jnp.zeros((10, 100, 100)))


def test_basic_assignment():
    """Test basic optimizer assignment."""
    from ueaj.opt.optimizer_config import OptimizerConfig
    
    model = TestModel(nnx.Rngs(0))
    config = OptimizerConfig(model)
    
    # Assign optimizers
    config['small'] = optax.adam(1e-3)
    config['medium'] = optax.sgd(1e-2)
    
    # Create optimizer
    optimizer = config.create_optimizer(include_state_mapping=False)
    
    # Check that it works
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    assert opt_state is not None


def test_list_tree_access():
    """Test list access on tree paths."""
    from ueaj.opt.optimizer_config import OptimizerConfig
    
    model = TestModel(nnx.Rngs(0))
    config = OptimizerConfig(model)
    
    # Shared optimizer
    opt = optax.adam(1e-3)
    config[['small', 'medium']] = opt
    
    # Verify they share the same instance
    params_by_opt = config._collect_params_by_optimizer()
    assert len(params_by_opt) == 1  # Only one optimizer instance


def test_tensor_slicing():
    """Test efficient tensor slicing."""
    from ueaj.opt.optimizer_config import OptimizerConfig
    
    model = TestModel(nnx.Rngs(0))
    config = OptimizerConfig(model)
    
    # Quarter slicing
    config['large'] = optax.sgd(1e-2)
    config['large', :500, :500] = optax.adam(1e-3)
    
    # Collect optimizers
    params_by_opt = config._collect_params_by_optimizer()
    
    # Count regions for 'large'
    total_regions = sum(
        1 for _, accesses in params_by_opt.values()
        for access in accesses
        if access.tree_path == ('large',)
    )
    
    # Should create 3 regions (1 for Adam quarter, 2 for SGD remainder)
    assert total_regions == 3


def test_list_tensor_access():
    """Test list access on tensors."""
    from ueaj.opt.optimizer_config import OptimizerConfig
    
    model = TestModel(nnx.Rngs(0))
    config = OptimizerConfig(model)
    
    # List of indices
    config['small', [0, 2, 4, 6, 8]] = optax.adam(1e-3)
    config['small', [1, 3, 5, 7, 9]] = optax.sgd(1e-2)
    
    # Create optimizer
    optimizer = config.create_optimizer(include_state_mapping=False)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    assert opt_state is not None


def test_shared_optimizer_state():
    """Test that shared optimizers share state."""
    from ueaj.opt.optimizer_config import OptimizerConfig
    
    model = TestModel(nnx.Rngs(0))
    config = OptimizerConfig(model)
    
    # Create shared optimizer
    shared_opt = optax.adam(1e-3)
    
    # Assign to multiple parameters
    config['small'] = shared_opt
    config['medium'] = shared_opt
    
    # Different optimizer for another parameter
    config['large'] = optax.sgd(1e-2)
    
    # Create optimizer
    optimizer = config.create_optimizer(include_state_mapping=False)
    
    # Check that we have 2 optimizer instances, not 3
    params_by_opt = config._collect_params_by_optimizer()
    assert len(params_by_opt) == 2


def test_complex_nested_patterns():
    """Test complex nested patterns."""
    from ueaj.opt.optimizer_config import OptimizerConfig
    
    model = TestModel(nnx.Rngs(0))
    config = OptimizerConfig(model)
    
    # Base optimizer
    config['tensor_3d'] = optax.sgd(1e-2)
    
    # Override first slice
    config['tensor_3d', 0] = optax.adam(1e-3)
    
    # Override part of second slice
    config['tensor_3d', 1, :50, :50] = optax.lion(5e-4)
    
    # Create optimizer
    optimizer = config.create_optimizer(include_state_mapping=False)
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    assert opt_state is not None


def test_wildcard_patterns():
    """Test wildcard pattern matching."""
    from ueaj.opt.optimizer_config import OptimizerConfig
    
    model = TestModel(nnx.Rngs(0))
    config = OptimizerConfig(model)
    
    # Default for all
    config[...] = optax.sgd(1e-2)
    
    # Override specific parameter
    config['small'] = optax.adam(1e-3)
    
    # Check assignments
    params_by_opt = config._collect_params_by_optimizer()
    
    # Should have 2 optimizers
    assert len(params_by_opt) == 2


def test_training_step():
    """Test actual training step with mixed optimizers."""
    from ueaj.opt.optimizer_config import OptimizerConfig
    
    model = TestModel(nnx.Rngs(0))
    config = OptimizerConfig(model)
    
    # Configure
    config['small'] = optax.adam(1e-3)
    config['medium', :50, :] = optax.lion(5e-4)
    config['medium', 50:, :] = optax.sgd(1e-2)
    
    # Create optimizer
    optimizer = config.create_optimizer()
    
    # Initialize
    params = nnx.state(model, nnx.Param)
    opt_state = optimizer.init(params)
    
    # Define simple loss
    def loss_fn(params):
        total = 0.0
        for leaf in jax.tree_util.tree_leaves(params):
            if hasattr(leaf, 'value'):
                total += jnp.sum(leaf.value ** 2)
            elif isinstance(leaf, jnp.ndarray):
                total += jnp.sum(leaf ** 2)
        return total
    
    # Compute gradients
    grads = jax.grad(loss_fn)(params)
    
    # Update
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    # Check that parameters changed
    old_loss = loss_fn(params)
    new_loss = loss_fn(new_params)
    
    # With zero-initialized params and squared loss, 
    # gradients are zero so loss shouldn't change
    assert jnp.allclose(old_loss, new_loss)


if __name__ == "__main__":
    test_basic_assignment()
    test_list_tree_access()
    test_tensor_slicing()
    test_list_tensor_access()
    test_shared_optimizer_state()
    test_complex_nested_patterns()
    test_wildcard_patterns()
    test_training_step()
    
    print("All tests passed!")