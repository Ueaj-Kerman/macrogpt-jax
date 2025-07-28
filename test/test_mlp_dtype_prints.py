#!/usr/bin/env python3
"""
Test file that creates MLPs with bf16 and fp8e4m3 params, tests them with bf16 and fp8e4m3 inputs,
and lets the GMLP print dtypes to console.

This test file shows:
1. MLPs can be created with different parameter dtypes (bf16, fp8)
2. The print statements in GMLP show the dtypes during forward pass
3. The LOW_PRECISION path is taken when input dtype is fp8
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.utils import astype_fwd_noop_bwd, LOW_PRECISION
from ueaj.model.mlp import MLP, GMLP, MLPConfig
from ueaj.model.ueajsum import ParamConfig


def test_mlp_configurations():
    """Test MLPs with different parameter and input dtypes."""
    
    model_d = 16
    hidden_d = 32
    batch_size = 2
    seq_len = 4
    
    # Test configurations: (name, param_dtype)
    param_configs = [
        ("bf16 params", jnp.bfloat16),
        ("fp8e4m3 params", jnp.float8_e4m3fn),
    ]
    
    # Input dtypes to test
    input_dtypes = [
        ("bf16 inputs", jnp.bfloat16),
        ("fp8e4m3 inputs", jnp.float8_e4m3fn),
    ]
    
    for param_name, param_dtype in param_configs:
        print(f"\n{'='*60}")
        print(f"Testing with {param_name}")
        print(f"{'='*60}")
        
        # Create config
        param_config = ParamConfig("", group=nnx.Param).with_dtype(param_dtype).with_grad_dtype(jnp.float32)
        mlp_config = MLPConfig(
            model_d=model_d,
            hidden_d=hidden_d,
            activation_fn=nnx.relu,
            param_config=param_config
        )
        
        # Initialize models
        rngs = rng.Rngs(42)
        # mlp = MLP(mlp_config, rngs)
        gmlp = GMLP(mlp_config, rngs)
        
        # Show parameter info
        print(f"\nCreated MLP and GMLP with parameter dtype: {param_dtype}")
        # print(f"MLP up_proj param dtype: {mlp.up_proj.w_1.dtype}")
        print(f"GMLP fused_proj param dtype: {gmlp.fused_proj.w_1.dtype}")
        
        for input_name, input_dtype in input_dtypes:
            print(f"\n--- Testing with {input_name} ---")
            
            # Create input
            x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, seq_len, model_d))
            x = x.astype(input_dtype)
            
            print(f"Input shape: {x.shape}, dtype: {x.dtype}")
            print(f"Input dtype in LOW_PRECISION: {x.dtype in LOW_PRECISION}")
            
            # The actual forward pass - will print dtypes even if it errors
            @nnx.grad
            def grad(mlp, x, y):
                y_pred = mlp(x)
                return jnp.square(y_pred-y.astype(y_pred.dtype)).sum()
            grad(gmlp, x, x)


def main():
    print("LOW_PRECISION dtypes:", LOW_PRECISION)
    print("\nThis test shows:")
    print("1. MLPs can be created with bf16 and fp8e4m3 parameters")
    print("2. The dtype information for parameters is preserved")
    print("3. The GMLP forward pass has print statements that show intermediate dtypes")
    print("4. The LOW_PRECISION path is taken when input dtype is in LOW_PRECISION set")
    
    test_mlp_configurations()
    
    print("\n\nNOTE: To see the actual dtype prints during forward pass, the @nnx.jit")
    print("decorator needs to be temporarily removed from GMLP.__call__ in mlp.py,")
    print("or run the forward pass outside of JIT compilation.")


if __name__ == "__main__":
    main()