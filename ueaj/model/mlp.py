from typing import *

import jax
from jax import numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng
from ueaj.model.einsum import *
from ueaj.utils import *
from ueaj.utils.configurator import *


@config
class MLP(nnx.Module):
    def __init__(self, 
        model_d: int,
        rngs: rng.Rngs,
        hidden_d: int | None = None,  # Defaults to 4 * model_d
        act_fn: Callable[[jax.Array], jax.Array] = nnx.swish,
        param_dtype: jnp.dtype = jnp.bfloat16
    ):
        super().__init__()
        if hidden_d is None:
            hidden_d = 4 * model_d
        size_dict = {'d': model_d, 'h': hidden_d}
        
        # Create projections with LeCun initialization
        self.up_proj = Einsum(
            "bnd,dh->bnh",
            nnx.initializers.lecun_normal(),
            size_dict,
            rngs=rngs,
            dtype=param_dtype
        )
        self.down_proj = Einsum(
            "bnh,hd->bnd",
            nnx.initializers.zeros_init(),
            size_dict,
            rngs=rngs,
            dtype=param_dtype
        )
        self.activation_fn = act_fn

    def __call__(self, x):
        x = self.up_proj(x)
        x = self.activation_fn(x)
        x = self.down_proj(x)
        return x


@config
class GMLP(nnx.Module):
    def __init__(self, 
        model_d: int,
        rngs: rng.Rngs,
        hidden_d: int | None = None,  # Defaults to 4 * model_d
        activation_fn: Callable[[jax.Array], jax.Array] = nnx.swish,
        param_dtype: jnp.dtype = jnp.bfloat16
    ):
        super().__init__()
        if hidden_d is None:
            hidden_d = 4 * model_d
        
        # Create fused gate/up projection with LeCun initialization
        size_dict_fused = {'d': model_d, 'h': hidden_d, 'i': 2}
        self.fused_proj = Einsum(
            "bnd,idh->ibnh",
            nnx.initializers.lecun_normal(),
            size_dict_fused,
            batch_dims="i",
            rngs=rngs,
            dtype=param_dtype
        )
        
        size_dict = {'d': model_d, 'h': hidden_d}
        self.down_proj = Einsum(
            "bnh,hd->bnd",
            nnx.initializers.zeros_init(),
            size_dict,
            rngs=rngs,
            dtype=param_dtype
        )
        self.activation_fn = activation_fn

    def __call__(self, x):
        up, gate = self.fused_proj(x)

        gate = self.activation_fn(gate)

        # Apply gating
        if x.dtype not in LOW_PRECISION:
            x = up * gate
        else:
            s = jnp.max(jnp.abs(up), axis=(0, 1), keepdims=True) + 1
            x = (s * up) * gate
            x = x / s
        
        # Apply down projection
        x = self.down_proj(x)

        return x


if __name__ == "__main__":
    rngs = rng.Rngs(0)
    x = jnp.ones((1, 1, 16))
    
    # Direct instantiation with defaults
    m = MLP(
        model_d=16,
        hidden_d=32,
        rngs=rngs,
        act_fn=nnx.relu
    )
    
    # Using override to create custom GMLP
    GMLPCustom = GMLP.override(
        activation_fn=nnx.relu,
        param_dtype=jnp.float32
    )
    gm = GMLPCustom(model_d=16, hidden_d=32, rngs=rngs)
    
    print(m(x))
    print(gm(x))