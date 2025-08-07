from ueaj.model.einsum import *
from ueaj.utils import *
from ueaj.utils.configurator import *

# TODO: MoE load balancing
#   shard_map across batch, sequence
#   tally up expert activations
#   if expert activation is too high, where+cumsum+where to to obtain k
#   then fill in removed with experts with too little

@config
class MLP(nnx.Module):
    def __init__(self, 
        model_d: int,
        hidden_d: int | None = None,  # Defaults to 4 * model_d
        act_fn: Callable[[jax.Array], jax.Array] = nnx.swish,
        param_dtype: jnp.dtype = jnp.bfloat16,
        up_proj: Callable = Einsum,
        down_proj: Callable = Einsum,
        *,
        rngs: rng.Rngs,
        mesh: Optional[jax.sharding.Mesh] = None
    ):
        super().__init__()
        if hidden_d is None:
            hidden_d = 4 * model_d
        size_dict = {'d': model_d, 'h': hidden_d}
        
        # Create projections with LeCun initialization
        self.up_proj = up_proj(
            "bnd,dh->bnh",
            size_dict=size_dict,
            rngs=rngs,
            dtype=param_dtype,
            mesh=mesh,
            sharding=(None, 'tensor')
        )
        self.down_proj = down_proj(
            "bnh,hd->bnd",
            initializer=zeros_init,
            size_dict=size_dict,
            rngs=rngs,
            dtype=param_dtype,
            mesh=mesh,
            sharding=('tensor', None)
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
        hidden_d: int | None = None,  # Defaults to 4 * model_d
        activation_fn: Callable[[jax.Array], jax.Array] = nnx.swish,
        param_dtype: jnp.dtype = jnp.bfloat16,
        fused_proj: Callable = Einsum,
        down_proj: Callable = Einsum,
        *,
        rngs: rng.Rngs,
        mesh: Optional[jax.sharding.Mesh] = None
    ):
        super().__init__()
        if hidden_d is None:
            hidden_d = 4 * model_d
        
        # Create fused gate/up projection with LeCun initialization
        size_dict_fused = {'d': model_d, 'h': hidden_d, 'i': 2}
        self.fused_proj = fused_proj(
            "bnd,idh->ibnh",
            size_dict=size_dict_fused,
            batch_dims="i",
            rngs=rngs,
            dtype=param_dtype,
            mesh=mesh,
            sharding=(None, None, 'tensor')
        )
        
        size_dict = {'d': model_d, 'h': hidden_d}
        self.down_proj = down_proj(
            "bnh,hd->bnd",
            initializer=zeros_init,
            size_dict=size_dict,
            rngs=rngs,
            dtype=param_dtype,
            mesh=mesh,
            sharding=('tensor', None)
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
