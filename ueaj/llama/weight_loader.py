"""
Cleaner weight loading utilities for Llama models using regex pattern matching.
"""
import re
from pathlib import Path
from typing import *
import jax
import jax.numpy as jnp
from flax.nnx import rnglib as rng


class WeightMapper:
    """Maps HuggingFace weight names to model attributes using regex patterns."""
    
    def __init__(self, model):
        self.model = model
        self.config = model.config
        
        # Track fused MLP weights that need to be combined
        self.pending_fused_weights = {}
        
        # Define weight mapping patterns
        self.patterns = [
            # Embeddings
            (r"^model\.embed_tokens\.weight$", self._set_embed_tokens),
            
            # Layer components
            (r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$", self._set_attention_weight),
            (r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$", self._set_mlp_weight),
            (r"^model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$", self._set_layer_norm),
            
            # Final components
            (r"^model\.norm\.weight$", self._set_final_norm),
            (r"^lm_head\.weight$", self._set_lm_head),
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [(re.compile(pattern), handler) for pattern, handler in self.patterns]
    
    def set_weight(self, key: str, value: jax.Array) -> bool:
        """
        Set a weight in the model based on the key.
        
        Returns:
            True if weight was set, False if key was not recognized
        """
        for pattern, handler in self.compiled_patterns:
            match = pattern.match(key)
            if match:
                handler(match, value)
                return True
        return False
    
    def _set_embed_tokens(self, match: re.Match, value: jax.Array):
        """Set embedding weights."""
        self.model.embed_tokens.embedding.value = value.astype(self.config.tensor_config.dtype)
    
    def _set_attention_weight(self, match: re.Match, value: jax.Array):
        """Set attention projection weights."""
        layer_idx = int(match.group(1))
        proj_type = match.group(2)
        
        if proj_type == "q":
            # Reshape Q projection: (num_heads * head_dim, hidden_size) -> (hidden_size, num_kv_heads, kv_q_ratio, head_dim)
            q_weight = value.T.reshape(
                self.config.model_d,
                self.config.layer_config.attention_config.kv_heads,
                self.config.layer_config.attention_config.kv_q_ratio,
                self.config.layer_config.attention_config.kq_d
            )
            self.model.layers.attn.q.w_1.value = self.model.layers.attn.q.w_1.value.at[layer_idx].set(q_weight.astype(self.config.tensor_config.dtype))
            
        elif proj_type in ["k", "v"]:
            # Check if using fused KV
            if self.config.layer_config.attention_config.fused:
                # For fused KV, we need to handle K and V together
                # Reshape: (num_kv_heads * head_dim, hidden_size) -> (hidden_size, num_kv_heads, head_dim)
                weight = value.T.reshape(
                    self.config.model_d,
                    self.config.layer_config.attention_config.kv_heads,
                    self.config.layer_config.attention_config.kq_d
                )
                
                if not hasattr(self, '_kv_buffer'):
                    self._kv_buffer = {}
                
                # Store K or V weight temporarily
                layer_key = f"layer_{layer_idx}"
                if layer_key not in self._kv_buffer:
                    self._kv_buffer[layer_key] = {}
                
                self._kv_buffer[layer_key][proj_type] = weight
                
                # If we have both K and V, combine them into fused KV
                if 'k' in self._kv_buffer[layer_key] and 'v' in self._kv_buffer[layer_key]:
                    k_weight = self._kv_buffer[layer_key]['k']
                    v_weight = self._kv_buffer[layer_key]['v']
                    
                    # Stack K and V along the fused dimension: (2, hidden_size, num_kv_heads, head_dim)
                    kv_weight = jnp.stack([k_weight, v_weight], axis=0)
                    self.model.layers.attn.kv.w_1.value = self.model.layers.attn.kv.w_1.value.at[layer_idx].set(kv_weight.astype(self.config.tensor_config.dtype))
                    
                    # Clean up buffer
                    del self._kv_buffer[layer_key]
            else:
                # Non-fused case: handle K and V separately
                weight = value.T.reshape(
                    self.config.model_d,
                    self.config.layer_config.attention_config.kv_heads,
                    self.config.layer_config.attention_config.kq_d
                )
                if proj_type == "k":
                    self.model.layers.attn.k.w_1.value = self.model.layers.attn.k.w_1.value.at[layer_idx].set(weight.astype(self.config.tensor_config.dtype))
                else:
                    self.model.layers.attn.v.w_1.value = self.model.layers.attn.v.w_1.value.at[layer_idx].set(weight.astype(self.config.tensor_config.dtype))
            
        elif proj_type == "o":
            # Reshape O projection: (hidden_size, hidden_size) -> (kv_heads, kv_q_ratio, head_dim, hidden_size)
            o_weight = value.reshape(self.config.model_d, self.config.model_d).T
            o_weight = o_weight.reshape(
                self.config.layer_config.attention_config.kv_heads,
                self.config.layer_config.attention_config.kv_q_ratio,
                self.config.layer_config.attention_config.kq_d,
                self.config.model_d
            )
            self.model.layers.attn.o.w_1.value = self.model.layers.attn.o.w_1.value.at[layer_idx].set(o_weight.astype(self.config.tensor_config.dtype))
    
    def _set_mlp_weight(self, match: re.Match, value: jax.Array):
        """Set MLP projection weights."""
        layer_idx = int(match.group(1))
        proj_type = match.group(2)
        
        # Transpose all MLP weights
        weight = value.T.astype(self.config.tensor_config.dtype)
        
        if hasattr(self.model.layers.mlp, "fused_proj"):
            # Gated MLP with fused projections
            if proj_type in ["gate", "up"]:
                # Store weights temporarily
                key = f"layer_{layer_idx}_fused"
                if key not in self.pending_fused_weights:
                    self.pending_fused_weights[key] = {}
                
                # Store the weight
                self.pending_fused_weights[key][proj_type] = weight
                
                # Check if we have both gate and up projections
                if len(self.pending_fused_weights[key]) == 2:
                    # Create the fused weight tensor
                    gate_weight = self.pending_fused_weights[key]["gate"]
                    up_weight = self.pending_fused_weights[key]["up"]
                    
                    # Stack along the first dimension: shape (2, hidden_size, intermediate_size)
                    fused_weight = jnp.stack([up_weight, gate_weight], axis=0)
                    self.model.layers.mlp.fused_proj.w_1.value = self.model.layers.mlp.fused_proj.w_1.value.at[layer_idx].set(fused_weight)
                    
                    # Clean up
                    del self.pending_fused_weights[key]
                    
            elif proj_type == "down":
                self.model.layers.mlp.down_proj.w_2.value = self.model.layers.mlp.down_proj.w_2.value.at[layer_idx].set(weight)
        else:
            # Regular MLP
            if proj_type == "up":
                self.model.layers.mlp.up_proj.w_1.value = self.model.layers.mlp.up_proj.w_1.value.at[layer_idx].set(weight)
            elif proj_type == "down":
                self.model.layers.mlp.down_proj.w_1.value = self.model.layers.mlp.down_proj.w_1.value.at[layer_idx].set(weight)
    
    def _set_layer_norm(self, match: re.Match, value: jax.Array):
        """Set layer normalization weights."""
        layer_idx = int(match.group(1))
        norm_type = match.group(2)
        
        if norm_type == "input_layernorm":
            self.model.layers.attn_norm.scale.value = self.model.layers.attn_norm.scale.value.at[layer_idx].set(value.astype(self.config.tensor_config.dtype))
        else:  # post_attention_layernorm
            self.model.layers.mlp_norm.scale.value = self.model.layers.mlp_norm.scale.value.at[layer_idx].set(value.astype(self.config.tensor_config.dtype))
    
    def _set_final_norm(self, match: re.Match, value: jax.Array):
        """Set final layer norm weight."""
        self.model.norm.scale.value = value.astype(self.config.tensor_config.dtype)
    
    def _set_lm_head(self, match: re.Match, value: jax.Array):
        """Set language model head weight."""
        if self.model.lm_head is not None:
            # Transpose for flax convention
            self.model.lm_head.kernel.value = value.T.astype(self.config.tensor_config.dtype)


def load_weights_from_safetensors(model, safetensor_files):
    """
    Load weights into a model from safetensor files using the WeightMapper.
    
    Args:
        model: The model to load weights into
        safetensor_files: List of safetensor file paths
    """
    from safetensors import safe_open
    
    mapper = WeightMapper(model)
    loaded_keys = []
    skipped_keys = []
    
    for st_file in safetensor_files:
        with safe_open(st_file, framework="flax") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if mapper.set_weight(key, tensor):
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)
    
    return loaded_keys, skipped_keys


def create_llama_model_config(model_id: str, dtype: Optional[jax.typing.DTypeLike] = None):
    """Create a configured LlamaModel class using HuggingFace config and override API."""
    from transformers import LlamaConfig as HFLlamaConfig
    from ueaj.model import SoftmaxAttention, GMLP, TransformerLayer, LlamaModel, RMSNorm

    # Load config using transformers
    hf_config = HFLlamaConfig.from_pretrained(model_id)

    kv_heads = getattr(hf_config, "num_key_value_heads", hf_config.num_attention_heads)
    model_d = hf_config.hidden_size
    
    # Get dtype
    if dtype is None:
        dtype = jax.dtypes.canonicalize_dtype(hf_config.torch_dtype)

    # Create configured model class using override API
    return LlamaModel.override(
        vocab_size=hf_config.vocab_size,
        model_d=model_d,
        num_layers=hf_config.num_hidden_layers,
        max_position_embeddings=hf_config.max_position_embeddings,
        tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
        param_dtype=dtype,
        # Configure transformer layer
        transformer_layer=TransformerLayer.override(
            # Configure attention
            attn=SoftmaxAttention.override(
                kq_d=hf_config.head_dim,
                kv_heads=kv_heads,
                kv_q_ratio=hf_config.num_attention_heads // kv_heads,
                rope_theta=getattr(hf_config, "rope_theta", 10000.0),
                attention_scale='sp',  # Use SP scaling for LLaMA models
                param_dtype=dtype
            ),
            # Configure MLP
            mlp=GMLP.override(
                hidden_d=hf_config.intermediate_size,
                param_dtype=dtype
            ),
            # Configure normalization
            norm=RMSNorm.override(
                eps=getattr(hf_config, "rms_norm_eps", 1e-5),
                scale="uncentered",
                param_dtype=dtype
            )
        ),
        # Configure final norm
        norm=RMSNorm.override(
            eps=getattr(hf_config, "rms_norm_eps", 1e-5),
            scale="uncentered",
            param_dtype=dtype
        )
    )


def from_pretrained(
    model_class,
    model_path: str,
    rngs: Optional[rng.Rngs] = None,
    dtype: Optional[jax.typing.DTypeLike] = None,
    abstract: bool = False,
) -> Any:
    """
    Load a pretrained Llama model from safetensors files.

    Args:
        model_class: The model class to instantiate (ignored, uses LlamaModel)
        model_path: Path to directory containing safetensors files
        rngs: Random number generators
        dtype: Data type for model parameters
        abstract: If True, create abstract model without allocating weights

    Returns:
        Loaded model instance
    """
    from flax import nnx
    
    if rngs is None:
        rngs = rng.Rngs(0)

    # Create configured model class using override API
    configured_model_class = create_llama_model_config(model_path, dtype)

    # Create model instance
    if abstract:
        model = nnx.eval_shape(lambda: configured_model_class(rngs=rng.Rngs(0)))
    else:
        model = configured_model_class(rngs=rngs)

    # Load weights from safetensors
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise ValueError(f"Model directory {model_path} does not exist")

    # Find all safetensors files
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    # Load weights using the cleaner weight loader
    loaded_keys, skipped_keys = load_weights_from_safetensors(model, safetensor_files)

    if skipped_keys:
        print(f"Warning: Skipped {len(skipped_keys)} unrecognized keys during weight loading")
        if len(skipped_keys) < 10:
            print(f"Skipped keys: {skipped_keys}")

    return model