"""
Cleaner weight loading utilities for Llama models using regex pattern matching.
"""
import re
from pathlib import Path
from typing import *
import os
import json
import warnings
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

# HuggingFace imports
try:
    from transformers import AutoConfig, AutoTokenizer
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    warnings.warn("transformers or huggingface_hub not available. HuggingFace loading will not work.")


# Removed old WeightMapper (state-based) in favor of direct in-place loading


class ModelWeightMapper:
    """Maps HuggingFace weight names directly into a live nnx.Module.

    This mapper mutates the provided model's parameters in place.
    """

    def __init__(self, target_model):
        self.model = target_model

        # Extract config info from model structure
        self.model_d = target_model.model_d
        self.kv_heads = target_model.layers.attn.kv_heads
        self.kv_q_ratio = target_model.layers.attn.kv_q_ratio
        self.kq_d = target_model.layers.attn.kq_d

        # Determine dtype from embedding
        embed = target_model.embed_tokens
        if hasattr(embed, 'base'):
            self.dtype = embed.base.embedding.value.dtype
        else:
            self.dtype = embed.embedding.value.dtype

        # Track fused MLP weights that need to be combined
        self.pending_fused_weights: Dict[str, Dict[str, jax.Array]] = {}

        # Patterns
        self.patterns = [
            (r"^model\.embed_tokens\.weight$", self._set_embed_tokens),
            (r"^model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight$", self._set_attention_weight),
            (r"^model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$", self._set_mlp_weight),
            (r"^model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight$", self._set_layer_norm),
            (r"^model\.norm\.weight$", self._set_final_norm),
            (r"^lm_head\.weight$", self._set_lm_head),
        ]
        self.compiled_patterns = [(re.compile(pattern), handler) for pattern, handler in self.patterns]

    def set_weight(self, key: str, value: jax.Array) -> bool:
        for pattern, handler in self.compiled_patterns:
            m = pattern.match(key)
            if m:
                handler(m, value)
                return True
        return False

    def _set_embed_tokens(self, match: re.Match, value: jax.Array):
        self.model.embed_tokens.embedding.value = value.astype(self.dtype)

    def _set_attention_weight(self, match: re.Match, value: jax.Array):
        layer_idx = int(match.group(1))
        proj_type = match.group(2)

        if proj_type == "q":
            q_weight = value.T.reshape(self.model_d, self.kv_heads, self.kv_q_ratio, self.kq_d)
            q_param = self.model.layers.attn.q.w
            q_param.value = q_param.value.at[layer_idx].set(q_weight.astype(self.dtype))
            return

        if proj_type in ["k", "v"]:
            fused_kv = hasattr(self.model.layers.attn, 'kv')
            weight = value.T.reshape(self.model_d, self.kv_heads, self.kq_d)
            if fused_kv:
                if not hasattr(self, '_kv_buffer'):
                    self._kv_buffer = {}
                layer_key = f"layer_{layer_idx}"
                if layer_key not in self._kv_buffer:
                    self._kv_buffer[layer_key] = {}
                self._kv_buffer[layer_key][proj_type] = weight
                if 'k' in self._kv_buffer[layer_key] and 'v' in self._kv_buffer[layer_key]:
                    k_weight = self._kv_buffer[layer_key]['k']
                    v_weight = self._kv_buffer[layer_key]['v']
                    kv_weight = jnp.stack([k_weight, v_weight], axis=0)
                    kv_param = self.model.layers.attn.kv.w
                    kv_param.value = kv_param.value.at[layer_idx].set(kv_weight.astype(self.dtype))
                    del self._kv_buffer[layer_key]
            else:
                if proj_type == 'k':
                    k_param = self.model.layers.attn.k.w
                    k_param.value = k_param.value.at[layer_idx].set(weight.astype(self.dtype))
                else:
                    v_param = self.model.layers.attn.v.w
                    v_param.value = v_param.value.at[layer_idx].set(weight.astype(self.dtype))
            return

        if proj_type == "o":
            o_weight = value.reshape(self.model_d, self.model_d).T
            o_weight = o_weight.reshape(self.kv_heads, self.kv_q_ratio, self.kq_d, self.model_d)
            o_param = self.model.layers.attn.o.w
            o_param.value = o_param.value.at[layer_idx].set(o_weight.astype(self.dtype))

    def _set_mlp_weight(self, match: re.Match, value: jax.Array):
        layer_idx = int(match.group(1))
        proj_type = match.group(2)
        weight = value.T.astype(self.dtype)
        if hasattr(self.model.layers.mlp, 'fused_proj'):
            if proj_type in ["gate", "up"]:
                key = f"layer_{layer_idx}_fused"
                if key not in self.pending_fused_weights:
                    self.pending_fused_weights[key] = {}
                self.pending_fused_weights[key][proj_type] = weight
                if len(self.pending_fused_weights[key]) == 2:
                    gate_weight = self.pending_fused_weights[key]["gate"]
                    up_weight = self.pending_fused_weights[key]["up"]
                    fused_weight = jnp.stack([up_weight, gate_weight], axis=0)
                    fp_param = self.model.layers.mlp.fused_proj.w
                    fp_param.value = fp_param.value.at[layer_idx].set(fused_weight)
                    del self.pending_fused_weights[key]
            elif proj_type == 'down':
                dp_param = self.model.layers.mlp.down_proj.w
                dp_param.value = dp_param.value.at[layer_idx].set(weight)
        else:
            if proj_type == 'up':
                up_param = self.model.layers.mlp.up_proj.w
                up_param.value = up_param.value.at[layer_idx].set(weight)
            elif proj_type == 'down':
                down_param = self.model.layers.mlp.down_proj.w
                down_param.value = down_param.value.at[layer_idx].set(weight)

    def _set_layer_norm(self, match: re.Match, value: jax.Array):
        layer_idx = int(match.group(1))
        norm_type = match.group(2)
        if norm_type == 'input_layernorm':
            an = self.model.layers.attn_norm.scale
            an.value = an.value.at[layer_idx].set(value.astype(self.dtype))
        else:
            mn = self.model.layers.mlp_norm.scale
            mn.value = mn.value.at[layer_idx].set(value.astype(self.dtype))

    def _set_final_norm(self, match: re.Match, value: jax.Array):
        self.model.norm.scale.value = value.astype(self.dtype)

    def _set_lm_head(self, match: re.Match, value: jax.Array):
        if hasattr(self.model, 'lm_head') and self.model.lm_head is not None:
            self.model.lm_head.w.value = value.T.astype(self.dtype)
    


def load_safetensors_into_model(model_like, safetensor_files) -> Tuple[List[str], List[str]]:
    """Load safetensors directly into a concrete model (in place).

    Returns:
        (loaded_keys, skipped_keys)
    """
    from safetensors import safe_open
    mapper = ModelWeightMapper(model_like)
    loaded_keys: List[str] = []
    skipped_keys: List[str] = []
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
        tie_word_embeddings=getattr(hf_config, "tie_word_embeddings", False),
        head_cap=None,  # Disable logit soft-capping for LLaMA (Gemma 2 feature)
        # Configure transformer layer
        transformer_layer=TransformerLayer.override(
            # Configure attention
            attn=SoftmaxAttention.override(
                kq_d=hf_config.head_dim,
                kv_heads=kv_heads,
                kv_q_ratio=hf_config.num_attention_heads // kv_heads,
                rope_theta=getattr(hf_config, "rope_theta", 10000.0),
                attn_scale='sp',  # Use SP scaling for LLaMA models
            ),
            # Configure MLP
            mlp=GMLP.override(
                hidden_d=hf_config.intermediate_size,
            ),
            # Configure normalization
            norm=RMSNorm.override(
                eps=getattr(hf_config, "rms_norm_eps", 1e-5),
                scale_mode='uncentered',  # LLaMA uses uncentered RMSNorm
            )
        ),
        # Configure final norm
        norm=RMSNorm.override(
            eps=getattr(hf_config, "rms_norm_eps", 1e-5),
            scale_mode='uncentered',  # LLaMA uses uncentered RMSNorm
        )
    )


def from_pretrained(
    model_path: str,
    rngs: Optional[rng.Rngs] = None,
    dtype: Optional[jax.typing.DTypeLike] = None,
    existing_model: Optional[Any] = None,
) -> Any:
    """
    Load pretrained LLaMA weights and modify a concrete model in place.

    If `existing_model` is provided, its parameters are updated and it is
    returned. Otherwise a new model is instantiated from HF config,
    updated with loaded weights, and returned.
    """
    from flax import nnx

    if rngs is None:
        rngs = rng.Rngs(0)

    # Ensure a concrete model instance to write into
    if existing_model is None:
        configured_model_class = create_llama_model_config(model_path, dtype)
        model = configured_model_class(rngs=rngs)
    else:
        model = existing_model

    # Resolve safetensors
    model_dir = Path(model_path)
    if not model_dir.exists():
        raise ValueError(f"Model directory {model_path} does not exist")
    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No safetensors files found in {model_path}")

    # Load weights directly into the model in place
    loaded_keys, skipped_keys = load_safetensors_into_model(model, safetensor_files)

    if skipped_keys:
        print(f"Warning: Skipped {len(skipped_keys)} unrecognized keys during weight loading")
        if len(skipped_keys) < 10:
            print(f"Skipped keys: {skipped_keys}")

    return model


# Removed state_from_pretrained; use from_pretrained for in-place loading


def load_llama_from_hf(
    model_name: str = "meta-llama/Llama-3.2-1B",
    cache_dir: Optional[str] = None,
    dtype: Optional[jax.typing.DTypeLike] = None,
    rngs: Optional[rng.Rngs] = None,
) -> Tuple[Any, Any]:
    """Load a LLaMA model and tokenizer from HuggingFace.

    This is the main entry point for loading pretrained LLaMA models. It handles:
    - Downloading from HuggingFace (or using cached models)
    - Loading tokenizer
    - Creating model with correct configuration
    - Loading weights via in-place mapper

    Args:
        model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B")
        cache_dir: Optional cache directory for downloads
        dtype: Optional dtype for model weights (default: from model config)
        rngs: Optional RNG state (default: Rngs(0))

    Returns:
        Tuple of (model, tokenizer)

    Example:
        >>> model, tokenizer = load_llama_from_hf("meta-llama/Llama-3.2-1B")
        >>> tokens = tokenizer("Hello world", return_tensors="np")["input_ids"]
        >>> logits = model(tokens)
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "HuggingFace libraries not available. "
            "Install with: pip install transformers huggingface_hub"
        )

    print(f"Loading {model_name} from HuggingFace...")

    # Download model files
    print("Downloading model files...")
    model_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Load model using from_pretrained
    print("Creating corresponding JAX model...")
    print("Loading weights from safetensors...")
    model = from_pretrained(
        model_path,
        rngs=rngs,
        dtype=dtype,
    )

    return model, tokenizer
