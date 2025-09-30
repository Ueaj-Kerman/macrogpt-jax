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


# ============================================================================
# PEFT LoRA Import/Export for vLLM compatibility
# ============================================================================

def save_lora_to_peft(
    model,
    output_dir: str,
    base_model_name: str = "meta-llama/Llama-3.1-8B",
):
    """Save LoRA weights to disk in HuggingFace PEFT format.

    Emits per-layer PEFT keys (model.layers.{i}.self_attn.*) for vmapped models.
    """
    from pathlib import Path
    from ueaj.model.lora import LoRAEinsum, LoRAEmbed

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    num_layers = getattr(model, 'num_layers', None)
    if num_layers is None:
        raise ValueError("Model must expose num_layers to export LoRA")

    peft_weights: Dict[str, np.ndarray] = {}
    target_modules = set()

    # Helper to export a vmapped LoRAEinsum module
    def export_einsum(module, hf_proj_name: str):
        nonlocal peft_weights
        A = np.array(module.lora_A.value, dtype=np.float32)
        B = np.array(module.lora_B.value, dtype=np.float32)
        # Expected shapes: (L, in, r) and (L, r, out) or with equivalent axes order
        # Normalize to (L, in, r) and (L, r, out)
        if A.ndim == 2:
            A = A[None, ...]
        if B.ndim == 2:
            B = B[None, ...]
        if A.shape[-1] != B.shape[-2]:
            # swap last two dims of A if needed
            if A.shape[-2] == B.shape[-2]:
                A = np.swapaxes(A, -1, -2)
        # Emit per-layer
        for i in range(A.shape[0]):
            a_i = A[i]
            b_i = B[i]
            # Flatten any remaining batch dims on A except last (rank)
            if a_i.ndim > 2:
                a_i = a_i.reshape(-1, a_i.shape[-1])
            if b_i.ndim > 2:
                b_i = b_i.reshape(-1, b_i.shape[-1])
            peft_weights[f'base_model.model.model.layers.{i}.self_attn.{hf_proj_name}.lora_A.weight'] = a_i.astype(np.float32)
            peft_weights[f'base_model.model.model.layers.{i}.self_attn.{hf_proj_name}.lora_B.weight'] = b_i.astype(np.float32)
        target_modules.add(hf_proj_name)

    # Export attention projections if present
    attn = getattr(model.layers, 'attn', None)
    if attn is not None:
        for name, hf_name in [('q', 'q_proj'), ('k', 'k_proj'), ('v', 'v_proj'), ('o', 'o_proj')]:
            mod = getattr(attn, name, None)
            if isinstance(mod, LoRAEinsum):
                export_einsum(mod, hf_name)

    # Export embedding LoRA if present
    embed = getattr(model, 'embed_tokens', None)
    if isinstance(embed, LoRAEmbed):
        A = np.array(embed.lora_A.value, dtype=np.float32)
        B = np.array(embed.lora_B.value, dtype=np.float32)
        if A.ndim > 2:
            A = A.reshape(-1, A.shape[-1])
        if B.ndim > 2:
            B = B.reshape(-1, B.shape[-1])
        peft_weights['base_model.model.model.embed_tokens.lora_A.weight'] = A.astype(np.float32)
        peft_weights['base_model.model.model.embed_tokens.lora_B.weight'] = B.astype(np.float32)
        target_modules.add('embed_tokens')

    if not peft_weights:
        raise ValueError("No LoRA parameters found to export")

    # Infer rank/alpha from one module if available
    lora_rank = None
    lora_alpha = None
    probe = None
    if attn is not None and isinstance(getattr(attn, 'q', None), LoRAEinsum):
        probe = attn.q
    elif isinstance(embed, LoRAEmbed):
        probe = embed
    if probe is not None:
        lora_rank = int(getattr(probe, 'rank', 16))
        lora_alpha = float(getattr(probe, 'alpha', 32.0))
    else:
        lora_rank = 16
        lora_alpha = 32.0

    # Save weights using safetensors (PyTorch-free)
    try:
        from safetensors.numpy import save_file
        save_file(peft_weights, output_path / "adapter_model.safetensors")
        print("  - Saved in safetensors format (PyTorch-free)")
    except ImportError:
        np.savez(output_path / "adapter_model.npz", **peft_weights)
        print("  - Saved in numpy format (.npz)")

    # Create PEFT config
    peft_config = {
        "base_model_name_or_path": base_model_name,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": 0.0,
        "target_modules": sorted(list(target_modules)),
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": None,
    }

    with open(output_path / "adapter_config.json", 'w') as f:
        json.dump(peft_config, f, indent=2)

    print(f"LoRA adapter saved to {output_dir}")
    print(f"  - Rank: {lora_rank}, Alpha: {lora_alpha}")
    print(f"  - Target modules: {sorted(target_modules)}")
    print(f"  - Format: HuggingFace PEFT (compatible with vLLM)")

    return nnx.state(model, nnx.LoRAParam)


def load_lora_from_peft(adapter_dir: str, model: nnx.Module = None) -> nnx.State:
    """Load LoRA weights from HuggingFace PEFT format.

    Args:
        adapter_dir: Directory containing PEFT adapter
        model: Optional model with LoRA structure. If provided, loads weights directly.

    Returns:
        nnx.State that can be merged into a model with nnx.update()

    Example:
        >>> # Option 1: Get state and merge
        >>> lora_state = load_lora_from_peft("./my_lora")
        >>> nnx.update(model, lora_state)
        >>>
        >>> # Option 2: Load directly
        >>> load_lora_from_peft("./my_lora", model=model)
    """
    import json
    from pathlib import Path
    from flax import nnx
    from ueaj.model.lora import _build_lora_path_map

    adapter_path = Path(adapter_dir)

    # Load PEFT config
    with open(adapter_path / "adapter_config.json", 'r') as f:
        peft_config = json.load(f)

    print(f"Loading LoRA adapter from {adapter_dir}")
    print(f"  - Rank: {peft_config['r']}, Alpha: {peft_config['lora_alpha']}")
    target_modules = peft_config.get('target_modules', 'not specified')
    print(f"  - Target modules: {target_modules}")

    # Load weights - try safetensors first (modern format), then fall back to others
    safetensors_file = adapter_path / "adapter_model.safetensors"
    bin_file = adapter_path / "adapter_model.bin"
    npz_file = adapter_path / "adapter_model.npz"

    if safetensors_file.exists():
        # Load from safetensors (preferred format)
        from safetensors import safe_open
        with safe_open(str(safetensors_file), framework="numpy") as f:
            peft_weights_np = {key: f.get_tensor(key) for key in f.keys()}
    elif npz_file.exists():
        # Load from numpy format
        peft_weights_np = dict(np.load(npz_file))
    elif bin_file.exists():
        # Try torch format (requires torch)
        try:
            import torch
            peft_weights = torch.load(bin_file, map_location='cpu')
            peft_weights_np = {k: v.numpy() for k, v in peft_weights.items()}
        except ImportError:
            raise ImportError("torch is required to load .bin format adapters. Please save in safetensors format instead.")
    else:
        raise FileNotFoundError(f"No adapter weights found in {adapter_dir}")

    # If model provided, use direct module-based loading
    # Strategy: Collect per-layer weights and stack them into batch dimension
    if model is not None:
        # Group weights by module path (without layer index)
        from collections import defaultdict
        module_weights = defaultdict(lambda: {})  # {module_path: {layer_idx: {lora_A/B: weight}}}

        # First pass: collect weights by module and layer
        for peft_key, value in peft_weights_np.items():
            if 'lora_A.weight' not in peft_key and 'lora_B.weight' not in peft_key:
                continue

            # Parse PEFT key
            is_a = 'lora_A.weight' in peft_key
            param_name = 'lora_A' if is_a else 'lora_B'

            # Remove prefixes
            path_str = peft_key
            if path_str.startswith('base_model.model.base_model.model.'):
                path_str = path_str.replace('base_model.model.base_model.model.', '', 1)
            elif path_str.startswith('base_model.model.'):
                path_str = path_str.replace('base_model.model.', '', 1)
            path_str = path_str.replace(f'.{param_name}.weight', '')

            # Parse layer index and module path
            if 'layers' not in path_str:
                continue  # Skip non-layer modules for now

            parts = path_str.split('.')
            layer_idx = int(parts[2])

            # Build module path (without layer index)
            if 'self_attn' in path_str:
                if 'q_proj' in path_str:
                    module_key = 'attn.q'
                elif 'k_proj' in path_str:
                    module_key = 'attn.k'
                elif 'v_proj' in path_str:
                    module_key = 'attn.v'
                elif 'o_proj' in path_str:
                    module_key = 'attn.o'
                else:
                    continue
            elif 'mlp' in path_str:
                if 'up_proj' in path_str:
                    module_key = 'mlp.up_proj'
                elif 'down_proj' in path_str:
                    module_key = 'mlp.down_proj'
                elif 'gate_proj' in path_str:
                    module_key = 'mlp.gate_proj'
                else:
                    continue
            else:
                continue

            # PEFT format matches our format - no transpose needed!
            # PEFT: lora_A (in_features, rank), lora_B (rank, out_features)
            # Ours: lora_A (in_features, rank), lora_B (rank, out_features)

            # Store in grouped structure
            if layer_idx not in module_weights[module_key]:
                module_weights[module_key][layer_idx] = {}
            module_weights[module_key][layer_idx][param_name] = value

        # Second pass: stack weights and set them
        modules_updated = 0
        num_layers = getattr(model, 'num_layers', 16)  # Default to 16 if not specified

        for module_key, layer_weights in module_weights.items():
            try:
                # Navigate to the module
                parts = module_key.split('.')
                lora_module = model.layers
                for part in parts:
                    lora_module = getattr(lora_module, part)

                # Check if module has LoRA parameters
                if not (hasattr(lora_module, 'lora_A') and hasattr(lora_module, 'lora_B')):
                    continue

                # Stack weights for all layers
                lora_a_layers = []
                lora_b_layers = []
                for layer_idx in range(num_layers):
                    if layer_idx in layer_weights:
                        lora_a_layers.append(layer_weights[layer_idx].get('lora_A'))
                        lora_b_layers.append(layer_weights[layer_idx].get('lora_B'))
                    else:
                        # If missing, use zeros (though this shouldn't happen)
                        if 'lora_A' in layer_weights[0]:
                            lora_a_layers.append(np.zeros_like(layer_weights[0]['lora_A']))
                            lora_b_layers.append(np.zeros_like(layer_weights[0]['lora_B']))

                # Stack along batch dimension and convert to JAX
                if lora_a_layers[0] is not None:
                    lora_a_stacked = np.stack(lora_a_layers, axis=0)  # (num_layers, ...)
                    lora_b_stacked = np.stack(lora_b_layers, axis=0)

                    # Convert to JAX bfloat16
                    lora_a_jax = jnp.array(lora_a_stacked.astype(np.float32)).astype(jnp.bfloat16)
                    lora_b_jax = jnp.array(lora_b_stacked.astype(np.float32)).astype(jnp.bfloat16)

                    # Set the stacked weights
                    lora_module.lora_A.value = lora_a_jax
                    lora_module.lora_B.value = lora_b_jax
                    modules_updated += 1

            except (AttributeError, KeyError) as e:
                continue

        print(f"Loaded {modules_updated} LoRA modules")

        # Return the updated LoRA state
        return nnx.state(model, nnx.LoRAParam)

    else:
        raise NotImplementedError(
            "Loading LoRA without a model is not yet implemented. "
            "Please pass the model as an argument: load_lora_from_peft(path, model=model)"
        )
