#!/usr/bin/env python3
"""
Script to sample from a loaded Llama model.
"""
import argparse
import json
import os
import sys
from typing import Optional, List
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download, HfFolder

from ueaj.model import LlamaModel, apply_lora_to_model
from ueaj.llama import load_lora_from_peft, save_lora_to_peft, from_pretrained
from ueaj.llama.weight_loader import create_llama_model_config
from flax.nnx import rnglib as rng


def sample_from_logits(
    logits: jax.Array,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    min_p: Optional[float] = None,
    rng_key: Optional[jax.random.PRNGKey] = None,
) -> int:
    """
    Sample a token from logits using temperature, top-k, top-p, and min-p.
    
    Args:
        logits: Logits for next token prediction (vocab_size,)
        temperature: Temperature for sampling
        top_k: If set, only sample from top k tokens
        top_p: If set, only sample from tokens with cumulative probability < top_p
        min_p: If set, only sample from tokens with probability >= min_p * top_probability
        rng_key: Random key for sampling
        
    Returns:
        Sampled token ID
    """
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)
        
    # Apply temperature
    if temperature > 0:
        logits = logits / temperature
    else:
        # Temperature 0 means greedy decoding
        return jnp.argmax(logits).item()
    
    # Apply min-p filtering (do this before other filters for better results)
    if min_p is not None and min_p > 0:
        # Get probabilities
        probs = jax.nn.softmax(logits)
        # Find the top probability
        top_prob = jnp.max(probs)
        # Calculate dynamic threshold
        threshold = min_p * top_prob
        # Filter tokens below threshold
        mask = probs >= threshold
        logits = jnp.where(mask, logits, -jnp.inf)
    
    # Apply top-k filtering
    if top_k is not None and top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        top_k_indices = jax.lax.top_k(logits, top_k)[1]
        mask = jnp.zeros_like(logits, dtype=bool)
        mask = mask.at[top_k_indices].set(True)
        logits = jnp.where(mask, logits, -jnp.inf)
    
    # Apply top-p (nucleus) filtering
    if top_p is not None and 0 < top_p < 1.0:
        sorted_indices = jnp.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        sorted_probs = jax.nn.softmax(sorted_logits)
        cumsum_probs = jnp.cumsum(sorted_probs)
        
        # Find cutoff
        cutoff_idx = jnp.searchsorted(cumsum_probs, top_p) + 1
        cutoff_idx = min(cutoff_idx, len(sorted_indices))
        
        # Create mask
        mask = jnp.zeros_like(logits, dtype=bool)
        mask = mask.at[sorted_indices[:cutoff_idx]].set(True)
        logits = jnp.where(mask, logits, -jnp.inf)
    
    # Sample from distribution
    probs = jax.nn.softmax(logits)
    token_id = jax.random.categorical(rng_key, logits)
    
    return token_id.item()


def generate(
    model: LlamaModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.95,
    min_p: Optional[float] = None,
    seed: int = 42,
) -> str:
    """
    Generate text from a prompt using the model.
    
    Args:
        model: The Llama model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        seed: Random seed
        
    Returns:
        Generated text
    """
    # Tokenize prompt
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    
    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    
    # Generate tokens
    generated_ids = input_ids.copy()
    
    for i in range(max_new_tokens):
        # Get logits for next token
        logits = model(generated_ids)
        
        next_token_logits = logits[0, -1, :]  # Get logits for last position
        
        # Sample next token
        rng, sample_rng = jax.random.split(rng)
        next_token_id = sample_from_logits(
            next_token_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            rng_key=sample_rng,
        )
        
        # Append to sequence
        generated_ids = jnp.concatenate([
            generated_ids,
            jnp.array([[next_token_id]])
        ], axis=1)
        
        # Check for EOS token
        if next_token_id == tokenizer.eos_token_id:
            break
        else:
            print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    
    # Decode generated text
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return generated_text


def download_model_if_needed(model_id: str, cache_dir: Optional[str] = None) -> str:
    """Download model from HuggingFace if not already cached."""
    print(f"Downloading/loading model: {model_id}")
    
    # Download model (will use cache if already downloaded)
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False,
        token=HfFolder.get_token(),
        allow_patterns=["*.safetensors", "*.json", "*.txt"],
        ignore_patterns=["*.bin", "*.h5", "*.msgpack", "*.onnx", "*.pt"],
    )
    
    print(f"Model loaded from: {local_dir}")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="Sample from a Llama model")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model ID or local path (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer name (defaults to same as model)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k filtering parameter",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) filtering parameter",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min-p filtering parameter (e.g., 0.05)",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="HuggingFace LoRA adapter repo or local path (PEFT format)",
    )
    parser.add_argument(
        "--save-random-adapter",
        type=str,
        default=None,
        help="Path to save a randomly initialized LoRA adapter (PEFT format)",
    )
    parser.add_argument(
        "--random-rank",
        type=int,
        default=16,
        help="Rank for randomly initialized LoRA (used with --save-random-adapter)",
    )
    parser.add_argument(
        "--random-alpha",
        type=float,
        default=32.0,
        help="Alpha for randomly initialized LoRA (used with --save-random-adapter)",
    )
    parser.add_argument(
        "--random-std",
        type=float,
        default=0.1,
        help="Stddev for random LoRA params (used with --save-random-adapter)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reserved flag (no-op)",
    )

    args = parser.parse_args()
    
    # Set up JAX
    print(f"JAX devices: {jax.devices()}")

    print(f"Loading model: {args.model}")
    print(f"Using dtype: {args.dtype}")

    # Download base model files (uses cache if available)
    model_path = snapshot_download(
        repo_id=args.model,
        cache_dir=None,
        allow_patterns=["*.safetensors", "*.json"],
        ignore_patterns=["*.bin", "*.h5"],
    )

    # Resolve dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    cfg_dtype = dtype_map.get(args.dtype, jnp.bfloat16)

    # Create model structure from HF config
    model_class = create_llama_model_config(args.model, dtype=cfg_dtype)
    print("Creating model structure...")
    model = model_class(rngs=rng.Rngs(0))

    # Build a pure nnx.State from safetensors and merge
    print("Loading base model weights into model...")
    model = from_pretrained(model_path, existing_model=model)

    # Load tokenizer early (may be reused for random LoRA sample)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=None)
    print("Model loaded successfully!")

    # Optionally apply and load a LoRA adapter
    if args.adapter:
        print(f"\nLoading LoRA adapter: {args.adapter}")
        # Download/load adapter files (PEFT format)
        adapter_path = snapshot_download(
            repo_id=args.adapter,
            allow_patterns=["adapter_config.json", "adapter_model.safetensors", "*.json", "*.safetensors"],
        )

        # Read PEFT adapter config to set rank/alpha and module targets
        adapter_cfg_path = Path(adapter_path) / "adapter_config.json"
        if not adapter_cfg_path.exists():
            raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

        with open(adapter_cfg_path, "r") as f:
            peft_cfg = json.load(f)

        rank = int(peft_cfg.get("r", 16))
        alpha = float(peft_cfg.get("lora_alpha", 32.0))
        target_modules = peft_cfg.get("target_modules", None)

        # Map PEFT target module names to substrings in our module paths
        # e.g., q_proj -> 'q', k_proj -> 'k', embed_tokens -> 'embed_tokens'
        if isinstance(target_modules, list) and target_modules:
            mapping = {
                "q_proj": "q",
                "k_proj": "k",
                "v_proj": "v",
                "o_proj": "o",
                "up_proj": "up_proj",
                "down_proj": "down_proj",
                "gate_proj": "gate_proj",
                "embed_tokens": "embed_tokens",
            }
            targets = [mapping.get(m, m) for m in target_modules]
        else:
            targets = None  # Apply to all supported modules (excluding lm_head)

        # Apply LoRA structure (pure: returns a new model copy)
        model = apply_lora_to_model(
            model,
            rank=rank,
            alpha=alpha,
            target_modules=targets,
            rngs=rng.Rngs(0),
        )

        # Load adapter weights into the model
        _ = load_lora_from_peft(adapter_path, model=model)
        print("LoRA adapter loaded.")

    # Optionally create a random LoRA adapter, save it, and verify load
    if args.save_random_adapter:
        out_dir = args.save_random_adapter
        print(f"\nCreating randomly initialized LoRA (rank={args.random_rank}, alpha={args.random_alpha})...")
        # Apply LoRA structure to a copy
        model_with_lora = apply_lora_to_model(
            model,
            rank=args.random_rank,
            alpha=args.random_alpha,
            target_modules=['q', 'k', 'v', 'o'],
            rngs=rng.Rngs(args.seed),
        )

        # Randomize LoRA parameters in-place
        key = jax.random.PRNGKey(args.seed)
        def rand_like(arr):
            nonlocal key
            key, sk = jax.random.split(key)
            return jax.random.normal(sk, arr.shape, dtype=arr.dtype) * args.random_std

        # For vmapped LoRA modules on attention projections
        for name in ['q', 'k', 'v', 'o']:
            mod = getattr(model_with_lora.layers.attn, name)
            if hasattr(mod, 'lora_A') and hasattr(mod, 'lora_B'):
                mod.lora_A.value = rand_like(mod.lora_A.value)
                mod.lora_B.value = rand_like(mod.lora_B.value)

        # Save adapter
        print(f"Saving random LoRA adapter to: {out_dir}")
        save_lora_to_peft(model_with_lora, out_dir, base_model_name=args.model)

        # Verify we can load it back into a fresh LoRA-applied copy
        print("Verifying reload of random LoRA adapter...")
        verify_model = apply_lora_to_model(
            model,
            rank=args.random_rank,
            alpha=args.random_alpha,
            target_modules=['q', 'k', 'v', 'o'],
            rngs=rng.Rngs(args.seed + 1),
        )
        _ = load_lora_from_peft(out_dir, model=verify_model)

        # Generate with the random LoRA (likely nonsensical)
        print("\nRandom LoRA sample (expect nonsensical output):")
        gen_random = generate(
            verify_model,
            tokenizer,
            args.prompt,
            max_new_tokens=max(32, args.max_new_tokens // 2),
            temperature=max(0.8, args.temperature),
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            seed=args.seed + 2,
        )
        print(f"\nRandom LoRA generated text:\n{gen_random}")

    # Override tokenizer if specified
    if args.tokenizer:
        print(f"Loading custom tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("Generating...")
    
    generated_text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        seed=args.seed,
    )
    
    print(f"\nGenerated text:\n{generated_text}")


if __name__ == "__main__":
    main()
