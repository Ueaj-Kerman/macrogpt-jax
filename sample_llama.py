#!/usr/bin/env python3
"""
Script to sample from a loaded Llama model.
"""
import argparse
import os
from typing import Optional, List
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from ueaj.model.llama import LlamaModel, LlamaConfig


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
        help="Use KV cache for faster generation",
    )
    
    args = parser.parse_args()
    
    # Set up JAX
    print(f"JAX devices: {jax.devices()}")
    
    # Check if model is a local path or HuggingFace ID
    if os.path.exists(args.model):
        model_path = args.model
    else:
        # Download from HuggingFace
        model_path = download_model_if_needed(args.model)
    
    # Load tokenizer
    tokenizer_name = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Determine dtype
    dtype_map = {
        "float32": jnp.float32,
        "float16": jnp.float16,
        "bfloat16": jnp.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # Load model
    print(f"Loading model from: {model_path}")
    print(f"Using dtype: {args.dtype}")
    print(f"Using KV cache: {args.use_cache}")
    
    model = LlamaModel.from_pretrained(
        model_path,
        dtype=dtype,
        abstract=True,
    )
    
    print(f"Model loaded successfully!")
    print(f"Model config: {model.config}")
    
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