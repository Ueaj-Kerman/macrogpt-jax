#!/usr/bin/env python3
"""Example script for LoRA fine-tuning of LLaMA models.

This script demonstrates:
1. Loading a pretrained LLaMA model (or creating a fresh one)
2. Applying LoRA adaptations
3. Training on a dataset
4. Saving the LoRA adapter in HuggingFace PEFT format

Usage:
    # Train with default UEAJ 150M model
    ./scripts/run_python.sh scripts/train_lora.py

    # Train with pretrained LLaMA model
    ./scripts/run_python.sh scripts/train_lora.py --model meta-llama/Llama-3.2-1B

    # Customize LoRA settings
    ./scripts/run_python.sh scripts/train_lora.py --rank 32 --alpha 64 --lr 1e-4

Environment variables:
    LORA_OUTPUT_DIR: Where to save the LoRA adapter (default: ./lora_adapter)
"""

import os
import argparse
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.nnx import rnglib as rng
import datasets
import transformers

from ueaj import data
from ueaj.model import configs, apply_lora_to_model
from ueaj.train import make_lora_optimizer, print_lora_info, get_lora_param_count
from ueaj.llama import save_lora_to_peft, load_lora_from_peft

# Try to import wandb, but don't require it
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    print("wandb not installed, skipping logging")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for LLaMA models")

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="HuggingFace model to load (default: use UEAJ_150M)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for HuggingFace models"
    )

    # LoRA arguments
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="LoRA rank (default: 16)"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=32.0,
        help="LoRA alpha scaling (default: 32.0)"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules to adapt (default: all except lm_head)"
    )

    # Training arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1)"
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Sequence length (default: 2048)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum training steps (default: 1000)"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Warmup steps (default: 100)"
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log every N steps (default: 10)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help="Save checkpoint every N steps (default: 500)"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="HuggingFace dataset (default: HuggingFaceFW/fineweb-edu)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sample-10BT",
        help="Dataset config name (default: sample-10BT)"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("LORA_OUTPUT_DIR", "./lora_adapter"),
        help="Where to save LoRA adapter (default: $LORA_OUTPUT_DIR or ./lora_adapter)"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lora-finetuning",
        help="Wandb project name (default: lora-finetuning)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name for logging"
    )

    return parser.parse_args()


def create_or_load_model(args):
    """Create a fresh model or load from HuggingFace."""
    if args.model is None:
        print("Creating UEAJ_150M model...")
        model = configs.UEAJ_150M(rngs=rng.Rngs(0))
    else:
        print(f"Loading model from {args.model}...")
        # TODO: Implement HF model loading with weight_loader
        # For now, fall back to UEAJ model
        print("Warning: HF loading not yet implemented, using UEAJ_150M")
        model = configs.UEAJ_150M(rngs=rng.Rngs(0))

    return model


def setup_tokenizer(seq_len):
    """Setup tokenizer."""
    print("Loading tokenizer...")
    tokenizer = transformers.GPT2TokenizerFast.from_pretrained("openai-community/gpt2")
    tokenizer.model_max_length = seq_len
    return tokenizer


def setup_dataset(tokenizer, batch_size, seq_len, dataset_name, dataset_config):
    """Setup training dataset."""
    print(f"Loading dataset: {dataset_name}...")
    dataset = datasets.load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
    )

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 50256

    print("Setting up data iterator...")
    dataset, (_, _) = data.prepare_dataset(
        dataset,
        tokenizer,
        batch_size=batch_size,
        seq_len=seq_len,
        pad_token_id=pad_token_id,
        buffer_size=32
    )

    return dataset, pad_token_id


def train_step(graph_def, full_state, lora_state, opt_state, optimizer, tokens, doc_ids, pad_token):
    """Single training step - only LoRA parameters are trained.

    Args:
        graph_def: Model graph definition
        full_state: Full model state (all parameters)
        lora_state: LoRA parameter state
        opt_state: Optimizer state
        optimizer: Optimizer
        tokens: Input tokens
        doc_ids: Document IDs
        pad_token: Padding token

    Returns:
        Updated full_state, lora_state, opt_state, loss, stats
    """
    from ueaj.data.loss import next_token_loss

    def loss_fn(lora_state):
        """Loss function w.r.t. LoRA parameters only."""
        # Merge LoRA params into full state for forward pass
        merged_state = full_state.copy()
        jax.tree.map(lambda a, b: a.update(b) if hasattr(a, 'update') else b, merged_state, lora_state)

        # Create temporary model for forward pass
        model = nnx.merge(graph_def, merged_state)

        # Forward pass
        logits = model(tokens)
        loss, stats = next_token_loss(logits, tokens, document_ids=doc_ids, pad_token=pad_token)
        return loss, stats

    # Compute gradients w.r.t. LoRA state only (base params frozen)
    (loss, stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(lora_state)

    # Update LoRA parameters
    updates, opt_state = optimizer.update(grads, opt_state, lora_state)
    lora_state = optax.apply_updates(lora_state, updates)

    # Update full state with new LoRA params
    full_state = full_state.copy()
    nnx.update(full_state, lora_state)

    return full_state, lora_state, opt_state, loss, stats


def main():
    args = parse_args()

    # Initialize wandb if available
    if HAS_WANDB and args.run_name:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args)
        )

    # Create/load model
    model = create_or_load_model(args)

    # Apply LoRA
    print(f"\nApplying LoRA (rank={args.rank}, alpha={args.alpha})...")
    model = apply_lora_to_model(
        model,
        rank=args.rank,
        alpha=args.alpha,
        target_modules=args.target_modules,
        rngs=rng.Rngs(42)
    )

    # Print LoRA info
    print_lora_info(model)

    # Setup optimizer
    print(f"\nSetting up optimizer (lr={args.lr})...")
    optimizer = make_lora_optimizer(
        lr=args.lr,
        dtype=jnp.float32
    )

    # Split model into graph definition and state
    graph_def, full_state = nnx.split(model, nnx.Param)

    # Extract LoRA state for training
    lora_state = nnx.state(model, nnx.LoRAParam)

    # Initialize optimizer with LoRA state only
    opt_state = optimizer.init(lora_state)

    # Setup tokenizer and dataset
    tokenizer = setup_tokenizer(args.seq_len)
    train_dataset, pad_token = setup_dataset(
        tokenizer,
        args.batch_size,
        args.seq_len,
        args.dataset,
        args.dataset_name
    )

    # Training loop
    print(f"\nStarting training for {args.max_steps} steps...")
    print("=" * 60)

    step = 0
    start_time = time.time()

    for batch in train_dataset:
        if step >= args.max_steps:
            break

        tokens, doc_ids = batch

        # Compute warmup factor
        warmup = min(step / args.warmup_steps, 1.0) if args.warmup_steps > 0 else 1.0

        # Training step (only trains LoRA params)
        full_state, lora_state, opt_state, loss, stats = train_step(
            graph_def, full_state, lora_state, opt_state, optimizer,
            tokens, doc_ids, pad_token
        )

        # Request next batch
        train_dataset.send(None)

        # Wait for computation to finish
        loss.block_until_ready()

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) * args.batch_size * args.seq_len / elapsed

            print(f"[{step:4d}] Loss: {float(loss):.4f}, "
                  f"Warmup: {warmup:.2f}, "
                  f"Tokens/s: {tokens_per_sec:.0f}")

            if HAS_WANDB and args.run_name:
                wandb.log({
                    "loss": float(loss),
                    "warmup": warmup,
                    "tokens_per_sec": tokens_per_sec,
                    "step": step
                })

        # Save checkpoint
        if step > 0 and step % args.save_every == 0:
            print(f"\nSaving checkpoint to {args.output_dir}...")
            # Merge state back into model for saving
            model_for_save = nnx.merge(graph_def, full_state)
            save_lora_to_peft(
                model_for_save,
                args.output_dir,
                base_model_name=args.model or "ueaj_150m"
            )
            print("Checkpoint saved!")

        step += 1

    # Final save
    print(f"\nTraining complete! Saving final adapter to {args.output_dir}...")
    # Model already has updated LoRA params from last train_step
    save_lora_to_peft(
        model,
        args.output_dir,
        base_model_name=args.model or "ueaj_150m"
    )

    print("\nDone!")
    print("=" * 60)

    if HAS_WANDB and args.run_name:
        wandb.finish()


if __name__ == "__main__":
    main()