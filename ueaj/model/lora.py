"""LoRA (Low-Rank Adaptation) implementation for Einsum layers.

This module provides LoRA fine-tuning capabilities compatible with HuggingFace PEFT format.
"""
import operator
from functools import reduce
from typing import Optional, List, Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import rnglib as rng

from ueaj.model.einsum import Einsum, EinsumMetadata

# Use Flax NNX's built-in LoRAParam
LoRAParam = nnx.LoRAParam


class LoRAEinsum(nnx.Module):
    """Einsum layer with LoRA adaptation.

    Wraps an existing Einsum layer and adds low-rank adaptation matrices A and B.
    The effective weight becomes: W_eff = W_frozen + (alpha / rank) * B @ A

    Args:
        base_einsum: The frozen Einsum layer to adapt
        rank: LoRA rank (typically 4-64)
        alpha: LoRA scaling parameter (typically 16-32)
        rngs: Random number generators

    Example:
        >>> base = Einsum("bnd,dh->bnh", size_dict={'d': 512, 'h': 2048}, rngs=rngs)
        >>> lora = LoRAEinsum(base, rank=16, alpha=32, rngs=rngs)
        >>> output = lora(input)  # Uses base + LoRA adaptation
    """

    def __init__(
        self,
        base_einsum: Einsum,
        rank: int = 16,
        alpha: float = 32.0,
        *,
        rngs: rng.Rngs,
    ):
        super().__init__()

        # Store base einsum (weights stay frozen)
        self.base = base_einsum

        # Get dimensions from metadata and actual weight shape
        # The reduced form is: (...batch_dims, reducing_size, non_reducing_size)
        metadata = base_einsum.metadata

        # Compute flattened dimensions
        reducing_size = reduce(operator.mul, metadata.reducing_shape, 1)
        non_reducing_size = reduce(operator.mul, metadata.non_reducing_shape, 1)

        # Account for nnx.vmap/scan leading batch axes that are not encoded
        # in Einsum.metadata.batch_shape by comparing actual weight dims.
        canonical_ndim = len(metadata.canonical_shape)
        actual_w_shape = base_einsum.w.value.shape
        extra_ndim = len(actual_w_shape) - canonical_ndim
        if extra_ndim > 0:
            vmap_batch_shape = actual_w_shape[:extra_ndim]
        else:
            vmap_batch_shape = ()

        effective_batch_shape = tuple(vmap_batch_shape) + tuple(metadata.batch_shape)

        # Total dimensions for flattened 2D LoRA matrices
        if reduce(operator.mul, effective_batch_shape, 1) > 1:
            lora_a_shape = effective_batch_shape + (reducing_size, rank)
            lora_b_shape = effective_batch_shape + (rank, non_reducing_size)
        else:
            lora_a_shape = (reducing_size, rank)
            lora_b_shape = (rank, non_reducing_size)

        # Initialize LoRA matrices
        # A: normal distribution with small std
        # B: zeros (standard LoRA practice - ensures initial adaptation is zero)
        self.lora_A = LoRAParam(
            jax.random.normal(rngs.params(), lora_a_shape, dtype=base_einsum.w.value.dtype) * 0.01
        )
        self.lora_B = LoRAParam(
            jnp.zeros(lora_b_shape, dtype=base_einsum.w.value.dtype)
        )

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Store metadata for dimension handling
        self.metadata = metadata

        # Cache canonical metadata transforms for mapping reduced -> canonical
        self._canonical_tail_ndim = len(metadata.canonical_shape)
        # Permutation from canonical -> [batch, reducing, non_reducing]
        self._to_reduced_axes = metadata.transpose_axes
        # Inverse permutation to go back to canonical
        inv = [0] * len(self._to_reduced_axes)
        for i, a in enumerate(self._to_reduced_axes):
            inv[a] = i
        self._to_canonical_axes = inv

        # No runtime prints in production

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with LoRA adaptation.

        Args:
            x: Input tensor

        Returns:
            Output tensor: base_output + lora_adaptation * scaling
        """
        # Forward through frozen base
        base_out = self.base(x)

        # Compute LoRA weight in reduced shape and map to canonical
        # (...batch, in, r) @ (...batch, r, out) -> (...batch, in, out)
        # Normalize orientation so A: (..., in, rank), B: (..., rank, out)
        A = self.lora_A.value
        B = self.lora_B.value
        if A.ndim >= 2 and A.shape[-1] != self.rank and A.shape[-2] == self.rank:
            A = jnp.swapaxes(A, -1, -2)
        if B.ndim >= 2 and B.shape[-2] != self.rank and B.shape[-1] == self.rank:
            B = jnp.swapaxes(B, -1, -2)

        lora_w_red = jnp.einsum('...ir,...ro->...io', A, B)

        # Reshape reduced to split dims: (...batch, reducing_shape..., non_reducing_shape...)
        batch_ndim = lora_w_red.ndim - 2
        batch_prefix = lora_w_red.shape[:batch_ndim]
        lora_w_split = lora_w_red.reshape(
            batch_prefix + self.metadata.reducing_shape + self.metadata.non_reducing_shape
        )

        # Transpose tail dims back to canonical order, leaving any extra vmap dims in front
        if self._canonical_tail_ndim > 0:
            # Build full permutation for total dims: [batch_prefix..., tail...] where tail permutes
            perm = list(range(batch_ndim)) + [batch_ndim + i for i in self._to_canonical_axes]
            lora_w_canonical = jnp.transpose(lora_w_split, axes=perm)
        else:
            lora_w_canonical = lora_w_split

        # Einsum with the LoRA canonical weight using the same expression as base
        lora_out = jnp.einsum(self.base.einsum_expr, x, lora_w_canonical)

        # Combine: base + scaled LoRA
        return base_out + lora_out * self.scaling



class LoRAEmbed(nnx.Module):
    """Embedding layer with LoRA adaptation.

    Wraps an existing Embed layer and adds low-rank adaptation matrices A and B.
    The effective embedding becomes: Emb_eff = Emb_frozen + (alpha / rank) * B @ A

    Args:
        base_embed: The frozen Embed layer to adapt
        rank: LoRA rank (typically 4-64)
        alpha: LoRA scaling parameter (typically 16-32)
        rngs: Random number generators

    Example:
        >>> base = nnx.Embed(num_embeddings=50000, features=512, rngs=rngs)
        >>> lora = LoRAEmbed(base, rank=16, alpha=32, rngs=rngs)
        >>> output = lora(token_ids)  # Uses base + LoRA adaptation
    """

    def __init__(
        self,
        base_embed: nnx.Embed,
        rank: int = 16,
        alpha: float = 32.0,
        *,
        rngs: rng.Rngs,
    ):
        super().__init__()

        # Store base embedding (weights stay frozen)
        self.base = base_embed

        # Get embedding dimensions
        num_embeddings = base_embed.num_embeddings
        features = base_embed.features
        dtype = base_embed.embedding.value.dtype

        # Initialize LoRA matrices
        # A: (num_embeddings, rank) - one rank vector per token
        # B: (rank, features) - projects rank back to embedding dimension
        self.lora_A = LoRAParam(
            jax.random.normal(rngs.params(), (num_embeddings, rank), dtype=dtype) * 0.01
        )
        self.lora_B = LoRAParam(
            jnp.zeros((rank, features), dtype=dtype)
        )

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Forward pass with LoRA adaptation.

        Args:
            inputs: Token IDs of shape (...,)

        Returns:
            Embeddings of shape (..., features)
        """
        # Forward through frozen base
        base_out = self.base(inputs)

        # LoRA path: lookup A, then multiply by B
        # inputs: (...,) token indices
        # lora_A[inputs]: (..., rank)
        # lora_B: (rank, features)
        # result: (..., features)
        lora_a_selected = self.lora_A.value[inputs]  # (..., rank)
        lora_out = lora_a_selected @ self.lora_B.value  # (..., features)

        # Combine: base + scaled LoRA
        return base_out + lora_out * self.scaling

    def attend(self, query: jax.Array) -> jax.Array:
        """Attend method for tied embeddings (if used as lm_head).

        Args:
            query: Query vectors of shape (..., features)

        Returns:
            Logits of shape (..., num_embeddings)
        """
        # For the tied embedding case, we need to compute query @ embedding.T
        # With LoRA: query @ (base.T + scaling * (B @ A).T)
        #         = query @ base.T + scaling * query @ A.T @ B.T

        # Base attend
        base_logits = self.base.attend(query)

        # LoRA path
        # query: (..., features)
        # lora_B.T: (features, rank)
        # lora_A.T: (rank, num_embeddings)
        lora_logits = (query @ self.lora_B.value.T) @ self.lora_A.value.T

        return base_logits + lora_logits * self.scaling


def apply_lora_to_model(
    model: nnx.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
    *,
    rngs: rng.Rngs,
) -> nnx.Module:
    """Apply LoRA adaptation to specific modules and return a new model.

    This function is pure with respect to the input `model`: it first
    constructs a copy via `nnx.split`/`nnx.merge`, then performs the LoRA
    module surgery on the copy and returns it. The original model remains
    unchanged.

    Manually traverses the model copy and applies LoRA to Einsum and Embed
    layers whose path matches any of the target_modules strings.

    Args:
        model: The model to adapt
        rank: LoRA rank
        alpha: LoRA scaling parameter
        target_modules: List of module name patterns to adapt (e.g., ['q', 'v', 'k', 'o', 'embed']).
                       Matches if the pattern appears anywhere in the module path.
                       If None, adapts ALL Einsum/Embed modules except those in exclude_modules.
        exclude_modules: List of module name patterns to exclude (e.g., ['lm_head']).
                        Default: ['lm_head'] to avoid adapting output projection.
        rngs: Random number generators

    Returns:
        A new model instance with LoRA applied

    Example:
        >>> # Apply LoRA to all attention projections
        >>> model = apply_lora_to_model(model, rank=16, target_modules=['q', 'k', 'v', 'o'], rngs=rngs)
        >>>
        >>> # Apply LoRA to everything except lm_head (default, includes embeddings)
        >>> model = apply_lora_to_model(model, rank=16, rngs=rngs)
        >>>
        >>> # Apply LoRA only to embeddings
        >>> model = apply_lora_to_model(model, rank=16, target_modules=['embed'], rngs=rngs)
    """
    # Work on a copy to keep this function pure
    graph_def, state = nnx.split(model)
    model_copy = nnx.merge(graph_def, state)

    if exclude_modules is None:
        # Default: don't adapt lm_head
        exclude_modules = ['lm_head']

    # Track modules to replace (path -> LoRA module)
    modules_to_replace = {}

    # Helper function to traverse the model tree
    def traverse_modules(obj, path=()):
        """Recursively traverse module attributes."""
        # Check if this is a module we want to adapt
        if isinstance(obj, (Einsum, nnx.Embed)):
            # Path components for exact matching
            path_comps = [str(p) for p in path]

            # Check if module should be excluded (exact component match)
            if any(excl in path_comps for excl in exclude_modules):
                return

            # Check if module matches target patterns
            if target_modules is None:
                # Adapt all Einsums/Embeds (except excluded)
                should_adapt = True
            else:
                # Check if any target equals a path component
                should_adapt = any(target in path_comps for target in target_modules)

            if should_adapt:
                # Create appropriate LoRA wrapper
                if isinstance(obj, Einsum):
                    lora_module = LoRAEinsum(obj, rank, alpha, rngs=rngs)
                elif isinstance(obj, nnx.Embed):
                    lora_module = LoRAEmbed(obj, rank, alpha, rngs=rngs)
                else:
                    return

                modules_to_replace[path] = lora_module

        # Traverse child attributes if this is a module
        if isinstance(obj, nnx.Module):
            # Get all attributes
            for attr_name in dir(obj):
                # Skip private/magic attributes
                if attr_name.startswith('_'):
                    continue

                try:
                    attr = getattr(obj, attr_name)
                    # Only traverse Module attributes
                    if isinstance(attr, nnx.Module):
                        traverse_modules(attr, path + (attr_name,))
                except:
                    # Skip attributes that can't be accessed
                    pass

    # Traverse the model
    traverse_modules(model_copy)

    # Apply replacements by traversing the path and setting attributes
    for path, lora_module in modules_to_replace.items():
        # Navigate to parent and set the child
        parent = model_copy
        for key in path[:-1]:
            parent = getattr(parent, key)

        # Set the LoRA module
        setattr(parent, path[-1], lora_module)

    return model_copy


def _build_lora_path_map(model: nnx.Module) -> Dict[str, Tuple[Tuple, nnx.Module]]:
    """Build a mapping of string paths to (path_tuple, module) for LoRA modules.

    Uses Flax's built-in flat state representation for efficient path traversal.

    Args:
        model: Model to traverse

    Returns:
        Dict mapping path strings like 'layers/0/attn/q' to (path_tuple, module)
    """
    path_map = {}

    # Get all LoRA parameters using flat state
    lora_state = nnx.state(model, nnx.LoRAParam)
    flat_state = nnx.to_flat_state(lora_state)

    # Extract unique module paths (remove 'lora_A'/'lora_B' suffix)
    module_paths = set()
    for key_tuple in flat_state._keys:
        # key_tuple looks like: ('layers', 'attn', 'q', 'lora_A')
        # We want the module path: ('layers', 'attn', 'q')
        module_path = key_tuple[:-1]  # Remove 'lora_A' or 'lora_B'
        module_paths.add(module_path)

    # For each module path, navigate to get the actual module
    for module_path in module_paths:
        path_str = '/'.join(str(p) for p in module_path)

        # Navigate to the module
        obj = model
        for key in module_path:
            obj = getattr(obj, key)

        # Store the mapping
        path_map[path_str] = (module_path, obj)

    return path_map
