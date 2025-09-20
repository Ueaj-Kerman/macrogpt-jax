"""Utilities for canonicalizing Einsum parameters within optimizer chains."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.nnx import statelib
from flax.nnx.variablelib import VariableState

from ueaj.model.einsum import Einsum, EinsumMetadata


Path = Tuple[str, ...]


@dataclass(frozen=True)
class _EinsumEntry:
    metadata: EinsumMetadata


def _collect_einsum_metadata(module: nnx.Module, prefix: Path = ()) -> Dict[Path, _EinsumEntry]:
    """Traverse the module tree and collect metadata for all Einsum weights."""

    entries: Dict[Path, _EinsumEntry] = {}

    if isinstance(module, Einsum):
        entries[prefix + ("w",)] = _EinsumEntry(metadata=module.metadata)
        return entries

    for name, value in vars(module).items():
        if isinstance(value, nnx.Module):
            entries.update(_collect_einsum_metadata(value, prefix + (name,)))
    return entries


def _is_state_mapping(value: Any) -> bool:
    return isinstance(value, (statelib.State, MutableMapping))


def _iter_mapping_items(mapping: Any) -> Iterable[Tuple[str, Any]]:
    if isinstance(mapping, statelib.State):
        return mapping.raw_mapping.items()
    if isinstance(mapping, Mapping):
        return mapping.items()
    return ()


def _get_child(mapping: Any, key: str) -> Any:
    if isinstance(mapping, statelib.State):
        return mapping.raw_mapping.get(key)
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


def _to_canonical(array: jax.Array, metadata: EinsumMetadata) -> jax.Array:
    canonical_ndim = len(metadata.canonical_shape)
    leading_ndim = array.ndim - canonical_ndim
    if leading_ndim < 0:
        raise ValueError(
            f"Array with shape {array.shape} is smaller than canonical dims {metadata.canonical_shape}"
        )

    perm = list(range(leading_ndim)) + [leading_ndim + ax for ax in metadata.transpose_axes]
    permuted = jnp.transpose(array, perm)

    batch_ndim = metadata.batch_ndim
    reducing_ndim = metadata.reducing_ndim
    non_reducing_ndim = metadata.non_reducing_ndim

    head_shape = permuted.shape[: leading_ndim + batch_ndim]
    reducing_dims = permuted.shape[
        leading_ndim + batch_ndim : leading_ndim + batch_ndim + reducing_ndim
    ]
    non_reducing_dims = permuted.shape[
        leading_ndim + batch_ndim + reducing_ndim : leading_ndim + batch_ndim + reducing_ndim + non_reducing_ndim
    ]

    reducing_size = int(math.prod(reducing_dims)) if reducing_dims else 1
    non_reducing_size = int(math.prod(non_reducing_dims)) if non_reducing_dims else 1

    return jnp.reshape(permuted, head_shape + (reducing_size, non_reducing_size))


def _from_canonical(array: jax.Array, metadata: EinsumMetadata, target: jax.Array) -> jax.Array:
    canonical_ndim = len(metadata.canonical_shape)
    target_shape = target.shape
    leading_ndim = len(target_shape) - canonical_ndim
    if leading_ndim < 0:
        raise ValueError(
            f"Target shape {target_shape} smaller than canonical dims {metadata.canonical_shape}"
        )

    trailing_shape = target_shape[leading_ndim:]
    transposed_trailing = tuple(trailing_shape[i] for i in metadata.transpose_axes)

    batch_ndim = metadata.batch_ndim
    reducing_ndim = metadata.reducing_ndim
    non_reducing_ndim = metadata.non_reducing_ndim

    batch_shape = transposed_trailing[:batch_ndim]
    reducing_shape = transposed_trailing[batch_ndim : batch_ndim + reducing_ndim]
    non_reducing_shape = transposed_trailing[batch_ndim + reducing_ndim :]

    leading_shape = target_shape[:leading_ndim]

    expected = leading_shape + batch_shape + (
        int(math.prod(reducing_shape)) if reducing_shape else 1,
        int(math.prod(non_reducing_shape)) if non_reducing_shape else 1,
    )
    reshaped = jnp.reshape(array, expected)

    expanded = jnp.reshape(
        reshaped,
        leading_shape + batch_shape + reducing_shape + non_reducing_shape,
    )

    total_ndim = len(leading_shape) + canonical_ndim
    inv_perm = list(range(total_ndim))
    for idx, axis in enumerate(metadata.transpose_axes):
        inv_perm[len(leading_shape) + axis] = len(leading_shape) + idx

    restored = jnp.transpose(expanded, inv_perm)
    return jnp.reshape(restored, target_shape)


def _transform_tree(
    tree: Any,
    params_ref: Any,
    entries: Dict[Path, _EinsumEntry],
    forward: bool,
    path: Path = (),
) -> Any:
    if isinstance(tree, VariableState):
        entry = entries.get(path)
        if entry is None:
            return tree
        array = tree.value
        metadata = entry.metadata
        if forward:
            transformed = _to_canonical(array, metadata)
        else:
            if params_ref is None or not isinstance(params_ref, VariableState):
                raise ValueError(
                    f"Parameters required to uncanonicalize path {path}, got {type(params_ref)}"
                )

            transformed = _from_canonical(array, metadata, params_ref.value)
        return tree.replace(value=transformed)

    if isinstance(tree, jax.Array):
        entry = entries.get(path)
        if entry is None:
            return tree
        metadata = entry.metadata
        if forward:
            return _to_canonical(tree, metadata)
        if params_ref is None:
            raise ValueError(f"Parameters required to uncanonicalize path {path}")
        return _from_canonical(tree, metadata, params_ref)

    if _is_state_mapping(tree):
        container: Dict[str, Any] = {}
        for key, child_value in _iter_mapping_items(tree):
            child_params = _get_child(params_ref, key) if params_ref is not None else None
            transformed_child = _transform_tree(
                child_value,
                child_params,
                entries,
                forward,
                path + (key,),
            )
            if isinstance(transformed_child, statelib.State):
                container[key] = transformed_child.raw_mapping
            else:
                container[key] = transformed_child
        if isinstance(tree, statelib.State):
            return statelib.State(container)
        if isinstance(tree, MutableMapping):
            new_mapping = type(tree)()
            new_mapping.update(container)
            return new_mapping
        return container

    return tree


def canonicalize_einsums(
    model: nnx.Module,
    transform: optax.GradientTransformation | None = None,
) -> optax.GradientTransformation:
    """Wrap a gradient transform so Einsum weights use canonical layout."""

    if transform is None:
        transform = optax.identity()

    entries = _collect_einsum_metadata(model)

    def init_fn(params: Any):
        canonical_params = _transform_tree(params, None, entries, forward=True)
        return transform.init(canonical_params)

    def update_fn(updates, state, params=None, **extra):
        canonical_updates = _transform_tree(updates, None, entries, forward=True)
        canonical_params = (
            _transform_tree(params, None, entries, forward=True)
            if params is not None
            else None
        )
        canonical_result, new_state = transform.update(
            canonical_updates, state, canonical_params, **extra
        )
        if params is None:
            restored_updates = canonical_result
        else:
            restored_updates = _transform_tree(
                canonical_result,
                params,
                entries,
                forward=False,
            )
        return restored_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)
