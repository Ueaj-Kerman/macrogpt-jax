#!/usr/bin/env python3
"""Tests for canonicalizing Einsum weights within optimizer transforms."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from ueaj.model import configs
from ueaj.opt import canonicalize_einsums


def _debug_state_transform(label: str) -> optax.GradientTransformation:
    """Create a gradient transform that prints state structure for inspection."""

    def init_fn(_):
        return ()

    def update_fn(updates, state, params=None, **_):
        flat = nnx.to_flat_state(updates)
        print(f"{label} state:")
        for path, leaf in zip(flat._keys, flat._values):
            obj = leaf.value if hasattr(leaf, "value") else leaf
            shape = getattr(obj, "shape", None)
            dtype = getattr(obj, "dtype", None)
            path_str = "/".join(path)
            print(f"  {path_str or '(root)'}: shape={shape} dtype={dtype}")
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def test_roundtrip_preserves_structure_and_values():
    model = configs.UEAJ_NH(rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)

    updates = jax.tree.map(
        lambda node: node.replace(value=jnp.asarray(node.value, dtype=jnp.float32))
        if hasattr(node, "replace") and hasattr(node, "value")
        else node,
        params,
    )

    wrapped = canonicalize_einsums(
        model,
        _debug_state_transform("canonicalized"),
    )
    tx = optax.chain(wrapped, _debug_state_transform("restored"))
    state = tx.init(params)

    restored, _ = tx.update(updates, state, params)

    flattened_original = nnx.to_flat_state(updates)
    flattened_restored = nnx.to_flat_state(restored)

    assert flattened_original._keys == flattened_restored._keys
    for (path, orig), (_, new) in zip(
        zip(flattened_original._keys, flattened_original._values),
        zip(flattened_restored._keys, flattened_restored._values),
    ):
        if hasattr(orig, "value"):
            assert isinstance(new, orig.__class__)
            assert jnp.array_equal(orig.value, new.value)
        else:
            assert isinstance(new, orig.__class__)
            assert jnp.array_equal(orig, new)


def test_canonical_shape_for_query_projection():
    model = configs.UEAJ_NH(rngs=nnx.Rngs(0))
    params = nnx.state(model, nnx.Param)
    updates = jax.tree.map(lambda x: x, params)

    canon_tx = canonicalize_einsums(
        model,
        _debug_state_transform("canonicalized"),
    )
    state = canon_tx.init(params)
    canon_updates, _ = canon_tx.update(updates, state, params)

    q_leaf = canon_updates["layers"]["attn"]["q"]["w"]
    assert hasattr(q_leaf, "value")
    expected = model.layers.attn.q.metadata.reduced_shape
    assert q_leaf.value.shape[-len(expected):] == expected


if __name__ == "__main__":
    test_roundtrip_preserves_structure_and_values()
