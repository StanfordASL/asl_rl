from typing import Any

import jax
import jax.numpy as jnp


def tree_to_float32(tree: Any) -> Any:
    """Cast the float leaves of a pytree to float32"""

    def cast(x):
        x = jnp.asarray(x)
        return x.astype(jnp.float32) if jnp.issubdtype(x.dtype, jnp.floating) else x

    return jax.tree_util.tree_map(cast, tree)


def replicate_attrs(obj, sharding, _seen=None):
    """In-place: replicate every jax.Array reachable from obj's attributes."""
    _seen = set() if _seen is None else _seen
    if id(obj) in _seen or not hasattr(obj, "__dict__"):
        return
    _seen.add(id(obj))
    for name, val in vars(obj).items():
        leaves = jax.tree.leaves(val)
        if any(isinstance(x, jax.Array) for x in leaves):
            val = jax.tree.map(
                lambda x: (
                    jax.device_put(x, sharding) if isinstance(x, jax.Array) else x
                ),
                val,
            )
            setattr(obj, name, val)
        elif len(leaves) == 1 and leaves[0] is val:
            # Not a pytree container: recurse into plain helper objects
            replicate_attrs(val, sharding, _seen)
