from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

NORMALIZATION_EPS = 1e-8


class NormalizerState(NamedTuple):
    mean: Any  # pytree
    var: Any  # pytree
    count: jnp.ndarray


def init_normalizer(obs_sample) -> NormalizerState:
    return NormalizerState(
        mean=jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape[-1]), obs_sample),
        var=jax.tree_util.tree_map(lambda x: jnp.ones(x.shape[-1]), obs_sample),
        count=jnp.array(1e-4),  # small nonzero prior on init
    )


def update_normalizer(
    state: NormalizerState, obs, axis_name: str | None = None
) -> NormalizerState:

    def _psum(x):
        # Helper to handle psums whether we are on 1 device or many
        # Reduces to identity operation on a single device
        if axis_name is None:
            return x
        return jax.lax.psum(x, axis_name)

    local_count = jax.tree_util.tree_leaves(obs)[0].shape[0]
    batch_count = _psum(local_count)
    tot_count = state.count + batch_count

    batch_mean = jax.tree_util.tree_map(
        lambda x: _psum(jnp.sum(x, axis=0)) / batch_count, obs
    )

    batch_var = jax.tree_util.tree_map(
        lambda x, m: _psum(jnp.sum(jnp.square(x - m), axis=0)) / batch_count,
        obs,
        batch_mean,
    )

    delta = jax.tree_util.tree_map(jnp.subtract, batch_mean, state.mean)
    new_mean = jax.tree_util.tree_map(
        lambda m, d: m + d * batch_count / tot_count, state.mean, delta
    )

    def _new_var(var, b_var, d):
        m_a = var * state.count
        m_b = b_var * batch_count
        M2 = m_a + m_b + jnp.square(d) * state.count * batch_count / tot_count
        return jnp.maximum(M2 / tot_count, NORMALIZATION_EPS)  # floor on variance

    new_var = jax.tree_util.tree_map(_new_var, state.var, batch_var, delta)
    return NormalizerState(mean=new_mean, var=new_var, count=tot_count)


def normalize_obs(state: NormalizerState, obs, eps: float = NORMALIZATION_EPS):
    return jax.tree_util.tree_map(
        lambda m, v, x: (x - m) / jnp.sqrt(v + eps), state.mean, state.var, obs
    )
