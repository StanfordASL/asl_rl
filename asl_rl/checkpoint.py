"""Checkpointing and policy loading"""

import glob
import os
import pickle
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from asl_rl.model import ActorCritic
from asl_rl.normalizer import NormalizerState, normalize_obs


def checkpoint_iteration(path: str) -> int:
    # Parse the iteration number from a model_iter.pkl filename
    name = os.path.basename(path)
    return int(name.removeprefix("model_").removesuffix(".pkl"))


def latest_checkpoint(log_dir: str) -> str:
    # Select the highest-numbered model_iter.pkl in log_dir
    paths = glob.glob(os.path.join(log_dir, "model_*.pkl"))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints found in {log_dir}. Run train.py first."
        )
    return max(paths, key=checkpoint_iteration)


def load_policy(
    checkpoint_path: str, network: Any | None = None
) -> tuple[Callable[[jax.Array], jax.Array], int]:
    with open(checkpoint_path, "rb") as f:
        save = pickle.load(f)

    cfg = save["cfg"]
    params = jax.tree_util.tree_map(jnp.asarray, save["params"])

    # Can pass custom network stucture, otherwise rebuilds our 
    # default actor critic structure
    if network is None:
        network = ActorCritic(
            action_dim=int(params["log_std"].shape[0]),
            actor_hidden_dims=cfg["actor_hidden_dims"],
            critic_hidden_dims=cfg["critic_hidden_dims"],
            activation=cfg["activation"],
        )

    normalizer = NormalizerState(
        mean=jnp.asarray(save["normalizer"]["mean"]["actor"]),
        var=jnp.asarray(save["normalizer"]["var"]["actor"]),
        count=jnp.asarray(save["normalizer"]["count"]),
    )
    normalize = cfg["obs_normalization"]
    obs_dim = int(save["obs_shapes"]["actor"][-1])
    critic_dummy = jnp.zeros(save["obs_shapes"]["critic"])

    def policy(obs: jax.Array) -> jax.Array:
        if normalize:
            obs = normalize_obs(normalizer, obs)
        mean, _, _ = network.apply({"params": params}, obs, critic_dummy)
        return mean

    return policy, obs_dim
