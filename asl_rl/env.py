from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from flax import struct

from asl_rl.utils import tree_to_float32

State = Any  # Pytree, defined on a per-env basis

# Named axes
DEVICE_AXIS = "devices"
ENV_AXIS = "envs"


class Observation(NamedTuple):
    actor: Any
    critic: Any

    @classmethod
    def symmetric(cls, obs: Any) -> "Observation":
        return cls(actor=obs, critic=obs)


class Environment:
    def context(self):
        # (Advanced feature) Static data that is not closed over
        # Usually None for most envs, unless using a large dataset
        # NOTE: This might not be needed after jax>=0.12.1 (handled
        # automatically, see docs on JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS)
        return None

    def reset(self, key: jax.Array, ctx=None) -> tuple[Observation, State]:
        raise NotImplementedError

    def reset_from(
        self, key: jax.Array, state: State, ctx=None
    ) -> tuple[Observation, State]:
        # (Advanced feature) Reset given the final state of the previous episode
        # This is necessary for cross-episode stats (for instance, adaptive RSI
        # which continually tracks the states at which failures occur)
        del state  # Default implementation just reverts to standard reset()
        return self.reset(key, ctx)

    def step(
        self, key: jax.Array, state: State, action: jnp.ndarray, ctx=None
    ) -> tuple[Observation, State, float, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    @property
    def action_dim(self) -> int:
        # Size of the action vector. Note: actions are not clipped or scaled by
        # the runner; any action processing is part of the env's step()
        raise NotImplementedError


@struct.dataclass
class LogWrapperState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper:
    def __init__(self, env: Environment):
        self._env = env

    @staticmethod
    def _fresh_wrapper_state(env_state: State) -> LogWrapperState:
        # Helper function to create a fresh wrapper with reset logs
        return LogWrapperState(
            env_state=env_state,
            episode_returns=jnp.float32(0.0),
            episode_lengths=0,
            returned_episode_returns=jnp.float32(0.0),
            returned_episode_lengths=0,
            timestep=0,
        )

    def reset(self, key: jax.Array, ctx=None) -> tuple[Observation, LogWrapperState]:
        obs, env_state = self._env.reset(key, ctx)
        state = self._fresh_wrapper_state(env_state)
        return tree_to_float32(obs), state

    def reset_from(
        self, key: jax.Array, state: LogWrapperState, ctx=None
    ) -> tuple[Observation, LogWrapperState]:
        obs, env_state = self._env.reset_from(key, state.env_state, ctx)
        state = self._fresh_wrapper_state(env_state)
        return tree_to_float32(obs), state

    def step(
        self, key: jax.Array, state: LogWrapperState, action: jnp.ndarray, ctx=None
    ) -> tuple[Observation, LogWrapperState, float, bool, bool, dict[str, Any]]:
        obs, env_state, reward, terminated, truncated, info = self._env.step(
            key, state.env_state, action, ctx
        )
        obs = tree_to_float32(obs)
        reward = tree_to_float32(reward)

        done = jnp.logical_or(terminated, truncated)

        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1

        next_state = LogWrapperState(
            env_state=env_state,
            episode_returns=new_episode_return * (1.0 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1.0 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )

        info = info.copy()
        info["returned_episode_returns"] = next_state.returned_episode_returns
        info["returned_episode_lengths"] = next_state.returned_episode_lengths
        info["returned_episode"] = done
        info["timestep"] = state.timestep

        return obs, next_state, reward, terminated, truncated, info

    @property
    def action_dim(self) -> int:
        return self._env.action_dim


def _has_where(x: Any) -> bool:
    # Pytree nodes that know how to merge themselves, e.g. mjx.Data.where
    return hasattr(x, "where")


class AutoResetWrapper:
    def __init__(self, env: Environment):
        self._env = env

    def reset(self, key: jax.Array, ctx=None) -> tuple[Observation, State]:
        return self._env.reset(key, ctx)

    def step(
        self, key: jax.Array, state: State, action: jnp.ndarray, ctx=None
    ) -> tuple[Observation, State, float, bool, bool, dict[str, Any]]:
        reset_key, step_key = jax.random.split(key)
        obs, next_state, reward, terminated, truncated, info = self._env.step(
            step_key, state, action, ctx
        )

        # Reset whenever the episode ends, for either reason
        done = jnp.logical_or(terminated, truncated)

        # Note: use reset_from to carry cross-episode stats, if needed
        reset_obs, reset_state = self._env.reset_from(reset_key, next_state, ctx)

        def where_done(reset_v, step_v):
            if _has_where(step_v):
                return step_v.where(done, reset_v)
            return jnp.where(done, reset_v, step_v)

        # Select reset state/obs if done is True
        final_obs = jax.tree_util.tree_map(
            where_done, reset_obs, obs, is_leaf=_has_where
        )
        final_state = jax.tree_util.tree_map(
            where_done, reset_state, next_state, is_leaf=_has_where
        )

        return final_obs, final_state, reward, terminated, truncated, info

    @property
    def action_dim(self) -> int:
        return self._env.action_dim


class VecEnv:
    def __init__(self, env: Environment):
        self._env = env
        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None), axis_name=ENV_AXIS)
        self.step_fn = jax.vmap(
            self._env.step, in_axes=(0, 0, 0, None), axis_name=ENV_AXIS
        )

    def reset(self, keys: jax.Array, ctx=None) -> tuple[Observation, State]:
        return self.reset_fn(keys, ctx)

    def step(
        self, keys: jax.Array, state: State, action: jnp.ndarray, ctx=None
    ) -> tuple[
        Observation, State, jnp.ndarray, jnp.ndarray, jnp.ndarray, dict[str, Any]
    ]:
        return self.step_fn(keys, state, action, ctx)

    @property
    def action_dim(self) -> int:
        return self._env.action_dim
