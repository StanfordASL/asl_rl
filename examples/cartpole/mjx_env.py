from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
from flax import struct
from mujoco import mjx

from asl_rl.env import Environment, Observation


@struct.dataclass
class MjxCartpoleState:
    data: mjx.Data


class MjxCartpoleEnv(Environment):
    def __init__(self, swing_up: bool = False):
        xml_path = str(Path(__file__).parent / "cartpole.xml")
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mjx_model = mjx.put_model(self.mj_model, impl="warp")
        self.swing_up = swing_up

        # Allocate warp Data buffers once
        self.data_template = mjx.make_data(self.mj_model, impl="warp")

        # Note: physics timestep (0.01) is set in the xml
        self.decimation = 5  # 20 Hz control
        # 50 sec; the runner truncates episodes at this length
        self.max_episode_length = 1000

    def reset(self, key: jax.Array, ctx=None) -> tuple[Observation, MjxCartpoleState]:
        key_q, key_v = jax.random.split(key)
        q = jax.random.uniform(
            key_q,
            (2,),
            minval=jnp.array([-0.1, -0.034]),
            maxval=jnp.array([0.1, 0.034]),
            dtype=jnp.float32,
        )
        if self.swing_up:
            # Cart starts centered, pole hanging down
            q = jnp.array([0.0, jnp.pi + q[1]])
        v = jax.random.uniform(
            key_v, minval=-0.01, maxval=0.01, shape=(2,), dtype=jnp.float32
        )

        data = self.data_template.replace(qpos=q, qvel=v)
        state = MjxCartpoleState(data=mjx.forward(self.mjx_model, data))
        return self._get_obs(state), state

    def step(
        self, key: jax.Array, state: MjxCartpoleState, action: jnp.ndarray, ctx=None
    ) -> tuple[Observation, MjxCartpoleState, float, bool, bool, dict[str, Any]]:
        action = jnp.clip(action, -1.0, 1.0).astype(jnp.float32)
        data = state.data.replace(ctrl=action)

        # Integration step
        new_data = jax.lax.scan(
            lambda d, _: (mjx.step(self.mjx_model, d), None),
            data,
            None,
            length=self.decimation,
        )[0]

        next_state = MjxCartpoleState(data=new_data)
        obs = self._get_obs(next_state)
        reward = self._get_reward(next_state.data.qpos, next_state.data.qvel, action[0])
        terminated = jnp.abs(next_state.data.qpos[0]) >= 1.8  # Hit end of rail
        truncated = jnp.bool_(False)  # time limit: max_episode_length (runner)
        info = {}

        return obs, next_state, reward, terminated, truncated, info

    def _get_obs(self, state: MjxCartpoleState) -> Observation:
        obs = jnp.array(
            [
                state.data.qpos[0],
                jnp.cos(state.data.qpos[1]),
                jnp.sin(state.data.qpos[1]),
                state.data.qvel[0],
                state.data.qvel[1],
            ]
        )
        return Observation.symmetric(obs)

    def _get_reward(self, q: jnp.ndarray, v: jnp.ndarray, u: float) -> float:
        upright = (jnp.cos(q[1]) + 1.0) / 2.0
        centered = (1.0 + self._gaussian_tolerance(q[0], margin=2.0)) / 2.0
        small_control = (4.0 + self._quadratic_tolerance(u, margin=1.0)) / 5.0
        small_velocity = (1.0 + self._gaussian_tolerance(v[1], margin=5.0)) / 2.0
        return upright * centered * small_control * small_velocity

    def _gaussian_tolerance(self, x: float, margin: float) -> float:
        gaussian_scale = jnp.sqrt(-2.0 * jnp.log(0.1))
        scaled = (x / margin) * gaussian_scale
        return jnp.exp(-0.5 * scaled**2)

    def _quadratic_tolerance(self, x: float, margin: float) -> float:
        quadratic_scale = jnp.sqrt(1.0 - 0.1)
        scaled = (x / margin) * quadratic_scale
        return jnp.maximum(0.0, 1.0 - scaled**2)

    @property
    def action_dim(self) -> int:
        return 1
