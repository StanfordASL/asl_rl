from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from frax import Robot

from asl_rl.env import Environment, Observation


@struct.dataclass
class FraxCartpoleState:
    q: jnp.ndarray
    v: jnp.ndarray


class FraxCartpoleEnv(Environment):
    def __init__(self, swing_up: bool = False):
        urdf_path = str(Path(__file__).parent / "cartpole.urdf")
        self.robot = Robot(urdf_path)
        self.swing_up = swing_up

        self.dt_physics = 0.01  # 100 Hz sim
        self.decimation = 5  # 20 Hz control
        # 50 sec; the runner truncates episodes at this length
        self.max_episode_length = 1000

    def reset(self, key: jax.Array, ctx=None) -> tuple[Observation, FraxCartpoleState]:
        key_q, key_v = jax.random.split(key)
        q = jax.random.uniform(
            key_q,
            (2,),
            minval=jnp.array([-0.1, -0.034]),
            maxval=jnp.array([0.1, 0.034]),
        )
        if self.swing_up:
            # Cart starts centered, pole hanging down
            q = jnp.array([0.0, jnp.pi + q[1]])
        v = jax.random.uniform(key_v, minval=-0.01, maxval=0.01, shape=(2,))

        state = FraxCartpoleState(q=q, v=v)
        return self._get_obs(state), state

    def step(
        self, key: jax.Array, state: FraxCartpoleState, action: jnp.ndarray, ctx=None
    ) -> tuple[jnp.ndarray, FraxCartpoleState, float, bool, bool, dict[str, Any]]:
        action = jnp.clip(action, -1.0, 1.0)

        # Joint efforts (only on the cart, no torque on the pole)
        tau = jnp.array([action[0] * 10.0, 0.0])

        def integration_step(carry, _):
            q_curr, v_curr = carry
            a = self.robot.forward_dynamics(q_curr, v_curr, tau, None)
            # Semi-implicit euler
            v_next = v_curr + a * self.dt_physics
            q_next = q_curr + v_next * self.dt_physics

            # Joint limits on slider
            q_next = q_next.at[0].set(jnp.clip(q_next[0], -1.8, 1.8))
            hit_limit = (q_next[0] <= -1.8) | (q_next[0] >= 1.8)
            v_next = jnp.where(hit_limit, v_next.at[0].set(0.0), v_next)

            return (q_next, v_next), None

        (q_next, v_next), _ = jax.lax.scan(
            integration_step, (state.q, state.v), None, length=self.decimation
        )

        next_state = FraxCartpoleState(q=q_next, v=v_next)
        obs = self._get_obs(next_state)
        reward = self._get_reward(next_state.q, next_state.v, action[0])
        terminated = jnp.abs(next_state.q[0]) >= 1.8  # Hit end of rail
        truncated = jnp.bool_(False)  # time limit: max_episode_length (runner)
        info = {}

        return obs, next_state, reward, terminated, truncated, info

    def _get_obs(self, state: FraxCartpoleState) -> Observation:
        obs = jnp.array(
            [
                state.q[0],
                jnp.cos(state.q[1]),
                jnp.sin(state.q[1]),
                state.v[0],
                state.v[1],
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
