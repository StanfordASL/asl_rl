from collections.abc import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal

_ACTIVATIONS = {"relu": nn.relu, "tanh": jnp.tanh, "elu": nn.elu}


class ActorCritic(nn.Module):
    action_dim: int
    actor_hidden_dims: Sequence[int] = (64, 64)
    critic_hidden_dims: Sequence[int] = (64, 64)
    activation: str = "elu"
    init_std: float = 1.0

    def _mlp(
        self, x: jnp.ndarray, act_fn: Callable, hidden_dims: Sequence[int]
    ) -> jnp.ndarray:
        for dim in hidden_dims:
            x = nn.Dense(
                dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(x)
            x = act_fn(x)
        return x

    @nn.compact
    def __call__(
        self, actor_obs: jnp.ndarray, critic_obs: jnp.ndarray = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Returns (action_mean, action_std, value): a diagonal Gaussian policy
        # with mean and std of shape (..., action_dim), and value of shape (...)
        if critic_obs is None:
            critic_obs = actor_obs

        if self.activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation function: {self.activation}")
        act_fn = _ACTIVATIONS[self.activation]

        # Actor MLP
        actor_x = self._mlp(actor_obs, act_fn, self.actor_hidden_dims)
        action_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_x)

        # Learnable state-independent log_std parameter
        log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.full(
                shape, jnp.log(self.init_std), dtype=jnp.float32
            ),
            (self.action_dim,),
        )
        action_std = jnp.broadcast_to(jnp.exp(log_std), action_mean.shape)

        # Critic MLP
        critic_x = self._mlp(critic_obs, act_fn, self.critic_hidden_dims)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_x
        )
        value = jnp.squeeze(value, axis=-1)

        return action_mean, action_std, value
