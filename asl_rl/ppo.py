import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from asl_rl.config import PPOConfig
from asl_rl.env import DEVICE_AXIS, Observation


class Batch(NamedTuple):
    """Rollout data used by the PPO update. Leading dims (num_steps, num_envs)."""

    obs: Observation  # (normalized) actor / critic observations
    action: jnp.ndarray
    log_prob: jnp.ndarray  # log prob of action under the rollout policy
    mean: jnp.ndarray  # rollout policy mean, for the KL
    std: jnp.ndarray  # rollout policy std, for the KL
    value: jnp.ndarray  # rollout value estimate, for value clipping
    advantage: jnp.ndarray  # raw GAE advantages (normalized inside ppo_update)
    target: jnp.ndarray  # value target (returns)


def calculate_gae(
    rewards: jnp.ndarray,  # (num_steps, num_envs)
    values: jnp.ndarray,  # (num_steps, num_envs)
    terminated: jnp.ndarray,  # (num_steps, num_envs)
    truncated: jnp.ndarray,  # (num_steps, num_envs)
    last_val: jnp.ndarray,  # (num_envs,) value of the observation after the rollout
    gamma: float,
    lam: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Add next value back in advance of GAE zeroing it at the end of an episode.
    # Note: observations after truncations are lost, so V(s_t) is
    # used as a proxy for V(s_{t+1}) (as in rsl_rl)
    truncations = jnp.logical_and(truncated, jnp.logical_not(terminated)).astype(
        rewards.dtype
    )
    rewards = rewards + gamma * values * truncations
    dones = jnp.logical_or(terminated, truncated)

    def _get_advantages(gae_and_next_value, step_data):
        gae, next_value = gae_and_next_value
        reward, value, done = step_data
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lam * (1.0 - done) * gae
        return (gae, value), gae

    step_data = (rewards, values, dones)
    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        step_data,
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns


# Diagonal Gaussian policy

_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_sample(key: jax.Array, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return mean + std * jax.random.normal(key, mean.shape, mean.dtype)


def gaussian_log_prob(
    mean: jnp.ndarray, std: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    z = (x - mean) / std
    return -jnp.sum(0.5 * jnp.square(z) + jnp.log(std) + 0.5 * _LOG_2PI, axis=-1)


def gaussian_entropy(std: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(jnp.log(std) + 0.5 * (1.0 + _LOG_2PI), axis=-1)


def gaussian_kl(
    old_mean: jnp.ndarray,
    old_std: jnp.ndarray,
    new_mean: jnp.ndarray,
    new_std: jnp.ndarray,
) -> jnp.ndarray:
    kl = (
        jnp.log(new_std / old_std)
        + (jnp.square(old_std) + jnp.square(old_mean - new_mean))
        / (2.0 * jnp.square(new_std))
        - 0.5
    )
    return jnp.sum(kl, axis=-1)


def ppo_loss_fn(
    params: Any,
    apply_fn: Callable,
    batch: Batch,
    cfg: PPOConfig,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    mean, std, values = apply_fn({"params": params}, batch.obs.actor, batch.obs.critic)
    log_probs = gaussian_log_prob(mean, std, batch.action)
    entropy = gaussian_entropy(std).mean()

    # Policy surrogate loss
    ratio = jnp.exp(log_probs - batch.log_prob)
    surr1 = ratio * batch.advantage
    surr2 = (
        jnp.clip(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * batch.advantage
    )
    policy_loss = -jnp.minimum(surr1, surr2).mean()

    # Value function loss
    if cfg.use_clipped_value_loss:
        value_pred_clipped = batch.value + jnp.clip(
            values - batch.value, -cfg.clip_param, cfg.clip_param
        )
        value_losses = jnp.square(values - batch.target)
        value_losses_clipped = jnp.square(value_pred_clipped - batch.target)
        value_loss = jnp.maximum(value_losses, value_losses_clipped).mean()
    else:
        value_loss = jnp.square(values - batch.target).mean()

    # KL from the rollout policy, for the adaptive learning rate and logging
    kl = jnp.mean(gaussian_kl(batch.mean, batch.std, mean, std))

    # Total PPO loss
    total_loss = (
        policy_loss + cfg.value_loss_coef * value_loss - cfg.entropy_coef * entropy
    )
    return total_loss, {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "kl": kl,
    }


def normalize_advantages(advantages: jnp.ndarray) -> jnp.ndarray:
    mean = jax.lax.pmean(jnp.mean(advantages), DEVICE_AXIS)
    variance = jax.lax.pmean(jnp.mean(jnp.square(advantages - mean)), DEVICE_AXIS)
    return (advantages - mean) / (jnp.sqrt(variance) + 1e-8)


def make_optimizer(cfg: PPOConfig) -> optax.GradientTransformation:
    """Adam with global gradient norm clipping.

    Only the adaptive-KL schedule needs to change the learning rate during
    training, so only it pays for the inject_hyperparams wrapper (which carries
    the learning rate in the optimizer state, see get_lr / _set_lr).
    """
    if cfg.schedule == "adaptive":
        return optax.inject_hyperparams(
            lambda learning_rate: optax.chain(
                optax.clip_by_global_norm(cfg.max_grad_norm),
                optax.adam(learning_rate, eps=1e-5),
            )
        )(learning_rate=cfg.learning_rate)
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(cfg.learning_rate, eps=1e-5),
    )


def get_lr(train_state: TrainState, cfg: PPOConfig) -> jnp.ndarray:
    if cfg.schedule == "adaptive":
        return train_state.opt_state.hyperparams["learning_rate"]
    return jnp.asarray(cfg.learning_rate)


def _set_lr(train_state: TrainState, lr: jnp.ndarray) -> TrainState:
    opt_state = train_state.opt_state
    hyperparams = dict(opt_state.hyperparams)
    hyperparams["learning_rate"] = lr
    return train_state.replace(opt_state=opt_state._replace(hyperparams=hyperparams))


def _adapt_lr(lr: jnp.ndarray, kl: jnp.ndarray, cfg: PPOConfig) -> jnp.ndarray:
    """rsl_rl-style KL-targeting LR update"""
    too_high = kl > cfg.desired_kl * 2.0
    too_low = (kl < cfg.desired_kl / 2.0) & (kl > 0.0)
    lr = jnp.where(too_high, jnp.maximum(cfg.lr_min, lr / 1.5), lr)
    lr = jnp.where(too_low, jnp.minimum(cfg.lr_max, lr * 1.5), lr)
    return lr


def ppo_update(
    train_state: TrainState,
    apply_fn: Callable,  # network.apply
    batch: Batch,
    key: jax.Array,
    cfg: PPOConfig,
) -> tuple[TrainState, dict[str, jnp.ndarray]]:
    """PPO epoch / minibatch SGD over a collected rollout.

    Returns the updated train state and per-minibatch metrics, each of shape
    (update_epochs, num_minibatches). Metrics are local to this device.
    """
    batch = batch._replace(advantage=normalize_advantages(batch.advantage))

    # Flatten (num_steps, num_envs, *) into a single batch axis
    batch_size = batch.action.shape[0] * batch.action.shape[1]
    flat_batch = jax.tree_util.tree_map(
        lambda x: x.reshape(batch_size, *x.shape[2:]), batch
    )

    def _update_epoch(train_state, perm_key):
        permutation = jax.random.permutation(perm_key, batch_size)
        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.take(x, permutation, axis=0).reshape(
                (cfg.num_minibatches, -1) + x.shape[1:]
            ),
            flat_batch,
        )

        def _update_minibatch(train_state, minibatch):
            grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)
            (_, metrics), grads = grad_fn(train_state.params, apply_fn, minibatch, cfg)

            # Average gradients across devices
            grads = jax.tree_util.tree_map(
                lambda x: jax.lax.pmean(x, DEVICE_AXIS), grads
            )

            # Adaptive LR towards a desired KL
            if cfg.schedule == "adaptive":
                kl = jax.lax.pmean(metrics["kl"], DEVICE_AXIS)
                train_state = _set_lr(
                    train_state, _adapt_lr(get_lr(train_state, cfg), kl, cfg)
                )

            train_state = train_state.apply_gradients(grads=grads)
            return train_state, metrics

        return jax.lax.scan(_update_minibatch, train_state, minibatches)

    perm_keys = jax.random.split(key, cfg.update_epochs)
    return jax.lax.scan(_update_epoch, train_state, perm_keys)
