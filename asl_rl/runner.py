import os
import pickle
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.training.train_state import TrainState
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from asl_rl.config import PPOConfig
from asl_rl.env import DEVICE_AXIS, AutoResetWrapper, LogWrapper, Observation, VecEnv
from asl_rl.model import ActorCritic
from asl_rl.normalizer import init_normalizer, normalize_obs, update_normalizer
from asl_rl.ppo import Batch, calculate_gae, get_lr, make_optimizer, ppo_update
from asl_rl.utils import replicate_attrs

# Theses keys are used by the LogWrapper in env.py
_PROTECTED_INFO_KEYS = frozenset(
    {
        "returned_episode",
        "returned_episode_returns",
        "returned_episode_lengths",
        "timestep",
    }
)


class Transition(NamedTuple):
    terminated: jnp.ndarray  # (num_steps, num_envs)
    truncated: jnp.ndarray  # (num_steps, num_envs)
    action: jnp.ndarray  # (num_steps, num_envs, action_dim)
    value: jnp.ndarray  # (num_steps, num_envs)
    reward: jnp.ndarray  # (num_steps, num_envs)
    log_prob: jnp.ndarray  # (num_steps, num_envs)
    mean: jnp.ndarray  # (num_steps, num_envs, action_dim)
    std: jnp.ndarray  # (num_steps, num_envs, action_dim)
    obs: Observation  # (num_steps, num_envs, obs_dim) for actor and critic
    info: dict[str, jnp.ndarray]


def _collect_rollout(
    apply_fn: Any,  # network.apply
    step_fn: Any,  # vec_env.step
    params: Any,
    normalizer_state: Any,
    env_state: Any,
    last_obs: Any,
    key: jax.Array,
    num_steps: int,
    num_envs: int,
    obs_normalization: bool,
    ctx: Any,
) -> tuple[tuple[Any, Any], Transition]:
    """Roll out `num_steps` transitions across the vectorized envs via scan."""

    def _env_step(carry, step_key):
        env_state, last_obs = carry

        norm_obs = (
            normalize_obs(normalizer_state, last_obs) if obs_normalization else last_obs
        )

        # Query policy distribution and sample actions
        action_key, env_key = jax.random.split(step_key)
        action_dist, value = apply_fn(
            {"params": params}, norm_obs.actor, norm_obs.critic
        )
        action = action_dist.sample(seed=action_key)
        log_prob = action_dist.log_prob(action)

        # Environment step
        keys_step = jax.random.split(env_key, num_envs)
        next_obs, next_env_state, reward, terminated, truncated, info = step_fn(
            keys_step, env_state, action, ctx
        )

        transition = Transition(
            terminated=terminated,
            truncated=truncated,
            action=action,
            value=value,
            reward=reward,
            log_prob=log_prob,
            mean=action_dist.mean(),
            std=action_dist.stddev(),
            obs=last_obs,
            info=info,
        )
        return (next_env_state, next_obs), transition

    step_keys = jax.random.split(key, num_steps)
    carry = (env_state, last_obs)
    return jax.lax.scan(_env_step, carry, step_keys)


def _compute_metrics(
    rollout: Transition,
    update_metrics: dict[str, jnp.ndarray],
    learning_rate: jnp.ndarray,
    num_devices: int,
) -> dict[str, jnp.ndarray]:
    """Training metrics for one iteration, reduced across devices."""
    # Extract completed episode statistics
    completed_mask = rollout.info["returned_episode"]
    completed_returns = rollout.info["returned_episode_returns"]
    completed_lengths = rollout.info["returned_episode_lengths"]

    # Core metrics: Note: some are summed over devices, others are averaged
    # scale handles the cross-device averaging, assuming psum is next
    scale = 1.0 / num_devices
    core_metrics = {
        # summed over devices
        "num_completed_episodes": jnp.sum(completed_mask),
        "episode_return_sum": jnp.sum(completed_returns * completed_mask),
        "episode_length_sum": jnp.sum(completed_lengths * completed_mask),
        # averaged over devices
        **{k: jnp.mean(v) * scale for k, v in update_metrics.items()},
        "rewards_mean": jnp.mean(rollout.reward) * scale,
        "action_std": jnp.mean(rollout.std) * scale,
    }
    # Handle custom metrics defined in the env for that task
    custom_metrics = {
        k: jnp.mean(rollout.info[k]) * scale
        for k in rollout.info
        if k not in _PROTECTED_INFO_KEYS
    }
    # Handle metrics reduction across devices
    metrics = core_metrics | custom_metrics
    metric_vec = jnp.stack(list(metrics.values()))
    metric_vec = jax.lax.psum(metric_vec, DEVICE_AXIS)
    metrics = dict(zip(metrics.keys(), metric_vec))
    # Add other metrics that are not reduced over devices
    episodes = jnp.maximum(metrics["num_completed_episodes"], 1.0)
    other_metrics = {
        "learning_rate": learning_rate,
        "mean_episode_return": metrics["episode_return_sum"] / episodes,
        "mean_episode_length": metrics["episode_length_sum"] / episodes,
    }
    # Pop any metrics that don't need to be logged, but still needed reduction
    metrics.pop("episode_return_sum")
    metrics.pop("episode_length_sum")
    # Check for name collisions with custom metrics
    overlap = custom_metrics.keys() & (core_metrics.keys() | other_metrics.keys())
    if overlap:
        raise ValueError(f"Env info keys {overlap} conflict with built-in metrics")
    return metrics | other_metrics


class OnPolicyRunner:
    def __init__(
        self,
        env: Any,
        cfg: PPOConfig,
        log_dir: str | None = None,
        logger: Any = None,  # asl_rl.wandb_utils.Logger if wandb installed
        network: Any | None = None,
    ):
        self.env = env
        self.cfg = cfg
        self.log_dir = None if log_dir is None else str(log_dir)
        self.logger = logger

        # Wrap environment with LogWrapper and AutoResetWrapper
        log_env = LogWrapper(env)
        reset_env = AutoResetWrapper(log_env)
        self.vec_env = VecEnv(reset_env)

        # Dimensions & parameters
        self.num_envs = self.cfg.num_envs
        self.num_steps = self.cfg.num_steps_per_env
        self.obs_normalization = self.cfg.obs_normalization
        self.num_devices = self.cfg.num_devices

        available_devices = jax.local_devices()
        if self.num_devices < 1:
            raise ValueError("num_devices must be at least 1")
        if self.num_devices > len(available_devices):
            raise ValueError(
                f"Requested {self.num_devices} devices, but JAX only sees "
                f"{len(available_devices)} local device(s)"
            )
        if self.num_envs % self.num_devices != 0:
            raise ValueError(
                f"num_envs ({self.num_envs}) must be divisible by num_devices "
                f"({self.num_devices})"
            )

        self.devices = available_devices[: self.num_devices]
        self.multi_device = self.num_devices > 1
        self.num_envs_per_device = self.num_envs // self.num_devices
        self.mesh = Mesh(np.array(self.devices), (DEVICE_AXIS,))
        self.sharded = NamedSharding(self.mesh, P(DEVICE_AXIS))
        self.replicated = NamedSharding(self.mesh, P())
        local_batch_size = self.num_steps * self.num_envs_per_device
        if local_batch_size % self.cfg.num_minibatches != 0:
            raise ValueError(
                "num_steps_per_env * (num_envs / num_devices) must be "
                f"divisible by num_minibatches; got {local_batch_size} and "
                f"{self.cfg.num_minibatches}"
            )

        # Handle replications
        replicate_attrs(env, self.replicated)
        self.env_ctx = jax.device_put(env.context(), self.replicated)

        # Seed / keys
        self.seed = self.cfg.seed
        self.key = jax.random.PRNGKey(self.seed)

        # Dimensions
        self.action_dim = env.action_dim

        # Vectorized reset
        self._reset = jax.jit(
            jax.shard_map(
                self.vec_env.reset,
                mesh=self.mesh,
                in_specs=(P(DEVICE_AXIS), P()),
                out_specs=P(DEVICE_AXIS),
                check_vma=False,  # TODO: Check if needed (MJX-Warp)
            )
        )

        # Probe the observation structure with a dummy reset
        self.key, probe_key = jax.random.split(self.key)
        probe_keys = jax.device_put(
            jax.random.split(probe_key, self.num_envs), self.sharded
        )
        batch_obs, _ = jax.eval_shape(self._reset, probe_keys, self.env_ctx)
        obs_template = jax.tree_util.tree_map(
            lambda x: jnp.zeros(x.shape[1:], x.dtype), batch_obs
        )

        # Allow for passing custom network. Expects call structure:
        # (actor_obs, critic_obs) -> (action_distribution, value)
        if network is None:
            # Otherwise, just use our standard actor/critic structure
            network = ActorCritic(
                action_dim=self.action_dim,
                actor_hidden_dims=self.cfg.actor_hidden_dims,
                critic_hidden_dims=self.cfg.critic_hidden_dims,
                activation=self.cfg.activation,
                init_std=self.cfg.init_std,
            )
        self.network = network

        self.optimizer = make_optimizer(self.cfg)

        # Initialize neural network parameters and TrainState
        self.key, init_key = jax.random.split(self.key)
        variables = self.network.init(init_key, obs_template.actor, obs_template.critic)
        self.params = variables["params"]
        self.train_state = TrainState.create(
            apply_fn=self.network.apply,
            params=self.params,
            tx=self.optimizer,
        )

        # Initialize observation normalizer (one running mean/var per obs leaf)
        self.normalizer_state = init_normalizer(obs_template)
        # Saved with checkpoints so inference can build inputs for any network
        self.obs_shapes = {k: tuple(v.shape) for k, v in obs_template._asdict().items()}
        self.current_learning_iteration = 0

        self._update_step = self._build_update_step()

    def _build_update_step(self):
        num_devices = self.num_devices

        def update_step(train_state, normalizer_state, env_state, last_obs, key, ctx):
            next_key, local_key = jax.random.split(key)
            # Maintain randomness across devices even though the key is replicated
            local_key = jax.random.fold_in(local_key, jax.lax.axis_index(DEVICE_AXIS))
            rollout_key, update_key = jax.random.split(local_key)

            # Collect rollout trajectory
            (next_env_state, next_obs), rollout = _collect_rollout(
                self.network.apply,
                self.vec_env.step,
                train_state.params,
                normalizer_state,
                env_state,
                last_obs,
                rollout_key,
                num_steps=self.num_steps,
                num_envs=self.num_envs_per_device,
                obs_normalization=self.obs_normalization,
                ctx=ctx,
            )

            # Calculate advantages & returns, bootstrapping from the final obs
            norm_last_obs = (
                normalize_obs(normalizer_state, next_obs)
                if self.obs_normalization
                else next_obs
            )
            _, last_val = self.network.apply(
                {"params": train_state.params},
                norm_last_obs.actor,
                norm_last_obs.critic,
            )
            advantages, targets = calculate_gae(
                rewards=rollout.reward,
                values=rollout.value,
                terminated=rollout.terminated,
                truncated=rollout.truncated,
                last_val=last_val,
                gamma=self.cfg.gamma,
                lam=self.cfg.lam,
            )

            batch = Batch(
                obs=(
                    normalize_obs(normalizer_state, rollout.obs)
                    if self.obs_normalization
                    else rollout.obs
                ),
                action=rollout.action,
                log_prob=rollout.log_prob,
                mean=rollout.mean,
                std=rollout.std,
                value=rollout.value,
                advantage=advantages,
                target=targets,
            )
            train_state, update_metrics = ppo_update(
                train_state, self.network.apply, batch, update_key, self.cfg
            )
            all_metrics = _compute_metrics(
                rollout,
                update_metrics,
                learning_rate=get_lr(train_state, self.cfg),
                num_devices=num_devices,
            )

            # Update normalizer stats for the next iteration (one running
            # mean/var per observation leaf).
            if self.obs_normalization:
                flat_obs = jax.tree_util.tree_map(
                    lambda x: x.reshape(-1, x.shape[-1]), rollout.obs
                )
                new_normalizer_state = update_normalizer(
                    normalizer_state, flat_obs, axis_name=DEVICE_AXIS
                )
            else:
                new_normalizer_state = normalizer_state

            return (
                train_state,
                new_normalizer_state,
                next_env_state,
                next_obs,
                next_key,
                all_metrics,
            )

        # Note: we always use sharding, even on a single device
        # (less complicated code with no noticeable performance gap)
        update_step_jit = jax.jit(
            jax.shard_map(
                update_step,
                mesh=self.mesh,
                in_specs=(P(), P(), P(DEVICE_AXIS), P(DEVICE_AXIS), P(), P()),
                out_specs=(P(), P(), P(DEVICE_AXIS), P(DEVICE_AXIS), P(), P()),
                check_vma=False,  # TODO: Check if needed (MJX-Warp)
            ),
            # Allow XLA to reuse rollout / env state buffers in place
            donate_argnums=(0, 1, 2, 3, 4),
        )
        return update_step_jit

    def learn(self, num_learning_iterations: int) -> dict[str, Any]:
        self.key, reset_key = jax.random.split(self.key)
        reset_keys = jax.device_put(
            jax.random.split(reset_key, self.num_envs), self.sharded
        )
        last_obs, env_state = self._reset(reset_keys, self.env_ctx)

        if self.multi_device:
            print(
                f"Using {self.num_devices} devices "
                f"({self.num_envs_per_device} envs per device)."
            )
        self.key, training_key = jax.random.split(self.key)

        # Replicate params / optimizer / normalizer state / key across the mesh.
        train_state, normalizer_state = jax.tree.map(
            jnp.copy, (self.train_state, self.normalizer_state)
        )
        train_state = jax.device_put(train_state, self.replicated)
        normalizer_state = jax.device_put(normalizer_state, self.replicated)
        training_key = jax.device_put(training_key, self.replicated)

        print("Compiling training loop...")
        start_compile = time.time()
        compiled_update = self._update_step.lower(
            train_state,
            normalizer_state,
            env_state,
            last_obs,
            training_key,
            self.env_ctx,
        ).compile()
        compile_time = time.time() - start_compile
        print(f"Compilation finished in {compile_time:.2f} seconds.")

        print(
            f"\n{'Iteration':<10} | {'Return':<10} | {'Length':<10} | {'Policy Loss':<12} | {'Value Loss':<10} | {'Entropy':<8} | {'Act Std':<8} | {'FPS':<6}"
        )
        print("-" * 90)

        first_iteration = self.current_learning_iteration + 1
        last_iteration = self.current_learning_iteration + num_learning_iterations

        history = {
            "iteration": [],
            "mean_episode_return": [],
            "mean_episode_length": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
        }

        interval_start = time.time()
        iters_since_log = 0
        should_save = False
        for iteration in range(first_iteration, last_iteration + 1):
            (
                train_state,
                normalizer_state,
                env_state,
                last_obs,
                training_key,
                metrics,
            ) = compiled_update(
                train_state,
                normalizer_state,
                env_state,
                last_obs,
                training_key,
                self.env_ctx,
            )
            iters_since_log += 1

            should_save = (
                self.log_dir is not None
                and self.cfg.save_interval > 0
                and iteration % self.cfg.save_interval == 0
            )
            if (
                iters_since_log >= self.cfg.log_interval
                or iteration == last_iteration
                or should_save
            ):
                host_metrics = jax.device_get(metrics)
                elapsed = time.time() - interval_start
                fps = (self.num_steps * self.num_envs * iters_since_log) / elapsed

                mean_return = float(host_metrics["mean_episode_return"])
                mean_length = float(host_metrics["mean_episode_length"])
                num_completed = int(host_metrics["num_completed_episodes"])

                return_str = f"{mean_return:.2f}" if num_completed > 0 else "N/A"
                length_str = f"{mean_length:.1f}" if num_completed > 0 else "N/A"

                print(
                    f"{iteration:<10} | {return_str:<10} | {length_str:<10} | {float(host_metrics['policy_loss']):<12.5f} | {float(host_metrics['value_loss']):<10.5f} | {float(host_metrics['entropy']):<8.4f} | {float(host_metrics['action_std']):<8.4f} | {fps:<6.0f}"
                )

                history["iteration"].append(iteration)
                history["mean_episode_return"].append(
                    mean_return if num_completed > 0 else None
                )
                history["mean_episode_length"].append(
                    mean_length if num_completed > 0 else None
                )
                history["policy_loss"].append(float(host_metrics["policy_loss"]))
                history["value_loss"].append(float(host_metrics["value_loss"]))
                history["entropy"].append(float(host_metrics["entropy"]))

                if self.logger is not None:  # Upload to wandb
                    wandb_metrics = {k: float(v) for k, v in host_metrics.items()}
                    if num_completed == 0:  # don't log N/A returns as garbage values
                        wandb_metrics.pop("mean_episode_return")
                        wandb_metrics.pop("mean_episode_length")
                    wandb_metrics["fps"] = fps
                    self.logger.log(wandb_metrics, step=iteration)

                iters_since_log = 0
                interval_start = time.time()

            if should_save:
                self.save(
                    os.path.join(self.log_dir, f"model_{iteration}.pkl"),
                    train_state,
                    normalizer_state,
                    iteration,
                )
                interval_start = time.time()

        self.train_state = train_state
        self.normalizer_state = normalizer_state
        self.current_learning_iteration = last_iteration
        self.key = training_key

        # Final save, unless the last iteration already landed on a save interval
        if self.log_dir is not None and not should_save:
            self.save(
                os.path.join(self.log_dir, f"model_{last_iteration}.pkl"),
                train_state,
                normalizer_state,
                last_iteration,
            )

        return history

    def save(
        self,
        path: str,
        train_state: TrainState,
        normalizer_state: Any,
        iteration: int,
    ):
        opt_state = jax.tree_util.tree_map(np.asarray, train_state.opt_state)
        save_dict = {
            "params": jax.tree_util.tree_map(np.asarray, train_state.params),
            "opt_state": serialization.to_state_dict(opt_state),
            "normalizer": serialization.to_state_dict(
                jax.tree_util.tree_map(np.asarray, normalizer_state)
            ),
            "iter": iteration,
            "obs_shapes": self.obs_shapes,
            "cfg": self.cfg.to_dict(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        print(f"Saved checkpoint to {path}")
        if self.logger is not None:  # Upload to wandb if configured
            self.logger.save(path)

    def load(self, path: str):
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        params = jax.tree_util.tree_map(jnp.asarray, save_dict["params"])
        opt_state = serialization.from_state_dict(
            self.train_state.opt_state, save_dict["opt_state"]
        )
        opt_state = jax.tree_util.tree_map(jnp.asarray, opt_state)
        self.train_state = self.train_state.replace(params=params, opt_state=opt_state)

        normalizer_state = serialization.from_state_dict(
            self.normalizer_state, save_dict["normalizer"]
        )
        self.normalizer_state = jax.tree_util.tree_map(jnp.asarray, normalizer_state)
        self.current_learning_iteration = save_dict["iter"]
        # Fresh RNG stream per resume point
        self.key = jax.random.fold_in(self.key, self.current_learning_iteration)
        print(
            f"Loaded checkpoint from {path} (iteration {self.current_learning_iteration})"
        )
