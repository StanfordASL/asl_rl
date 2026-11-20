from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal


@dataclass(frozen=True)
class PPOConfig:
    # Rollout collection
    num_envs: int = 1024
    """Number of parallel environments"""
    num_steps_per_env: int = 24
    """Rollout length per environment"""

    # Optimization
    update_epochs: int = 4
    """Number of passes over each rollout"""
    num_minibatches: int = 4
    """Number of minibatches per epoch"""
    learning_rate: float = 3e-4
    """(Initial) learning rate. Mutated if schedule='adaptive'"""
    max_grad_norm: float = 0.5
    """Max L2 norm of the global gradient update; larger updates are scaled down (preserving direction)"""

    # Learning rate
    schedule: Literal["fixed", "adaptive"] = "fixed"
    """Learning rate schedule type (fixed or adaptive)"""
    desired_kl: float = 0.01
    """Target KL for an adaptive learning rate schedule"""
    lr_min: float = 1e-5
    """Minimum learning rate for an adaptive learning rate schedule"""
    lr_max: float = 1e-2
    """Maximum learning rate for an adaptive learning rate schedule"""

    # PPO objective
    gamma: float = 0.99
    """MDP discount factor"""
    lam: float = 0.95
    """GAE lambda: bias-variance tradeoff param (0=low variance/high bias, 1=high variance/low bias)"""
    clip_param: float = 0.2
    """PPO surrogate clipping parameter"""
    entropy_coef: float = 0.0
    """Weight of the entropy term in the PPO loss. Higher values encourage exploration; 0=disabled"""
    value_loss_coef: float = 1.0
    """Weight of the value/critic term in the PPO loss. Balances critic learning relative to the policy"""
    use_clipped_value_loss: bool = True
    """Whether to clip the value function / critic update (mirroring the policy / actor)"""

    # Network
    # TODO decide if these config values should be separate from the rest of these
    actor_hidden_dims: Sequence[int] = (64, 64)
    """Actor MLP hidden dimensions"""
    critic_hidden_dims: Sequence[int] = (64, 64)
    """Critic MLP hidden dimensions"""
    activation: Literal["elu", "relu", "tanh"] = "elu"
    """MLP activation function. Currently, one of 'elu', 'relu', or 'tanh'"""
    init_std: float = 1.0
    """Initial standard deviation for the Gaussian policy"""

    # Episodes
    randomize_initial_episode_phase: bool = True
    """For envs that set max_episode_length: start each env's first episode at a random
    step so time-outs don't happen in lockstep (rsl_rl's init_at_random_ep_len)"""

    # Misc
    obs_normalization: bool = False
    """Whether to normalize the observations"""
    save_interval: int = 100
    """Number of iterations between checkpoints"""
    log_interval: int = 1
    """Number of iterations between metrics logging"""
    seed: int = 42
    """Random seed"""
    num_devices: int = 1
    """Number of GPUs/TPUs to use"""

    def __post_init__(self):
        assert self.schedule in ("fixed", "adaptive"), self.schedule
        assert self.activation in ("elu", "relu", "tanh"), self.activation
        assert self.num_envs >= 1
        assert self.num_steps_per_env >= 1
        assert self.update_epochs >= 1
        assert self.num_minibatches >= 1
        assert self.num_devices >= 1
        assert self.log_interval >= 1
        assert self.lr_min <= self.lr_max

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "PPOConfig":
        valid = {f.name for f in fields(cls)}
        unknown = set(cfg) - valid
        if unknown:
            raise ValueError(
                f"Unknown PPOConfig keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}"
            )
        return cls(**cfg)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
