from asl_rl import PPOConfig


def get_config():
    return PPOConfig(
        num_envs=1024,
        num_steps_per_env=32,
        update_epochs=4,
        num_minibatches=4,
        learning_rate=5e-4,
        gamma=0.95,
        lam=0.95,
        clip_param=0.2,
        entropy_coef=0.01,
        value_loss_coef=1.0,
        max_grad_norm=0.5,
        use_clipped_value_loss=True,
        obs_normalization=False,
        save_interval=50,
        seed=42,
        actor_hidden_dims=(64, 64),
        critic_hidden_dims=(64, 64),
        activation="elu",
        init_std=1.0,
        log_interval=10,
    )
