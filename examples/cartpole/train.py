import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from config import get_config

from asl_rl import OnPolicyRunner


def make_env(backend, swing_up):
    # Import lazily so each backend only needs its own physics dependency
    if backend == "jax":
        from env import CartpoleEnv

        return CartpoleEnv(swing_up)
    if backend == "frax":
        from frax_env import FraxCartpoleEnv

        return FraxCartpoleEnv(swing_up)
    from mjx_env import MjxCartpoleEnv

    return MjxCartpoleEnv(swing_up)


def main():
    parser = argparse.ArgumentParser(description="Train PPO on Cartpole.")
    parser.add_argument("--backend", choices=["jax", "frax", "mjx"], default="jax")
    parser.add_argument("--swing-up", action="store_true", help="Start hanging down")
    args = parser.parse_args()

    cfg = get_config()
    env = make_env(args.backend, args.swing_up)
    task = "cartpole_swingup" if args.swing_up else "cartpole"

    # Initialize runner with log output directory
    repo_dir = Path(__file__).parent.parent.parent
    log_dir = repo_dir / "logs" / f"{task}_{args.backend}"
    runner = OnPolicyRunner(env, cfg, log_dir=log_dir)

    num_iterations = 250
    print(f"Starting training for {num_iterations} iterations...")
    history = runner.learn(num_iterations)
    print("Training finished!")

    # Filter out None values in completed returns (where no episodes finished)
    iters = []
    returns = []
    for it, ret in zip(history["iteration"], history["mean_episode_return"]):
        if ret is not None:
            iters.append(it)
            returns.append(ret)

    plt.figure(figsize=(10, 5))
    plt.plot(
        iters, returns, label="Mean Episode Return", color="royalblue", linewidth=2
    )
    plt.xlabel("Iteration")
    plt.ylabel("Mean Return")
    plt.title(f"PPO {task} Training Curve (asl_rl, {args.backend})")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    artifacts_dir = repo_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plot_path = artifacts_dir / f"{task}_{args.backend}_training.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved learning curve plot to {plot_path}")


if __name__ == "__main__":
    main()
