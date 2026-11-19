import os

# Inference is faster for small MLPs on CPU
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["JAX_ENABLE_X64"] = "True"
os.environ["JAX_PLATFORMS"] = "cpu"


import argparse
import time
from pathlib import Path

import jax
import mujoco
import mujoco.viewer
import numpy as np

from asl_rl import latest_checkpoint, load_policy


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained Cartpole policy.")
    parser.add_argument(
        "--backend",
        choices=["jax", "frax", "mjx"],
        default="jax",
        help="Which backend's checkpoint to load",
    )
    parser.add_argument("--swing-up", action="store_true", help="Start hanging down")
    args = parser.parse_args()

    repo_dir = Path(__file__).parent.parent.parent
    task = "cartpole_swingup" if args.swing_up else "cartpole"
    checkpoint_path = latest_checkpoint(repo_dir / "logs" / f"{task}_{args.backend}")
    policy, obs_dim = load_policy(checkpoint_path)
    policy_fn = jax.jit(policy)
    print(f"Loaded policy from {checkpoint_path}")

    # timestep 0.01 is set in the xml; decimation 5 -> 20 Hz control
    mj_model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "cartpole.xml"))
    mj_data = mujoco.MjData(mj_model)
    if args.swing_up:
        mj_data.qpos[1] = np.pi  # pole hanging down
    decimation = 5

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        while viewer.is_running():
            obs = np.array(
                [
                    mj_data.qpos[0],
                    np.cos(mj_data.qpos[1]),
                    np.sin(mj_data.qpos[1]),
                    mj_data.qvel[0],
                    mj_data.qvel[1],
                ]
            )
            assert obs.shape == (obs_dim,), "Obs layout doesn't match checkpoint"
            # The slider motor has gear 10, matching the envs' action scale
            mj_data.ctrl[:] = np.asarray(policy_fn(obs))

            for _ in range(decimation):
                step_start = time.time()
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()
                elapsed = time.time() - step_start
                if elapsed < mj_model.opt.timestep:
                    time.sleep(mj_model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
