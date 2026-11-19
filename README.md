# asl-rl

> [!WARNING]
> Work in progress!

A minimal, performant implementation of PPO in JAX

*(you've heard of rsl-rl, now get ready for asl-rl)*

### Why asl-rl?

There are lots of great libraries out there for RL, and for most people, I would highly recommend [mjlab](https://github.com/mujocolab/mjlab) (and [rsl-rl](https://github.com/leggedrobotics/rsl_rl)). However, I have my own personal opinions, which maybe you might share:

- I'm not a fan of manager-based RL environments as I find digging through many layers of abstraction hard to debug
- On this point, I also prefer functional rather than stateful code, as it can be very easy to introduce tricky bugs when dealing with state management
- JAX is great, and can deliver a higher level of performance than torch, if done well.
- I essentially only care about PPO, and don't want to introduce bloat with more algorithms.

If you agree with any of these thoughts, give `asl-rl` a try!

## Installation

```
git clone https://github.com/StanfordASL/asl_rl
cd asl_rl
uv venv --python 3.12.3  # Feel free to handle your venv differently
source .venv/bin/activate
uv pip install -e ".[all]"  # Installs cuda13 JAX by default
```

I prefer to manage my dependencies with an explicit `uv` venv, hence the above installation guide. But, feel free to use `pip`, `conda`, or `uv run` instead.

The `[all]` flag on installation, by default, installs all backends, cuda13-compatible JAX, `wandb`, and `onnx`. If you want to keep things light or CPU-only, you can go for a standard install.


## Examples

`asl-rl` comes bundled with a (continuous) cartpole example, based on the [cartpole example from `mjlab`](https://github.com/mujocolab/mjlab/tree/8bb2bc9e187797689ec82c78c35019afd2fd0376/src/mjlab/tasks/cartpole). 

Although cartpole is quite simple, this example is great for getting a sense of how to work with `asl-rl`, and how to design envs for different JAX-compatible physics backends.

Currently, we support three different physics backends, each selectable via the `--backend` flag:

- `jax` (default): Analytical dynamics model of cartpole in JAX
- `frax`: Fast robot kinematics and dynamics in JAX: see [StanfordASL/frax](https://github.com/StanfordASL/frax)
- `mjx`: MuJoCo MJX-Warp. Note: we focus on the Warp implementation exclusively, as this gives much better performance and simulation quality than the original JAX implementation

To train,
```
python examples/cartpole/train.py
```
(Or, add the backend flag, e.g. `python examples/cartpole/train.py --backend mjx`)

On an RTX 4090, I'm getting the following steps/second (cartpole, 16384 envs), depending on the physics backend:

- `jax`: 31 million steps/second
- `frax`: 14 million steps/second
- `mjx`: 6 million steps/second

If you don't have a GPU, each of these will run on CPU, but much slower.

To play the trained policy in an interactive MuJoCo viewer:
```
python examples/cartpole/play.py
```
(Or, add the backend flag, e.g. `python examples/cartpole/play.py --backend mjx`. You'll need to have already trained a policy with that backend.)


## Speed

A reasonable question is, if both `asl-rl` and `mjlab` have a MuJoCo Warp interface, why would I ever use `asl-rl`?

Setting aside any opinions on JAX vs Torch, and manager-based APIs, here's a comparison of steps/second on the same humanoid whole-body control example, with the same versions of MuJoCo Warp, and the same xml. Roughly, `asl-rl` is 2x faster:

1 GPU (L40S)

- `mjlab`: 112k steps/second
- `asl-rl`: 227k steps/second

4 GPUs (L40S)

- `mjlab`: 420k steps/second
- `asl-rl`: 880k steps/second


## Acknowledgment

`asl-rl` is inspired by the following excellent libraries:

- [mjlab](https://github.com/mujocolab/mjlab)
- [rsl-rl](https://github.com/leggedrobotics/rsl_rl)
- [mujoco-playground](https://github.com/google-deepmind/mujoco_playground/)
- [mujoco-mjx](https://github.com/google-deepmind/mujoco/tree/fded2dae42a22b497a3b8498a477b1167324d99c/mjx)
- [brax](https://github.com/google/brax)
- [gymnax](https://github.com/RobertTLange/gymnax)
- [purejaxrl](https://github.com/luchris429/purejaxrl)
