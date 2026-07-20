# BipedalWalker-v3 — Deep RL with PPO

Train a bipedal robot to **walk, run, and handle terrain** using **Proximal Policy Optimization (PPO)** on Gymnasium’s `BipedalWalker-v3`.

| | |
|---|---|
| **Algorithm** | PPO (Stable-Baselines3) |
| **Policy** | MLP `[64, 64]` |
| **Timesteps** | 1,000,000 |
| **Hardware** | Tesla T4 (Kaggle) · ~35 min |
| **Final episode reward** | **+133** |
| **Artifacts** | Checkpoints + progression videos |

> Classic Gym “solved” threshold is often cited as **~300** mean return. This run shows **clear locomotion skill** (stable walking → running) without claiming a full solve — metrics are reported honestly.

---

## Training progression (best portfolio asset)

Watch the policy improve from random collapse to coordinated running:

| Stage | Steps | Video | Behavior |
|-------|------:|-------|----------|
| Untrained | 0 | [`video/00_untrained.mp4`](./video/00_untrained.mp4) | Random flailing, falls in ~2s |
| Early | 200K | [`video/200k_steps.mp4`](./video/200k_steps.mp4) | Wobbly shuffle forward |
| Mid | 400K | [`video/400k_steps.mp4`](./video/400k_steps.mp4) | Coordinated walking gait |
| Strong | 600K | [`video/600k_steps.mp4`](./video/600k_steps.mp4) | Confident running · [`model_600000.zip`](./model_600000.zip) |
| Late | 800K | [`video/800k_steps.mp4`](./video/800k_steps.mp4) | Fast, stable locomotion |
| Final | 1M | [`video/1M_final.mp4`](./video/1M_final.mp4) | Running + obstacle handling · [`model_final.zip`](./model_final.zip) |

---

## Method

```
Observation (24-D state)
        │
        ▼
   Actor MLP [64, 64]  ──▶ continuous actions (4 torque joints)
   Critic MLP [64, 64] ──▶ value estimate V(s)
        │
        ▼
   PPO clipped surrogate objective
   (GAE-λ advantages, entropy / value coeffs as configured)
```

### Hyperparameters (published run)

| Param | Value |
|-------|------:|
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| Network | pi/vf `[64, 64]` |

Full config: [`configs/ppo_default.json`](./configs/ppo_default.json)

---

## Repository layout

```
BipedalWalker-PPO/
├── train.py              # PPO training + checkpoints + eval callback
├── evaluate.py           # mean ± std return over N episodes
├── record_video.py       # render policy (or random) to MP4
├── model_final.zip       # 1M-step policy
├── model_600000.zip      # mid-training checkpoint
├── configs/ppo_default.json
├── results/training_summary.json
├── video/                # progression rollouts
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/danielsolo707/BipedalWalker-PPO.git
cd BipedalWalker-PPO
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** `gymnasium[box2d]` needs a Box2D build. On some platforms you may need:

```bash
pip install swig
pip install gymnasium[box2d]
```

---

## Usage

### Run the trained agent (interactive)

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("BipedalWalker-v3", render_mode="human")
model = PPO.load("model_final.zip")

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

### Evaluate (mean return)

```bash
python evaluate.py --model model_final.zip --episodes 20
```

### Record a video

```bash
python record_video.py --model model_final.zip --output video/my_rollout.mp4
python record_video.py --untrained --output video/random.mp4
```

### Train from scratch (optional — models already provided)

```bash
python train.py --timesteps 1000000 --n-envs 4 --output-dir models --tensorboard
```

Intermediate checkpoints are saved under `models/checkpoints/`; final weights go to `models/model_final.zip`.

---

## Results summary

| Metric | Value |
|--------|------:|
| Training steps | 1,000,000 |
| Wall time | ~35 minutes (T4) |
| Final episode reward (reported run) | **+133** |
| Behavior at 1M | Stable run + obstacle handling |
| Solved threshold (literature / Gym) | ~300 mean return |

See [`results/training_summary.json`](./results/training_summary.json) for the progression table in machine-readable form.

---

## What this repo demonstrates

- End-to-end **deep RL experiment** with a standard continuous-control benchmark
- **PPO** usage via Stable-Baselines3 with explicit hyperparameters
- Checkpointing + **visual training curriculum** (strong storytelling for portfolios)
- Evaluation and video tooling — not just a notebook dump

---

## Author

**Daniel Soleimani** · [github.com/danielsolo707](https://github.com/danielsolo707)

---

## License

Code is provided for portfolio / educational use. Gymnasium / Box2D remain under their respective licenses.
