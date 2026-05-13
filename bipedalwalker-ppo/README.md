# BipedalWalker-v3 - Deep RL with PPO

Training an agent to walk, run, and jump using Proximal Policy Optimization (PPO) on OpenAI Gymnasium's BipedalWalker-v3.

## Training Progression Videos

| Stage | Steps | Description |
|-------|-------|-------------|
| 00_untrained.mp4 | 0 | Random flailing, falls in 2 seconds |
| 200k_steps.mp4 | 200K | Wobbly walking, shuffling forward |
| 400k_steps.mp4 | 400K | Coordinated walking gait |
| 600k_steps.mp4 | 600K | Confident running |
| 800k_steps.mp4 | 800K | Fast, stable locomotion |
| 1M_final.mp4 | 1M | Full running with obstacle handling |

## Training Stats
- Algorithm: PPO (Stable Baselines3)
- Policy: MLP
- Total steps: 1,000,000
- Hardware: Tesla T4 (Kaggle)
- Time: ~35 minutes
- Final reward: +133

## Quick Start
```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make('BipedalWalker-v3', render_mode='human')
model = PPO.load('model_final.zip')

obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, _, term, trunc, _ = env.step(action)
    if term or trunc:
        break
```

## Author
Daniel - Creative Developer | ML Engineer