"""Evaluate a saved PPO checkpoint on BipedalWalker-v3.

Example::

    python evaluate.py --model model_final.zip --episodes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PPO BipedalWalker agent")
    p.add_argument("--model", type=str, default="model_final.zip")
    p.add_argument("--env-id", type=str, default="BipedalWalker-v3")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--render", action="store_true")
    p.add_argument("--max-steps", type=int, default=1600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    render_mode = "human" if args.render else None
    env = gym.make(args.env_id, render_mode=render_mode)
    model = PPO.load(str(model_path))

    returns = []
    lengths = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total = 0.0
        steps = 0
        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        returns.append(total)
        lengths.append(steps)
        print(f"Episode {ep + 1:02d}: return={total:8.2f}  steps={steps}")

    env.close()
    summary = {
        "model": str(model_path),
        "episodes": args.episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "reference_training_reward": 133,
        "classic_solved_threshold": 300,
    }
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
