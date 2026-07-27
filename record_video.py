"""Record an MP4 of a saved policy (or a random untrained agent).

Matches the Kaggle progression-video cell: imageio write at 30 FPS, default
max_steps=500 (the length used for published ``video/*.mp4`` assets).

Examples::

    python record_video.py --model model_final.zip --output video/custom_final.mp4
    python record_video.py --untrained --output video/00_untrained.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record BipedalWalker episode video")
    p.add_argument("--model", type=str, default="model_final.zip")
    p.add_argument("--untrained", action="store_true", help="Random actions only")
    p.add_argument("--env-id", type=str, default="BipedalWalker-v3")
    p.add_argument("--hardcore", action="store_true")
    p.add_argument("--output", type=str, default="video/rollout.mp4")
    p.add_argument("--seed", type=int, default=None)
    # Kaggle notebook used max_steps=500 and fps=30 for progression videos.
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, hardcore=args.hardcore, render_mode="rgb_array")

    model = None
    if not args.untrained:
        model_path = Path(args.model)
        if not model_path.exists():
            env.close()
            raise SystemExit(f"Model not found: {model_path}")
        model = PPO.load(str(model_path), env=env)

    reset_kwargs = {}
    if args.seed is not None:
        reset_kwargs["seed"] = args.seed
    obs, _ = env.reset(**reset_kwargs)

    frames = []
    total = 0.0
    steps = 0
    done = False
    while not done and steps < args.max_steps:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        total += float(reward)
        steps += 1
        done = terminated or truncated

    env.close()

    if not frames:
        raise SystemExit("No frames captured. Is the env render_mode='rgb_array'?")

    imageio.mimsave(str(out_path), frames, fps=args.fps)
    print(f"Recorded return={total:.2f}, steps={steps}, frames={len(frames)} → {out_path}")


if __name__ == "__main__":
    main()
