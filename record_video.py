"""Record an MP4 of a saved policy (or a random untrained agent).

Examples::

    python record_video.py --model model_final.zip --output video/custom_final.mp4
    python record_video.py --untrained --output video/00_untrained.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record BipedalWalker episode video")
    p.add_argument("--model", type=str, default="model_final.zip")
    p.add_argument("--untrained", action="store_true", help="Random actions only")
    p.add_argument("--env-id", type=str, default="BipedalWalker-v3")
    p.add_argument("--output", type=str, default="video/rollout.mp4")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # RecordVideo expects a directory + episode trigger; we copy/rename after.
    tmp_dir = out_path.parent / "_tmp_record"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(tmp_dir),
        name_prefix=out_path.stem,
        episode_trigger=lambda e: True,
        disable_logger=True,
    )

    model = None
    if not args.untrained:
        model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        steps = 0
        total = 0.0
        while not done and steps < args.max_steps:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
            done = terminated or truncated
        print(f"Recorded episode {ep + 1}: return={total:.2f}, steps={steps}")

    env.close()

    # Move the newest mp4 to the requested path
    videos = sorted(tmp_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not videos:
        raise SystemExit("No video file was produced. Is moviepy/ffmpeg available?")
    target = videos[-1]
    if out_path.exists():
        out_path.unlink()
    target.replace(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
