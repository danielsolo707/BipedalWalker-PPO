"""Train a PPO agent on Gymnasium BipedalWalker-v3 (Stable-Baselines3).

This script matches the completed portfolio run:
  - Algorithm: PPO (MLP policy)
  - Total timesteps: 1_000_000
  - Intermediate checkpoints every 200k steps (for progression videos)
  - Final reward on the finished run: ~+133 mean return

Example::

    python train.py --timesteps 1000000 --output-dir models
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor


# Hyperparameters used for the published 1M-step run (Tesla T4 / Kaggle).
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO BipedalWalker-v3 training")
    p.add_argument("--env-id", type=str, default="BipedalWalker-v3")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=4, help="Parallel envs (1 = DummyVecEnv)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="models")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--checkpoint-freq", type=int, default=200_000)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logs")
    return p.parse_args()


def make_env(env_id: str, seed: int, rank: int = 0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(env_id: str, n_envs: int, seed: int):
    if n_envs <= 1:
        env = DummyVecEnv([make_env(env_id, seed, 0)])
    else:
        env = SubprocVecEnv([make_env(env_id, seed, i) for i in range(n_envs)])
    return VecMonitor(env)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_env = build_vec_env(args.env_id, args.n_envs, args.seed)
    eval_env = DummyVecEnv([make_env(args.env_id, args.seed + 10_000, 0)])

    tb_log = str(log_dir / "tb") if args.tensorboard else None

    model = PPO(
        env=train_env,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=tb_log,
        **DEFAULT_HYPERPARAMS,
    )

    # Save intermediate policies (used to render progression videos)
    # SB3 multiplies checkpoint_freq by n_envs for VecEnvs — adjust so
    # saves land near every `checkpoint_freq` env steps.
    save_freq = max(1, args.checkpoint_freq // max(1, args.n_envs))
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="ppo_bipedal",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(1, args.eval_freq // max(1, args.n_envs)),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Persist the exact config used for this run
    meta = {
        "env_id": args.env_id,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "hyperparams": {
            **{k: v for k, v in DEFAULT_HYPERPARAMS.items() if k != "policy_kwargs"},
            "policy_kwargs": {"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
        },
        "reference_run": {
            "total_steps": 1_000_000,
            "hardware": "Tesla T4 (Kaggle)",
            "wall_time_minutes": 35,
            "final_episode_reward": 133,
            "notes": "Partial locomotion success; classic 'solved' threshold is ~300 mean return.",
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Training PPO on {args.env_id} for {args.timesteps:,} steps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    final_path = out_dir / "model_final"
    model.save(str(final_path))
    print(f"Saved final model → {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
