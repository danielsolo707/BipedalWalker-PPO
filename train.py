"""Train a PPO agent on Gymnasium BipedalWalker-v3 (Stable-Baselines3).

Matches the published Kaggle run that produced ``model_*.zip`` and progression
videos in this repo:

  - Algorithm: PPO (MlpPolicy, SB3 defaults for net arch)
  - Env: BipedalWalker-v3 (hardcore=False), single DummyVecEnv
  - Timesteps: 1_000_000
  - Checkpoints every 200k steps as ``model_{steps}.zip``
  - Entropy coef: 0.001 (not the SB3 default of 0.0)

Example::

    python train.py --timesteps 1000000 --output-dir .
    python train.py --config configs/ppo_default.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


# Hyperparameters from the Kaggle notebook that produced the published artifacts.
DEFAULT_HYPERPARAMS: dict[str, Any] = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # SB3 MlpPolicy default for continuous control is pi/vf [64, 64].
    # Left implicit so we match the notebook (no policy_kwargs).
}


class SaveCheckpoint(BaseCallback):
    """Save ``model_{num_timesteps}.zip`` every ``save_freq`` env steps.

    Same naming scheme as the Kaggle training notebook (not SB3 CheckpointCallback
    prefixes), so progression videos can load ``model_200000.zip``, etc.
    """

    def __init__(self, save_freq: int, save_path: str | Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.num_timesteps > 0:
            if self.num_timesteps % self.save_freq == 0:
                path = self.save_path / f"model_{self.num_timesteps}"
                self.model.save(str(path))
                print(f"\nCheckpoint saved: {self.num_timesteps} steps → {path}.zip")
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO BipedalWalker-v3 training (Kaggle-aligned)")
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional JSON config (e.g. configs/ppo_default.json)",
    )
    p.add_argument("--env-id", type=str, default="BipedalWalker-v3")
    p.add_argument("--hardcore", action="store_true", help="Use hardcore terrain (default: off)")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Parallel envs (published Kaggle run used 1 × DummyVecEnv)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (Kaggle run did not set one; omit for non-deterministic)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Where to write model_*.zip and run_config.json (default: repo root)",
    )
    p.add_argument("--checkpoint-freq", type=int, default=200_000)
    p.add_argument("--device", type=str, default="auto", help="'auto' | 'cuda' | 'cpu'")
    p.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard under logs/tb")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--progress-bar", action="store_true", help="SB3 progress bar (needs tqdm/rich)")
    return p.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def apply_config_to_args(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Overlay non-CLI defaults from JSON when the user did not need them."""
    if "env_id" in cfg:
        args.env_id = cfg["env_id"]
    if "total_timesteps" in cfg:
        args.timesteps = int(cfg["total_timesteps"])
    if "n_envs" in cfg:
        args.n_envs = int(cfg["n_envs"])
    if "seed" in cfg and cfg["seed"] is not None:
        args.seed = int(cfg["seed"])
    if "checkpoint_freq" in cfg:
        args.checkpoint_freq = int(cfg["checkpoint_freq"])
    if cfg.get("hardcore"):
        args.hardcore = True


def hyperparams_from_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    hp = dict(DEFAULT_HYPERPARAMS)
    if not cfg:
        return hp
    raw = cfg.get("hyperparams") or {}
    for key in (
        "learning_rate",
        "n_steps",
        "batch_size",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "clip_range",
        "ent_coef",
        "vf_coef",
        "max_grad_norm",
    ):
        if key in raw:
            hp[key] = raw[key]
    if "policy" in cfg:
        hp["policy"] = cfg["policy"]
    elif "policy" in raw:
        hp["policy"] = raw["policy"]
    # Optional explicit net_arch; Kaggle left this as SB3 default.
    net_arch = raw.get("net_arch")
    if net_arch is not None:
        hp["policy_kwargs"] = {"net_arch": net_arch}
    return hp


def make_env(env_id: str, hardcore: bool, seed: int | None, rank: int = 0):
    def _init():
        env = gym.make(env_id, hardcore=hardcore, render_mode=None)
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(env_id: str, n_envs: int, hardcore: bool, seed: int | None):
    # Published run: DummyVecEnv with a single env. Multi-env is optional for speed.
    thunks = [make_env(env_id, hardcore, seed, i) for i in range(n_envs)]
    if n_envs <= 1:
        return DummyVecEnv(thunks)
    return SubprocVecEnv(thunks)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
        apply_config_to_args(args, cfg)

    hp = hyperparams_from_config(cfg if cfg else None)
    policy = hp.pop("policy")
    policy_kwargs = hp.pop("policy_kwargs", None)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    if args.tensorboard:
        log_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    train_env = build_vec_env(args.env_id, args.n_envs, args.hardcore, args.seed)

    ppo_kwargs: dict[str, Any] = {
        "policy": policy,
        "env": train_env,
        "verbose": 1,
        "device": device,
        "tensorboard_log": str(log_dir / "tb") if args.tensorboard else None,
        **hp,
    }
    if args.seed is not None:
        ppo_kwargs["seed"] = args.seed
    if policy_kwargs is not None:
        ppo_kwargs["policy_kwargs"] = policy_kwargs

    model = PPO(**ppo_kwargs)

    # With n_envs > 1, num_timesteps advances by n_envs per call; divide so
    # wall-clock checkpoints still land near every checkpoint_freq env steps.
    # Kaggle used n_envs=1 so save_freq == checkpoint_freq.
    save_freq = max(1, args.checkpoint_freq // max(1, args.n_envs))
    checkpoint_cb = SaveCheckpoint(save_freq=save_freq, save_path=out_dir)

    meta = {
        "env_id": args.env_id,
        "hardcore": bool(args.hardcore),
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "device": device,
        "checkpoint_freq": args.checkpoint_freq,
        "hyperparams": {
            **{k: v for k, v in DEFAULT_HYPERPARAMS.items() if k != "policy"},
            "policy": policy,
            **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
            "note": "Matches Kaggle bipedalwalker-ppo notebook (ent_coef=0.001, DummyVecEnv×1).",
        },
        "reference_run": {
            "total_steps": 1_000_000,
            "hardware": "Tesla T4 (Kaggle)",
            "wall_time_minutes": 35,
            "final_episode_reward": 133,
            "sb3_version": "2.8.0",
            "gymnasium_version": "1.2.0",
            "notes": (
                "Partial locomotion success; classic 'solved' threshold is ~300 mean return. "
                "Published checkpoints: model_600000.zip, model_final.zip."
            ),
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Training PPO on {args.env_id} (hardcore={args.hardcore}) for {args.timesteps:,} steps")
    print(f"  n_envs={args.n_envs}  device={device}  ent_coef={hp.get('ent_coef')}  seed={args.seed}")
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        progress_bar=args.progress_bar,
    )

    final_path = out_dir / "model_final"
    model.save(str(final_path))
    print(f"Saved final model → {final_path}.zip")

    train_env.close()


if __name__ == "__main__":
    main()
