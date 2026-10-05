"""Train a PPO policy on the gamified quadruped environment."""

from __future__ import annotations

import argparse
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quadruped_locomotion import QuadrupedGameEnv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--model-out", type=str, default="runs/ppo_quadruped")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--track-length", type=float, default=60.0)
    parser.add_argument("--start-level", type=int, default=1)
    parser.add_argument(
        "--reward-profile",
        choices=("spec", "balanced"),
        default="balanced",
        help="'balanced' rescales the per-step reward terms so locomotion "
        "outscores standing still; use 'spec' for the literal brief.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('Install training dependencies with `pip install -e ".[train]"`.') from exc

    def make_env(rank: int | None = None):
        def _factory():
            env = QuadrupedGameEnv(
                seed=None if rank is None else args.seed + rank,
                start_level=args.start_level,
                track_length=args.track_length,
                reward_profile=args.reward_profile,
            )
            return Monitor(env)

        return _factory

    env = make_vec_env(make_env(), n_envs=args.n_envs, seed=args.seed)
    eval_env = Monitor(
        QuadrupedGameEnv(
            seed=args.seed + 9999,
            start_level=args.start_level,
            track_length=args.track_length,
            reward_profile=args.reward_profile,
        )
    )

    runs_dir = pathlib.Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)

    # SB3 raises if tensorboard_log is set but tensorboard is missing, so only
    # enable it when the package is actually importable.
    try:
        import tensorboard  # noqa: F401
    except ImportError:
        tensorboard_log = None
        print(
            "note: tensorboard is not installed, logging to CSV only "
            '(pip install tensorboard, or "pip install -e .[train,tensorboard]").'
        )
    else:
        tensorboard_log = str(runs_dir / "tensorboard")

    callbacks = [
        CheckpointCallback(save_freq=max(1, 10_000 // args.n_envs), save_path=str(runs_dir / "checkpoints")),
        EvalCallback(
            eval_env,
            best_model_save_path=str(runs_dir / "best"),
            log_path=str(runs_dir / "eval"),
            eval_freq=max(1, 10_000 // args.n_envs),
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        device=args.device,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)

    out_path = pathlib.Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    env.close()
    eval_env.close()
    print(f"Saved PPO policy to {out_path}.zip (reward profile: {args.reward_profile})")


if __name__ == "__main__":
    main()