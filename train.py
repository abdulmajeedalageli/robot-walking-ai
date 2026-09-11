"""
Train / fine-tune a SAC humanoid policy for a more natural-looking gait.

Wraps the Humanoid env with NaturalGaitRewardWrapper (arm-swing symmetry +
straight-knee-at-stance bonuses) on top of the default forward-velocity
reward, then trains with Stable-Baselines3's SAC.

Examples
--------
Train from scratch on Humanoid-v5:
    python train.py --env-id Humanoid-v5 --total-timesteps 3000000

Fine-tune from the pretrained Hub checkpoint (env-id has to match what
the checkpoint was originally trained on, since obs/action shapes differ
between v4 and v5):
    python train.py --env-id Humanoid-v4 --start-from-pretrained \
        --repo jren123/sac-humanoid-v4 --filename SAC-Humanoid-v4.zip \
        --total-timesteps 1000000 --learning-rate 3e-5

Check joint/body names if you swap in a custom MuJoCo model:
    python train.py --debug-names
"""

import argparse
import logging
import os

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from huggingface_sb3 import load_from_hub

from src.reward_wrappers import NaturalGaitRewardWrapper

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def make_env(env_id, arm_swing_weight, knee_weight, seed=None):
    def _init():
        env = gym.make(env_id)
        env = NaturalGaitRewardWrapper(
            env,
            arm_swing_weight=arm_swing_weight,
            knee_weight=knee_weight,
        )
        env = Monitor(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return _init


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train/fine-tune SAC on the Humanoid environment with "
        "natural-gait reward shaping."
    )
    parser.add_argument("--env-id", type=str, default="Humanoid-v5")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--arm-swing-weight", type=float, default=0.05)
    parser.add_argument("--knee-weight", type=float, default=0.03)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--start-from-pretrained", action="store_true")
    parser.add_argument("--repo", type=str, default="jren123/sac-humanoid-v4")
    parser.add_argument("--filename", type=str, default="SAC-Humanoid-v4.zip")

    parser.add_argument("--save-path", type=str, default="models/sac_natural_gait")
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--tensorboard-log", type=str, default="tb_logs")

    parser.add_argument("--debug-names", action="store_true",
                         help="Print joint/body names for the env's MuJoCo "
                              "model and exit, without training.")
    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    if args.debug_names:
        env = gym.make(args.env_id)
        wrapped = NaturalGaitRewardWrapper(env)
        wrapped.reset()
        wrapped.debug_print_names()
        return

    env = make_vec_env(
        make_env(args.env_id, args.arm_swing_weight, args.knee_weight, args.seed),
        n_envs=args.n_envs,
        seed=args.seed,
    )
    eval_env = make_vec_env(
        make_env(args.env_id, args.arm_swing_weight, args.knee_weight, args.seed + 1),
        n_envs=1,
    )

    if args.start_from_pretrained:
        logging.info(f"Loading checkpoint from {args.repo}/{args.filename} to fine-tune.")
        checkpoint = load_from_hub(args.repo, args.filename)
        model = SAC.load(
            checkpoint,
            env=env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            tensorboard_log=args.tensorboard_log,
        )
    else:
        logging.info(f"Training SAC from scratch on {args.env_id}.")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            tensorboard_log=args.tensorboard_log,
            verbose=1,
            seed=args.seed,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // max(args.n_envs, 1), 1),
        save_path=os.path.dirname(args.save_path) or ".",
        name_prefix=os.path.basename(args.save_path),
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(args.save_path) or ".",
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    logging.info(f"Starting training for {args.total_timesteps} timesteps.")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(args.save_path)
    logging.info(f"Done. Saved to {args.save_path}.zip")


if __name__ == "__main__":
    main()