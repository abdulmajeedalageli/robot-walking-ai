"""
CLI entry point for viewing a trained humanoid policy.

Supports either:
  - a Hub checkpoint: python main.py --repo ... --filename ...
  - a local checkpoint from train.py:
        python main.py --model-path models/sac_natural_gait.zip --env-id Humanoid-v5
"""

import argparse
import sys
import logging
from src.simulation import NativeHumanoidSimulation


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained RL policy in a native MuJoCo Humanoid environment."
    )

    parser.add_argument(
        "--repo",
        type=str,
        default="jren123/sac-humanoid-v4",
        help="Hugging Face repo ID for the pretrained brain (ignored if --model-path is set).",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="SAC-Humanoid-v4.zip",
        help="Filename inside the Hugging Face repo (ignored if --model-path is set).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a local .zip checkpoint (e.g. from train.py). Overrides --repo/--filename.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="Humanoid-v4",
        help="Gymnasium env ID to run the policy in. Has to match whatever "
        "the loaded policy was trained on.",
    )

    args = parser.parse_args()

    try:
        sim = NativeHumanoidSimulation(
            repo_id=args.repo,
            filename=args.filename,
            local_model_path=args.model_path,
            env_id=args.env_id,
        )
        sim.load_brain()
        sim.run_simulation()
    except KeyboardInterrupt:
        logging.info("Interrupted, exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()