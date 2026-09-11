"""
Loads a trained SAC humanoid policy and runs it live in a MuJoCo viewer.

Can load from either a local checkpoint (e.g. something train.py just
saved) or a Hugging Face Hub repo. env_id is configurable so the viewer
always matches whatever environment the loaded policy was trained on
(Humanoid-v4 vs v5 have different obs/action shapes, so this has to match).
"""

import sys
import time
import logging
import warnings
import gymnasium as gym
import mujoco
import mujoco.viewer
from stable_baselines3 import SAC
from huggingface_sb3 import load_from_hub

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class NativeHumanoidSimulation:
    """
    Loads a trained SAC policy (from Hub or a local file) and runs it in
    a native MuJoCo viewer.
    """

    def __init__(
        self,
        repo_id="jren123/sac-humanoid-v4",
        filename="SAC-Humanoid-v4.zip",
        local_model_path=None,
        env_id="Humanoid-v4",
    ):
        self.repo_id = repo_id
        self.filename = filename
        self.local_model_path = local_model_path
        self.env_id = env_id
        self.model = None
        self.env = None

    def load_brain(self):
        # freeze everything training-related since this is inference only,
        # and shrink the buffer so it doesn't allocate memory we don't need
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
            "buffer_size": 1,
        }

        try:
            if self.local_model_path:
                logging.info(f"Loading local checkpoint: {self.local_model_path}")
                self.model = SAC.load(self.local_model_path, custom_objects=custom_objects)
            else:
                logging.info(f"Fetching weights from Hub: {self.repo_id}")
                checkpoint = load_from_hub(self.repo_id, self.filename)
                logging.info("Loading network...")
                self.model = SAC.load(checkpoint, custom_objects=custom_objects)

            logging.info("Brain loaded.")

        except Exception:
            logging.exception("Failed to load the brain.")
            sys.exit(1)

    def run_simulation(self):
        logging.info(f"Starting env: {self.env_id}")
        self.env = gym.make(self.env_id)
        obs, info = self.env.reset()

        mj_model = self.env.unwrapped.model
        mj_data = self.env.unwrapped.data
        sim_dt = self.env.unwrapped.dt

        logging.info("Launching viewer...")
        logging.info("Controls:")
        logging.info("  - Rotate camera: left-click + drag")
        logging.info("  - Pan camera: right-click + drag")
        logging.info("  - Apply force: double-click a limb + right-click drag")

        with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
            while viewer.is_running():
                step_start = time.time()

                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)

                if terminated or truncated:
                    logging.info("Fell over, resetting.")
                    obs, info = self.env.reset()

                viewer.sync()

                # pace the loop to real time so it doesn't run faster than
                # the physics timestep
                elapsed = time.time() - step_start
                time_to_wait = sim_dt - elapsed
                if time_to_wait > 0:
                    time.sleep(time_to_wait)

        logging.info("Viewer closed.")
        self.env.close()


if __name__ == "__main__":
    sim = NativeHumanoidSimulation()
    sim.load_brain()
    sim.run_simulation()