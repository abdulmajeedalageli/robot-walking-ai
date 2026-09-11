# Robot Walking AI

A reinforcement learning project where an AI agent controls a simulated humanoid robot and learns to walk in a physics-accurate 3D environment. The project includes both a live viewer for watching a trained policy walk, and a training pipeline for fine-tuning the agent toward a more natural-looking gait.

## What This Does

The humanoid is a physically simulated character with 17 controllable joints (hips, knees, shoulders, elbows, etc.), running inside [MuJoCo](https://mujoco.org/), a physics engine used widely in robotics research. An AI agent, trained with **Soft Actor-Critic (SAC)**, decides how to move each joint at every timestep to keep the humanoid balanced and walking forward.

Out of the box, the agent starts from a pretrained expert policy hosted on Hugging Face. On top of that, this project adds custom reward shaping aimed at making the walk look more human — rewarding natural arm-swing coordination and discouraging a crouched, shuffling gait.

## Project Structure

```
robot-walking-ai/
├── main.py                  # Launches the viewer to watch a trained policy walk
├── train.py                 # Trains / fine-tunes the policy
├── requirements.txt
├── src/
│   ├── simulation.py        # Loads a policy and runs it in the MuJoCo viewer
│   └── reward_wrappers.py   # Custom reward shaping for a more natural gait
└── models/                  # Local checkpoints saved during training (not tracked in git)
```

## Setup

```bash
git clone https://github.com/abdulmajeedalageli/robot-walking-ai.git
cd robot-walking-ai
python -m venv .venv
.venv\Scripts\activate      # on Windows
pip install -r requirements.txt
```

## Watching the Humanoid Walk

Run the viewer with the default pretrained expert:

```bash
python main.py
```

A MuJoCo window opens showing the humanoid walking, with camera controls:

- **Rotate camera:** left-click + drag
- **Pan camera:** right-click + drag
- **Apply force:** double-click a limb, then right-click + drag

If the humanoid falls over, it automatically resets and keeps walking.

To view a locally fine-tuned checkpoint instead of the default Hub model:

```bash
python main.py --model-path models/sac_natural_gait.zip --env-id Humanoid-v4
```

## Training / Fine-Tuning

`train.py` fine-tunes the pretrained expert with extra reward terms designed to encourage a more human-looking walk:

- **Arm-swing symmetry** — rewards the arms swinging in opposite directions, like a natural human walk.
- **Straight-knee-at-stance** — discourages bent knees while a foot is planted on the ground, reducing the crouched/shuffling look typical of naive RL policies.

Fine-tune starting from the pretrained checkpoint:

```bash
python train.py --start-from-pretrained --env-id Humanoid-v4 --total-timesteps 500000 --learning-rate 3e-5
```

Train a policy completely from scratch instead:

```bash
python train.py --env-id Humanoid-v5 --total-timesteps 2000000
```

Track training progress with TensorBoard:

```bash
tensorboard --logdir tb_logs
```

## Tech Stack

- **[MuJoCo](https://mujoco.org/)** — physics simulation engine
- **[Gymnasium](https://gymnasium.farama.org/)** — RL environment interface (Humanoid-v4 / v5)
- **[Stable-Baselines3](https://stable-baselines3.readthedocs.io/)** — SAC implementation
- **[Hugging Face Hub](https://huggingface.co/)** — hosting/loading the pretrained expert checkpoint

## Notes

Reward shaping alone has limits — it nudges the policy toward more natural-looking motion but can't fully replicate genuine human gait dynamics, which would require training against real motion-capture reference data (e.g. an Adversarial Motion Priors / DeepMimic-style setup). This project currently focuses on the lighter-weight reward-shaping approach as a practical first step.
