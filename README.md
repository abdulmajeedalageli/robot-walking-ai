# Humanoid-v4 SAC Simulation

A high-performance, real-time reinforcement learning simulation of a bipedal humanoid robot navigating DeepMind's MuJoCo physics engine. This project executes a pre-trained Soft Actor-Critic (SAC) neural network natively without legacy Gym dependencies or memory buffer leaks.

---

## Features

* **Modern Stack:** Built on native `gymnasium` and `mujoco` (v4 environment).
* **Pre-trained Expert Brain:** Uses SAC weights pulled directly from Hugging Face (`jren123/sac-humanoid-v4`).
* **Memory-Optimized:** Replay buffer capped at runtime to eliminate multi-gigabyte memory allocations.
* **Interactive Viewer:** Multi-threaded native viewer supporting dynamic perturbation and real-time physical disturbance recovery.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/humanoid-sac-simulation.git](https://github.com/your-username/humanoid-sac-simulation.git)
   cd humanoid-sac-simulation