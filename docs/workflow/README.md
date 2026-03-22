# Workflow

Step-by-step guides for the full pipeline: from setting up the environment to training and evaluating policies.

| Step | Guide | Description |
|------|-------|-------------|
| 0 | [Setup](00_setup.md) | Install Isaac Lab, LeRobot, configure venvs and `.env` |
| 0a | [Try demo](00a_try_demo.md) | Download demo datasets + checkpoint, visualize, eval, train |
| 1 | [Environment design](01_env_design.md) | Create or modify a task environment in Isaac Lab |
| 2 | [Recording](02_recording.md) | Record teleoperated episodes as LeRobot datasets |
| 3 | [BC training](03_bc_training.md) | Train ACT, Diffusion Policy, or SmolVLA via behavior cloning |
| 4 | [Offline RL](04_offline_rl.md) | IQL critics + advantage-weighted BC (VIP reward labeling) |
| 5 | [Online RL](05_online_rl.md) | Flow-Noise PPO with frozen VLA backbone |
| 5b | [SAC RL](05b_sac_rl.md) | SAC with RLPD (off-policy, *in progress*) |
| 6 | [Adapting to new tasks](06_adapting.md) | Checklist for adding new environments |
| 7 | [Demo recording](07_demo_recording.md) | Record eval/teleop GIFs for documentation |
