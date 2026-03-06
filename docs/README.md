# Documentation Map

This is the reference index. For task-oriented instructions (how to record, train, evaluate, deploy) see **[guide.md](guide.md)**.

- **[guide.md](guide.md)** — how to work with the project (by task: record, train, eval, deploy)
- **[architecture.md](architecture.md)** — venv split, code boundaries, data flow, data format
- **[troubleshooting.md](troubleshooting.md)** — common errors and fixes
- **[attribution.md](attribution.md)** — sources and licenses for adapted code
- **[results/easy_task.md](results/easy_task.md)** — benchmark results on `figure_shape_placement_easy`

---

## Experiment workflow

Step-by-step account of running the full pipeline on `figure_shape_placement_easy` — what was done, what parameters were used, what worked and what didn't.

| File | Covers |
|------|--------|
| [workflow/00_setup.md](workflow/00_setup.md) | Isaac Sim, Isaac Lab, LeRobot, venv setup |
| [workflow/01_env_design.md](workflow/01_env_design.md) | Task env creation, spawn zones, sim rewards, DR |
| [workflow/02_recording.md](workflow/02_recording.md) | Teleoperation, dataset recording, post-processing |
| [workflow/03_bc_training.md](workflow/03_bc_training.md) | ACT, Diffusion Policy, SmolVLA BC training and eval |
| [workflow/04_offline_rl.md](workflow/04_offline_rl.md) | IQL critics, VIP reward labeling, weighted BC fine-tune |
| [workflow/05_online_rl.md](workflow/05_online_rl.md) | Noise Prior, Learned Noise Sampler, Flow-Noise PPO |
| [workflow/05b_sac_rl.md](workflow/05b_sac_rl.md) | SAC online RL |

---

## Reference

Detailed documentation for each script and library module. Full index with venvs and test coverage: [reference/index.md](reference/index.md).

### Environment & tasks

| Doc | Covers |
|-----|--------|
| [reference/env_design.md](reference/env_design.md) | DirectRLEnv vs ManagerBasedRLEnv, task structure, spawn zones, obs/action spaces, how to add a new task |
| [reference/isaac_lab_gym_env.md](reference/isaac_lab_gym_env.md) | Gymnasium wrapper for SAC/PPO (ManagerBasedRLEnv → gym) |
| [reference/performance_tuning.md](reference/performance_tuning.md) | Isaac Sim rate limiting, FPS optimization |
| [reference/physics_tuning.md](reference/physics_tuning.md) | Contact sensors, friction, solver settings |
| [reference/isaac_sim_standalone.md](reference/isaac_sim_standalone.md) | Running Isaac Sim without Isaac Lab |

### Data & recording

| Doc | Covers |
|-----|--------|
| [reference/record_episodes.md](reference/record_episodes.md) | `record_episodes.py` — teleop recording with optional reward annotations |
| [reference/lerobot_dataset.md](reference/lerobot_dataset.md) | LeRobotDatasetWriter, RecordingManager, v3.0 format internals |
| [reference/replay_dataset.md](reference/replay_dataset.md) | Replaying a recorded dataset in simulation |
| [reference/trim_after_success.md](reference/trim_after_success.md) | Trim trailing frames after success event |
| [reference/filter_dataset.md](reference/filter_dataset.md) | Filter episodes by criteria |
| [reference/visualize_rerun.md](reference/visualize_rerun.md) | Visualize episodes in Rerun (multi-file video, JPEG compression) |

### Teleoperation & real robot

| Doc | Covers |
|-----|--------|
| [reference/teleop_devices.md](reference/teleop_devices.md) | Keyboard, gamepad, SO-101 leader arm, calibration |
| [reference/sim2real.md](reference/sim2real.md) | Camera calibration, sim-to-real gap, domain randomization |

### Policies — BC / VLA

| Doc | Covers |
|-----|--------|
| [reference/train_act.md](reference/train_act.md) | ACT training |
| [reference/train_diffusion.md](reference/train_diffusion.md) | Diffusion Policy training and eval |
| [reference/eval_act_policy.md](reference/eval_act_policy.md) | ACT eval (single-env, parallel, sweep) |
| [reference/eval_vla_policy.md](reference/eval_vla_policy.md) | SmolVLA eval with optional noise prior |
| [reference/train_vla_weighted_bc.md](reference/train_vla_weighted_bc.md) | SmolVLA fine-tune with IQL advantage weights |

### Reward models

| Doc | Covers |
|-----|--------|
| [reference/reward_classifier.md](reference/reward_classifier.md) | Binary success classifier (ResNet18), training and eval |
| [reference/vip_reward.md](reference/vip_reward.md) | VIP reward (ResNet50, Ego4D embeddings), goal-conditioned |
| [reference/prepare_reward_dataset.md](reference/prepare_reward_dataset.md) | Label dataset with VIP or sim rewards |

### RL

| Doc | Covers |
|-----|--------|
| [reference/collect_rollouts.md](reference/collect_rollouts.md) | IK scripted policy, rollout collection with DR |
| [reference/neural_ik.md](reference/neural_ik.md) | Neural IK MLP (pinocchio FK dataset, training, iterative refinement) |
| [reference/train_iql_critics.md](reference/train_iql_critics.md) | IQL V/Q critics, advantage weight computation |
| [reference/train_iql_pretrain.md](reference/train_iql_pretrain.md) | SAC-side IQL pretraining (actor + critic warm-start) |
| SAC gRPC training | `train_sac_composite_grpc.py` + `sac_server.py` (composite/sim/classifier/VIP reward) |
| [reference/critics.md](reference/critics.md) | Critics, LearnedNoiseSampler modules |

### Online RL — flow & noise

| Doc | Covers |
|-----|--------|
| [reference/flow_noise_smolvla.md](reference/flow_noise_smolvla.md) | SmolVLA with stochastic flow noise head |
| [reference/train_flow_noise_ppo.md](reference/train_flow_noise_ppo.md) | PPO fine-tuning via flow noise (single-process) |
| [reference/train_flow_noise_ppo_grpc.md](reference/train_flow_noise_ppo_grpc.md) | PPO via gRPC — two-process split for lerobot 0.5.1 (architecture, data flow, RPC reference) |
| [reference/train_noise_sampler.md](reference/train_noise_sampler.md) | Learned Noise Sampler (offline + online) |

### Tools & utilities

| Doc | Covers |
|-----|--------|
| [reference/viewer_tools.md](reference/viewer_tools.md) | OpenCV camera viewers (recording, eval, HIL) + shm architecture |
| [reference/experiment_tracking.md](reference/experiment_tracking.md) | trackio, wandb, system metrics |
| [reference/scripts.md](reference/scripts.md) | All scripts described in prose, grouped by workflow stage — good for browsing |
| [reference/index.md](reference/index.md) | Full registry: every script and module with venv, doc link, test coverage |
