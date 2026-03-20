# Scripts Reference

All runnable scripts grouped by workflow stage. For a full table with venv and test info see [index.md](index.md).

---

## Teleoperation and recording

**`scripts/teleop/teleop_agent.py`** — run teleoperation without recording. Useful for practicing or checking robot response before a recording session. Supports keyboard, gamepad, and SO-101 leader arm.

**`scripts/teleop/record_episodes.py`** — record teleoperated episodes to a LeRobot v3.0 dataset (parquet + H.264 video). Each episode is saved on success/discard. Supports spawn diversity (`--diversity-keys`, `--diversity-ratio`) to ensure even coverage of the spawn zone. Use `--reward-mode success|sim|sim+success` to record reward annotations alongside the dataset (for IQL, classifier training, or analysis).

**`scripts/teleop/replay_dataset.py`** — replay recorded episodes in simulation. Useful for verifying dataset quality visually.

**`scripts/calibrate_leader.py`** — calibrate the SO-101 leader arm. Writes homing offsets to motor registers. Run once before first use, or after reassembly.

---

## Evaluation

### Single checkpoint

**`scripts/eval/eval_act_policy.py`** — evaluate an ACT checkpoint in simulation. Reports success rate over N episodes.

**`scripts/eval/eval_diffusion_policy.py`** — evaluate a Diffusion Policy checkpoint. Use `--num-inference-steps 10` for faster DDIM denoising.

**`scripts/eval/eval_vla_policy.py`** — evaluate a SmolVLA checkpoint. Supports `--noise-prior` (learned noise sampler or fixed mu), `--n-action-steps` (how many steps to execute per inference call). See [eval_vla_policy.md](eval_vla_policy.md) for full options.

**`scripts/eval/eval_sac_policy.py`** — evaluate a SAC policy checkpoint.

**`scripts/eval/eval_reward_classifier.py`** — evaluate the reward classifier on a dataset. Reports accuracy and confidence distribution.

### Sweep eval (multiple checkpoints)

**`scripts/eval/sweep_act_eval_single.py`** — evaluate multiple ACT checkpoints with a shared random seed using single-env eval. Reliable results. Recommended.

**`scripts/eval/sweep_act_eval.py`** — parallel variant (faster, uses N envs).

**`scripts/eval/sweep_diffusion_eval_single.py`** — same for Diffusion Policy (single-env).

**`scripts/eval/sweep_diffusion_eval.py`** — parallel variant for Diffusion Policy.

**`scripts/eval/sweep_vla_eval.py`** — same for SmolVLA. Accepts `--all` to sweep all saved checkpoints automatically.

### Parallel eval

**`scripts/eval/eval_act_policy_parallel.py`** and **`eval_diffusion_policy_parallel.py`** — run N environments in parallel for faster evaluation.

### Diagnostics

**`tests/sim/test_env_spawn.py`** — visual spawn zone verification. Runs N resets, prints cube and platform positions, holds GUI open. Use this after creating or modifying an env to check spawn zones look correct.

**`scripts/eval/analyze_classifier_distribution.py`** — plot classifier confidence histogram on a dataset. Useful for checking whether the classifier separates success/failure cleanly.

**`scripts/eval/collect_rollouts.py`** — collect IK-based rollouts for Neural IK dataset generation. See [collect_rollouts.md](collect_rollouts.md).

---

## Training

### Imitation learning

**`scripts/train/train_act.py`** — train ACT policy on a recorded dataset. Uses `configs/policy/act/baseline.yaml` by default; variants: `high_kl.yaml`, `low_kl.yaml`.

**`scripts/train/train_diffusion.py`** — train Diffusion Policy. Uses `configs/policy/diffusion/baseline.yaml`.

For SmolVLA BC training use `lerobot-train` directly (see [workflow/03_bc_training.md](../workflow/03_bc_training.md)).

### Reward preparation

**`scripts/train/prepare_classifier_dataset.py`** — label dataset frames for binary reward classifier training (gripper-based success detection).

**`scripts/train/prepare_reward_dataset.py`** — label dataset with VIP rewards or sim rewards. Output dataset is used for IQL critics training.

### Reward model training

**`scripts/train/train_reward_classifier_v2.py`** — train binary success classifier. Unfrozen ResNet18 backbone, differential learning rates, balanced sampling. More accurate than the frozen-backbone version.

### Offline RL

**`scripts/train/train_iql_critics.py`** — train IQL V/Q critics on a labeled dataset. Produces `critics.pt` used to compute advantage weights.

**`scripts/train/train_vla_weighted_bc.py`** — fine-tune SmolVLA with IQL advantage weights. Wraps `lerobot-train` with a monkey-patch that multiplies per-sample loss by the IQL weight. Good demonstrations (high advantage) get stronger gradient signal.

### Online RL

**`scripts/train/train_sac_composite_grpc.py`** — SAC training via gRPC two-process split. Client (isaaclab-env) runs env loop + HIL; server (lerobot-env) holds SACPolicy + replay buffers + reward models. Works with lerobot 0.5.1. Use `--auto-server` to auto-start the server.

**`scripts/train/sac_server.py`** — gRPC server for SAC training. Holds SACPolicy, 3 replay buffers (online/offline/intervention), RewardClassifier, VIPReward, and all SAC update logic.

**`scripts/train/train_iql_pretrain.py`** — SAC-side IQL pretraining: warm-starts the SAC actor and critic before online training.

**`scripts/train/train_flow_noise_ppo.py`** — PPO fine-tuning of SmolVLA via a stochastic noise head in the flow matching ODE. Trains only the noise head + value MLP (~640k params), VLA weights stay frozen. See [train_flow_noise_ppo.md](train_flow_noise_ppo.md).

**`scripts/train/train_noise_sampler_offline.py`** — train Learned Noise Sampler offline by backpropagating through the frozen VLA ODE. Faster than online, no simulator needed.

**`scripts/train/train_noise_sampler_online.py`** — train Learned Noise Sampler online via REINFORCE in simulation. Slower but directly optimizes success rate.

---

## Dataset utilities

**`scripts/tools/trim_after_success.py`** — remove frames after the success event in each episode, re-encode video. Use after recording to clean up trailing frames where the robot holds position.

**`scripts/tools/filter_dataset.py`** — filter episodes by criteria (success rate, length, etc.).

**`scripts/tools/recompute_stats.py`** — recompute `stats.json` after modifying a dataset.

**`scripts/tools/compute_iql_weights.py`** — compute IQL advantage weights from trained critics and write them into the dataset. Required before `train_vla_weighted_bc.py`.

**`scripts/verify_lerobot_dataset.py`** — verify LeRobot v3.0 dataset integrity (parquet schema, video files, stats).

**`scripts/visualize_lerobot_rerun.py`** — visualize dataset episodes in Rerun. Requires `venvs/rerun`.

---

## Camera and diagnostics

**`scripts/tools/camera_preview.py`** — live preview from USB cameras (real robot). Useful for checking camera placement and focus.

**`scripts/tools/camera_snapshot.py`** — save snapshots from USB cameras. Use for sim-vs-real visual comparison.

**`scripts/tools/camera_viewer.py`** — OpenCV viewer that reads sim camera frames from `/dev/shm` during recording. Run in a second terminal alongside `record_episodes.py`. Requires `venvs/viewer`.

**`scripts/tools/eval_viewer.py`** — same as `camera_viewer.py` but for eval scripts. Requires `venvs/viewer`.

**`scripts/tools/hil_viewer.py`** — camera viewer for HIL teleoperation sessions.

**`tests/sim/test_vip_reward.py`** — test VIP reward computation on a dataset: prints reward range, mean, and visualizes goal frames.

**`scripts/tools/verify_rewards.py`** — verify that sim reward values in a recorded dataset are sensible.

**`scripts/tools/measure_x0_sensitivity.py`** — measure how shifting x_0 affects SmolVLA action output. Used to calibrate noise prior scale.

**`scripts/tools/trackio_cleanup.py`** — remove old or duplicate trackio experiment runs.

---

## Neural IK

**`scripts/tools/generate_ik_dataset.py`** — generate a dataset of (end-effector pose → joint angles) pairs using the Isaac Lab IK solver. Input for neural IK training.

**`scripts/tools/train_neural_ik.py`** — train a small MLP to approximate the IK solver.

**`scripts/tools/verify_neural_ik.py`** — compare neural IK predictions against the sim IK solver.

---

## USD / URDF utilities

**`scripts/convert_usd.py`** — convert URDF to USD format for Isaac Sim.

**`scripts/utils/interactive_camera_setup.py`** — interactive tool for adjusting camera pose in Isaac Sim and printing the resulting config.

---

## Tests

**`tests/sim/`** — functional tests that require Isaac Sim. Run these after making changes to envs or devices to catch regressions. See also `tests/` for unit tests that do not require Isaac Sim.

**`tests/sim/test_robot.py`** — interactive joint movement test for the real robot. Modes: `interactive` (manual), `preset` (fixed positions), `test` (automated sweep).

**`scripts/isaac_sim_standalone/test_robot.py`** — same test modes using pure Isaac Sim API (no Isaac Lab). Useful when Isaac Lab is not installed or for low-level debugging. Known limitation: robot spawns at world origin, not on table. See [isaac_sim_standalone.md](isaac_sim_standalone.md).
