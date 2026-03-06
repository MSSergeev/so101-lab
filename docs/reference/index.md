# Reference Index

All scripts and library modules — description, documentation, tests, venv.

For module boundaries, venv rationale, and how to adapt the project: [structure.md](structure.md)

Venv legend: `il` = Isaac Lab env, `lr` = lerobot-env, `rerun` = venvs/rerun, `viewer` = venvs/viewer

---

## Scripts

### Teleoperation and recording

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/teleop/teleop_agent.py` | Teleoperation without recording (keyboard, gamepad, leader arm) | reference/teleop_devices.md | — | il |
| `scripts/teleop/record_episodes.py` | Record teleoperated episodes to LeRobot v3.0 dataset; `--reward-mode` for optional reward annotations | reference/record_episodes.md | — | il |
| `scripts/teleop/replay_dataset.py` | Replay recorded dataset in simulation | reference/replay_dataset.md | — | il |
| `scripts/calibrate_leader.py` | Calibrate SO-101 leader arm (writes homing offsets to motors) | reference/teleop_devices.md | — | lr |

### Evaluation

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/eval/eval_act_policy.py` | Evaluate ACT checkpoint in simulation | reference/eval_act_policy.md | — | il |
| `scripts/eval/eval_act_policy_parallel.py` | Evaluate ACT with N parallel envs | reference/eval_act_policy.md | — | il |
| `scripts/eval/eval_diffusion_policy.py` | Evaluate Diffusion Policy checkpoint | reference/train_diffusion.md | — | il |
| `scripts/eval/eval_diffusion_policy_parallel.py` | Evaluate Diffusion Policy with N parallel envs | reference/train_diffusion.md | — | il |
| `scripts/eval/eval_vla_policy.py` | Evaluate SmolVLA/Pi0 via gRPC policy server; `--auto-server` starts server automatically, `--noise-prior` patches model | reference/eval_vla_policy.md | — | il |
| `scripts/eval/smolvla_server.py` | gRPC policy server (lerobot-env, Python 3.12) — loads SmolVLA/Pi0, serves inference via `AsyncInference` proto | (docstring) | — | lr |
| `scripts/eval/eval_sac_policy.py` | Evaluate SAC checkpoint via gRPC (isaaclab-env + sac_server) | (docstring) | — | il |
| `scripts/eval/eval_reward_classifier.py` | Evaluate reward classifier on a dataset | reference/reward_classifier.md | — | lr |
| `scripts/eval/sweep_act_eval.py` | Sweep ACT checkpoints (parallel eval) | reference/eval_act_policy.md | — | il |
| `scripts/eval/sweep_act_eval_single.py` | Sweep ACT checkpoints (single-env eval) | reference/eval_act_policy.md | — | il |
| `scripts/eval/sweep_diffusion_eval.py` | Sweep Diffusion checkpoints (parallel eval) | reference/train_diffusion.md | — | il |
| `scripts/eval/sweep_diffusion_eval_single.py` | Sweep Diffusion checkpoints (single-env eval) | reference/train_diffusion.md | — | il |
| `scripts/eval/sweep_vla_eval.py` | Sweep SmolVLA checkpoints; auto-starts/stops gRPC policy server | reference/eval_vla_policy.md | — | il |
| `scripts/eval/collect_rollouts.py` | Collect IK rollouts with DR for offline dataset | reference/collect_rollouts.md | — | il |
| `scripts/eval/test_env_spawn.py` | Visual spawn zone verification (N resets, prints positions) | reference/env_design.md | — | il |
| `scripts/eval/analyze_classifier_distribution.py` | Classifier confidence histogram diagnostic | (docstring) | — | lr |

### Training — BC / VLA

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/train/train_act.py` | Train ACT policy | reference/train_act.md | — | lr |
| `scripts/train/train_diffusion.py` | Train Diffusion Policy | reference/train_diffusion.md | — | lr |
| `scripts/train/train_reward_classifier_v2.py` | Train binary reward classifier (unfrozen backbone, balanced sampling) | reference/reward_classifier.md | — | lr |
| `scripts/train/train_vla_weighted_bc.py` | Fine-tune SmolVLA with IQL advantage weights | reference/train_vla_weighted_bc.md | — | lr |
| `scripts/train/prepare_classifier_dataset.py` | Label dataset frames for reward classifier training | reference/reward_classifier.md | — | lr |
| `scripts/train/prepare_reward_dataset.py` | Label dataset with VIP or sim rewards | reference/prepare_reward_dataset.md | — | lr |

### Training — RL

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/train/train_sac_composite_grpc.py` | SAC training via gRPC — client (env only), works with lerobot 0.5.1 | (docstring) | — | il |
| `scripts/train/sac_server.py` | gRPC SAC server — holds SACPolicy + replay buffers + reward models + SAC update | (docstring) | — | lr |
| `scripts/train/train_iql_critics.py` | Train IQL critics (V, Q) for advantage weight computation | reference/train_iql_critics.md | — | lr |
| `scripts/train/train_iql_pretrain.py` | SAC-side IQL pretraining (actor + critic warm-start) | reference/train_iql_pretrain.md | — | lr |
| `scripts/train/train_flow_noise_ppo.py` | PPO fine-tuning of SmolVLA via flow noise head (single-process, requires sys.path hack) | reference/train_flow_noise_ppo.md | — | il |
| `scripts/train/train_flow_noise_ppo_grpc.py` | PPO fine-tuning via gRPC — client (env only), works with lerobot 0.5.1 | reference/train_flow_noise_ppo_grpc.md | — | il |
| `scripts/train/ppo_server.py` | gRPC PPO server — holds SmolVLA + VIP + rollout buffer + PPO update | reference/train_flow_noise_ppo_grpc.md | — | lr |
| `scripts/train/train_noise_sampler_offline.py` | Train Learned Noise Sampler offline (backprop through ODE) | reference/train_noise_sampler.md | — | lr |
| `scripts/train/train_noise_sampler_online.py` | Train Learned Noise Sampler online (REINFORCE in sim) | reference/train_noise_sampler.md | — | il |

### Tools — dataset utilities

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/tools/trim_after_success.py` | Remove frames after success event, re-encode video | reference/trim_after_success.md | — | lr |
| `scripts/tools/filter_dataset.py` | Filter episodes by criteria | reference/filter_dataset.md | — | lr |
| `scripts/tools/recompute_stats.py` | Recompute dataset statistics (stats.json) | (docstring) | — | lr |
| `scripts/tools/compute_iql_weights.py` | Compute IQL advantage weights from trained critics | reference/train_iql_critics.md | — | lr |
| `scripts/verify_lerobot_dataset.py` | Verify LeRobot v3.0 dataset integrity | (docstring) | — | lr |
| `scripts/visualize_lerobot_rerun.py` | Visualize dataset episodes in Rerun | reference/visualize_rerun.md | — | rerun |

### Tools — diagnostics and camera

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/tools/camera_preview.py` | Live preview from USB cameras (real robot) | reference/sim2real.md | — | lr |
| `scripts/tools/camera_snapshot.py` | Save snapshots from USB cameras for sim-vs-real comparison | reference/sim2real.md | — | lr |
| `scripts/tools/camera_viewer.py` | OpenCV preview of sim camera feed during recording | reference/viewer_tools.md | — | viewer |
| `scripts/tools/eval_viewer.py` | OpenCV viewer for sim camera feed during eval | reference/viewer_tools.md | — | viewer |
| `scripts/tools/hil_viewer.py` | Camera viewer for HIL teleoperation | reference/viewer_tools.md | — | viewer |
| `scripts/tools/test_vip_reward.py` | Test VIP reward computation on a dataset | (docstring) | — | lr |
| `scripts/tools/verify_rewards.py` | Verify sim reward values in recorded dataset | (docstring) | — | il |
| `scripts/tools/measure_x0_sensitivity.py` | Measure how x_0 shift affects SmolVLA action output | (docstring) | — | il |
| `scripts/tools/trackio_cleanup.py` | Clean up old trackio experiment runs | (docstring) | — | lr |

### Tools — Neural IK

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/tools/generate_ik_dataset.py` | Generate FK dataset via pinocchio for Neural IK training | reference/neural_ik.md | — | il |
| `scripts/tools/train_neural_ik.py` | Train Neural IK MLP on generated dataset | reference/neural_ik.md | — | il |
| `scripts/tools/verify_neural_ik.py` | Verify Neural IK accuracy via pinocchio FK | reference/neural_ik.md | — | il |

### Tools — USD / URDF utilities

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/convert_usd.py` | Convert URDF to USD | (docstring) | — | il |
| `scripts/utils/interactive_camera_setup.py` | Interactive tool for camera pose setup in Isaac Sim | reference/performance_tuning.md | — | il |

### Isaac Sim standalone (no Isaac Lab)

| Script | Description | Doc | Test | Venv |
|--------|-------------|-----|------|------|
| `scripts/isaac_sim_standalone/test_robot.py` | Joint movement test using pure Isaac Sim API | reference/isaac_sim_standalone.md | — | il |

### Functional tests (require Isaac Sim)

| Script | Description | Venv |
|--------|-------------|------|
| `scripts/test/test_template_env.py` | Verify template env loads and steps | il |
| `scripts/test/test_pick_cube.py` | Verify pick_cube env | il |
| `scripts/test/test_rl_gym_wrapper.py` | Verify gym wrapper (ManagerBasedRLEnv → Gymnasium) | il |
| `scripts/test/test_contact_sensor.py` | Verify contact sensor in sim | il |
| `scripts/test/inspect_template_env.py` | Print template env observation/action shapes | il |
| `scripts/test_robot.py` | Interactive joint movement test for real robot (modes: interactive/preset/test/all) | il |

---

## Library modules (`so101_lab/`)

### assets/

| Module | Description | Doc |
|--------|-------------|-----|
| `assets/robots/so101.py` | SO-101 ArticulationCfg for Isaac Lab (`SO101_CFG`) | (docstring) |
| `assets/scenes/workbench.py` | Scene configs: `WORKBENCH_CLEAN_CFG`, `GROUND_PLANE_CFG` | (docstring) |

### tasks/

| Module | Description | Doc |
|--------|-------------|-----|
| `tasks/template/env.py` | Base DirectRLEnv for all tasks | reference/env_design.md |
| `tasks/template/env_cfg.py` | Scene config, camera setup, teleop device support | reference/env_design.md |
| `tasks/pick_cube/` | Pick-and-place example task | (docstring) |
| `tasks/figure_shape_placement/` | Main task: place cube into slot | reference/env_design.md |
| `tasks/figure_shape_placement_easy/` | Easy variant: small spawn zone, fixed platform | reference/env_design.md |

### devices/

| Module | Description | Doc |
|--------|-------------|-----|
| `devices/keyboard.py` | Keyboard teleoperation device | reference/teleop_devices.md |
| `devices/gamepad.py` | Gamepad teleoperation device | reference/teleop_devices.md |
| `devices/device_base.py` | Abstract base for teleop devices | reference/teleop_devices.md |
| `devices/action_process.py` | Convert device output to env action tensor | reference/teleop_devices.md |
| `devices/lerobot/so101_leader.py` | SO-101 leader arm device | reference/teleop_devices.md |
| `devices/lerobot/common/motors/` | FeetechMotorsBus, Motor, MotorCalibration | reference/teleop_devices.md |

### data/

| Module | Description | Doc |
|--------|-------------|-----|
| `data/lerobot_dataset.py` | LeRobotDatasetWriter — writes parquet + H.264 video (v3.0 format) | reference/lerobot_dataset.md |
| `data/collector.py` | RecordingManager — orchestrates env + device + dataset writer | reference/lerobot_dataset.md |
| `data/converters.py` | Isaac Lab rad → LeRobot normalized conversion | reference/lerobot_dataset.md |
| `data/video_utils.py` | H.264 encoding utilities | reference/lerobot_dataset.md |

### policies/

| Module | Description | Doc |
|--------|-------------|-----|
| `policies/act/` | ACT inference wrapper | (docstring) |
| `policies/diffusion/` | Diffusion Policy inference wrapper | (docstring) |
| `policies/grpc_client.py` | Synchronous gRPC client for remote policy inference (Python 3.11); action chunking, connects to `smolvla_server.py` | (docstring) |
| `policies/rl/critics.py` | IQL V/Q critics for advantage weight computation | reference/critics.md |
| `policies/rl/flow_noise_smolvla.py` | SmolVLA with stochastic noise head for PPO | reference/flow_noise_smolvla.md |
| `policies/rl/noise_prior.py` | Fixed mu shift for flow matching x_0 | reference/critics.md |
| `policies/rl/learned_noise_sampler.py` | State-conditional MLP: obs → (mu, sigma) for x_0 | reference/critics.md |
| `policies/rl/neural_ik.py` | Neural IK MLP: ee_pose → joint_angles | reference/neural_ik.md |
| `policies/rl/ik_policy.py` | IK-based policy for rollout collection | reference/collect_rollouts.md + reference/neural_ik.md |

### rewards/

| Module | Description | Doc |
|--------|-------------|-----|
| `rewards/classifier.py` | Binary success classifier (ResNet18 backbone) | reference/reward_classifier.md |
| `rewards/vip_reward.py` | VIP reward: ResNet50 Ego4D embeddings, goal-conditioned | reference/vip_reward.md |

### rl/

| Module | Description | Doc |
|--------|-------------|-----|
| `rl/isaac_lab_gym_env.py` | ManagerBasedRLEnv → Gymnasium wrapper for SAC/PPO | reference/isaac_lab_gym_env.md |
| `rl/domain_rand.py` | `apply_domain_rand_flags()` — shared DR helper for all RL scripts | (docstring) |
| `rl/hil_input.py` | HIL toggle via keyboard/shared memory *(incomplete)* | — |
| `rl/hil_device.py` | HIL leader arm reader *(incomplete)* | — |

### transport/

| Module | Description | Doc |
|--------|-------------|-----|
| `transport/ppo.proto` | Protobuf service definition for PPO training (9 RPCs) | reference/train_flow_noise_ppo_grpc.md |
| `transport/utils.py` | Standalone chunked gRPC transport (no lerobot imports, works in 3.11 and 3.12) | (docstring) |
| `transport/ppo_client.py` | `PPOTrainingClient` — gRPC client for isaaclab-env | reference/train_flow_noise_ppo_grpc.md |
| `transport/ppo_pb2.py` | Generated protobuf stubs (from `ppo.proto`) | — |
| `transport/ppo_pb2_grpc.py` | Generated gRPC stubs (from `ppo.proto`) | — |
| `transport/sac.proto` | Protobuf service definition for SAC training (8 RPCs) | (docstring) |
| `transport/sac_client.py` | `SACTrainingClient` — gRPC client for isaaclab-env | (docstring) |
| `transport/sac_pb2.py` | Generated protobuf stubs (from `sac.proto`) | — |
| `transport/sac_pb2_grpc.py` | Generated gRPC stubs (from `sac.proto`) | — |

### utils/

| Module | Description | Doc |
|--------|-------------|-----|
| `utils/compat.py` | `patch_hf_custom_models()` — fix transformers ≥5.3 breaking custom HF models (ResNet10) | (docstring) |
| `utils/checkpoint.py` | `resolve_checkpoint_path()` — shared checkpoint resolution for eval/sweep scripts | (docstring) |
| `utils/scene_state.py` | `extract_scene_state()`, `get_object_state()`, `get_gripper_state()` — scene state for eval registry | (docstring) |
| `utils/spawn_diversity.py` | Spawn diversity tracker for balanced episode coverage | (docstring) |
| `utils/policy_server.py` | `start_policy_server()`, `start_ppo_server()`, `start_sac_server()` — start gRPC servers in lerobot-env subprocess | (docstring) |
| `utils/shm_preview.py` | Shared memory camera preview (write/read `/dev/shm`) | reference/viewer_tools.md |
| `utils/performance.py` | Timing and FPS measurement utilities | (docstring) |
| `utils/tracker.py` | `setup_tracker()`, `cleanup_tracker()` — unified trackio/wandb/none setup for train scripts | reference/experiment_tracking.md |
| `utils/system_monitor.py` | CPU/GPU/RAM metrics for experiment tracking | reference/experiment_tracking.md |

---

## Unit tests (`tests/`)

| Test | What it tests | Venv |
|------|--------------|------|
| `test_lerobot_v3_writer.py` | LeRobotDatasetWriter parquet + video output | lr |
| `test_keyboard_device.py` | Keyboard device action output | il |
| `test_gamepad.py` | Gamepad device | il |
| `test_motor_bus.py` | FeetechMotorsBus connect/read/write | lr |
| `test_teleop_integration.py` | Device + env integration | il |
| `test_device_gui.py` | Device with Isaac Sim GUI | il |
| `test_teleop_mode_gui.py` | Teleop mode with GUI | il |
| `test_backward_compatibility_gui.py` | Template env direct mode (no teleop device) | il |
