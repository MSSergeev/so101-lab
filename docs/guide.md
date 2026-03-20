# Project Guide

How to work with SO-101 Lab — by task, not by component.

For component details: [docs/README.md](README.md) → reference docs.
For the full experiment walkthrough: [docs/workflow/](workflow/).
For project structure, module boundaries, and how to adapt: [reference/structure.md](reference/structure.md).

---

## Getting started

```bash
# Setup venvs and install
cp .env.example .env          # set ISAACLAB_ENV and LEROBOT_ENV
act-isaac && uv pip install -e .
act-lerobot && uv pip install -e ".[hardware]"
```

Full setup (Isaac Sim, Isaac Lab, LeRobot, tool venvs): [workflow/00_setup.md](workflow/00_setup.md).

---

## I want to record a dataset

```bash
act-isaac  # activate Isaac Lab env

# With leader arm + sim rewards labeled at reset
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=so101leader \
    --calibration-file=leader_1.json \
    --port=/dev/ttyACM0 \
    --reward-mode success

# With keyboard (no real robot)
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=keyboard
```

Open camera preview in a separate terminal:
```bash
source venvs/viewer/bin/activate
python scripts/tools/camera_viewer.py
```

**Expected time:** ~1.3× total episode duration including discarded attempts. 50 episodes of 13–15 s each takes ~20 minutes.

Details: [reference/record_episodes.md](reference/record_episodes.md), [reference/viewer_tools.md](reference/viewer_tools.md).

---

## I want to inspect a dataset

```bash
# Check structure and metadata
act-lerobot  # activate lerobot-env
python scripts/verify_lerobot_dataset.py data/recordings/my_task_v1

# Visualize episodes in Rerun
source venvs/rerun/bin/activate
python scripts/visualize_lerobot_rerun.py data/recordings/my_task_v1
python scripts/visualize_lerobot_rerun.py data/recordings/my_task_v1 -e 0 5 10
python scripts/visualize_lerobot_rerun.py data/recordings/my_task_v1 --skip-images  # fast, state only
```

Details: [reference/visualize_rerun.md](reference/visualize_rerun.md).

---

## I want to clean up a dataset

```bash
act-lerobot  # activate lerobot-env

# Remove frames after success event (trim trailing motion)
python scripts/tools/trim_after_success.py \
    --dataset data/recordings/my_task_v1 \
    --output  data/recordings/my_task_v1_trimmed

# Filter episodes by success rate, length, etc.
python scripts/tools/filter_dataset.py \
    --dataset data/recordings/my_task_v1 \
    --min-reward 1.0

# Recompute stats after manual edits
python scripts/tools/recompute_stats.py data/recordings/my_task_v1
```

Details: [reference/trim_after_success.md](reference/trim_after_success.md), [reference/filter_dataset.md](reference/filter_dataset.md).

---

## I want to train a policy

**ACT:**
```bash
act-lerobot  # activate lerobot-env
python scripts/train/train_act.py \
    --dataset data/recordings/my_task_v1 \
    --config configs/policy/act/baseline.yaml \
    --name act_my_task_v1
```

**Diffusion Policy:**
```bash
python scripts/train/train_diffusion.py \
    --dataset data/recordings/my_task_v1 \
    --name diffusion_my_task_v1
```

**SmolVLA:**
```bash
lerobot-train --policy.type=smolvla \
    --dataset.repo_id=local/my_task_v1 \
    --dataset.root=data/recordings/my_task_v1
```

Details: [reference/train_act.md](reference/train_act.md), [reference/train_diffusion.md](reference/train_diffusion.md).

---

## I want to evaluate a policy in simulation

```bash
act-isaac  # activate Isaac Lab env

# ACT — single env
python scripts/eval/eval_act_policy.py \
    --checkpoint outputs/act_my_task_v1 \
    --env figure_shape_placement_easy \
    --episodes 50 --headless

# Sweep all checkpoints
python scripts/eval/sweep_act_eval_single.py \
    --checkpoint outputs/act_my_task_v1 \
    --all --episodes 50 --max-steps 500
```

Open eval preview in a separate terminal:
```bash
source venvs/viewer/bin/activate
python scripts/tools/eval_viewer.py
```

Details: [reference/eval_act_policy.md](reference/eval_act_policy.md).

---

## I want to improve a policy with RL

For full command sequences, parameters, and what worked/didn't work in practice:
- Offline RL: [workflow/04_offline_rl.md](workflow/04_offline_rl.md)
- Online RL: [workflow/05_online_rl.md](workflow/05_online_rl.md)

**Offline RL (IQL advantage-weighted BC):**

1. Label dataset with rewards → [reference/prepare_reward_dataset.md](reference/prepare_reward_dataset.md)
2. Train IQL critics → [reference/train_iql_critics.md](reference/train_iql_critics.md)
3. Compute advantage weights → [reference/train_iql_critics.md](reference/train_iql_critics.md)
4. Fine-tune VLA → [reference/train_vla_weighted_bc.md](reference/train_vla_weighted_bc.md)

**Online RL (Flow-Noise PPO / Learned Noise Sampler):**

1. Collect IK rollouts for offline buffer → [reference/collect_rollouts.md](reference/collect_rollouts.md)
2. Train Flow-Noise PPO → [reference/train_flow_noise_ppo_grpc.md](reference/train_flow_noise_ppo_grpc.md) (gRPC, lerobot 0.5.1)
   or train Noise Sampler → [reference/train_noise_sampler.md](reference/train_noise_sampler.md)

---

## I want to add a new task

Checklist:

1. Copy `so101_lab/tasks/figure_shape_placement/` → `so101_lab/tasks/<new_task>/`
2. Rename classes, update internal imports
3. Adjust spawn params, thresholds, reward terms in `env_cfg.py`
4. Register in `so101_lab/tasks/__init__.py` (BC + RL registries)
5. Verify: `python tests/sim/test_env_spawn.py --gui --env <new_task>`

Details: [reference/env_design.md](reference/env_design.md). Full adaptation guide (new task, scene, camera, robot): [workflow/06_adapting.md](workflow/06_adapting.md).

---

## I want to deploy on the real robot

```bash
act-lerobot  # activate lerobot-env

# Calibrate leader arm (one-time)
python scripts/calibrate_leader.py

# Deploy policy
lerobot-record \
    --robot.type=so101 \
    --policy.path=outputs/act_my_task_v1/best
```

Details: [reference/teleop_devices.md](reference/teleop_devices.md), [reference/sim2real.md](reference/sim2real.md).
