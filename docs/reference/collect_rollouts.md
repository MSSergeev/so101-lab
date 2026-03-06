# collect_rollouts.py

Collect scripted IK rollouts with domain randomization for dataset generation. Isaac Lab venv.

Output: LeRobot v3 dataset with `next.reward` + `sim_rewards.pt` side-file.

---

Runs in **Isaac Lab env** (`act-isaac`). See [setup](../workflow/00_setup.md).

## Overview

Teleoperated demos (200 episodes) are often insufficient for IQL pretraining and BC augmentation. The IK scripted policy generates 300+ episodes automatically with DR on every reset — randomized cube/platform positions, lighting, physics, camera noise — producing data for use as an offline buffer in IQL and SAC.

Uses `ManagerBasedRLEnv` directly (not `IsaacLabGymEnv`) to support `num_envs > 1`.

---

## CLI

```bash
act-isaac  # activate Isaac Lab env

# Smoke test
python scripts/eval/collect_rollouts.py \
    --policy ik --episodes 5 \
    --output-dataset /tmp/test_rollouts \
    --headless --num_envs 1 --no-domain-rand

# Full collection (4 parallel envs, DR enabled)
python scripts/eval/collect_rollouts.py \
    --policy ik --episodes 300 \
    --output-dataset data/recordings/ik_dr_v1 \
    --headless --num_envs 4

# Collect only successful episodes
python scripts/eval/collect_rollouts.py \
    --policy ik --episodes 300 \
    --output-dataset data/recordings/ik_success_v1 \
    --headless --num_envs 4 --success-only
```

### Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--policy` | `ik` | Policy type (only `ik` supported) |
| `--env` | `figure_shape_placement` | Task name |
| `--episodes` | 300 | Target number of saved episodes |
| `--output-dataset` | required | Output LeRobot dataset path |
| `--episode-length` | 30.0 | Max episode length (s) |
| `--physics-hz` | 120 | Physics simulation rate |
| `--policy-hz` | 30 | Policy execution rate (decimation = physics/policy) |
| `--reward-mode` | `sim+success` | `success` / `sim` / `sim+success` |
| `--success-bonus` | 10.0 | Added to reward on successful episodes |
| `--reward-weights` | `""` | Override sim reward term weights, e.g. `"drop_penalty=-10"` |
| `--num_envs` | 1 | Parallel environments |
| `--phase-timeout` | from yaml (80) | Max steps per IK phase (0 = distance-based only) |
| `--success-only` | false | Save only successful episodes; keeps running until N collected |
| `--config` | `configs/collect_rollouts.yaml` | IK parameters config |
| `--realtime` | false | Rate-limit simulation to `physics_hz` |
| `--gui` | false | Run with GUI |
| `--crf` | 23 | Video encoding quality |
| `--gop` | `auto` | Video GOP size |
| `--no-domain-rand` / `--no-randomize-light` / `--no-randomize-physics` / `--no-camera-noise` / `--no-distractors` | false | Disable DR components |

---

## IK Policy State Machine

```
NEUTRAL → APPROACH → DESCEND → GRASP → LIFT → TRANSIT → DESCEND_SLOT → RELEASE → DONE
```

| Phase | Action | Transition condition |
|-------|--------|---------------------|
| NEUTRAL | EE to safe position in robot base frame | dist_3d < threshold OR timeout |
| APPROACH | EE directly above grasp point (same XY as DESCEND) | dist_xy < 10mm OR timeout |
| DESCEND | EE descends to cube (grasp_height along cube normal) | dist_xy < threshold AND dist_z < z_threshold |
| GRASP | Close gripper, hold `grasp_hold_steps` | steps ≥ grasp_hold_steps |
| LIFT | EE returns to NEUTRAL position | dist_3d < threshold OR timeout |
| TRANSIT | EE moves to slot (slot_approach_height above) | dist_3d < threshold OR timeout |
| DESCEND_SLOT | EE descends into slot | dist_3d < threshold OR timeout |
| RELEASE | Open gripper, hold `grasp_hold_steps` | steps ≥ grasp_hold_steps |
| DONE | Hold current EE position | — |

**Key design decisions:**
- APPROACH → DESCEND uses XY-only threshold — ensures alignment before descent (DESCEND is purely vertical)
- DESCEND → GRASP checks XY and Z separately — prevents closing gripper above the cube
- NEUTRAL is always reachable (fixed position in robot base frame) — eliminates diagonal approach failures
- LIFT targets the same NEUTRAL point — avoids IK singularities when lifting directly above the cube
- TRANSIT/DESCEND_SLOT/RELEASE compensate for grasp imprecision: `target_xy = slot_xy - (cube_pos - ee_pos)` captured at GRASP→LIFT

### IK Controller

`DifferentialIKController(command_type="pose", ik_method="dls")` from `isaaclab.controllers`. Controls 5 arm joints; gripper is controlled separately.

**Critical: Jacobian frame correction.** `get_jacobians()` returns Jacobian in world frame, but target/EE are in robot base frame. SO-101 is rotated −90° around Z in the scene — without correction, IK drives the arm away from the target. Fix:

```python
base_rot_mat = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))  # R^T
jac_b[:, :3, :] = torch.bmm(base_rot_mat, jac_w[:, :3, :])
jac_b[:, 3:, :] = torch.bmm(base_rot_mat, jac_w[:, 3:, :])
```

### Orientation Control

- APPROACH/DESCEND/GRASP: `quat_mul(cube_quat, R_y(π))` — perpendicular approach to cube top face
- 4 yaw candidates (0°/90°/180°/270°) — cube is symmetric; selects closest to current EE orientation
- LIFT/TRANSIT/DESCEND_SLOT/RELEASE: holds current EE orientation

### Trajectory Noise (for diversity)

At each `reset()`, two per-env Gaussian noise buffers are sampled:
- `grasp_noise_xy_std` — added to APPROACH/DESCEND/GRASP XY targets
- `slot_noise_xy_std` — added to TRANSIT/DESCEND_SLOT XY targets (set to 0 for tight-fit platforms)

Noise is constant within an episode, creating varied trajectories across episodes.

---

## IK Config (`configs/collect_rollouts.yaml`)

```yaml
ik:
  neutral_pos: [x, y, z]       # safe position in robot base frame; z = height above robot root Z
  approach_height: 0.07         # above cube center (APPROACH phase)
  grasp_height: 0.005           # grasp level along cube normal
  grasp_side_offset: 0.030      # offset along gripper local +X (compensates finger asymmetry)
  slot_approach_height: 0.12    # above slot (TRANSIT)
  slot_descent_height: 0.008    # depth into slot
  position_threshold: 0.035     # 3D threshold for NEUTRAL/LIFT/TRANSIT/DESCEND_SLOT
  approach_xy_threshold: 0.010  # XY-only threshold for APPROACH→DESCEND
  transition_z_threshold: 0.007 # Z threshold for DESCEND→GRASP
  phase_timeout_steps: 80       # max steps per phase
  grasp_hold_steps: 25          # steps to hold gripper open/closed
  gripper_open_rad: 0.60
  gripper_closed_rad: -0.174
  grasp_noise_xy_std: 0.002
  slot_noise_xy_std: 0.0

success:
  xy_threshold: 0.030           # cube→slot XY distance (m)
  z_tolerance: 0.003            # Z tolerance (m)
  yaw_threshold_deg: 30.0       # yaw error with 90° symmetry
```

---

## Output Format

LeRobot v3 dataset at `--output-dataset` with `next.reward` column, plus a side-file:

```
<output-dataset>/
├── meta/info.json, stats.json, tasks.parquet, episodes/
├── data/chunk-000/file-000.parquet    # includes next.reward
├── videos/observation.images.*/...
└── sim_rewards.pt                     # per-frame sim reward breakdown
```

### sim_rewards.pt

```python
{
    "episode_index": torch.int64,   # frame → episode mapping
    "drop_penalty":  torch.float32,
    "jerky_motion_penalty": torch.float32,
    # ... all RewardManager terms
}
```

Used as an offline buffer in IQL/SAC training — provides dense sim rewards with episode-level indexing.

### Reward modes

| Mode | `next.reward` |
|------|--------------|
| `success` | 1.0 if success else 0.0 |
| `sim` | Σ weight_i × term_i (from RewardManager) |
| `sim+success` | sim + `success_bonus` on success |

Success: `dist_xy(cube, slot) < xy_threshold` AND `|cube_z − (TABLE_HEIGHT + 0.006)| < z_tolerance`.

Success is checked **before** `env.step()` — avoids stale cube position after auto-reset.

---

## Multi-env Recording Loop

Isaac Lab `ManagerBasedRLEnv.step()` auto-resets envs that reach `terminated | truncated`. For envs where the IK policy finishes (DONE) before timeout, manual reset is triggered:

```python
action = policy.compute()
build_frame(obs_dict, action, env_i)   # record BEFORE step
obs_dict, _, terminated, truncated, _ = env.step(action)

done = policy.is_done() | terminated | truncated
for i in done.nonzero():
    flush_episode(i) → dataset
    policy.reset([i])
    if ik_done[i] and not env_done[i]:
        env._reset_idx([i])
        obs_dict = env.observation_manager.compute(update_history=True)
```

---

## Notes

- Expected success rate: 60–80% depending on cube/platform spawn positions and IK quality
- `--success-only` runs until N successful episodes are collected — actual total episodes may be 1.3–2× higher
- `env._reset_idx()` and `observation_manager.compute()` are Isaac Lab internal APIs — may need adaptation on Isaac Lab upgrades
- `generate_ik_dataset.py` and `train_neural_ik.py` in `scripts/tools/` use the same IK controller to build a neural IK approximation (ee_pose → joint_angles)
