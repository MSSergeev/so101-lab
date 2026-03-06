# replay_dataset.md

Re-record a dataset with updated camera configuration (new FOV, angle, position) without redoing teleoperation. Isaac Lab env.

---

Runs in **Isaac Lab env** (`act-isaac`). See [setup](../workflow/00_setup.md).

## Overview

When sim camera parameters change, existing datasets become stale — images don't match new config. `replay_dataset.py` replays saved actions in sim with the new camera setup and writes a new dataset. Actions are identical to source; only observations (images) are re-rendered.

## CLI

```bash
act-isaac  # activate Isaac Lab env

# Full replay, headless
python scripts/teleop/replay_dataset.py \
    --source data/recordings/figure_shape_placement_v1 \
    --output data/recordings/figure_shape_placement_v1_newcam \
    --headless

# Test with 3 episodes first
python scripts/teleop/replay_dataset.py \
    --source data/recordings/figure_shape_placement_v1 \
    --output data/recordings/figure_shape_placement_v1_newcam \
    --episodes 0,1,2

# With GUI for visual verification
python scripts/teleop/replay_dataset.py \
    --source data/recordings/figure_shape_placement_v1 \
    --output data/recordings/figure_shape_placement_v1_newcam
```

## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | **required** | Source dataset path |
| `--output` | **required** | New dataset path |
| `--env` | `figure_shape_placement` | Environment type |
| `--physics-hz` | 120 | PhysX frequency |
| `--policy-hz` | 30 | Policy/recording frequency (must match dataset fps) |
| `--render-hz` | 30 | Camera render frequency |
| `--episodes` | all | Comma-separated subset (e.g., `0,1,2`) |
| `--crf` | 23 | Video quality |
| `--gop` | 2 | Keyframe interval |
| `--headless` | false | No Isaac Sim GUI |

## How it works

```
source dataset                   replay_dataset.py               new dataset
─────────────                    ─────────────────               ───────────
episode_metadata.json  ──────►  reset_to_state()
data parquet (actions) ──────►  env.step()  ────────────────►   new images
                                                                 same actions
```

1. Reads `episode_metadata.json` (initial_state per episode), data parquet (actions), `info.json` (fps, task)
2. For each episode: `env.reset_to_state(initial_state)` — exact deterministic reset, no randomization
3. Replays actions: `motor_normalized_to_joint_rad()` → tensor → `env.step()`, recording new observations
4. New dataset written via `RecordingManager` + `LeRobotDatasetWriter` (existing infrastructure)

`reset_to_state(state)` accepts: `platform_x`, `platform_y`, `platform_yaw_deg`, `cube_x`, `cube_y`, `cube_yaw_deg`, `light_intensity`, `light_color_temp` (optional).

Action round-trip: dataset (normalized) → `motor_normalized_to_joint_rad()` → `env.step()` → `joint_rad_to_motor_normalized()` → new dataset. Result is identical to source within float32 precision.

## Verification

```bash
source venvs/rerun/bin/activate
python scripts/visualize_lerobot_rerun.py data/recordings/figure_shape_placement_v1_newcam
```

Compare new vs old images visually. Actions should be identical.

## Notes

- PhysX is not 100% deterministic — trajectories may diverge slightly from source. Effect is minor in practice.
- Only meaningful when changing cameras (FOV, angle, position). If physics changed, use regular teleoperation.
- `--gop 2` default (vs `auto` in record_episodes) — replay datasets are typically used for training.
