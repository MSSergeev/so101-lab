# sim2real.md

Deploy sim-trained policies (ACT, Diffusion) to the real SO-101 robot. lerobot-env.

---

Runs in **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

Checkpoints produced by `train_act.py` / `train_diffusion.py` are directly compatible with LeRobot API — no conversion needed. Deployment uses `lerobot-record --policy.path=...`.

Key insight: `rad ↔ normalized` conversion only exists in Isaac Lab. On the real robot, motors already operate in normalized space `[-100, 100]` — the same space the policy was trained on.

## Hardware

| Component | Device | ID |
|-----------|--------|----|
| Leader arm | `/dev/ttyACM2` | `5A46081978` |
| Follower arm | `/dev/ttyACM0` | `5A68011143` |
| Top camera | `/dev/video0` | Logitech Brio 90, 640×480 MJPG |
| Wrist camera | `/dev/video2` | OV2710 + 6205 lens, 640×480 MJPG |

## Calibration sync

Calibration writes `homing_offset` directly to motors (STS3215 register 31). Both so101-lab and LeRobot use identical calibration file format — after calibrating in one system, copy to the other:

```bash
# After calibrating leader in so101-lab → sync to LeRobot
cp so101_lab/devices/lerobot/.cache/leader_1.json \
   ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/leader_arm_1.json
```

**Do not calibrate the same motor with both systems without syncing** — offsets overwrite each other and `range_min`/`range_max` become invalid.

LeRobot calibration files location:
```
~/.cache/huggingface/lerobot/calibration/
├── teleoperators/so101_leader/leader_arm_1.json
└── robots/so101_follower/follower_arm_1.json
```

## Teleoperation (leader → follower)

```bash
act-lerobot  # activate lerobot-env

# Basic (no cameras)
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm_1 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM2 \
    --teleop.id=leader_arm_1 \
    --fps=30

# With cameras + visualization
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm_1 \
    --robot.cameras='{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM2 \
    --teleop.id=leader_arm_1 \
    --display_data=true
```

## Policy deployment

```bash
act-lerobot  # activate lerobot-env

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm_1 \
    --robot.cameras='{top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: MJPG}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM2 \
    --teleop.id=leader_arm_1 \
    --policy.path=outputs/act_figure_shape_placement_v4/best \
    --dataset.repo_id=local/eval_sim2real \
    --dataset.single_task="Pick up the cube and place it on the platform" \
    --dataset.push_to_hub=false \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=30 \
    --display_data=true
```

**`dataset.repo_id` must start with `eval_`** when using `--policy.path`.

## Camera utilities

```bash
act-lerobot  # activate lerobot-env

# Live preview (requires opencv-python, not headless)
python scripts/tools/camera_preview.py --cameras 0 2 --names top wrist

# Save snapshots for sim-vs-real comparison
python scripts/tools/camera_snapshot.py --cameras 0 2 --names top wrist --output-dir data/reference_images
```

Snapshots stored in `data/reference_images/` (gitignored).

---

## Sim2real camera gap

Zero-shot transfer failed on first test (arm hovered, repetitive motions, didn't reach the cube). Root cause: camera FOV and angle mismatch between sim and real.

### FOV analysis

| Camera | Sim H-FOV | Real H-FOV | Gap |
|--------|-----------|------------|-----|
| Top (Brio 90) | 67° | ~78° | +11° |
| Wrist (OV2710 + 6205) | 53.5° | ~78° | **+24°** |

Top camera angle also differs: sim looks from side-above, real camera points nearly straight down.

### FOV formula

```
H-FOV = 2 × atan(horizontal_aperture / (2 × focal_length))
```

Both `horizontal_aperture` and `focal_length` are set per-camera in `env_cfg.py`.

To match real cameras in sim:

```python
# Top camera — target H-FOV ~78°:
# Current: focal_length=28.7, horizontal_aperture=38.11 → 67°
# Fix:     focal_length=23.5, horizontal_aperture=38.11 → 78°

# Wrist camera — target H-FOV ~78°:
# Current: focal_length=36.5, horizontal_aperture=36.83 → 53.5°
# Fix:     focal_length=22.7, horizontal_aperture=36.83 → 78°
```

### OV2710 lens reference

| Lens | Diagonal FOV | H-FOV (4:3) |
|------|-------------|-------------|
| 6205 no-distortion | ~90° | ~78° |
| 3.9mm no-distortion | ~70° | ~60° |
| 4mm | ~85° | ~73° |
| 6mm | ~55° | ~47° |

### Joint distribution gap (observed)

| Joint | Sim mean | Real mean | Diff |
|-------|----------|-----------|------|
| shoulder_pan | 17.9 | 32.8 | +14.9 |
| shoulder_lift | -0.8 | 4.3 | +5.1 |
| elbow_flex | 2.0 | -17.5 | -19.5 |
| wrist_flex | 70.5 | 95.8 | **+25.3** |
| wrist_roll | 60.1 | 73.6 | +13.5 |
| gripper | 18.8 | 10.3 | -8.5 |

---

## Replay dataset

After fixing camera FOV/angle, re-record with new cameras without redoing teleoperation — see `reference/replay_dataset.md`.
