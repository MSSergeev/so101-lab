# Teleoperation Devices

Teleoperation device classes and the `teleop_agent.py` script. **Isaac Lab env.**

For recording episodes during teleoperation, see [workflow/02_recording.md](../workflow/02_recording.md).

---

Runs in **Isaac Lab env** (`act-isaac`) and **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

Three device types are supported:

| Device | Class | Input | Action space | IK |
|--------|-------|-------|--------------|----|
| `keyboard` | `SO101Keyboard` | carb.input | 8 DOF (SE3 + 2 joints) | DifferentialIK |
| `gamepad` | `SO101Gamepad` | carb.input / pygame | 8 DOF (SE3 + 2 joints) | DifferentialIK |
| `so101leader` | `SO101Leader` | FeetechMotorsBus (USB) | 6 DOF (joint positions) | None |

**Keyboard and gamepad are simulation-only** — they use DifferentialIK which requires Isaac Sim.
**SO101Leader** maps hardware joint positions directly; works in sim and for real-robot recording.

---

## Action spaces

### Keyboard / Gamepad (8 DOF)

```
[dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper]
```

- Indices 0–5: SE(3) delta in gripper frame → transformed to base frame → DifferentialIK
- Index 6: shoulder_pan delta (direct)
- Index 7: gripper delta (direct)

DifferentialIK controls 4 joints (shoulder_lift, elbow_flex, wrist_flex, wrist_roll).
shoulder_pan and gripper are controlled directly via `RelativeJointPositionActionCfg`.

### SO101Leader (6 DOF)

```
[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
```

Absolute joint positions in normalized space [-100, 100] (gripper [0, 100]).
Mapped directly to `JointPositionActionCfg` with `use_default_offset=False`.

---

## Keyboard bindings

| Key | Action | Delta |
|-----|--------|-------|
| W / S | Forward / Backward | dz ±0.01 m |
| Q / E | Up / Down | dx ±0.01 m |
| A / D | Shoulder pan L/R | d_shoulder_pan ±0.15 rad |
| J / L | Yaw left / right | dyaw ±0.15 rad |
| K / I | Pitch down / up | dpitch ±0.15 rad |
| U / O | Gripper open / close | d_gripper ±0.15 rad |
| Space | Start control | — |
| X | Reset environment | — |

---

## Xbox gamepad bindings

| Control | Action | Notes |
|---------|--------|-------|
| Left Stick ↑↓ | Forward / backward (dz) | Analog |
| Right Stick ↑↓ | Up / down (dx) | Analog |
| Right Stick ←→ | Yaw | Analog |
| D-Pad ↑↓ | Pitch | Digital, 0.12 sensitivity |
| D-Pad ←→ | Roll | Digital, 0.12 sensitivity |
| LB / RB | Shoulder pan L/R | Binary |
| LT / RT | Gripper open / close | Analog |
| B | Start control | — |
| X | Reset environment | — |

**Linux note:** `carb.input` has limited gamepad support on Linux. A pygame fallback reads `/dev/input/js0` directly. Requires `uv pip install pygame` in Isaac Lab env.

Dead zone: 0.15 (Xbox One S). If stick drift persists, increase `self.dead_zone` in `SO101Gamepad.__init__`.

---

## SO101Leader: calibration

The leader arm requires a calibration file before use. Calibration writes homing offsets and range limits to a JSON file; it does NOT modify motor registers.

```bash
act-lerobot  # activate lerobot-env
python scripts/calibrate_leader.py --port /dev/ttyACM2 --name leader_1
```

Interactive steps:
1. Move arm to **middle** of range → press Enter
2. Move all joints through **full range of motion** → press Enter
3. Saved to `so101_lab/devices/lerobot/.cache/leader_1.json`

> To sync with LeRobot calibration cache after calibrating here:
> ```bash
> cp so101_lab/devices/lerobot/.cache/leader_1.json \
>    ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/leader_arm_1.json
> ```

---

## teleop_agent.py

Teleoperation-only script (no recording). Used for testing env behavior.

```bash
act-isaac  # activate Isaac Lab env

# Keyboard
python scripts/teleop/teleop_agent.py \
    --env figure_shape_placement_easy \
    --teleop-device keyboard

# Gamepad
python scripts/teleop/teleop_agent.py \
    --env figure_shape_placement_easy \
    --teleop-device gamepad

# Leader arm
python scripts/teleop/teleop_agent.py \
    --env figure_shape_placement_easy \
    --teleop-device so101leader \
    --port /dev/ttyACM2 \
    --calibration-file leader_1.json
```

For recording with the leader arm, use `record_episodes.py` instead — see [workflow/02_recording.md](../workflow/02_recording.md).

---

## Implementation notes

### Frame naming

Isaac Lab has two frame naming systems:

| System | Example | Where used |
|--------|---------|------------|
| FrameTransformer name | `"gripper"` | Visualization only (`env_cfg.py`) |
| USD body name | `"gripper_frame_link"` | `find_bodies()`, DifferentialIK |

Always use USD body names for DifferentialIK and `find_bodies()`.

### Dual buffer (gamepad)

Gamepad sticks send two events per axis (e.g., `LEFT_STICK_UP=0.0` and `LEFT_STICK_DOWN=0.8`).
The gamepad device stores values in a `[2, 6]` buffer (positive/negative directions) and resolves
the net command each step.

### Device base class

`Device` (in `device_base.py`) requires `carb.input` — it only works with Isaac Sim running.
Do not inherit from it for headless or lerobot-env use.

`advance()` returns `None` until Space/B is pressed, then returns:
```python
{"reset": bool, "started": bool, "joint_state": dict}
```

### Hardware setup

| Device | Default port |
|--------|-------------|
| Leader arm | `/dev/ttyACM2` |
| Follower arm | `/dev/ttyACM0` |

Serial port permissions:
```bash
sudo usermod -a -G dialout $USER  # then logout/login
```

Motor specs: 6 × STS3215, baudrate 1 Mbps, normalized range [-100, 100] (gripper [0, 100]).
