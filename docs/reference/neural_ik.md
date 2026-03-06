# Neural IK

MLP-based inverse kinematics for SO-101, replacing DLS IK in the scripted rollout policy. Isaac Lab venv (generate/train/verify require `pinocchio`).

---

Runs in **Isaac Lab env** (`act-isaac`). See [setup](../workflow/00_setup.md).

## Overview

SO-101 has a 5-DOF arm but IK targets are 6-DOF (position + orientation). DLS IK finds a compromise that causes jitter near singularities and missed waypoints. A trained MLP avoids these issues by learning the IK mapping directly from FK data.

The neural IK is used inside `IKScriptedPolicy` (in `collect_rollouts.py`) when `method: "neural"` is set in the config.

---

## Workflow

```bash
act-isaac  # activate Isaac Lab env

# 1. Generate FK dataset
python scripts/tools/generate_ik_dataset.py \
    --output data/neural_ik_data.npz --samples 50000000

# 2. Train MLP (recommended config)
python scripts/tools/train_neural_ik.py \
    --dataset data/neural_ik_data.npz \
    --output  outputs/neural_ik \
    --hidden-size 512 --epochs 60

# 3. Verify
python scripts/tools/verify_neural_ik.py \
    --checkpoint outputs/neural_ik \
    --noise-std 0.3 --iters 2

# 4. Enable in collect_rollouts.yaml:
#    method: "neural"
#    neural_ik_checkpoint: "outputs/neural_ik"
```

---

## generate_ik_dataset.py

Generates FK samples via `pinocchio`. Reads `assets/robots/so101/urdf/so101_new_calib.urdf`.

```
uniform_sample(joint_limits) → q[5] → pinocchio FK → EE pose (pos + quat)
```

Output: `.npz` with arrays `joints (N,5)`, `ee_pos (N,3)`, `ee_quat (N,4)`.

Quaternions in **wxyz** (IsaacLab convention), canonical sign: `w ≥ 0`.

50M samples takes ~2 minutes.

```bash
python scripts/tools/generate_ik_dataset.py \
    --output data/neural_ik_data.npz \
    --samples 50000000          # 50M recommended
```

---

## train_neural_ik.py

Trains an MLP on the FK dataset.

### Model architecture

```
Input (12): ee_pos(3) + ee_quat(4) + current_joints(5)
  normalize:
    ee_pos  → (x - mean) / std
    ee_quat → L2-normalize, flip sign if w < 0
    joints  → (q - mid) / range → [-1, 1]

Hidden: num_layers × hidden_size (SiLU activations)
Output: Linear(hidden_size → 5) → denormalize → clamp(joint_limits)
```

`current_joints` in the input allows the network to pick the IK solution closest to the current configuration. In the training dataset, `current_joints = target + noise(std=0.3 rad)` — this teaches the network to return solutions near the current pose, producing smooth step-to-step transitions.

Model is saved via `torch.save(model)` — inference does not require knowing the architecture separately. `NeuralIKNet` class: `so101_lab/policies/rl/neural_ik.py`.

Training: Adam + CosineAnnealingLR, batch=4096.

```bash
python scripts/tools/train_neural_ik.py \
    --dataset     data/neural_ik_data.npz \
    --output      outputs/neural_ik \
    --hidden-size 512        # default 256
    --num-layers  3          # default 3
    --epochs      60         # default 30
```

### Results

Verification with `iters=2`, `noise_std=0.3`, `n=200k`:

| Architecture | MAE (°) | EE mean | EE 90th | EE max |
|-------------|---------|---------|---------|--------|
| 3×256, 10M, 30 epochs | 1.09° | 4 mm | 7 mm | 27 mm |
| 3×256, 50M, 30 epochs | 0.86° | 3.5 mm | 6.2 mm | 34.7 mm |
| 4×256, 50M, 30 epochs | 0.76° | 2.9 mm | 5.0 mm | 25.0 mm |
| 3×512, 30M, 60 epochs | 0.80° | 2.0 mm | 3.4 mm | 31.7 mm |
| **3×512, 50M, 60 epochs** | **0.77°** | **1.8 mm** | **3.2 mm** | **24.6 mm** |

Position threshold in the state machine = 35 mm. All configs reach 100% < 35 mm with `iters=2`. Without iterative refinement, max error reaches 54 mm.

---

## verify_neural_ik.py

Loads the trained model, predicts joint angles for validation samples, runs pinocchio FK on predictions and measures EE position error in mm.

```bash
python scripts/tools/verify_neural_ik.py \
    --checkpoint outputs/neural_ik \
    --dataset    data/neural_ik_data.npz \
    --noise-std  0.3 \
    --iters      2 \
    --n          5000          # validation samples
```

Reports MAE (°), EE mean/90th/max (mm).

---

## Iterative Refinement

The model is run N times: previous prediction becomes `current_joints` for the next pass. Effective when input has noise (real inference: arm in motion).

- `iters=1`: single pass
- `iters=2`: recommended — 50–60% reduction in max error
- `iters=3+`: no further gain

Set in `configs/collect_rollouts.yaml`:
```yaml
neural_ik_iters: 2
```

---

## Waypoint Interpolator

Without interpolation, phase transitions (e.g. GRASP → LIFT) send the arm 18+ cm in one step. Neural IK receives a target far outside its training distribution (`noise_std=0.3 rad`) and produces unstable results.

`WaypointInterpolator` (in `ik_policy.py`) moves the commanded EE position toward the final waypoint at `max_ee_speed` with trapezoidal velocity profile. IK always receives a nearby target — within its training distribution.

```
on phase change: _cmd_pos = current_ee_pos_w
each step:       _cmd_pos += direction × velocity × dt  (trapezoid)
IK input:        _cmd_pos  (near current pose)
transition check: against final waypoint (not interpolated)
```

**DESCEND phase** uses separate XY/Z thresholds:
- XY: `position_threshold = 0.035 m`
- Z: `transition_z_threshold = 0.015 m` (stricter — correct height required before grasping)

**Hold phases (GRASP, RELEASE):** joints are fixed at current positions — IK is not called. Without this, `cube_approach_quat` is recomputed each step from the fluctuating `ee_quat_w` (contact with cube) causing wrist rotation.

Config:
```yaml
max_ee_speed: 0.20           # m/s
ee_accel: 0.70               # m/s²
transition_z_threshold: 0.015
```

**Stale EE data after reset:** `ee_frame.data` goes stale after `env._reset_idx()` — PhysX recalculates FK only on the next `sim.step()`. Fix: `_initial_ee_pos_w` is cached on the first `compute()` call (after `env.reset()` when data is fresh) and used for all subsequent resets.

---

## Config (`configs/collect_rollouts.yaml`)

```yaml
ik:
  method: "neural"                          # "dls" or "neural"
  neural_ik_checkpoint: "outputs/neural_ik"
  neural_ik_iters: 2
  max_ee_speed: 0.20
  ee_accel: 0.70
  # ... other IK params (see collect_rollouts.md)
```

`method: "dls"` falls back to Jacobian DLS IK — the interpolator works in both modes.

---

## Files

```
scripts/tools/
├── generate_ik_dataset.py    # FK dataset via pinocchio
├── train_neural_ik.py        # MLP training
└── verify_neural_ik.py       # Accuracy verification (pinocchio FK)

so101_lab/policies/rl/
├── neural_ik.py              # NeuralIKNet class + inference wrapper
└── ik_policy.py              # IKScriptedPolicy + WaypointInterpolator

configs/
└── collect_rollouts.yaml     # IK method, neural_ik_checkpoint, interpolator params
```

---

## Notes

- Singularities: near fully-extended arm MAE is higher (wrist_flex contributes the most error). `iters=2` reduces max error significantly.
- Pinocchio vs USD: dataset is generated from URDF; simulator uses USD. Small discrepancies are possible but not critical in practice.
- `generate_ik_dataset.py` and `verify_neural_ik.py` require `pinocchio` — install in Isaac Lab env: `uv pip install pin`.
