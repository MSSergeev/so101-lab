"""Configuration constants for SO-101 robot testing."""

import numpy as np

# Joint limits (from so101.usd - verified manually in Isaac Sim GUI)
# USD stores limits in degrees, converted to radians here
DOF_LIMITS_LOWER = np.array([
    -1.91986,  # shoulder_pan: -110° (USD verified)
    -1.74533,  # shoulder_lift: -100° (USD verified)
    -1.68977,  # elbow_flex: -96.8° (USD verified)
    -1.65806,  # wrist_flex: -95° (USD verified)
    -2.74017,  # wrist_roll: -157° (USD verified)
    -0.17453,  # gripper: -10° (USD verified)
])

DOF_LIMITS_UPPER = np.array([
    1.91986,   # shoulder_pan: 110° (USD verified)
    1.74533,   # shoulder_lift: 100° (USD verified)
    1.53588,   # elbow_flex: 88.0° (USD verified)
    1.65806,   # wrist_flex: 95° (USD verified)
    2.82743,   # wrist_roll: 162° (USD verified)
    1.74533,   # gripper: 100° (USD verified)
])

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]

# PD controller gains (from NVIDIA examples)
PD_STIFFNESS = 17.8
PD_DAMPING = 0.60

# Robot positioning (table height = 0.73m)
ROBOT_POSITION = np.array([0.0, 0.0, 0.75])
ROBOT_ORIENTATION = np.array([1, 0, 0, 0])  # quaternion (w, x, y, z)

# Preset joint positions
PRESET_POSITIONS = {
    "HOME": np.array([0.021, -1.735, 1.529, 0.394, 0.014, -0.165]),
    "PICK_READY": np.array([0.0, -1.2, 1.0, 0.5, 0.0, -0.165]),
    "PLACE_READY": np.array([0.0, -0.8, 0.6, 0.3, 0.0, -0.165]),
    "GRIPPER_OPEN": np.array([0.021, -1.735, 1.529, 0.394, 0.014, 1.5]),
    "GRIPPER_CLOSE": np.array([0.021, -1.735, 1.529, 0.394, 0.014, -0.165]),
    "JOINT_MID": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    "REST": np.array([0.0, -0.5, 1.2, 0.8, 0.0, -0.165]),
}

# Simulation parameters
STEPS_PER_POSITION = 250
STATUS_PRINT_INTERVAL = 100
JOINT_STEP_SIZE = 0.02  # Radians per keypress (~1°)

# Physics
PHYSICS_DT = 1.0 / 360.0  # 360 Hz for stability


def validate_preset_positions():
    """Validate and clip preset positions against joint limits."""
    print("\nValidating preset positions...")
    for name, position in PRESET_POSITIONS.items():
        violations = (position < DOF_LIMITS_LOWER) | (position > DOF_LIMITS_UPPER)
        if violations.any():
            print(f"  ⚠️  {name}: clipping joints {np.where(violations)[0].tolist()}")
            position[:] = np.clip(position, DOF_LIMITS_LOWER, DOF_LIMITS_UPPER)
        else:
            print(f"  ✓  {name}: OK")
