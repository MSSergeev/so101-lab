"""Conversion utilities between Isaac Lab and LeRobot formats.

NOTE: Joint limits are duplicated here from so101_lab/assets/robots/so101.py
because that module depends on isaaclab, which is not available in lightweight
environments (e.g., venvs/rerun). Keep in sync with so101.py when updating limits.
"""

import numpy as np

# Joint limits - synced with SO101_USD_JOINT_LIMITS in so101.py (radians)
SO101_JOINT_LIMITS_RAD = {
    "shoulder_pan": (-1.91986, 1.91986),    # (-110.0, 110.0) deg
    "shoulder_lift": (-1.74533, 1.74533),   # (-100.0, 100.0) deg
    "elbow_flex": (-1.68977, 1.68977),      # (-96.8, 96.8) deg
    "wrist_flex": (-1.65806, 1.65806),      # (-95.0, 95.0) deg
    "wrist_roll": (-2.74017, 2.82743),      # (-157.0, 162.0) deg - asymmetric!
    "gripper": (-0.17453, 1.74533),         # (-10.0, 100.0) deg
}

# Motor limits from so101.py - SO101_LEADER_MOTOR_LIMITS
# Normalized range for LeRobot format
SO101_MOTOR_LIMITS_NORMALIZED = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


def joint_rad_to_motor_normalized(joint_pos: np.ndarray) -> np.ndarray:
    """
    Convert joint positions from radians to normalized motor positions.
    Used for sim2real transfer and LeRobot export.

    Args:
        joint_pos: [..., 6] array in radians (any batch shape)

    Returns:
        [..., 6] array in normalized range (gripper [0, 100], others [-100, 100])
    """
    joint_names = list(SO101_JOINT_LIMITS_RAD.keys())
    result = np.zeros_like(joint_pos, dtype=np.float32)

    for i, name in enumerate(joint_names):
        rad_min, rad_max = SO101_JOINT_LIMITS_RAD[name]
        motor_min, motor_max = SO101_MOTOR_LIMITS_NORMALIZED[name]

        # Normalize: radians → [0, 1] → motor range
        normalized = (joint_pos[..., i] - rad_min) / (rad_max - rad_min)
        result[..., i] = normalized * (motor_max - motor_min) + motor_min

    return result


def motor_normalized_to_joint_rad(motor_pos: np.ndarray) -> np.ndarray:
    """
    Convert normalized motor positions to joint radians.
    Used for real2sim transfer.

    Args:
        motor_pos: [..., 6] array in normalized range (any batch shape)

    Returns:
        [..., 6] array in radians
    """
    joint_names = list(SO101_JOINT_LIMITS_RAD.keys())
    result = np.zeros_like(motor_pos, dtype=np.float32)

    for i, name in enumerate(joint_names):
        rad_min, rad_max = SO101_JOINT_LIMITS_RAD[name]
        motor_min, motor_max = SO101_MOTOR_LIMITS_NORMALIZED[name]

        # Normalize: motor → [0, 1] → radians
        normalized = (motor_pos[..., i] - motor_min) / (motor_max - motor_min)
        result[..., i] = normalized * (rad_max - rad_min) + rad_min

    return result
