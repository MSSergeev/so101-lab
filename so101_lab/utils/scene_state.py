"""Scene state extraction for eval registry."""

from __future__ import annotations

import numpy as np


def get_object_state(env, obj_name: str) -> dict | None:
    """Extract object position and rotation from environment."""
    try:
        # Try scene dict first, then private attribute
        obj = None
        if obj_name in env.scene.keys():
            obj = env.scene[obj_name]
        elif hasattr(env, f"_{obj_name}"):
            obj = getattr(env, f"_{obj_name}")

        if obj is None:
            return None

        # Get position and orientation
        pos = obj.data.root_pos_w[0].cpu().numpy().tolist()
        quat = obj.data.root_quat_w[0].cpu().numpy()  # (w, x, y, z)

        # Convert quaternion to euler (simplified - just yaw for now)
        w, x, y, z = quat
        yaw_rad = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        yaw_deg = np.degrees(yaw_rad)

        return {
            "pos": [round(p, 4) for p in pos],
            "rot_deg": [0, 0, round(yaw_deg, 1)]
        }
    except Exception:
        return None


def get_gripper_state(env, joint_pos_override: np.ndarray | None = None) -> dict | None:
    """Extract gripper position and state.

    Args:
        env: Environment instance
        joint_pos_override: If provided, use this instead of reading from robot.data.joint_pos.
                           Useful when robot state may have been reset (e.g., after done=True).
    """
    try:
        # Get end-effector position from ee_frame if available
        ee_pos = None
        if "ee_frame" in env.scene.keys():
            ee_pos = env.scene["ee_frame"].data.target_pos_w[0, 0].cpu().numpy().tolist()

        # Get gripper joint position (last joint)
        if joint_pos_override is not None:
            joint_pos = joint_pos_override
        else:
            robot = env.scene["robot"]
            joint_pos = robot.data.joint_pos[0].cpu().numpy()

        gripper_pos = float(joint_pos[-1])  # Last joint is gripper
        # Gripper: -0.174 rad (-10°) = closed, +1.745 rad (+100°) = open
        gripper_closed = gripper_pos < 0.5

        result = {
            "joint_pos": round(gripper_pos, 4),
            "closed": gripper_closed
        }
        if ee_pos:
            result["pos"] = [round(p, 4) for p in ee_pos]

        return result
    except Exception:
        return None


def extract_scene_state(env, joint_pos_override: np.ndarray | None = None) -> dict:
    """Extract full scene state for eval registry.

    Args:
        env: Environment instance
        joint_pos_override: If provided, use this for gripper state instead of robot.data.joint_pos.
    """
    state = {"objects": {}}

    # Try common object names
    for obj_name in ["platform", "cube", "sorting_platform"]:
        obj_state = get_object_state(env, obj_name)
        if obj_state:
            state["objects"][obj_name] = obj_state

    # Gripper state
    gripper = get_gripper_state(env, joint_pos_override)
    if gripper:
        state["gripper"] = gripper

    return state
