# Copyright (c) 2025, SO-101 Lab Project
# Adapted from: leisaac (https://github.com/isaac-sim/IsaacLab)
# Original license: BSD-3-Clause
# Changes: Simplified for keyboard/gamepad/SO101Leader, updated SO101Leader conversion logic

"""Action processing utilities for teleoperation devices."""

import torch
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg, RelativeJointPositionActionCfg, JointPositionActionCfg

from so101_lab.assets.robots import SO101_USD_JOINT_LIMITS, SO101_LEADER_MOTOR_LIMITS


def init_action_cfg(action_cfg, device: str):
    """Initialize ActionsCfg for specified device type.

    Args:
        action_cfg: TemplateActionsCfg instance to configure
        device: Device type ("keyboard", "gamepad", "so101leader")
    """
    if device in ["keyboard", "gamepad"]:
        # DifferentialIK for 4 arm joints (shoulder_lift through wrist_roll)
        action_cfg.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper_frame_link",  # USD body name, NOT FrameTransformer name
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",  # Damped Least Squares
            ),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        )

        # Direct control for shoulder_pan and gripper
        action_cfg.gripper_action = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "gripper"],
        )

    elif device == "so101leader":
        # Absolute joint position control for SO101Leader (real robot)
        # SO101Leader returns normalized motor positions that are mapped to absolute joint positions
        action_cfg.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,  # Actions are already in radians
            use_default_offset=False,  # CRITICAL: Disable offset to use absolute positions
        )
        action_cfg.gripper_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,  # Actions are already in radians
        )

    else:
        raise ValueError(f"Unsupported device type: {device}")


# Joint name to motor ID mapping
joint_names_to_motor_ids = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper": 5,
}


def convert_action_from_so101_leader(
    joint_state: dict[str, float],
    motor_limits: dict[str, tuple[float, float]],
    joint_limits: dict[str, tuple[float, float]],
    device: torch.device,
) -> torch.Tensor:
    """Convert SO101Leader normalized motor positions to absolute joint positions in radians.

    Adapted from: leisaac (https://github.com/isaac-sim/IsaacLab)
    Original function: convert_action_from_so101_leader() in leisaac/devices/action_process.py
    Changes: Updated to use SO101 Lab's joint limits from USD file

    This function performs linear interpolation from motor normalized range to joint range:
    1. Motor outputs normalized values (e.g., [-100, 100] for arm, [0, 100] for gripper)
    2. We map these to the physical joint limits in the simulation (e.g., [-110°, +110°])
    3. Convert from degrees to radians

    Example for shoulder_pan:
        Motor: 50.0 in range [-100, 100]  (motor_limits)
        Joint: [-110°, +110°]              (joint_limits from USD)

        motor_degree = 50.0 - (-100.0) = 150.0
        motor_range = 100.0 - (-100.0) = 200.0
        joint_range = 110.0 - (-110.0) = 220.0

        processed_degree = (150.0 / 200.0) * 220.0 + (-110.0) = 55.0°
        processed_radius = 55.0° * π/180 = 0.96 rad

    Args:
        joint_state: Normalized motor positions from SO101Leader
                     e.g., {"shoulder_pan": 50.0, "gripper": 60.0, ...}
        motor_limits: Motor normalized ranges (from SO101_LEADER_MOTOR_LIMITS)
                      e.g., {"shoulder_pan": (-100.0, 100.0), "gripper": (0.0, 100.0)}
        joint_limits: Joint physical limits in degrees (from SO101_USD_JOINT_LIMITS)
                      e.g., {"shoulder_pan": (-110.0, 110.0), "gripper": (-10.0, 100.0)}
        device: Torch device for output tensor

    Returns:
        torch.Tensor: Absolute joint positions in radians [1, 6]
    """
    processed_action = torch.zeros(1, 6, device=device)

    for joint_name, motor_id in joint_names_to_motor_ids.items():
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]

        # Calculate ranges
        motor_range = motor_limit_range[1] - motor_limit_range[0]
        joint_range = joint_limit_range[1] - joint_limit_range[0]

        # Map motor position to joint degree
        motor_degree = joint_state[joint_name] - motor_limit_range[0]
        processed_degree = motor_degree / motor_range * joint_range + joint_limit_range[0]

        # Convert degree to radians
        processed_radius = processed_degree / 180.0 * torch.pi

        processed_action[:, motor_id] = processed_radius

    return processed_action


def preprocess_device_action(action_dict: dict, teleop_device) -> torch.Tensor:
    """Convert device action dict to tensor for environment.

    Args:
        action_dict: Dict from device.advance() with "joint_state" key
        teleop_device: Device instance (used to check device type)

    Returns:
        torch.Tensor: Action tensor [num_envs, action_dim]
    """
    from .lerobot import SO101Leader

    joint_state = action_dict["joint_state"]

    if isinstance(teleop_device, SO101Leader):
        # SO101Leader returns dict with normalized motor positions
        # Convert to absolute joint positions in radians
        action = convert_action_from_so101_leader(
            joint_state=joint_state,
            motor_limits=SO101_LEADER_MOTOR_LIMITS,
            joint_limits=SO101_USD_JOINT_LIMITS,
            device=teleop_device.env.device,
        )
    else:
        # Keyboard/Gamepad: joint_state is already array [8]
        if isinstance(joint_state, torch.Tensor):
            action = joint_state
        else:
            action = torch.tensor(joint_state, dtype=torch.float32)

        # Add batch dimension if needed
        if action.dim() == 1:
            action = action.unsqueeze(0)

    return action
