# Copyright (c) 2025, SO-101 Lab Project
# Adapted from: leisaac (https://github.com/isaac-sim/IsaacLab)
# Original license: BSD-3-Clause
# Changes: Adapted for so101-lab, added rotvec_to_euler helper, updated frame name

"""Keyboard teleoperation device for SO-101.

WARNING: Uses DifferentialIK which is SIMULATION-ONLY.
For real robot: use direct joint commands or leader device.
"""

import carb
import isaaclab.utils.math as math_utils
import numpy as np
import torch

from .device_base import Device


def rotvec_to_euler(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert a batch of rotation vectors (axis-angle) into Euler XYZ deltas.

    Args:
        rotvec: Rotation vector (axis-angle representation) [N, 3]

    Returns:
        Euler angles (roll, pitch, yaw) [3]
    """
    # |rotvec| gives the rotation magnitude for each environment (shape: [N, 1])
    rotvec_norm = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    rotvec_norm_clamped = torch.clamp(rotvec_norm, min=1.0e-8)
    axis = rotvec / rotvec_norm_clamped

    # when norm ~ 0, the axis direction is ill-defined; fall back to +X
    default_axis = torch.tensor([1.0, 0.0, 0.0], device=rotvec.device, dtype=axis.dtype).view(1, 3)
    axis = torch.where(rotvec_norm > 1.0e-8, axis, default_axis.repeat(rotvec.shape[0], 1))

    delta_quat = math_utils.quat_from_angle_axis(rotvec_norm.squeeze(-1), axis)
    delta_roll, delta_pitch, delta_yaw = math_utils.euler_xyz_from_quat(delta_quat)
    return torch.cat([delta_roll, delta_pitch, delta_yaw], dim=0)


class SO101Keyboard(Device):
    """A keyboard controller for sending SE(3) commands as delta poses for SO-101 single arm.

    Action space: [dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper]
    - First 6 DOF: SE(3) delta in gripper frame (transformed to base frame)
    - Last 2 DOF: Direct joint deltas for shoulder_pan and gripper

    Key bindings:
        ============================== ================= =================
        Description                    Key               Key
        ============================== ================= =================
        Forward / Backward              W                 S
        Shoulder Pan Left / Right       A                 D
        Up / Down                       Q                 E
        Rotate (Yaw) Left / Right       J                 L
        Rotate (Pitch) Down / Up        K                 I
        Gripper Open / Close            U                 O
        ============================== ================= =================
    """

    def __init__(self, env, sensitivity: float = 1.0):
        """Initialize keyboard device.

        Args:
            env: Isaac Lab environment containing the robot.
            sensitivity: Multiplier for all control sensitivities.
        """
        super().__init__(env, "keyboard")

        # Control sensitivities
        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity

        # Create key bindings
        self._create_key_bindings()

        # Command buffer: (dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper)
        self._delta_action = np.zeros(8)

        # Initialize target frame for gripper
        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]

        # Use USD body name for find_bodies (NOT FrameTransformer name)
        self.target_frame = "gripper_frame_link"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def _add_device_control_description(self):
        """Add keyboard control descriptions to display table."""
        self._display_controls_table.add_row(["W", "forward"])
        self._display_controls_table.add_row(["S", "backward"])
        self._display_controls_table.add_row(["A", "shoulder_pan left"])
        self._display_controls_table.add_row(["D", "shoulder_pan right"])
        self._display_controls_table.add_row(["Q", "up"])
        self._display_controls_table.add_row(["E", "down"])
        self._display_controls_table.add_row(["J", "yaw left"])
        self._display_controls_table.add_row(["L", "yaw right"])
        self._display_controls_table.add_row(["K", "pitch down"])
        self._display_controls_table.add_row(["I", "pitch up"])
        self._display_controls_table.add_row(["U", "gripper open"])
        self._display_controls_table.add_row(["O", "gripper close"])

    def get_device_state(self):
        """Get current device state with frame transformation.

        Returns:
            np.ndarray: Action in base frame [8]
        """
        return self._convert_delta_from_frame(self._delta_action)

    def reset(self):
        """Reset command buffer to zero."""
        self._delta_action[:] = 0.0

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard press/release events."""
        super()._on_keyboard_event(event, *args, **kwargs)

        # Apply command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_action += self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]

        # Remove command when released
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in self._INPUT_KEY_MAPPING.keys():
                self._delta_action -= self._ACTION_DELTA_MAPPING[self._INPUT_KEY_MAPPING[event.input.name]]

    def _create_key_bindings(self):
        """Create default key bindings for delta actions in gripper frame."""
        self._ACTION_DELTA_MAPPING = {
            "forward": np.asarray([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "backward": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]) * self.joint_sensitivity,
            "right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) * self.joint_sensitivity,
            "up": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "down": np.asarray([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * self.pos_sensitivity,
            "rotate_up": np.asarray([0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_down": np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_left": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "rotate_right": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "gripper_open": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) * self.joint_sensitivity,
            "gripper_close": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]) * self.joint_sensitivity,
        }
        self._INPUT_KEY_MAPPING = {
            "W": "forward",
            "S": "backward",
            "A": "left",
            "D": "right",
            "Q": "up",
            "E": "down",
            "K": "rotate_up",
            "I": "rotate_down",
            "J": "rotate_left",
            "L": "rotate_right",
            "U": "gripper_open",
            "O": "gripper_close",
        }

    def _convert_delta_from_frame(self, delta_action: np.ndarray) -> np.ndarray:
        """Convert delta action from gripper frame to robot base frame.

        Args:
            delta_action: Delta action in gripper frame [8]

        Returns:
            Delta action in robot base frame [8]
        """
        # Early exit if no SE(3) delta
        if np.allclose(delta_action[:3], 0.0) and np.allclose(delta_action[3:6], 0.0):
            return delta_action

        is_delta_rot = not np.allclose(delta_action[3:6], 0.0)

        torch_delta_action = torch.tensor(delta_action, device=self.env.device, dtype=torch.float32)

        # Expand to batch dimension
        delta_pos_f = torch_delta_action[:3].repeat(self.env.num_envs, 1)
        delta_rot_f = torch_delta_action[3:6].repeat(self.env.num_envs, 1)
        delta_quat_f = math_utils.quat_from_euler_xyz(delta_rot_f[:, 0], delta_rot_f[:, 1], delta_rot_f[:, 2])
        delta_rotvec_f = math_utils.axis_angle_from_quat(delta_quat_f)

        # Get gripper frame orientation (use body_quat_w)
        frame_pos, frame_quat = (
            self.robot_asset.data.root_pos_w,
            self.robot_asset.data.body_quat_w[:, self.target_frame_idx],
        )
        root_pos, root_quat = self.robot_asset.data.root_pos_w, self.robot_asset.data.root_quat_w

        # Compute gripper-to-base transformation
        _, frame2root = math_utils.subtract_frame_transforms(root_pos, root_quat, frame_pos, frame_quat)
        frame2root_quat = math_utils.quat_unique(frame2root)

        # Transform deltas from gripper frame to base frame
        delta_pos_r = math_utils.quat_apply(frame2root_quat, delta_pos_f)
        delta_rotvec_r = math_utils.quat_apply(frame2root_quat, delta_rotvec_f)
        delta_rot_r = rotvec_to_euler(delta_rotvec_r) if is_delta_rot else torch.zeros(3, device=self.env.device)

        # Combine: [pos, rot, shoulder_pan, gripper]
        delta_action_r = torch.cat([delta_pos_r.squeeze(0), delta_rot_r, torch_delta_action[6:]], dim=0)

        return delta_action_r.cpu().numpy()
