# Copyright (c) 2025, SO-101 Lab Project
# Adapted from: Isaac Lab Se3Gamepad (https://github.com/isaac-sim/IsaacLab)
# Original license: BSD-3-Clause
# Changes: Adapted for SO-101 robot, custom mapping (shoulder_pan on bumpers, gripper on triggers)

"""Xbox gamepad teleoperation device for SO-101.

WARNING: Uses DifferentialIK which is SIMULATION-ONLY.
For real robot: use direct joint commands or leader device.
"""

import carb
import isaaclab.utils.math as math_utils
import numpy as np
import threading
import time
import torch
import weakref

from .device_base import Device

# Pygame fallback for Linux (carb.input often doesn't detect gamepads)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[SO101Gamepad] WARNING: pygame not available. Install with: pip install pygame")


class SO101Gamepad(Device):
    """Xbox gamepad controller for SO-101 single arm with custom mapping.

    Action space: [dx, dy, dz, droll, dpitch, dyaw, d_shoulder_pan, d_gripper]
    - First 6 DOF: SE(3) delta in gripper frame (transformed to base frame)
    - Last 2 DOF: Direct joint deltas for shoulder_pan and gripper

    Xbox Controller bindings:
        ============================== ========================= =========================
        Description                    Control (+ve)             Control (-ve)
        ============================== ========================= =========================
        Start control                  B button                  -
        Reset environment              X button                  -
        Forward / Backward             Left Stick Up             Left Stick Down
        Up / Down                      Right Stick Up            Right Stick Down
        Yaw Rotation Left / Right      Right Stick Left          Right Stick Right
        Pitch Rotation Down / Up       D-Pad Down                D-Pad Up
        Roll Rotation Left / Right     D-Pad Left                D-Pad Right
        Shoulder Pan Left / Right      LB (bumper)               RB (bumper)
        Gripper Open / Close           LT (trigger, analog)      RT (trigger, analog)
        ============================== ========================= =========================
    """

    def __init__(self, env, sensitivity: float = 1.0):
        """Initialize Xbox gamepad device.

        Args:
            env: Isaac Lab environment containing the robot.
            sensitivity: Multiplier for all control sensitivities.
        """
        # CRITICAL: Disable simulator gamepad camera control BEFORE parent init
        carb_settings_iface = carb.settings.get_settings()
        camera_ctrl_before = carb_settings_iface.get("/persistent/app/omniverse/gamepadCameraControl")
        carb_settings_iface.set_bool("/persistent/app/omniverse/gamepadCameraControl", False)
        camera_ctrl_after = carb_settings_iface.get("/persistent/app/omniverse/gamepadCameraControl")

        print(f"[SO101Gamepad] Camera control: {camera_ctrl_before} -> {camera_ctrl_after}")

        # Initialize parent (sets up keyboard subscription for B/R keys)
        super().__init__(env, "gamepad")

        # Control sensitivities
        self.pos_sensitivity = 0.01 * sensitivity
        self.joint_sensitivity = 0.15 * sensitivity
        self.rot_sensitivity = 0.15 * sensitivity
        self.dead_zone = 0.15  # Xbox One S has significant stick drift

        # Try carb.input first
        self._gamepad = self._appwindow.get_gamepad(0)
        self._use_pygame = False

        if self._gamepad:
            gamepad_name = self._input.get_gamepad_name(self._gamepad)
            print(f"[SO101Gamepad] carb.input detected: {gamepad_name}")

            self._gamepad_sub = self._input.subscribe_to_gamepad_events(
                self._gamepad,
                lambda event, *args, obj=weakref.proxy(self): obj._on_gamepad_event(event, *args),
            )
            print("[SO101Gamepad] Using carb.input API")
        else:
            # Fallback to pygame
            if not PYGAME_AVAILABLE:
                raise RuntimeError(
                    "No gamepad detected via carb.input and pygame not available!\n"
                    "Install pygame: pip install pygame"
                )

            print("[SO101Gamepad] carb.input failed, trying pygame fallback...")
            self._init_pygame_gamepad()
            self._use_pygame = True

        # Create key bindings
        self._create_key_bindings()

        # Command buffers
        # Dual buffer for sticks: [2, 6] = (positive/negative direction, 6 DOF pose)
        self._delta_pose_raw = np.zeros([2, 6])
        # Single buffer for joints: [2] = (shoulder_pan, gripper)
        self._delta_joint_raw = np.zeros(2)

        # Initialize target frame for gripper
        self.asset_name = "robot"
        self.robot_asset = self.env.scene[self.asset_name]
        self.target_frame = "gripper_frame_link"
        body_idxs, _ = self.robot_asset.find_bodies(self.target_frame)
        self.target_frame_idx = body_idxs[0]

    def __del__(self):
        """Release gamepad and keyboard interfaces."""
        # Stop pygame polling thread
        if hasattr(self, "_pygame_thread_running"):
            self._pygame_thread_running = False
            if hasattr(self, "_pygame_thread"):
                self._pygame_thread.join(timeout=1.0)

        # Cleanup pygame
        if hasattr(self, "_use_pygame") and self._use_pygame:
            if hasattr(self, "_pygame_joystick"):
                self._pygame_joystick.quit()
            pygame.quit()

        # Cleanup carb
        if hasattr(self, "_input") and hasattr(self, "_gamepad") and hasattr(self, "_gamepad_sub"):
            self._input.unsubscribe_to_gamepad_events(self._gamepad, self._gamepad_sub)
            self._gamepad_sub = None

        super().__del__()

    def _add_device_control_description(self):
        """Add gamepad control descriptions to display table."""
        self._display_controls_table.add_row(["Left Stick Up/Down", "forward/backward"])
        self._display_controls_table.add_row(["Right Stick Up/Down", "up/down"])
        self._display_controls_table.add_row(["Right Stick Left/Right", "yaw rotation"])
        self._display_controls_table.add_row(["D-Pad Up/Down", "pitch rotation"])
        self._display_controls_table.add_row(["D-Pad Left/Right", "roll rotation"])
        self._display_controls_table.add_row(["LB / RB (bumpers)", "shoulder_pan left/right"])
        self._display_controls_table.add_row(["LT / RT (triggers)", "gripper open/close"])
        self._display_controls_table.add_row(["X button", "reset environment"])

    def get_device_state(self):
        """Get current device state with frame transformation.

        Returns:
            np.ndarray: Action in base frame [8]
        """
        # Resolve dual buffer to signed values
        delta_pose = self._resolve_command_buffer(self._delta_pose_raw)

        # Combine: [pose (6), shoulder_pan, gripper]
        delta_action = np.concatenate([delta_pose, self._delta_joint_raw])

        # Transform from gripper frame to base frame
        return self._convert_delta_from_frame(delta_action)

    def reset(self):
        """Reset command buffers to zero."""
        self._delta_pose_raw.fill(0.0)
        self._delta_joint_raw.fill(0.0)

    def _init_pygame_gamepad(self):
        """Initialize pygame gamepad (fallback for Linux)."""
        pygame.init()
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("No gamepad detected by pygame! Check /dev/input/js*")

        self._pygame_joystick = pygame.joystick.Joystick(0)
        self._pygame_joystick.init()

        print(f"[SO101Gamepad] pygame detected: {self._pygame_joystick.get_name()}")
        print(f"[SO101Gamepad] Axes: {self._pygame_joystick.get_numaxes()}, "
              f"Buttons: {self._pygame_joystick.get_numbuttons()}, "
              f"Hats: {self._pygame_joystick.get_numhats()}")

        # Start polling thread
        self._pygame_thread_running = True
        self._pygame_thread = threading.Thread(target=self._pygame_poll_loop, daemon=True)
        self._pygame_thread.start()

        print("[SO101Gamepad] Using pygame API (polling at 60 Hz)")

    def _pygame_poll_loop(self):
        """Poll pygame events in background thread."""
        while self._pygame_thread_running:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    self._handle_pygame_axis(event.axis, event.value)
                elif event.type == pygame.JOYBUTTONDOWN:
                    self._handle_pygame_button(event.button, 1.0)
                elif event.type == pygame.JOYBUTTONUP:
                    self._handle_pygame_button(event.button, 0.0)
                elif event.type == pygame.JOYHATMOTION:
                    self._handle_pygame_hat(event.value)

            time.sleep(1.0 / 60.0)  # 60 Hz polling

    def _handle_pygame_axis(self, axis, value):
        """Handle pygame axis events (sticks, triggers)."""
        # Apply dead zone
        if abs(value) < self.dead_zone:
            value = 0.0

        # Xbox One S Controller pygame mapping (Linux)
        # Axes: 0=LeftX, 1=LeftY, 2=RightX, 3=RightY, 4=RT, 5=LT
        if axis == 1:  # Left Stick Y (forward/backward)
            if value < 0:  # Up (negative in pygame)
                self._delta_pose_raw[1, 2] = -value * self.pos_sensitivity  # backward
                self._delta_pose_raw[0, 2] = 0
            else:  # Down
                self._delta_pose_raw[0, 2] = value * self.pos_sensitivity  # forward
                self._delta_pose_raw[1, 2] = 0

        elif axis == 3:  # Right Stick Y (up/down)
            if value < 0:  # Up
                self._delta_pose_raw[0, 0] = -value * self.pos_sensitivity
                self._delta_pose_raw[1, 0] = 0
            else:  # Down
                self._delta_pose_raw[1, 0] = value * self.pos_sensitivity
                self._delta_pose_raw[0, 0] = 0

        elif axis == 2:  # Right Stick X (yaw)
            if value < 0:  # Left
                self._delta_pose_raw[0, 5] = -value * self.rot_sensitivity
                self._delta_pose_raw[1, 5] = 0
            else:  # Right
                self._delta_pose_raw[1, 5] = value * self.rot_sensitivity
                self._delta_pose_raw[0, 5] = 0

        elif axis == 5:  # LT (gripper open)
            # Triggers in pygame: -1.0 (not pressed) to 1.0 (fully pressed)
            normalized = (value + 1.0) / 2.0  # Convert to 0.0-1.0
            self._delta_joint_raw[1] = normalized * self.joint_sensitivity

        elif axis == 4:  # RT (gripper close)
            normalized = (value + 1.0) / 2.0
            self._delta_joint_raw[1] = -normalized * self.joint_sensitivity

    def _handle_pygame_button(self, button, value):
        """Handle pygame button events."""
        # Xbox One S Controller pygame mapping (Linux)
        # Buttons: 0=A, 1=B, 2=?, 3=X, 4=Y, 5=?, 6=LB, 7=RB, ...

        if button == 1:  # B button - start control
            if value > 0.5:
                if not self._started:
                    print("[INFO] Gamepad control started!")
                self._started = True
                self._reset_state = False

        elif button == 3:  # X button - reset
            if value > 0.5:
                print("[INFO] Gamepad reset requested (X button)")
                # Stop control temporarily for reset
                self._started = False
                self._reset_state = True
                # Clear buffers immediately
                self._delta_pose_raw.fill(0.0)
                self._delta_joint_raw.fill(0.0)
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()

        elif button == 6:  # LB - shoulder pan left
            self._delta_joint_raw[0] = -self.joint_sensitivity if value > 0.5 else 0.0

        elif button == 7:  # RB - shoulder pan right
            self._delta_joint_raw[0] = self.joint_sensitivity if value > 0.5 else 0.0

    def _handle_pygame_hat(self, value):
        """Handle pygame D-Pad (hat) events."""
        # Hat value: (x, y) where x,y in {-1, 0, 1}
        x, y = value

        # D-Pad Up/Down → pitch
        if y == 1:  # Up
            self._delta_pose_raw[1, 4] = self.rot_sensitivity * 0.8
            self._delta_pose_raw[0, 4] = 0
        elif y == -1:  # Down
            self._delta_pose_raw[0, 4] = self.rot_sensitivity * 0.8
            self._delta_pose_raw[1, 4] = 0
        else:
            self._delta_pose_raw[:, 4] = 0

        # D-Pad Left/Right → roll
        if x == -1:  # Left
            self._delta_pose_raw[0, 3] = self.rot_sensitivity * 0.8
            self._delta_pose_raw[1, 3] = 0
        elif x == 1:  # Right
            self._delta_pose_raw[1, 3] = self.rot_sensitivity * 0.8
            self._delta_pose_raw[0, 3] = 0
        else:
            self._delta_pose_raw[:, 3] = 0

    def _on_gamepad_event(self, event, *args, **kwargs):
        """Handle gamepad events (sticks, buttons, triggers)."""
        # Apply dead zone filter
        val = event.value if abs(event.value) >= self.dead_zone else 0.0

        # Sticks (analog) → dual buffer for pose
        if event.input in self._INPUT_STICK_VALUE_MAPPING:
            direction, axis, sensitivity = self._INPUT_STICK_VALUE_MAPPING[event.input]
            self._delta_pose_raw[direction, axis] = sensitivity * val

        # D-Pad (digital buttons) → dual buffer for rotation
        if event.input in self._INPUT_DPAD_VALUE_MAPPING:
            direction, axis, sensitivity = self._INPUT_DPAD_VALUE_MAPPING[event.input]
            if val > 0.5:
                self._delta_pose_raw[direction, axis] = sensitivity
                self._delta_pose_raw[1 - direction, axis] = 0  # Clear opposite direction
            else:
                self._delta_pose_raw[:, axis] = 0

        # Bumpers (binary) → shoulder_pan
        if event.input == carb.input.GamepadInput.LEFT_SHOULDER:
            self._delta_joint_raw[0] = -self.joint_sensitivity if val > 0.5 else 0.0
        if event.input == carb.input.GamepadInput.RIGHT_SHOULDER:
            self._delta_joint_raw[0] = self.joint_sensitivity if val > 0.5 else 0.0

        # Triggers (analog) → gripper
        if event.input == carb.input.GamepadInput.LEFT_TRIGGER:
            self._delta_joint_raw[1] = val * self.joint_sensitivity  # Open
        if event.input == carb.input.GamepadInput.RIGHT_TRIGGER:
            self._delta_joint_raw[1] = -val * self.joint_sensitivity  # Close

        # B button → start control
        if event.input == carb.input.GamepadInput.B and val > 0.5:
            if not self._started:
                print("[INFO] Gamepad control started!")
            self._started = True
            self._reset_state = False

        # X button → reset environment
        if event.input == carb.input.GamepadInput.X and val > 0.5:
            print("[INFO] Gamepad reset requested (X button)")
            # Stop control temporarily for reset
            self._started = False
            self._reset_state = True
            # Clear buffers immediately
            self._delta_pose_raw.fill(0.0)
            self._delta_joint_raw.fill(0.0)
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()

        # Additional callbacks
        if event.input in self._additional_callbacks:
            self._additional_callbacks[event.input]()

        return True

    def _create_key_bindings(self):
        """Create gamepad input mappings.

        Mapping format: (direction, axis, sensitivity)
        - direction: 0 = positive, 1 = negative
        - axis: 0=x, 1=y, 2=z, 3=roll, 4=pitch, 5=yaw
        """
        # Stick mappings (analog)
        self._INPUT_STICK_VALUE_MAPPING = {
            # Left Stick Y → dz (forward/backward in gripper frame)
            carb.input.GamepadInput.LEFT_STICK_UP: (1, 2, self.pos_sensitivity),     # backward (negative z)
            carb.input.GamepadInput.LEFT_STICK_DOWN: (0, 2, self.pos_sensitivity),   # forward (positive z)

            # Right Stick Y → dx (up/down in gripper frame)
            carb.input.GamepadInput.RIGHT_STICK_UP: (0, 0, self.pos_sensitivity),    # up (positive x)
            carb.input.GamepadInput.RIGHT_STICK_DOWN: (1, 0, self.pos_sensitivity),  # down (negative x)

            # Right Stick X → dyaw (yaw rotation)
            carb.input.GamepadInput.RIGHT_STICK_LEFT: (0, 5, self.rot_sensitivity),  # yaw left (positive)
            carb.input.GamepadInput.RIGHT_STICK_RIGHT: (1, 5, self.rot_sensitivity), # yaw right (negative)
        }

        # D-Pad mappings (digital, with 0.8 multiplier for rotation)
        self._INPUT_DPAD_VALUE_MAPPING = {
            # D-Pad Up/Down → dpitch
            carb.input.GamepadInput.DPAD_UP: (1, 4, self.rot_sensitivity * 0.8),     # pitch up (negative)
            carb.input.GamepadInput.DPAD_DOWN: (0, 4, self.rot_sensitivity * 0.8),   # pitch down (positive)

            # D-Pad Left/Right → droll
            carb.input.GamepadInput.DPAD_LEFT: (0, 3, self.rot_sensitivity * 0.8),   # roll left (positive)
            carb.input.GamepadInput.DPAD_RIGHT: (1, 3, self.rot_sensitivity * 0.8),  # roll right (negative)
        }

    def _resolve_command_buffer(self, raw_command: np.ndarray) -> np.ndarray:
        """Resolve dual buffer to signed command values.

        Args:
            raw_command: Raw command from gamepad. Shape (2, 6) where:
                - First index: direction (0=positive, 1=negative)
                - Second index: axis (0=x, 1=y, 2=z, 3=roll, 4=pitch, 5=yaw)

        Returns:
            Resolved signed command. Shape (6,)
        """
        # Determine sign: if negative value > positive value, sign is negative
        delta_command_sign = raw_command[1, :] > raw_command[0, :]
        # Get maximum absolute value
        delta_command = raw_command.max(axis=0)
        # Apply sign
        delta_command[delta_command_sign] *= -1

        return delta_command

    def _convert_delta_from_frame(self, delta_action: np.ndarray) -> np.ndarray:
        """Convert delta action from gripper frame to robot base frame.

        Reuses the frame transformation logic from keyboard device.

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

        # Convert rotation vector back to euler
        if is_delta_rot:
            from .keyboard import rotvec_to_euler
            delta_rot_r = rotvec_to_euler(delta_rotvec_r)
        else:
            delta_rot_r = torch.zeros(3, device=self.env.device)

        # Combine: [pos, rot, shoulder_pan, gripper]
        delta_action_r = torch.cat([delta_pos_r.squeeze(0), delta_rot_r, torch_delta_action[6:]], dim=0)

        return delta_action_r.cpu().numpy()
