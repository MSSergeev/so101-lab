"""Interactive keyboard controller for manual joint control."""

import numpy as np
import carb.input
import omni.appwindow

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import configs


class InteractiveController:
    """Simple keyboard controller for joint-by-joint control."""

    def __init__(self, robot):
        self.robot = robot
        self.current_joint = 0
        # Flatten to 1D array (num_envs, num_dof) -> (num_dof,)
        self.target_pos = robot.get_joint_positions().flatten()[:6].copy()
        self.running = True

        # Keyboard state
        self._key_pressed = {"increase": False, "decrease": False}

        # Setup keyboard input
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event
        )

    def cleanup(self):
        """Explicit cleanup of keyboard subscription."""
        if hasattr(self, '_keyboard_sub') and self._keyboard_sub:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name

            # Joint selection (1-6)
            if key in ["KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5", "KEY_6",
                       "NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4", "NUMPAD_5", "NUMPAD_6"]:
                joint_num = int(key.split("_")[-1])
                self.current_joint = joint_num - 1
                print(f"\n>>> Selected joint {self.current_joint}: {configs.JOINT_NAMES[self.current_joint]}")

            # Adjust angle
            elif key in ["UP", "W"]:
                self._key_pressed["increase"] = True
            elif key in ["DOWN", "S"]:
                self._key_pressed["decrease"] = True

            # Preset positions
            elif key == "H":
                self._go_to_preset("HOME")
            elif key == "P":
                self._go_to_preset("PICK_READY")
            elif key == "L":
                self._go_to_preset("PLACE_READY")
            elif key == "M":
                self._go_to_preset("JOINT_MID")
            elif key == "R":
                self._go_to_preset("REST")

            # Toggle gripper
            elif key == "G":
                self._toggle_gripper()

            # Exit
            elif key in ["ESCAPE", "Q"]:
                print("\n>>> Exiting...")
                self.running = False

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input.name
            if key in ["UP", "W"]:
                self._key_pressed["increase"] = False
            elif key in ["DOWN", "S"]:
                self._key_pressed["decrease"] = False

        return True

    def _adjust_joint(self, delta):
        """Adjust current joint angle by delta."""
        lower = configs.DOF_LIMITS_LOWER[self.current_joint].item()
        upper = configs.DOF_LIMITS_UPPER[self.current_joint].item()

        old_val = self.target_pos[self.current_joint].item()
        new_val = max(lower, min(upper, old_val + delta))
        self.target_pos[self.current_joint] = new_val

        print(f"  {configs.JOINT_NAMES[self.current_joint]}: {old_val:.3f} → {new_val:.3f} rad "
              f"({new_val * 57.2958:.1f}°)")

    def _go_to_preset(self, name):
        """Move to preset position."""
        print(f"\n>>> Moving to {name}...")
        self.target_pos = configs.PRESET_POSITIONS[name].copy()

    def _toggle_gripper(self):
        """Toggle gripper between open and close."""
        current = self.target_pos[5]
        if current > 0.5:  # Currently open
            self.target_pos[5] = -0.165
            print("\n>>> Gripper: CLOSE")
        else:
            self.target_pos[5] = 1.5
            print("\n>>> Gripper: OPEN")

    def get_action(self):
        """Get current target position, applying continuous adjustments."""
        if self._key_pressed["increase"]:
            self._adjust_joint(configs.JOINT_STEP_SIZE)
        elif self._key_pressed["decrease"]:
            self._adjust_joint(-configs.JOINT_STEP_SIZE)

        return self.target_pos

    def print_status(self):
        """Print current joint states."""
        actual = self.robot.get_joint_positions().flatten()[:6]
        print(f"\nJoint {self.current_joint} ({configs.JOINT_NAMES[self.current_joint]}) selected")
        print(f"Target:  {self.target_pos}")
        print(f"Actual:  {actual}")
        error = np.linalg.norm(actual - self.target_pos)
        print(f"Error:   {error:.4f} rad")
