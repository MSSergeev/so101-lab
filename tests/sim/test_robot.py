"""Test robot joint movements in Isaac Sim. Isaac Lab env.

Modes:
    interactive  Manual keyboard control. Keys: 1-6 select joint, W/S move,
                 H=home, P=pick_ready, L=place_ready, G=toggle gripper, ESC=quit.
    preset       Cycle through preset positions automatically (250 steps each).
    test         Automated tests: individual joints, limits, preset positions.
                 Reports pass/fail (error threshold: 0.05 rad).
    all          preset → test → interactive (if not headless).

Usage:
    eval "$(./activate_isaaclab.sh)"
    python tests/sim/test_robot.py --mode interactive
    python tests/sim/test_robot.py --mode preset
    python tests/sim/test_robot.py --mode test --headless
    python tests/sim/test_robot.py --mode all

Joint limits (from URDF):
    shoulder_pan:  [-3.14, 3.14] rad
    shoulder_lift: [-1.75,  0.0] rad
    elbow_flex:    [ 0.0,  1.54] rad
    wrist_pitch:   [-1.66, 1.66] rad
    wrist_roll:    [-3.14, 3.14] rad
    gripper:       [-0.17, 1.57] rad
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


class PresetPositions:
    """Preset joint positions for SO-101 robot."""

    # From so101.py init_state - compact home position
    HOME = torch.tensor([0.021, -1.735, 1.529, 0.394, 0.014, -0.165])

    # Working positions
    PICK_READY = torch.tensor([0.0, -1.2, 1.0, 0.5, 0.0, -0.165])    # Above table
    PLACE_READY = torch.tensor([0.0, -0.8, 0.6, 0.3, 0.0, -0.165])   # For placement

    # Gripper tests
    GRIPPER_OPEN = torch.tensor([0.021, -1.735, 1.529, 0.394, 0.014, 1.5])
    GRIPPER_CLOSE = torch.tensor([0.021, -1.735, 1.529, 0.394, 0.014, -0.165])

    # Mid position - all joints centered
    JOINT_MID = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # REST - compact pose for storage/transport
    REST = torch.tensor([0.0, -0.5, 1.2, 0.8, 0.0, -0.165])

    @classmethod
    def validate_and_clip(cls, env):
        """Validate all preset positions against joint limits and clip if needed."""
        limits = env.scene["robot"].data.soft_joint_pos_limits[0, :6]  # (6, 2)
        lower, upper = limits[:, 0], limits[:, 1]

        presets = {
            "HOME": cls.HOME,
            "PICK_READY": cls.PICK_READY,
            "PLACE_READY": cls.PLACE_READY,
            "GRIPPER_OPEN": cls.GRIPPER_OPEN,
            "GRIPPER_CLOSE": cls.GRIPPER_CLOSE,
            "JOINT_MID": cls.JOINT_MID,
            "REST": cls.REST,
        }

        print("\nValidating preset positions...")
        for name, position in presets.items():
            position_device = position.to(env.device)
            violations = (position_device < lower) | (position_device > upper)
            if violations.any():
                print(f"  ⚠️  {name}: clipping joints {violations.nonzero().squeeze().tolist()}")
                position.clamp_(lower.cpu(), upper.cpu())
            else:
                print(f"  ✓  {name}: OK")


class ManualController:
    """Simple keyboard controller for joint-by-joint control."""

    def __init__(self, env):
        self.env = env
        self.current_joint = 0  # Selected joint index (0-5)
        self.step_size = 0.02   # Radians per keypress (~1°)

        # Current target position (start from current robot position)
        self.target_pos = env.scene["robot"].data.joint_pos[0, :6].clone()

        # Keyboard state tracking for continuous input
        self._key_pressed = {
            "increase": False,  # UP or W held
            "decrease": False,  # DOWN or S held
        }

        # Import carb and omni for keyboard input
        import carb.input
        import omni.appwindow
        import weakref

        # Store carb reference for use in callbacks
        self._carb = carb

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        # Subscribe to keyboard events (use weakref to avoid memory leaks)
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args)
        )

        # Joint names for display
        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper"
        ]

        self.running = True

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        # Get key code and event type
        if event.type == self._carb.input.KeyboardEventType.KEY_PRESS:
            key = event.input.name

            # Joint selection (1-6) - support both regular number keys and numpad
            if key in ["KEY_1", "KEY_2", "KEY_3", "KEY_4", "KEY_5", "KEY_6",
                       "NUMPAD_1", "NUMPAD_2", "NUMPAD_3", "NUMPAD_4", "NUMPAD_5", "NUMPAD_6"]:
                # Extract joint number from key name
                if key.startswith("NUMPAD_"):
                    joint_num = int(key.split("_")[1])
                elif key.startswith("KEY_"):
                    joint_num = int(key.split("_")[1])
                else:
                    joint_num = int(key)
                self.current_joint = joint_num - 1
                print(f"\n>>> Selected joint {self.current_joint}: {self.joint_names[self.current_joint]}")

            # Increase angle
            elif key in ["UP", "W"]:
                self._key_pressed["increase"] = True

            # Decrease angle
            elif key in ["DOWN", "S"]:
                self._key_pressed["decrease"] = True

            # Preset positions
            elif key == "H":
                self._go_to_preset("HOME", PresetPositions.HOME)
            elif key == "P":
                self._go_to_preset("PICK_READY", PresetPositions.PICK_READY)
            elif key == "L":
                self._go_to_preset("PLACE_READY", PresetPositions.PLACE_READY)
            elif key == "M":
                self._go_to_preset("JOINT_MID", PresetPositions.JOINT_MID)
            elif key == "R":
                self._go_to_preset("REST", PresetPositions.REST)

            # Toggle gripper
            elif key == "G":
                self._toggle_gripper()

            # Exit
            elif key in ["ESCAPE", "Q"]:
                print("\n>>> Exiting...")
                self.running = False

        # Handle key release to stop continuous movement
        elif event.type == self._carb.input.KeyboardEventType.KEY_RELEASE:
            key = event.input.name

            if key in ["UP", "W"]:
                self._key_pressed["increase"] = False
            elif key in ["DOWN", "S"]:
                self._key_pressed["decrease"] = False

        return True

    def _adjust_joint(self, delta):
        """Adjust current joint angle by delta."""
        limits = self.env.scene["robot"].data.soft_joint_pos_limits[0, self.current_joint]
        lower, upper = limits[0].item(), limits[1].item()

        old_val = self.target_pos[self.current_joint].item()
        new_val = max(lower, min(upper, old_val + delta))
        self.target_pos[self.current_joint] = new_val

        print(f"  {self.joint_names[self.current_joint]}: {old_val:.3f} → {new_val:.3f} rad "
              f"({new_val * 57.2958:.1f}°)")

    def _go_to_preset(self, name, position):
        """Move to preset position."""
        print(f"\n>>> Moving to {name}...")
        self.target_pos = position.to(self.env.device).clone()

    def _toggle_gripper(self):
        """Toggle gripper between open and close."""
        current = self.target_pos[5].item()
        if current > 0.5:  # Currently open
            self.target_pos[5] = -0.165  # Close
            print("\n>>> Gripper: CLOSE")
        else:  # Currently closed
            self.target_pos[5] = 1.5  # Open
            print("\n>>> Gripper: OPEN")

    def get_action(self) -> torch.Tensor:
        """Get current target position as action, applying continuous adjustments."""
        # Apply continuous adjustments if keys are held
        if self._key_pressed["increase"]:
            self._adjust_joint(self.step_size)
        elif self._key_pressed["decrease"]:
            self._adjust_joint(-self.step_size)

        return self.target_pos.unsqueeze(0)  # (1, 6)

    def print_status(self):
        """Print current joint states."""
        actual = self.env.scene["robot"].data.joint_pos[0, :6]
        print(f"\nJoint {self.current_joint} ({self.joint_names[self.current_joint]}) selected")
        print(f"Target:  {self.target_pos.cpu().numpy()}")
        print(f"Actual:  {actual.cpu().numpy()}")
        error = torch.norm(actual - self.target_pos)
        print(f"Error:   {error:.4f} rad")


class AutoTests:
    """Automated joint testing routines."""

    @staticmethod
    def test_individual_joints(env, steps_per_joint=250):
        """Move each joint individually to mid position."""
        print("\n" + "=" * 60)
        print("Test 1: Individual Joint Movement")
        print("=" * 60)

        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                       "wrist_flex", "wrist_roll", "gripper"]

        for joint_idx in range(6):
            # Get joint limits
            limits = env.scene["robot"].data.soft_joint_pos_limits[0, joint_idx]
            lower, upper = limits[0].item(), limits[1].item()
            mid = (lower + upper) / 2

            print(f"\n>>> Joint {joint_idx} ({joint_names[joint_idx]})")
            print(f"    Limits: [{lower:.3f}, {upper:.3f}]")
            print(f"    Target: {mid:.3f} rad ({mid * 57.2958:.1f}°)")

            # Create action with only this joint moving
            target = env.scene["robot"].data.joint_pos[0, :6].clone()
            target[joint_idx] = mid

            # Execute movement
            for _ in range(steps_per_joint):
                env.step(target.unsqueeze(0))

            # Check result
            actual = env.scene["robot"].data.joint_pos[0, joint_idx].item()
            error = abs(actual - mid)
            status = "✓" if error < 0.05 else "✗"
            print(f"    Actual: {actual:.3f} rad ({actual * 57.2958:.1f}°)")
            print(f"    Error:  {error:.4f} rad {status}")

            time.sleep(0.5)

    @staticmethod
    def test_joint_limits(env, steps=250):
        """Test upper/lower limits with 5% margin."""
        print("\n" + "=" * 60)
        print("Test 2: Joint Limits (with 5% safety margin)")
        print("=" * 60)

        joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                       "wrist_flex", "wrist_roll", "gripper"]

        for joint_idx in range(6):
            limits = env.scene["robot"].data.soft_joint_pos_limits[0, joint_idx]
            lower, upper = limits[0].item(), limits[1].item()
            margin = 0.05 * (upper - lower)

            print(f"\n>>> Joint {joint_idx} ({joint_names[joint_idx]})")

            # Test lower limit
            target_lower = lower + margin
            target = env.scene["robot"].data.joint_pos[0, :6].clone()
            target[joint_idx] = target_lower

            for _ in range(steps):
                env.step(target.unsqueeze(0))

            actual = env.scene["robot"].data.joint_pos[0, joint_idx].item()
            error = abs(actual - target_lower)
            print(f"    Lower: target={target_lower:.3f}, actual={actual:.3f}, error={error:.4f}")

            time.sleep(0.3)

            # Test upper limit
            target_upper = upper - margin
            target[joint_idx] = target_upper

            for _ in range(steps):
                env.step(target.unsqueeze(0))

            actual = env.scene["robot"].data.joint_pos[0, joint_idx].item()
            error = abs(actual - target_upper)
            print(f"    Upper: target={target_upper:.3f}, actual={actual:.3f}, error={error:.4f}")

            time.sleep(0.3)

    @staticmethod
    def test_preset_positions(env, steps=250):
        """Test moving to each preset position."""
        print("\n" + "=" * 60)
        print("Test 3: Preset Positions")
        print("=" * 60)

        presets = {
            "HOME": PresetPositions.HOME,
            "PICK_READY": PresetPositions.PICK_READY,
            "PLACE_READY": PresetPositions.PLACE_READY,
            "GRIPPER_OPEN": PresetPositions.GRIPPER_OPEN,
            "GRIPPER_CLOSE": PresetPositions.GRIPPER_CLOSE,
            "JOINT_MID": PresetPositions.JOINT_MID,
            "REST": PresetPositions.REST,
        }

        for name, position in presets.items():
            print(f"\n>>> {name}")
            position_device = position.to(env.device)

            for _ in range(steps):
                env.step(position_device.unsqueeze(0))

            actual = env.scene["robot"].data.joint_pos[0, :6]
            error = torch.norm(actual - position_device).item()
            status = "✓" if error < 0.05 else "✗"

            print(f"    Target: {position.numpy()}")
            print(f"    Actual: {actual.cpu().numpy()}")
            print(f"    Error:  {error:.4f} rad {status}")

            time.sleep(0.5)


def run_interactive_mode(env, simulation_app):
    """Manual keyboard control mode."""
    controller = ManualController(env)

    print("\n" + "=" * 60)
    print("Interactive Joint Control Mode")
    print("=" * 60)
    print("Controls:")
    print("  1-6: Select joint (shoulder_pan, shoulder_lift, elbow_flex,")
    print("                    wrist_flex, wrist_roll, gripper)")
    print("  ↑/↓ or W/S: Adjust angle (±0.02 rad / ~1°)")
    print("  H: Home position")
    print("  P: Pick ready position")
    print("  L: Place ready position")
    print("  M: Joint mid position")
    print("  R: Rest position")
    print("  G: Toggle gripper")
    print("  ESC or Q: Quit")
    print("=" * 60 + "\n")

    step_count = 0
    status_interval = 100

    while simulation_app.is_running() and controller.running:
        action = controller.get_action()
        env.step(action)

        step_count += 1

        # Print status periodically
        if step_count % status_interval == 0:
            controller.print_status()

    print("\nExiting interactive mode...")


def run_preset_mode(env):
    """Cycle through all preset positions."""
    print("\n" + "=" * 60)
    print("Preset Positions Demo")
    print("=" * 60)

    presets = {
        "HOME": PresetPositions.HOME,
        "PICK_READY": PresetPositions.PICK_READY,
        "PLACE_READY": PresetPositions.PLACE_READY,
        "GRIPPER_OPEN": PresetPositions.GRIPPER_OPEN,
        "GRIPPER_CLOSE": PresetPositions.GRIPPER_CLOSE,
        "JOINT_MID": PresetPositions.JOINT_MID,
        "REST": PresetPositions.REST,
    }

    for name, position in presets.items():
        print(f"\n>>> Moving to {name}...")
        position_device = position.to(env.device)

        for _ in range(250):
            env.step(position_device.unsqueeze(0))

        actual = env.scene["robot"].data.joint_pos[0, :6]
        error = torch.norm(actual - position_device).item()

        print(f"    Target: {position.numpy()}")
        print(f"    Actual: {actual.cpu().numpy()}")
        print(f"    Error:  {error:.4f} rad")

        time.sleep(1.0)

    print("\n" + "=" * 60)
    print("Preset demo completed!")
    print("=" * 60)


def run_test_mode(env):
    """Run all automated tests."""
    print("\n" + "=" * 60)
    print("Running Automated Joint Tests")
    print("=" * 60)

    tests = AutoTests()

    tests.test_individual_joints(env)
    tests.test_joint_limits(env)
    tests.test_preset_positions(env)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test SO-101 robot joint movements")
    parser.add_argument(
        "--mode",
        choices=["interactive", "preset", "test", "all"],
        default="interactive",
        help="Test mode: interactive (manual), preset (demo), test (auto), all (sequential)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no GUI)"
    )
    args = parser.parse_args()

    # Initialize Isaac Sim
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({
        "headless": args.headless,
        "enable_cameras": True  # Required by TemplateEnvCfg cameras
    })
    simulation_app = app_launcher.app

    # Import after Isaac Sim initialization
    from so101_lab.tasks.template.env_cfg import TemplateEnvCfg
    from so101_lab.tasks.template.env import TemplateEnv

    # Create environment
    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"

    print("Creating environment...")
    env = TemplateEnv(cfg)
    env.reset()
    print("Environment ready!\n")

    # Validate preset positions
    PresetPositions.validate_and_clip(env)

    # Run selected mode
    try:
        if args.mode == "interactive":
            run_interactive_mode(env, simulation_app)
        elif args.mode == "preset":
            run_preset_mode(env)
        elif args.mode == "test":
            run_test_mode(env)
        elif args.mode == "all":
            print("\n>>> Running all modes sequentially...\n")
            run_preset_mode(env)
            run_test_mode(env)
            if not args.headless:
                print("\nStarting interactive mode (press ESC to exit)...")
                run_interactive_mode(env, simulation_app)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Cleanup
    print("\nClosing environment...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
