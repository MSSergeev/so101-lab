"""Test SO-101 robot joint movements using pure Isaac Sim API (no Isaac Lab).

Usage:
    python scripts/isaac_sim_standalone/test_robot.py --mode interactive
    python scripts/isaac_sim_standalone/test_robot.py --mode preset
    python scripts/isaac_sim_standalone/test_robot.py --mode test
    python scripts/isaac_sim_standalone/test_robot.py --mode all
    python scripts/isaac_sim_standalone/test_robot.py --headless  # For CI/testing
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# CRITICAL: Create SimulationApp BEFORE any omni.* imports
from isaacsim import SimulationApp

# Parse args before creating SimulationApp
parser = argparse.ArgumentParser(description="Test SO-101 robot joint movements")
parser.add_argument(
    "--mode",
    choices=["interactive", "preset", "test", "all"],
    default="interactive",
    help="Test mode"
)
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
args = parser.parse_args()

# Check incompatibility
if args.mode == "interactive" and args.headless:
    print("ERROR: Interactive mode requires GUI (remove --headless)")
    sys.exit(1)

# Create SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

# NOW import omni modules
import omni.usd
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage

# Import configs
import configs


def configure_pd_gains(stage, robot_path):
    """Configure PD controller gains for all joints."""
    # Joints are in /World/Robot/joints/{joint_name}
    joints_path = f"{robot_path}/joints"

    for joint_name in configs.JOINT_NAMES:
        joint_prim = stage.GetPrimAtPath(f"{joints_path}/{joint_name}")
        if joint_prim.IsValid():
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.GetStiffnessAttr().Set(configs.PD_STIFFNESS)
            drive.GetDampingAttr().Set(configs.PD_DAMPING)
            print(f"  Configured PD gains for {joint_name}")
        else:
            print(f"  Warning: Joint {joint_name} not found at {joints_path}/{joint_name}")


def setup_scene():
    """Setup world, table, and robot."""
    print("Initializing Isaac Sim...")

    # Get paths
    script_dir = Path(__file__).resolve().parent
    assets_dir = script_dir.parent.parent / "assets"

    # Create World
    print("Creating World...")
    world = World(stage_units_in_meters=1.0, physics_dt=configs.PHYSICS_DT)
    world.scene.add_default_ground_plane()

    # Add scene (table)
    print("Adding table...")
    scene_path = str(assets_dir / "scenes" / "workbench_clean.usd")
    add_reference_to_stage(usd_path=scene_path, prim_path="/World/Scene")

    # Add robot (so101_w_cam.usd - with camera mount)
    print("Adding robot...")
    robot_usd_path = str(assets_dir / "robots" / "so101" / "usd" / "so101_w_cam.usd")
    add_reference_to_stage(usd_path=robot_usd_path, prim_path="/World/Robot")

    # Configure PD controller BEFORE creating articulation
    print("Configuring PD controller...")
    stage = omni.usd.get_context().get_stage()
    configure_pd_gains(stage, "/World/Robot")

    # Create articulation wrapper
    print("Initializing robot articulation...")
    robot = Articulation(
        prim_paths_expr="/World/Robot",
        name="so101"
    )
    world.scene.add(robot)

    # Reset world to initialize physics
    print("Resetting world...")
    world.reset()

    # NOTE: We don't set robot position because:
    # 1. Robot has fix_root_link=True in so101.py, which prevents base movement
    # 2. set_world_poses() causes "Invalid PhysX transform" errors with fixed root
    # 3. Position should be set in the USD file itself or in the scene composition
    # For joint testing, robot position at origin (0,0,0) is acceptable

    print(f"Robot initialized with {robot.num_dof} DOFs")
    print(f"Joint names: {robot.dof_names}")

    # Move to initial position with interpolation
    print("\nMoving to HOME position...")
    current_pos = robot.get_joint_positions().flatten()[:6]
    target_pos = configs.PRESET_POSITIONS["HOME"]

    # Smooth interpolation over 100 steps
    for step in range(100):
        world.step(render=True)
        if world.is_playing():
            # Linear interpolation
            alpha = (step + 1) / 100.0
            interpolated_pos = current_pos + alpha * (target_pos - current_pos)
            robot.set_joint_positions(interpolated_pos)
        elif world.is_stopped():
            world.reset()

    print("Environment ready!\n")
    return world, robot


def run_interactive_mode(robot, world):
    """Manual keyboard control mode."""
    from modes.interactive import InteractiveController

    controller = InteractiveController(robot)

    print("\n" + "=" * 60)
    print("Interactive Joint Control Mode")
    print("=" * 60)
    print("Controls:")
    print("  1-6: Select joint")
    print("  ↑/↓ or W/S: Adjust angle")
    print("  H: Home | P: Pick ready | L: Place ready")
    print("  M: Mid | R: Rest | G: Toggle gripper")
    print("  ESC or Q: Quit")
    print("=" * 60 + "\n")

    step_count = 0

    while simulation_app.is_running() and controller.running:
        world.step(render=True)

        if world.is_playing():
            action = controller.get_action()
            robot.set_joint_positions(action)

            step_count += 1
            if step_count % configs.STATUS_PRINT_INTERVAL == 0:
                controller.print_status()
        elif world.is_stopped():
            world.reset()

    controller.cleanup()
    print("\nExiting interactive mode...")


def run_preset_mode(robot, world):
    """Cycle through all preset positions."""
    print("\n" + "=" * 60)
    print("Preset Positions Demo")
    print("=" * 60)

    current_pos = robot.get_joint_positions().flatten()[:6]

    for name, position in configs.PRESET_POSITIONS.items():
        print(f"\n>>> Moving to {name}...")

        # Smooth interpolation from current to target
        target_pos = position
        steps = configs.STEPS_PER_POSITION

        for step in range(steps):
            world.step(render=True)
            if world.is_playing():
                # Linear interpolation
                alpha = (step + 1) / steps
                interpolated_pos = current_pos + alpha * (target_pos - current_pos)
                robot.set_joint_positions(interpolated_pos)
            elif world.is_stopped():
                world.reset()

        # Update current position for next movement
        current_pos = robot.get_joint_positions().flatten()[:6]
        error = np.linalg.norm(current_pos - position)
        print(f"    Error: {error:.4f} rad")
        time.sleep(1.0)

    print("\n" + "=" * 60)
    print("Preset demo completed!")
    print("=" * 60)


def run_test_mode(robot, world):
    """Run automated tests."""
    from modes.auto_tests import test_individual_joints, test_joint_limits, test_preset_positions

    print("\n" + "=" * 60)
    print("Running Automated Joint Tests")
    print("=" * 60)

    test_individual_joints(robot, world)
    test_joint_limits(robot, world)
    test_preset_positions(robot, world)

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


def main():
    """Main entry point."""
    # Setup scene
    world, robot = setup_scene()

    # Validate presets
    configs.validate_preset_positions()

    # Run selected mode
    try:
        if args.mode == "interactive":
            run_interactive_mode(robot, world)
        elif args.mode == "preset":
            run_preset_mode(robot, world)
        elif args.mode == "test":
            run_test_mode(robot, world)
        elif args.mode == "all":
            print("\n>>> Running all modes sequentially...\n")
            run_preset_mode(robot, world)
            run_test_mode(robot, world)
            if not args.headless:
                print("\nStarting interactive mode (press ESC to exit)...")
                run_interactive_mode(robot, world)
    except KeyboardInterrupt:
        print("\nInterrupted by user")

    # Keep simulation running unless headless
    if not args.headless:
        print("\nSimulation complete. Press Ctrl+C to exit or close window...")
        try:
            while simulation_app.is_running():
                world.step(render=True)
                if world.is_stopped():
                    world.reset()
        except KeyboardInterrupt:
            print("\nExiting...")

    # Cleanup
    print("\nClosing simulation...")
    world.clear()
    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
