"""Automated joint testing routines."""

import time
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import configs


def test_individual_joints(robot, world, steps_per_joint=None):
    """Move each joint individually to mid position."""
    if steps_per_joint is None:
        steps_per_joint = configs.STEPS_PER_POSITION

    print("\n" + "=" * 60)
    print("Test 1: Individual Joint Movement")
    print("=" * 60)

    for joint_idx in range(6):
        lower = configs.DOF_LIMITS_LOWER[joint_idx]
        upper = configs.DOF_LIMITS_UPPER[joint_idx]
        mid = (lower + upper) / 2

        print(f"\n>>> Joint {joint_idx} ({configs.JOINT_NAMES[joint_idx]})")
        print(f"    Limits: [{lower:.3f}, {upper:.3f}]")
        print(f"    Target: {mid:.3f} rad ({mid * 57.2958:.1f}°)")

        # Create action with only this joint moving
        current = robot.get_joint_positions().flatten()[:6].copy()
        target = current.copy()
        target[joint_idx] = mid

        # Execute movement with interpolation
        for step in range(steps_per_joint):
            world.step(render=True)
            if world.is_playing():
                alpha = (step + 1) / steps_per_joint
                interpolated = current + alpha * (target - current)
                robot.set_joint_positions(interpolated)
            elif world.is_stopped():
                world.reset()

        # Check result
        actual = robot.get_joint_positions().flatten()[joint_idx]
        error = abs(actual - mid)
        status = "✓" if error < 0.05 else "✗"
        print(f"    Actual: {actual:.3f} rad ({actual * 57.2958:.1f}°)")
        print(f"    Error:  {error:.4f} rad {status}")

        time.sleep(0.5)


def test_joint_limits(robot, world, steps=None):
    """Test upper/lower limits with 5% margin."""
    if steps is None:
        steps = configs.STEPS_PER_POSITION

    print("\n" + "=" * 60)
    print("Test 2: Joint Limits (with 5% safety margin)")
    print("=" * 60)

    for joint_idx in range(6):
        lower = configs.DOF_LIMITS_LOWER[joint_idx]
        upper = configs.DOF_LIMITS_UPPER[joint_idx]
        margin = 0.05 * (upper - lower)

        print(f"\n>>> Joint {joint_idx} ({configs.JOINT_NAMES[joint_idx]})")

        # Test lower limit
        target_lower = lower + margin
        current = robot.get_joint_positions().flatten()[:6].copy()
        target = current.copy()
        target[joint_idx] = target_lower

        for step in range(steps):
            world.step(render=True)
            if world.is_playing():
                alpha = (step + 1) / steps
                interpolated = current + alpha * (target - current)
                robot.set_joint_positions(interpolated)
            elif world.is_stopped():
                world.reset()

        actual = robot.get_joint_positions().flatten()[joint_idx]
        error = abs(actual - target_lower)
        print(f"    Lower: target={target_lower:.3f}, actual={actual:.3f}, error={error:.4f}")

        time.sleep(0.3)

        # Test upper limit
        target_upper = upper - margin
        current = robot.get_joint_positions().flatten()[:6].copy()
        target = current.copy()
        target[joint_idx] = target_upper

        for step in range(steps):
            world.step(render=True)
            if world.is_playing():
                alpha = (step + 1) / steps
                interpolated = current + alpha * (target - current)
                robot.set_joint_positions(interpolated)
            elif world.is_stopped():
                world.reset()

        actual = robot.get_joint_positions().flatten()[joint_idx]
        error = abs(actual - target_upper)
        print(f"    Upper: target={target_upper:.3f}, actual={actual:.3f}, error={error:.4f}")

        time.sleep(0.3)


def test_preset_positions(robot, world, steps=None):
    """Test moving to each preset position."""
    if steps is None:
        steps = configs.STEPS_PER_POSITION

    print("\n" + "=" * 60)
    print("Test 3: Preset Positions")
    print("=" * 60)

    current_pos = robot.get_joint_positions().flatten()[:6]

    for name, position in configs.PRESET_POSITIONS.items():
        print(f"\n>>> {name}")

        target_pos = position
        for step in range(steps):
            world.step(render=True)
            if world.is_playing():
                alpha = (step + 1) / steps
                interpolated = current_pos + alpha * (target_pos - current_pos)
                robot.set_joint_positions(interpolated)
            elif world.is_stopped():
                world.reset()

        current_pos = robot.get_joint_positions().flatten()[:6]
        error = np.linalg.norm(current_pos - position)
        status = "✓" if error < 0.05 else "✗"

        print(f"    Target: {position}")
        print(f"    Actual: {current_pos}")
        print(f"    Error:  {error:.4f} rad {status}")

        time.sleep(0.5)
