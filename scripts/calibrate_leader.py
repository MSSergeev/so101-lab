#!/usr/bin/env python3
# Copyright (c) 2025, SO-101 Lab Project

"""Calibrate SO-101 Leader arm motors.

This script creates a calibration file for SO101Leader device.
Run this once per leader arm before using teleop.

Usage:
    python scripts/calibrate_leader.py --port=/dev/ttyACM0 --name=leader_left
    python scripts/calibrate_leader.py --port=/dev/ttyACM1 --name=leader_right
"""

import argparse
import sys
from unittest.mock import MagicMock

# Mock Isaac Lab modules before any imports (avoid pxr dependency)
sys.modules['carb'] = MagicMock()
sys.modules['omni'] = MagicMock()
sys.modules['pxr'] = MagicMock()
sys.modules['isaaclab'] = MagicMock()
sys.modules['isaaclab.utils'] = MagicMock()
sys.modules['isaaclab.utils.math'] = MagicMock()


def calibrate_leader(port: str, calibration_name: str):
    """Run calibration procedure for SO101Leader."""
    from so101_lab.devices.lerobot.so101_leader import SO101Leader

    # Create mock env (SO101Leader needs env for Device base class)
    class MockEnv:
        device = "cpu"
        num_envs = 1

    print("=" * 60)
    print("SO-101 Leader Arm Calibration")
    print("=" * 60)
    print()
    print(f"Port: {port}")
    print(f"Calibration file: {calibration_name}.json")
    print()
    print("This will guide you through calibrating the leader arm.")
    print()

    try:
        # Create SO101Leader with recalibrate=True
        leader = SO101Leader(
            env=MockEnv(),
            port=port,
            recalibrate=True,
            calibration_file_name=f"{calibration_name}.json",
        )

        print()
        print("✓ Calibration complete!")
        print()
        print("Calibration saved to:")
        print(f"  {leader.calibration_path}")
        print()
        print("Next steps:")
        print(f"  python scripts/teleop/teleop_agent.py \\")
        print(f"      --teleop-device=so101leader \\")
        print(f"      --port={port} \\")
        print(f"      --calibration-file={calibration_name}.json")
        print()

        # Disconnect
        leader.disconnect()
        return True

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate SO-101 Leader arm motors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single arm setup
  python scripts/calibrate_leader.py --port=/dev/ttyACM0 --name=leader_main

  # Dual arm setup
  python scripts/calibrate_leader.py --port=/dev/ttyACM0 --name=leader_left
  python scripts/calibrate_leader.py --port=/dev/ttyACM1 --name=leader_right

Note:
  Calibration files are saved to:
    so101_lab/devices/lerobot/.cache/<name>.json
        """
    )
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port (e.g., /dev/ttyACM0, /dev/ttyUSB0)"
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Calibration name (e.g., leader_left, leader_right, leader_main)"
    )
    args = parser.parse_args()

    # Validate calibration name
    if not args.name.replace("_", "").isalnum():
        print("❌ Error: Calibration name must be alphanumeric (underscores allowed)")
        print(f"   Got: {args.name}")
        sys.exit(1)

    success = calibrate_leader(args.port, args.name)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
