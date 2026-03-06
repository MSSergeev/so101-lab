#!/usr/bin/env python3
# Copyright (c) 2025, SO-101 Lab Project

"""Test Feetech motor bus connection directly (no calibration needed).

This script tests motor communication at the lowest level.
Useful for debugging hardware issues without needing calibration.

Usage:
    python tests/test_motor_bus.py --port=/dev/ttyACM0
    python tests/test_motor_bus.py --port=/dev/ttyACM1
"""

import argparse
import sys
import time
from unittest.mock import MagicMock

# Mock Isaac Lab modules before any imports (avoid pxr dependency)
sys.modules['carb'] = MagicMock()
sys.modules['omni'] = MagicMock()
sys.modules['pxr'] = MagicMock()
sys.modules['isaaclab'] = MagicMock()
sys.modules['isaaclab.utils'] = MagicMock()
sys.modules['isaaclab.utils.math'] = MagicMock()


def test_motor_bus(port: str):
    """Test raw motor bus communication without SO101Leader wrapper."""
    from so101_lab.devices.lerobot.common.motors import (
        FeetechMotorsBus,
        Motor,
        MotorCalibration,
    )

    print(f"Testing Feetech motor bus on {port}...")
    print()

    # Define expected motors (SO-101 has 6 STS3215 motors)
    # Motor(id, model, norm_mode) - key is motor name
    from so101_lab.devices.lerobot.common.motors import MotorNormMode

    expected_motors = {
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }

    # Minimal calibration (use dummy values for testing)
    calibration = {
        "shoulder_pan": MotorCalibration(id=1, drive_mode=0, homing_offset=2048, range_min=0, range_max=4095),
        "shoulder_lift": MotorCalibration(id=2, drive_mode=0, homing_offset=2048, range_min=0, range_max=4095),
        "elbow_flex": MotorCalibration(id=3, drive_mode=0, homing_offset=2048, range_min=0, range_max=4095),
        "wrist_flex": MotorCalibration(id=4, drive_mode=0, homing_offset=2048, range_min=0, range_max=4095),
        "wrist_roll": MotorCalibration(id=5, drive_mode=0, homing_offset=2048, range_min=0, range_max=4095),
        "gripper": MotorCalibration(id=6, drive_mode=0, homing_offset=2048, range_min=0, range_max=4095),
    }

    try:
        print("1. Creating motor bus...")
        bus = FeetechMotorsBus(port=port, motors=expected_motors, calibration=calibration)

        print("2. Connecting to motors...")
        try:
            bus.connect()
            print("✓ Connected successfully!")
        except RuntimeError as e:
            if "motor check failed" in str(e):
                print("❌ Motor check failed")
                print()
                print(str(e))
                print()
                print("Troubleshooting:")
                print("  - Check USB connection")
                print("  - Verify power supply to motors")
                print("  - Check motor IDs (should be 1-6)")
                print("  - Try different port (/dev/ttyACM0, /dev/ttyACM1, /dev/ttyUSB0)")
                return False
            raise

        print()
        print("3. Motor information:")
        print("   ID | Name           | Model")
        print("   ---|----------------|-------")
        for motor_name, motor in expected_motors.items():
            print(f"   {motor.id}  | {motor_name:14s} | {motor.model}")

        print()
        print("4. Reading raw positions (press Ctrl+C to stop)...")
        print("   STS3215 range: 0-4095 (center=2048)")
        print("   Format: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]")
        print()

        try:
            motor_names = list(expected_motors.keys())
            while True:
                # Read raw positions (no normalization) for each motor
                positions = []
                for motor_name in motor_names:
                    pos = bus.read("Present_Position", motor_name, normalize=False)
                    positions.append(pos)

                # Format: show as integers
                pos_str = ", ".join([f"{int(p):4d}" for p in positions])
                print(f"   [{pos_str}]", end="\r")
                time.sleep(0.1)  # 10 Hz
        except KeyboardInterrupt:
            print()
            print()
            print("5. Disconnecting...")
            bus.disconnect()
            print("✓ Test complete!")
            return True

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test Feetech motor bus connection (no calibration needed)"
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Serial port (e.g., /dev/ttyACM0, /dev/ttyUSB0)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SO-101 Motor Bus Test")
    print("=" * 60)
    print()

    success = test_motor_bus(args.port)

    if success:
        print()
        print("Next steps:")
        print("  - Run calibration: python scripts/calibrate_leader.py --port=" + args.port)
        print("  - Start teleop: python scripts/teleop/teleop_agent.py --teleop-device=so101leader")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
