# Copyright (c) 2025, SO-101 Lab Project
# Adapted from: leisaac (https://github.com/isaac-sim/IsaacLab)
# Original license: BSD-3-Clause
# Changes: Adapted for SO-101 Lab project

"""SO-101 Leader arm device for real-hardware teleoperation.

Reads joint positions from 6 STS3215 motors via FeetechMotorsBus (USB serial)
and maps them directly to the simulated robot. No IK — direct joint-to-joint mapping.

Requires calibration file (JSON). Run scripts/calibrate_leader.py first.
"""

import json
import os

from so101_lab.devices.device_base import Device

from .common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from .common.motors import (
    FeetechMotorsBus,
    Motor,
    MotorCalibration,
    MotorNormMode,
    OperatingMode,
)


class SO101Leader(Device):
    """SO-101 Leader device for real robot teleoperation.

    Connects to SO-101 Leader arm via USB serial port (FeetechMotorsBus).
    Reads joint positions from 6 STS3215 motors: shoulder_pan, shoulder_lift,
    elbow_flex, wrist_flex, wrist_roll, gripper.

    Calibration is stored in .cache/so101_leader.json and must be created
    with --recalibrate flag on first run.
    """

    def __init__(
        self,
        env,
        port: str = "/dev/ttyACM0",
        recalibrate: bool = False,
        calibration_file_name: str = "so101_leader.json",
    ):
        """Initialize SO101Leader device.

        Args:
            env: Isaac Lab environment (for Device base class compatibility)
            port: Serial port for motor bus (e.g., /dev/ttyACM0, /dev/ttyUSB0)
            recalibrate: Force recalibration even if calibration file exists
            calibration_file_name: Name of calibration JSON file
        """
        super().__init__(env, "so101_leader")
        self.port = port

        # Calibration setup
        self.calibration_path = os.path.join(os.path.dirname(__file__), ".cache", calibration_file_name)

        # Load or create calibration
        if not os.path.exists(self.calibration_path):
            if not recalibrate:
                raise FileNotFoundError(
                    f"Calibration file not found: {self.calibration_path}\n"
                    "Run with --recalibrate to create calibration."
                )
            self.calibrate()
        elif recalibrate:
            self.calibrate()

        calibration = self._load_calibration()

        # Create motor bus with 6 STS3215 motors
        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calibration,
        )

        # Connect and configure
        self.connect()

    def _add_device_control_description(self):
        """Add SO101Leader controls to display table."""
        self._display_controls_table.add_row(["SO101-Leader", "Move leader arm to control follower/sim"])
        self._display_controls_table.add_row(
            ["[TIPS]", "Use --recalibrate flag to recalibrate motor ranges"]
        )

    def get_device_state(self) -> dict:
        """Get current joint positions from leader arm.

        Returns:
            dict: Joint positions in range [-100, 100] for arm, [0, 100] for gripper
                  {"shoulder_pan": 0.5, "shoulder_lift": -30.0, ...}
        """
        return self._bus.sync_read("Present_Position")

    @property
    def is_connected(self) -> bool:
        """Check if motor bus is connected."""
        return self._bus.is_connected

    def connect(self):
        """Connect to motor bus and configure motors."""
        if self.is_connected:
            raise DeviceAlreadyConnectedError("SO101-Leader is already connected.")
        self._bus.connect()
        self.configure()
        print("SO101-Leader connected.")

    def disconnect(self):
        """Disconnect from motor bus."""
        if not self.is_connected:
            raise DeviceNotConnectedError("SO101-Leader is not connected.")
        self._bus.disconnect()
        print("SO101-Leader disconnected.")

    def configure(self) -> None:
        """Configure motors: disable torque, set position mode."""
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def calibrate(self):
        """Interactive calibration process.

        Steps:
        1. Move arm to middle of range → press ENTER (homing)
        2. Move all joints through full range → press ENTER (min/max recording)
        3. Save calibration to JSON
        """
        # Create temporary bus without calibration
        self._bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
        )
        self.connect()

        print("\n=== Running calibration of SO101-Leader ===")
        self._bus.disable_torque()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        # Step 1: Homing
        input("Move SO101-Leader to the MIDDLE of its range and press ENTER...")
        homing_offset = self._bus.set_half_turn_homings()

        # Step 2: Record ranges
        print("Move all joints sequentially through their ENTIRE range.")
        print("Recording positions. Press ENTER to stop...")
        range_mins, range_maxes = self._bus.record_ranges_of_motion()

        # Step 3: Build calibration
        calibration = {}
        for motor, m in self._bus.motors.items():
            calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offset[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        # Save to motors and JSON
        self._bus.write_calibration(calibration)
        self._save_calibration(calibration)
        print(f"Calibration saved to {self.calibration_path}")

        self.disconnect()

    def _load_calibration(self) -> dict[str, MotorCalibration]:
        """Load calibration from JSON file."""
        with open(self.calibration_path) as f:
            json_data = json.load(f)

        calibration = {}
        for motor_name, motor_data in json_data.items():
            calibration[motor_name] = MotorCalibration(
                id=int(motor_data["id"]),
                drive_mode=int(motor_data["drive_mode"]),
                homing_offset=int(motor_data["homing_offset"]),
                range_min=int(motor_data["range_min"]),
                range_max=int(motor_data["range_max"]),
            )
        return calibration

    def _save_calibration(self, calibration: dict[str, MotorCalibration]):
        """Save calibration to JSON file."""
        save_calibration = {
            k: {
                "id": v.id,
                "drive_mode": v.drive_mode,
                "homing_offset": v.homing_offset,
                "range_min": v.range_min,
                "range_max": v.range_max,
            }
            for k, v in calibration.items()
        }

        # Create .cache directory if doesn't exist
        cache_dir = os.path.dirname(self.calibration_path)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        with open(self.calibration_path, "w") as f:
            json.dump(save_calibration, f, indent=4)

    def reset(self):
        """Reset device state (inherited from Device)."""
        pass  # No internal state to reset
