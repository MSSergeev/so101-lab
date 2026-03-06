# Status: incomplete — real robot HIL-SERL (Phase 5.7) not yet implemented
"""HIL device: read SO-101 leader arm positions as actions.

Lightweight wrapper around FeetechMotorsBus — does NOT inherit from Device
(Device.__init__ requires Isaac Sim GUI via carb.input).
"""

import json
import os

import numpy as np

from so101_lab.devices.lerobot.common.motors import (
    FeetechMotorsBus,
    Motor,
    MotorCalibration,
    MotorNormMode,
    OperatingMode,
)

# Motor order matches IsaacLabGymEnv action space
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class HILDeviceReader:
    """Reads leader arm joint positions as env actions in [-100, 100] format."""

    def __init__(self, port: str = "/dev/ttyACM0", calibration_path: str | None = None):
        if calibration_path is None:
            calibration_path = os.path.join(
                os.path.dirname(__file__), "..", "devices", "lerobot", ".cache", "so101_leader.json"
            )
        calibration_path = os.path.normpath(calibration_path)

        if not os.path.exists(calibration_path):
            raise FileNotFoundError(
                f"Calibration file not found: {calibration_path}\n"
                "Calibrate leader arm first via SO101Leader(recalibrate=True)."
            )

        calibration = self._load_calibration(calibration_path)

        self._bus = FeetechMotorsBus(
            port=port,
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
        self._bus.connect()
        self._bus.disable_torque()
        self._bus.configure_motors()
        for motor in self._bus.motors:
            self._bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        print(f"[HIL] Leader arm connected on {port}")

    def read_action(self) -> np.ndarray:
        """Read current joint positions as action array (6,) in [-100, 100]."""
        positions = self._bus.sync_read("Present_Position")
        return np.array([positions[name] for name in MOTOR_NAMES], dtype=np.float32)

    def disconnect(self):
        if self._bus.is_connected:
            self._bus.disconnect()
            print("[HIL] Leader arm disconnected")

    @staticmethod
    def _load_calibration(path: str) -> dict[str, MotorCalibration]:
        with open(path) as f:
            data = json.load(f)
        return {
            name: MotorCalibration(
                id=int(m["id"]),
                drive_mode=int(m["drive_mode"]),
                homing_offset=int(m["homing_offset"]),
                range_min=int(m["range_min"]),
                range_max=int(m["range_max"]),
            )
            for name, m in data.items()
        }
