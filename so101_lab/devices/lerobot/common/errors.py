# Copyright (c) 2025, SO-101 Lab Project
# Adapted from: leisaac (https://github.com/isaac-sim/IsaacLab)
# Original license: BSD-3-Clause
# Changes: Copied for SO-101 Lab project integration

class DeviceNotConnectedError(ConnectionError):
    """Exception raised when the device is not connected."""

    def __init__(self, message="This device is not connected. Try calling `connect()` first."):
        self.message = message
        super().__init__(self.message)


class DeviceAlreadyConnectedError(ConnectionError):
    """Exception raised when the device is already connected."""

    def __init__(
        self,
        message="This device is already connected. Try not calling `connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)
