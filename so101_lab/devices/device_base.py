# Copyright (c) 2025, SO-101 Lab Project
# Adapted from: leisaac (https://github.com/isaac-sim/IsaacLab)
# Original license: BSD-3-Clause
# Changes: Simplified advance() to return dict, removed leisaac-specific preprocessing

"""Base class for teleoperation devices."""

import weakref
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import carb
import numpy as np
import omni
import torch
from prettytable import PrettyTable


class DeviceBase(ABC):
    """An interface class for teleoperation devices."""

    def __init__(self):
        """Initialize the teleoperation interface."""
        pass

    def __str__(self) -> str:
        """Returns: A string containing the information of device."""
        return f"{self.__class__.__name__}"

    @abstractmethod
    def reset(self):
        """Reset the internals."""
        raise NotImplementedError

    @abstractmethod
    def add_callback(self, key: Any, func: Callable):
        """Add additional functions to bind keyboard.

        Args:
            key: The button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        raise NotImplementedError

    @abstractmethod
    def advance(self) -> Any:
        """Provides the device event state.

        Returns:
            The processed output from the device.
        """
        raise NotImplementedError


class Device(DeviceBase):
    """Keyboard-based teleoperation device."""

    def __init__(self, env, device_type: str):
        """Initialize keyboard device.

        Args:
            env: The environment which contains the robot(s) to control.
            device_type: The type of the device (e.g., "keyboard").
        """
        self.env = env
        self.device_type = device_type

        # Keyboard setup
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # Use weakref to allow object deletion
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # State flags
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        # Control table
        self._display_controls_table = PrettyTable()
        self._display_controls_table.title = f"Teleoperation Controls for {self.device_type}"
        self._display_controls_table.field_names = ["Key", "Description"]
        self._display_controls_table.align["Description"] = "l"
        # Basic controls
        self._display_controls_table.add_row(["Space", "start teleop"])
        self._display_controls_table.add_row(["X", "reset simulation"])
        self._display_controls_table.add_row(["T", "debug info"])
        self._display_controls_table.add_row(["Escape", "quit"])
        self._display_controls_table.add_row(["==========", "=========="])
        self._add_device_control_description()

    def __del__(self):
        """Release the keyboard interface."""
        self._stop_keyboard_listener()

    def get_device_state(self):
        """Get current device state. Must be implemented by subclass."""
        raise NotImplementedError

    def advance(self) -> dict | None:
        """Get device state and package as dict.

        Returns:
            dict with keys: "reset", "started", "keyboard", "joint_state"
            or None if not started
        """
        # Special case: return reset dict even if not started
        if self._reset_state:
            action_dict = {
                "reset": True,
                "started": self._started,
                "keyboard": True,
            }
            self._reset_state = False
            return action_dict

        if not self._started:
            return None

        action_dict = {
            "reset": False,
            "started": self._started,
            "keyboard": True,
        }

        # Get device-specific state (implemented by subclass)
        action_dict["joint_state"] = self.get_device_state()

        return action_dict

    def reset(self):
        """Reset device state."""
        pass

    def add_callback(self, key: str, func: Callable):
        """Add callback for specific key press."""
        self._additional_callbacks[key] = func

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            if key_name == "SPACE":
                self._started = True
                self._reset_state = False
            elif key_name == "X":
                self._started = False
                self._reset_state = True
            # Call additional callbacks for any registered key
            if key_name in self._additional_callbacks:
                self._additional_callbacks[key_name]()

    def _stop_keyboard_listener(self):
        """Unsubscribe from keyboard events."""
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _add_device_control_description(self):
        """Add device-specific control descriptions. Must be implemented by subclass."""
        raise NotImplementedError

    def display_controls(self):
        """Print control table to console."""
        print(self._display_controls_table)

    @property
    def started(self) -> bool:
        """Whether device control is started."""
        return self._started

    @property
    def reset_state(self) -> bool:
        """Whether reset is requested."""
        return self._reset_state

    @reset_state.setter
    def reset_state(self, reset_state: bool):
        self._reset_state = reset_state
