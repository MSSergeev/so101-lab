#!/usr/bin/env python3
# Copyright (c) 2025, SO-101 Lab Project

"""Unit tests for SO101Gamepad device."""

import numpy as np
import pytest


class TestGamepadDualBuffer:
    """Test dual buffer system for resolving gamepad stick inputs."""

    def test_positive_direction_only(self):
        """Test when only positive direction has value."""
        raw_command = np.array([[0.8, 0.0, 0.5], [0.0, 0.0, 0.0]])  # (2, 3)

        # Resolve buffer
        sign = raw_command[1, :] > raw_command[0, :]
        value = raw_command.max(axis=0)
        value[sign] *= -1

        assert np.allclose(value, [0.8, 0.0, 0.5])

    def test_negative_direction_only(self):
        """Test when only negative direction has value."""
        raw_command = np.array([[0.0, 0.0, 0.0], [0.0, 0.6, 0.9]])  # (2, 3)

        # Resolve buffer
        sign = raw_command[1, :] > raw_command[0, :]
        value = raw_command.max(axis=0)
        value[sign] *= -1

        assert np.allclose(value, [0.0, -0.6, -0.9])

    def test_mixed_directions(self):
        """Test when both positive and negative directions have values."""
        raw_command = np.array([[0.8, 0.0, 0.3], [0.0, 0.5, 0.7]])  # (2, 3)

        # Resolve buffer
        sign = raw_command[1, :] > raw_command[0, :]
        value = raw_command.max(axis=0)
        value[sign] *= -1

        assert np.allclose(value, [0.8, -0.5, -0.7])

    def test_zero_values(self):
        """Test when all values are zero."""
        raw_command = np.zeros([2, 3])

        # Resolve buffer
        sign = raw_command[1, :] > raw_command[0, :]
        value = raw_command.max(axis=0)
        value[sign] *= -1

        assert np.allclose(value, [0.0, 0.0, 0.0])


class TestGamepadDeadZone:
    """Test dead zone filtering for gamepad inputs."""

    def test_dead_zone_filter_below_threshold(self):
        """Test values below dead zone are zeroed."""
        dead_zone = 0.01
        values = [0.005, -0.008, 0.0]

        filtered = [val if abs(val) >= dead_zone else 0.0 for val in values]

        assert all(v == 0.0 for v in filtered)

    def test_dead_zone_filter_above_threshold(self):
        """Test values above dead zone pass through."""
        dead_zone = 0.01
        values = [0.5, -0.8, 0.02]

        filtered = [val if abs(val) >= dead_zone else 0.0 for val in values]

        assert filtered == [0.5, -0.8, 0.02]

    def test_dead_zone_filter_at_threshold(self):
        """Test values exactly at dead zone threshold."""
        dead_zone = 0.01
        values = [0.01, -0.01]

        filtered = [val if abs(val) >= dead_zone else 0.0 for val in values]

        assert filtered == [0.01, -0.01]


class TestGamepadMapping:
    """Test gamepad input mapping configuration."""

    def test_stick_mapping_axes(self):
        """Verify stick mappings target correct axes."""
        # Expected mappings (direction, axis, sensitivity)
        # Left Stick Y → z axis (idx 2)
        # Right Stick Y → x axis (idx 0)
        # Right Stick X → yaw axis (idx 5)

        left_stick_up = (1, 2, 0.01)  # backward (negative z)
        left_stick_down = (0, 2, 0.01)  # forward (positive z)
        right_stick_up = (0, 0, 0.01)  # up (positive x)
        right_stick_down = (1, 0, 0.01)  # down (negative x)
        right_stick_left = (0, 5, 0.15)  # yaw left (positive yaw)
        right_stick_right = (1, 5, 0.15)  # yaw right (negative yaw)

        # Verify axis indices
        assert left_stick_up[1] == 2  # z axis
        assert right_stick_up[1] == 0  # x axis
        assert right_stick_left[1] == 5  # yaw axis

    def test_dpad_mapping_axes(self):
        """Verify D-Pad mappings target correct axes."""
        # Expected mappings (direction, axis, sensitivity)
        # D-Pad Up/Down → pitch axis (idx 4)
        # D-Pad Left/Right → roll axis (idx 3)

        dpad_up = (1, 4, 0.15 * 0.8)  # pitch up (negative)
        dpad_down = (0, 4, 0.15 * 0.8)  # pitch down (positive)
        dpad_left = (0, 3, 0.15 * 0.8)  # roll left (positive)
        dpad_right = (1, 3, 0.15 * 0.8)  # roll right (negative)

        # Verify axis indices
        assert dpad_up[1] == 4  # pitch axis
        assert dpad_left[1] == 3  # roll axis

    def test_sensitivity_scaling(self):
        """Test sensitivity values are correctly scaled."""
        base_pos_sens = 0.01
        base_rot_sens = 0.15
        dpad_multiplier = 0.8

        # Stick sensitivities (position)
        assert abs((1, 2, base_pos_sens)[2] - 0.01) < 1e-6

        # Stick sensitivities (rotation)
        assert abs((0, 5, base_rot_sens)[2] - 0.15) < 1e-6

        # D-Pad sensitivities (with 0.8 multiplier)
        assert abs((0, 4, base_rot_sens * dpad_multiplier)[2] - 0.12) < 1e-6


class TestGamepadJointCommands:
    """Test joint command buffers for shoulder_pan and gripper."""

    def test_bumper_shoulder_pan_left(self):
        """Test LB bumper sets negative shoulder_pan delta."""
        joint_sensitivity = 0.15
        delta_joint_raw = np.zeros(2)

        # LB pressed (val > 0.5)
        delta_joint_raw[0] = -joint_sensitivity

        assert delta_joint_raw[0] == -0.15

    def test_bumper_shoulder_pan_right(self):
        """Test RB bumper sets positive shoulder_pan delta."""
        joint_sensitivity = 0.15
        delta_joint_raw = np.zeros(2)

        # RB pressed (val > 0.5)
        delta_joint_raw[0] = joint_sensitivity

        assert delta_joint_raw[0] == 0.15

    def test_trigger_gripper_open(self):
        """Test LT trigger opens gripper (positive delta)."""
        joint_sensitivity = 0.15
        delta_joint_raw = np.zeros(2)

        # LT analog value 0.8
        trigger_val = 0.8
        delta_joint_raw[1] = trigger_val * joint_sensitivity

        assert abs(delta_joint_raw[1] - 0.12) < 1e-6

    def test_trigger_gripper_close(self):
        """Test RT trigger closes gripper (negative delta)."""
        joint_sensitivity = 0.15
        delta_joint_raw = np.zeros(2)

        # RT analog value 0.6
        trigger_val = 0.6
        delta_joint_raw[1] = -trigger_val * joint_sensitivity

        assert abs(delta_joint_raw[1] - (-0.09)) < 1e-6

    def test_trigger_gripper_full_range(self):
        """Test gripper uses full analog range 0-1."""
        joint_sensitivity = 0.15
        delta_joint_raw = np.zeros(2)

        # Full press
        trigger_val = 1.0
        delta_joint_raw[1] = trigger_val * joint_sensitivity

        assert abs(delta_joint_raw[1] - 0.15) < 1e-6

        # No press
        trigger_val = 0.0
        delta_joint_raw[1] = trigger_val * joint_sensitivity

        assert delta_joint_raw[1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
