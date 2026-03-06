"""Unit tests for keyboard device."""

import numpy as np
import pytest


def test_keyboard_device_init():
    """Test device initialization."""
    from isaaclab.app import AppLauncher

    # Launch headless with cameras enabled
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    from so101_lab.devices import SO101Keyboard
    from so101_lab.tasks.template.env import TemplateEnv
    from so101_lab.tasks.template.env_cfg import TemplateEnvCfg

    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_teleop_device("keyboard")
    env = TemplateEnv(cfg)

    device = SO101Keyboard(env)

    assert device.device_type == "keyboard"
    assert device._delta_action.shape == (8,)
    assert not device.started

    env.close()
    simulation_app.close()


def test_frame_conversion():
    """Test gripper-to-base frame transformation."""
    from isaaclab.app import AppLauncher

    # Launch headless with cameras enabled
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    from so101_lab.devices import SO101Keyboard
    from so101_lab.tasks.template.env import TemplateEnv
    from so101_lab.tasks.template.env_cfg import TemplateEnvCfg

    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_teleop_device("keyboard")
    env = TemplateEnv(cfg)
    env.reset()

    device = SO101Keyboard(env)

    # Test zero delta passes through
    zero_delta = np.zeros(8)
    result = device._convert_delta_from_frame(zero_delta)
    assert np.allclose(result, zero_delta)

    # Test non-zero delta gets transformed
    nonzero_delta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = device._convert_delta_from_frame(nonzero_delta)
    assert result.shape == (8,)
    # Last 2 elements (shoulder_pan, gripper) should be unchanged
    assert np.allclose(result[6:], nonzero_delta[6:])

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
