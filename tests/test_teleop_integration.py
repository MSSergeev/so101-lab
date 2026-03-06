"""Integration tests for teleoperation."""

import pytest
import torch


def test_action_flow():
    """Test device → action → env.step() flow."""
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

    # Simulate device started
    device._started = True
    action_dict = device.advance()

    assert action_dict is not None
    assert "joint_state" in action_dict

    # Convert to action tensor
    action = cfg.preprocess_device_action(action_dict, device)
    assert action.shape == (env.num_envs, 8)

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    assert "policy" in obs

    env.close()
    simulation_app.close()


def test_backward_compatibility():
    """Test existing direct mode still works."""
    from isaaclab.app import AppLauncher

    # Launch headless with cameras enabled
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    from so101_lab.tasks.template.env import TemplateEnv
    from so101_lab.tasks.template.env_cfg import TemplateEnvCfg

    # Create env WITHOUT use_teleop_device()
    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1
    env = TemplateEnv(cfg)
    env.reset()

    # Verify direct mode properties
    assert cfg.action_space == 6
    assert cfg.actions is None
    assert env.action_manager is None

    # Test direct joint control
    actions = torch.zeros(env.num_envs, 6, device=env.device)
    obs, reward, terminated, truncated, info = env.step(actions)

    assert obs is not None
    assert "policy" in obs

    env.close()
    simulation_app.close()


def test_teleop_mode_properties():
    """Test teleop mode configuration."""
    from isaaclab.app import AppLauncher

    # Launch headless with cameras enabled
    app_launcher = AppLauncher(headless=True, enable_cameras=True)
    simulation_app = app_launcher.app

    from so101_lab.tasks.template.env import TemplateEnv
    from so101_lab.tasks.template.env_cfg import TemplateEnvCfg

    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_teleop_device("keyboard")
    env = TemplateEnv(cfg)

    # Verify teleop mode properties
    assert cfg.action_space == 8
    assert cfg.actions is not None
    assert env.action_manager is not None
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity is True

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
