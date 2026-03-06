#!/usr/bin/env python3
"""Test teleop mode configuration with GUI for visual inspection."""

from isaaclab.app import AppLauncher

# Launch with GUI and cameras
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import torch

from so101_lab.tasks.template.env import TemplateEnv
from so101_lab.tasks.template.env_cfg import TemplateEnvCfg

def main():
    print("\n" + "="*80)
    print("Testing Teleop Mode Configuration")
    print("="*80)

    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1

    print("\n[INFO] Enabling teleop mode with keyboard device...")
    cfg.use_teleop_device("keyboard")

    print("[INFO] Creating environment in TELEOP mode...")
    env = TemplateEnv(cfg)

    # Verify teleop mode properties
    print(f"\n[CHECK] action_space = {cfg.action_space} (expected: 8)")
    assert cfg.action_space == 8, "Action space should be 8 in teleop mode"

    print(f"[CHECK] cfg.actions = {cfg.actions} (expected: not None)")
    assert cfg.actions is not None, "Actions should be configured in teleop mode"

    print(f"[CHECK] env.action_manager = {env.action_manager} (expected: not None)")
    assert env.action_manager is not None, "ActionManager should be created in teleop mode"

    print(f"[CHECK] gravity disabled = {cfg.scene.robot.spawn.rigid_props.disable_gravity} (expected: True)")
    assert cfg.scene.robot.spawn.rigid_props.disable_gravity is True, "Gravity should be disabled for IK"

    print("\n[INFO] Checking ActionsCfg configuration...")
    print(f"  arm_action type: {type(cfg.actions.arm_action).__name__}")
    print(f"  gripper_action type: {type(cfg.actions.gripper_action).__name__}")

    print("\n[INFO] Resetting environment...")
    env.reset()

    print("[INFO] Testing teleop action (8 DOF with zero deltas)...")
    # Test with 8 DOF actions (SE3 delta + 2 joint deltas)
    actions = torch.zeros(env.num_envs, 8, device=env.device)

    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(actions)
        print(f"  Step {i+1}/10: obs keys = {list(obs.keys())}")

    assert obs is not None, "Observations should not be None"

    print("\n[SUCCESS] Teleop mode test passed!")
    print("  - Teleop mode (8 DOF) configured correctly")
    print("  - ActionManager created with DifferentialIK")
    print("  - Environment steps successfully")

    print("\n[INFO] Press Ctrl+C to exit...")

    # Keep running for visual inspection
    try:
        while simulation_app.is_running():
            env.step(actions)
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted, closing...")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
