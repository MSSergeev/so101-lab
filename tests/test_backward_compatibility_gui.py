#!/usr/bin/env python3
"""Test backward compatibility with GUI for visual inspection."""

from isaaclab.app import AppLauncher

# Launch with GUI and cameras
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import torch

from so101_lab.tasks.template.env import TemplateEnv
from so101_lab.tasks.template.env_cfg import TemplateEnvCfg

def main():
    print("\n" + "="*80)
    print("Testing Backward Compatibility (Direct Mode)")
    print("="*80)

    # Create env WITHOUT use_teleop_device()
    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1

    print("\n[INFO] Creating environment in DIRECT mode (no teleop)...")
    env = TemplateEnv(cfg)

    # Verify direct mode properties
    print(f"[CHECK] action_space = {cfg.action_space} (expected: 6)")
    assert cfg.action_space == 6, "Action space should be 6 in direct mode"

    print(f"[CHECK] cfg.actions = {cfg.actions} (expected: None)")
    assert cfg.actions is None, "Actions should be None in direct mode"

    print(f"[CHECK] env.action_manager = {env.action_manager} (expected: None)")
    assert env.action_manager is None, "ActionManager should be None in direct mode"

    print("\n[INFO] Resetting environment...")
    env.reset()

    print("[INFO] Testing direct joint control (stepping with zero actions)...")
    # Test direct joint control
    actions = torch.zeros(env.num_envs, 6, device=env.device)

    for i in range(10):
        obs, reward, terminated, truncated, info = env.step(actions)
        print(f"  Step {i+1}/10: obs keys = {list(obs.keys())}")

    assert obs is not None, "Observations should not be None"
    assert "policy" in obs, "Observations should contain 'policy' group"

    print("\n[SUCCESS] Backward compatibility test passed!")
    print("  - Direct mode (6 DOF) works correctly")
    print("  - No ActionManager created")
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
