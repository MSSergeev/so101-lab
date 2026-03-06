#!/usr/bin/env python3
"""Test keyboard device with GUI for visual inspection."""

from isaaclab.app import AppLauncher

# Launch with GUI and cameras
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

import numpy as np
import torch

from so101_lab.devices import SO101Keyboard
from so101_lab.tasks.template.env import TemplateEnv
from so101_lab.tasks.template.env_cfg import TemplateEnvCfg

def main():
    print("\n" + "="*80)
    print("Testing Keyboard Device")
    print("="*80)

    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1
    cfg.use_teleop_device("keyboard")

    print("\n[INFO] Creating environment and keyboard device...")
    env = TemplateEnv(cfg)
    device = SO101Keyboard(env, sensitivity=1.0)

    # Test device properties
    print(f"\n[CHECK] device_type = {device.device_type} (expected: 'keyboard')")
    assert device.device_type == "keyboard", "Device type should be 'keyboard'"

    print(f"[CHECK] delta_action shape = {device._delta_action.shape} (expected: (8,))")
    assert device._delta_action.shape == (8,), "Delta action should be 8-dimensional"

    print(f"[CHECK] started = {device.started} (expected: False)")
    assert not device.started, "Device should not be started initially"

    print("\n[INFO] Testing frame transformation...")
    env.reset()

    # Test zero delta
    zero_delta = np.zeros(8)
    result = device._convert_delta_from_frame(zero_delta)
    print(f"  Zero delta: {zero_delta}")
    print(f"  Result: {result}")
    assert np.allclose(result, zero_delta), "Zero delta should pass through unchanged"

    # Test non-zero delta
    nonzero_delta = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    result = device._convert_delta_from_frame(nonzero_delta)
    print(f"\n  Nonzero delta: {nonzero_delta}")
    print(f"  Result: {result}")
    assert result.shape == (8,), "Result should be 8-dimensional"
    assert np.allclose(result[6:], nonzero_delta[6:]), "Joint deltas should be unchanged"

    print("\n[SUCCESS] Device tests passed!")
    print("  - Device initialized correctly")
    print("  - Frame transformation working")

    print("\n[INFO] Display controls:")
    device.display_controls()

    print("\n[INFO] Press 'B' to start control, then use keyboard to test")
    print("[INFO] Press Ctrl+C to exit...")

    # Test with device input
    try:
        while simulation_app.is_running():
            action_dict = device.advance()

            if action_dict and action_dict.get("reset"):
                print("[INFO] Reset triggered")
                env.reset()
            elif action_dict:
                action = cfg.preprocess_device_action(action_dict, device)
                env.step(action)

                # Show active deltas
                if not np.allclose(device._delta_action, 0.0):
                    print(f"  Active delta: {device._delta_action}")
            else:
                env.sim.render()

    except KeyboardInterrupt:
        print("\n[INFO] User interrupted, closing...")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
