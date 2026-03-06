"""Test IsaacLabGymEnv wrapper — basic step/reset and obs shapes."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Run with GUI")
    args = parser.parse_args()

    # AppLauncher MUST be initialized before any Isaac Lab imports
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": not args.gui, "enable_cameras": True})

    import numpy as np
    from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv

    env = IsaacLabGymEnv()

    # Test reset
    obs, info = env.reset()
    print("=== Reset obs shapes ===")
    for k, v in obs.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    assert obs["observation.state"].shape == (6,), f"Expected (6,), got {obs['observation.state'].shape}"
    assert obs["observation.images.top"].shape == (480, 640, 3)
    assert obs["observation.images.wrist"].shape == (480, 640, 3)
    assert env.observation_space.contains(obs), "obs not in observation_space"
    print("Reset OK")

    # Test step with zero action
    action = np.zeros(6, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward == 0.0
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert env.observation_space.contains(obs), "obs not in observation_space after step"
    print(f"Step OK: reward={reward}, terminated={terminated}, truncated={truncated}")

    # Test multiple steps with random actions
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    print("10 random steps OK")

    env.close()
    app_launcher.app.close()
    print("All tests passed")


if __name__ == "__main__":
    main()
