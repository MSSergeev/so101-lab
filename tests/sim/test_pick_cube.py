"""Test pick cube environment loading and basic functionality."""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def main():
    """Test pick cube environment."""
    parser = argparse.ArgumentParser(description="Test SO-101 pick cube environment")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--headless", action="store_true", default=True, help="Run in headless mode")
    parser.add_argument("--gui", action="store_true", help="Run with GUI")
    args = parser.parse_args()

    headless = not args.gui

    # Initialize Isaac Sim first
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": headless, "enable_cameras": True})
    simulation_app = app_launcher.app

    # Now import Isaac Lab modules and our environment
    from so101_lab.tasks.pick_cube.env_cfg import PickCubeEnvCfg
    from so101_lab.tasks.pick_cube.env import PickCubeEnv

    # Create environment config
    cfg = PickCubeEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"

    # Create environment
    print(f"Creating pick cube environment with {args.num_envs} parallel environments...")
    sys.stdout.flush()
    env = PickCubeEnv(cfg)

    print("\n" + "=" * 60)
    print("Pick Cube Environment created successfully!")
    print("=" * 60)
    print(f"Action space: {cfg.action_space}")
    print(f"Observation space keys: {list(cfg.observation_space.keys())}")
    print(f"State space keys: {list(cfg.state_space.keys())}")
    print(f"Device: {env.device}")
    print(f"Number of environments: {env.num_envs}")
    print("=" * 60 + "\n")
    sys.stdout.flush()

    # Reset environment
    print("Resetting environment...")
    sys.stdout.flush()
    obs_dict, _ = env.reset()  # Returns (obs, info)
    print(f"Observation policy keys: {list(obs_dict['policy'].keys())}")
    print(f"  - joint_pos shape: {obs_dict['policy']['joint_pos'].shape}")
    print(f"  - actions shape: {obs_dict['policy']['actions'].shape}")
    print(f"  - joint_pos_target shape: {obs_dict['policy']['joint_pos_target'].shape}")
    if "top" in obs_dict['policy']:
        print(f"  - top camera shape: {obs_dict['policy']['top'].shape}")
    if "wrist" in obs_dict['policy']:
        print(f"  - wrist camera shape: {obs_dict['policy']['wrist'].shape}")
    print()
    sys.stdout.flush()

    # Check cube is registered
    print("Checking cube registration...")
    cube_pos = env.scene["cube"].data.root_pos_w
    print(f"  - Cube position shape: {cube_pos.shape}")
    print(f"  - Cube initial position: {cube_pos[0].cpu().numpy()}")
    print()
    sys.stdout.flush()

    # Take random actions for a few steps
    import torch
    print("Running 10 steps with random actions...")
    sys.stdout.flush()
    for i in range(10):
        # Small random actions around zero
        actions = torch.randn(env.num_envs, 6, device=env.device) * 0.05
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        if i % 5 == 0:
            print(f"  Step {i:2d}: reward={rewards[0].item():.3f}, "
                  f"terminated={terminated[0].item()}, truncated={truncated[0].item()}")
            sys.stdout.flush()

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)
    sys.stdout.flush()

    # Close environment
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
