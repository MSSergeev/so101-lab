"""Inspect template environment in Isaac Sim GUI - keeps window open for visual inspection."""

import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def main():
    """Inspect template environment with GUI."""
    parser = argparse.ArgumentParser(description="Inspect SO-101 template environment")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    args = parser.parse_args()

    # Initialize Isaac Sim with GUI
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": False, "enable_cameras": True})
    simulation_app = app_launcher.app

    # Now import Isaac Lab modules and our environment
    from so101_lab.tasks.template.env_cfg import TemplateEnvCfg
    from so101_lab.tasks.template.env import TemplateEnv

    # Create environment config
    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = "cuda:0"

    # Create environment
    print(f"Creating environment with {args.num_envs} parallel environments...")
    sys.stdout.flush()
    env = TemplateEnv(cfg)

    print("\n" + "=" * 60)
    print("Environment created successfully!")
    print("=" * 60)
    print(f"Number of environments: {env.num_envs}")
    print(f"Device: {env.device}")
    print("=" * 60)
    print("\nInstructions:")
    print("- Use mouse to navigate the scene")
    print("- Press 'q' in terminal or close window to exit")
    print("- Robot will perform slow random motions")
    print("=" * 60 + "\n")
    sys.stdout.flush()

    # Reset environment
    obs_dict, _ = env.reset()

    # Keep running with small random actions
    import torch
    step_count = 0

    try:
        while simulation_app.is_running():
            # Small random actions around current position
            actions = torch.randn(env.num_envs, 6, device=env.device) * 0.02
            obs_dict, rewards, terminated, truncated, info = env.step(actions)

            # Reset if episode ends
            if terminated.any() or truncated.any():
                print(f"Episode ended at step {step_count}, resetting...")
                obs_dict, _ = env.reset()

            step_count += 1

            # Print status every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}: Running... (press Ctrl+C to exit)")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print("\nClosing environment...")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
