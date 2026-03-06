"""Print per-step reward breakdown from env RewardManager.

Usage:
    python scripts/tools/verify_rewards.py --num-steps 50
    python scripts/tools/verify_rewards.py --num-steps 50 --gui
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Print env reward breakdown")
parser.add_argument("--num-steps", type=int, default=50)
parser.add_argument("--gui", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.gui:
    args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)

def _get_lerobot_src() -> str:
    import os
    from pathlib import Path
    if src := os.environ.get("LEROBOT_SRC"):
        return os.path.expanduser(src)
    env_file = Path(__file__).parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("LEROBOT_SRC="):
                return os.path.expanduser(line.split("=", 1)[1].strip())
    raise RuntimeError("LEROBOT_SRC not set. Add it to .env or set the environment variable.")

_lerobot_src = _get_lerobot_src()
if _lerobot_src not in sys.path:
    sys.path.insert(0, _lerobot_src)

from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv


def main():
    env = IsaacLabGymEnv()
    step_dt = env.step_dt
    print(f"step_dt = {step_dt:.6f} (1/step_dt = {1/step_dt:.1f} Hz)")

    obs, info = env.reset()

    print(f"\n{'step':>4} | {'reward':>9} | breakdown")
    print("-" * 100)

    total = 0.0
    for i in range(args.num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        details = env.get_reward_details()
        detail_str = "  ".join(f"{k.split('/')[-1]}={v:+.4f}" for k, v in details.items()
                               if k != "reward/sim_total")

        print(f"{i:4d} | {reward:+9.5f} | {detail_str}")
        total += reward

        if terminated or truncated:
            obs, info = env.reset()
            print(f"  -- RESET (terminated={terminated}, truncated={truncated}) --")

    print("-" * 100)
    print(f"Total reward: {total:+.4f} ({args.num_steps} steps)")
    print(f"Note: RewardManager multiplies raw values by weight * dt ({step_dt:.4f})")

    env.close()


if __name__ == "__main__":
    main()
    app_launcher.app.close()
