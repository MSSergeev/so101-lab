"""Visual verification of environment spawn zones.

Resets the env N times with GUI to visually inspect object placement.

Usage:
    python scripts/eval/test_env_spawn.py --gui --env figure_shape_placement_easy
    python scripts/eval/test_env_spawn.py --gui --env figure_shape_placement --resets 20
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test env spawn zones visually")
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment name")
parser.add_argument("--resets", type=int, default=10,
                    help="Number of resets to perform")
parser.add_argument("--steps-per-reset", type=int, default=60,
                    help="Sim steps between resets (for visual inspection)")
parser.add_argument("--gui", action="store_true", help="Run with GUI")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)

# --- Post-launch imports ---
import torch

from so101_lab.tasks import get_task

EnvClass, EnvCfgClass = get_task(args.env)
cfg = EnvCfgClass()
cfg.scene.num_envs = 1

env = EnvClass(cfg=cfg)

print(f"\nEnv: {args.env}")
print(f"Resets: {args.resets}, steps per reset: {args.steps_per_reset}\n")

for i in range(args.resets):
    obs, info = env.reset()
    state = env.get_initial_state() if hasattr(env, "get_initial_state") else {}
    if state:
        parts = [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in state.items()]
        print(f"Reset {i+1}/{args.resets}: {', '.join(parts)}")
    else:
        print(f"Reset {i+1}/{args.resets}")

    action_dim = env.action_space.shape[-1] if hasattr(env.action_space, "shape") else 6
    zero_action = torch.zeros(1, action_dim, device=env.device)
    for _ in range(args.steps_per_reset):
        env.step(zero_action)

print("\nDone. Close window to exit.")

if args.gui:
    try:
        while app_launcher.app.is_running():
            app_launcher.app.update()
    except KeyboardInterrupt:
        pass

env.close()
app_launcher.app.close()
