# Adapted from: LeRobot (https://github.com/huggingface/lerobot)
# Original license: Apache-2.0
# Changes: gRPC split — env runs in isaaclab-env, SACPolicy runs in lerobot-env
#   via sac_server.py. Success detection via sim reward milestones.

"""Evaluate trained SAC policy in Isaac Lab simulation.

Policy runs in a separate process (lerobot-env, Python 3.12) via gRPC sac_server.
Start the server automatically with --auto-server, or manually:
    act-lerobot
    python scripts/train/sac_server.py --port 8082

Then run eval:
    act-isaac
    python scripts/eval/eval_sac_policy.py \
        --checkpoint outputs/sac_v1/best/pretrained_model

Usage:
    # Basic evaluation (auto-starts server)
    python scripts/eval/eval_sac_policy.py \
        --checkpoint outputs/sac_v1/best/pretrained_model --auto-server

    # With GUI
    python scripts/eval/eval_sac_policy.py \
        --checkpoint outputs/sac_v1/best/pretrained_model --auto-server --gui

    # Disable domain randomization for clean eval
    python scripts/eval/eval_sac_policy.py \
        --checkpoint outputs/sac_v1/best/pretrained_model --auto-server --no-domain-rand

Key flags:
    --checkpoint PATH      Path to pretrained_model directory (required)
    --env NAME             Task name (default: figure_shape_placement)
    --episodes N           Number of evaluation episodes (default: 20)
    --max-steps N          Max steps per episode (default: 300)
    --image-size N         Resize images to NxN: 0=no resize, 128, 256 (default: 128)
    --no-domain-rand       Disable all DR (light, physics, camera noise, distractors)
    --auto-server          Auto-start sac_server.py in lerobot-env
    --gui                  Run with visualization
"""

# Section 1: argparse + AppLauncher
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate SAC policy in simulation")
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment name (default: figure_shape_placement)")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to pretrained_model directory",
)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gui", action="store_true", help="Run with GUI")
parser.add_argument(
    "--image-size",
    type=int,
    default=128,
    choices=[0, 128, 256],
    help="Resize images to NxN (0 = no resize, uses original 480x640)",
)
parser.add_argument("--no-randomize-light", action="store_true", help="Disable light randomization")
parser.add_argument("--no-randomize-physics", action="store_true", help="Disable physics randomization")
parser.add_argument("--no-camera-noise", action="store_true", help="Disable camera noise augmentation")
parser.add_argument("--no-distractors", action="store_true", help="Disable distractor objects")
parser.add_argument("--no-domain-rand", action="store_true", help="Disable all domain randomization")
parser.add_argument("--auto-server", action="store_true",
                    help="Auto-start sac_server.py in lerobot-env")
parser.add_argument("--server-host", type=str, default="127.0.0.1",
                    help="SAC server host (default: 127.0.0.1)")
parser.add_argument("--server-port", type=int, default=8082,
                    help="SAC server port (default: 8082)")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)

# Section 2: Imports
from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

import numpy as np
import torch
from tqdm import tqdm

from so101_lab.transport.sac_client import SACTrainingClient
from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv
from so101_lab.rl.domain_rand import apply_domain_rand_flags
from so101_lab.tasks import get_rl_task


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Start server if requested
    server_proc = None
    if args.auto_server:
        from so101_lab.utils.policy_server import start_sac_server
        server_proc = start_sac_server(port=args.server_port, host=args.server_host)

    # Connect to SAC server and load policy
    client = SACTrainingClient(host=args.server_host, port=args.server_port)
    client.connect()
    print(f"Connected to SAC server at {args.server_host}:{args.server_port}")

    init_config = {
        "resume": args.checkpoint,
        "image_size": args.image_size,
        "no_online": True,
        "no_offline": True,
    }
    client.init(init_config)
    client.set_mode("eval")
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Create env
    RLEnvCfgClass = get_rl_task(args.env)
    env_cfg = RLEnvCfgClass()
    env_cfg.scene.num_envs = 1
    apply_domain_rand_flags(env_cfg, args)
    env = IsaacLabGymEnv(env_cfg=env_cfg)

    # Run episodes
    results = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        pbar = tqdm(range(args.max_steps), desc=f"Ep {ep + 1}/{args.episodes}", leave=False)
        for step in pbar:
            action = client.sample_action_deterministic(obs)
            action = np.clip(action, -100, 100)
            action[5] = np.clip(action[5], 0, 100)  # gripper [0, 100]

            obs, _, terminated, truncated, _ = env.step(action)
            ep_steps += 1

            # Success detection via sim reward milestones
            reward_details = env.get_reward_details()
            milestone = reward_details.get("reward/milestone_placed_weighted", 0.0)
            ep_reward += milestone

            pbar.set_postfix({"placed": f"{ep_reward:.0f}"})

            if ep_reward > 0 or terminated or truncated:
                break

        pbar.close()
        success = ep_reward > 0
        results.append({"reward": ep_reward, "steps": ep_steps, "success": success})

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep + 1}/{args.episodes}: {status} | reward={ep_reward:.1f} | steps={ep_steps}")

    # Summary
    successes = sum(r["success"] for r in results)
    avg_reward = np.mean([r["reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])

    print(f"\n{'='*50}")
    print(f"SAC Evaluation Summary")
    print(f"{'='*50}")
    print(f"  Checkpoint:   {args.checkpoint}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Success rate: {successes}/{args.episodes} ({successes/args.episodes:.0%})")
    print(f"  Avg reward:   {avg_reward:.2f}")
    print(f"  Avg steps:    {avg_steps:.0f}")
    print(f"{'='*50}")

    # Cleanup
    client.close()
    if server_proc is not None:
        server_proc.terminate()
        server_proc.wait()
    env.close()


if __name__ == "__main__":
    main(args)
    app_launcher.app.close()
