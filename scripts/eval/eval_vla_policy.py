"""Evaluate VLA policy (SmolVLA, Pi0, GR00T) in Isaac Lab simulation.

Policy runs in a separate process (lerobot-env, Python 3.12) via gRPC.
Start the policy server first:
    act-lerobot
    python scripts/eval/smolvla_server.py --port 8080

Then run eval:
    act-isaac
    python scripts/eval/eval_vla_policy.py \
        --checkpoint outputs/smolvla_figure_v5/checkpoints/last/pretrained_model \
        --task "Place the cube into the matching slot on the platform" \
        --episodes 20
"""

# Section 1: argparse + AppLauncher
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate VLA policy in simulation")
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment name (default: figure_shape_placement)")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to pretrained_model directory or HuggingFace model ID",
)
parser.add_argument(
    "--task",
    type=str,
    default="Place the cube into the matching slot on the platform",
    help="Language instruction for VLA policy",
)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--max-steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=None, help="Random seed (random if not set)")
parser.add_argument("--episode-seed", type=int, default=None,
                    help="Run single episode with this exact seed (useful for reproducing specific scenarios)")
parser.add_argument("--start-paused", action="store_true",
                    help="Start paused before first episode (press Space in viewer to begin)")
parser.add_argument("--gui", action="store_true", help="Run with GUI")
parser.add_argument("--no-randomize-light", action="store_true", help="Disable light randomization")
parser.add_argument("--no-randomize-physics", action="store_true", help="Disable physics randomization")
parser.add_argument("--no-camera-noise", action="store_true", help="Disable camera noise augmentation")
parser.add_argument("--no-distractors", action="store_true", help="Disable distractor objects")
parser.add_argument("--no-domain-rand", action="store_true", help="Disable all domain randomization")
parser.add_argument("--preview", action="store_true", help="Enable camera preview window")
parser.add_argument("--output", type=str, default=None,
                    help="Output directory for summary.json (optional)")
parser.add_argument("--server", type=str, default="127.0.0.1",
                    help="Policy server host (default: 127.0.0.1)")
parser.add_argument("--port", type=int, default=8080,
                    help="Policy server port (default: 8080)")
parser.add_argument("--auto-server", action="store_true",
                    help="Auto-start policy server (for standalone use; do not use with sweep)")
parser.add_argument("--n-action-steps", type=int, default=15,
                    help="Actions to execute per inference call (default: 15)")
parser.add_argument("--noise-prior", type=str, default=None,
                    help="Path to noise_prior.pt — passed to policy server for patching")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)

# Section 2: Imports
from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from so101_lab.policies.grpc_client import GrpcPolicyClient
from so101_lab.utils.policy_server import start_policy_server
from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv
from so101_lab.rl.domain_rand import apply_domain_rand_flags
from so101_lab.tasks import get_rl_task
from so101_lab.utils.shm_preview import (
    cleanup_command_file,
    cleanup_shm,
    launch_viewer,
    read_command,
    stop_viewer,
    write_camera_to_shm,
    write_status_to_shm,
)

ACTION_DIM = 6


def main(args):
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 2**31))
    print(f"Seed: {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Start server if requested
    server_proc = None
    if args.auto_server:
        server_proc = start_policy_server(port=args.port, host=args.server)

    # Connect to policy server
    policy_client = GrpcPolicyClient(
        checkpoint=args.checkpoint,
        host=args.server,
        port=args.port,
        task=args.task,
        actions_per_chunk=args.n_action_steps,
        noise_prior=args.noise_prior,
    )
    print(f"Connecting to policy server at {args.server}:{args.port} ...")
    policy_client.connect()
    print(f"Connected. Checkpoint: {args.checkpoint}")
    print(f"Task: {args.task}")

    # Create env with domain randomization flags
    RLEnvCfgClass = get_rl_task(args.env)
    env_cfg = RLEnvCfgClass()
    env_cfg.scene.num_envs = 1
    apply_domain_rand_flags(env_cfg, args)
    env = IsaacLabGymEnv(env_cfg=env_cfg)

    # Launch preview if requested
    viewer_proc = None
    if args.preview:
        print("Launching camera preview...")
        viewer_proc = launch_viewer("eval_viewer.py")
        cleanup_command_file()
        import time
        time.sleep(1.0)

    # Run episodes
    import time as _time
    episode_rng = np.random.default_rng(args.seed)
    results = []
    quit_requested = False
    reuse_seed = None
    ep = 0

    if args.episode_seed is not None:
        args.episodes = 1
        print(f"Running single episode with seed: {args.episode_seed}")

    # Start paused — wait for Space press before first episode
    if args.start_paused and args.preview:
        print("[PAUSED] Press Space in viewer to start...")
        write_status_to_shm({"state": "EVAL", "episode": 0, "total_episodes": args.episodes, "step": 0, "max_steps": args.max_steps, "successes": 0, "status_text": "PAUSED — press Space to start"})
        while True:
            cmd = read_command()
            if cmd == "pause":
                break
            if cmd == "quit":
                quit_requested = True
                break
            _time.sleep(0.05)

    while ep < args.episodes and not quit_requested:
        if args.episode_seed is not None:
            episode_seed = args.episode_seed
        elif reuse_seed is not None:
            episode_seed = reuse_seed
            reuse_seed = None
        else:
            episode_seed = int(episode_rng.integers(0, 2**31))

        obs, _ = env.reset(seed=episode_seed)
        policy_client.reset()
        ep_reward = 0.0
        skip_episode = False
        restart_episode = False
        paused = False

        if args.preview:
            write_status_to_shm({"state": "EVAL", "episode": ep + 1, "total_episodes": args.episodes, "step": 0, "max_steps": args.max_steps, "successes": sum(r["success"] for r in results), "status_text": "Running..."})

        step = 0
        pbar = tqdm(total=args.max_steps, desc=f"Ep {ep + 1}/{args.episodes}", leave=False)
        while step < args.max_steps:
            # Handle commands from viewer
            if args.preview:
                cmd = read_command()
                if cmd == "quit":
                    print("\n[QUIT] Evaluation stopped by user")
                    quit_requested = True
                    break
                elif cmd == "next":
                    print(f"[NEXT] Skipping episode {ep + 1}")
                    skip_episode = True
                    break
                elif cmd == "restart":
                    print(f"[RESTART] Restarting episode {ep + 1}")
                    restart_episode = True
                    break
                elif cmd == "pause":
                    paused = not paused
                    if paused:
                        print(f"[PAUSED] Press Space to resume")
                        write_status_to_shm({"state": "EVAL", "episode": ep + 1, "total_episodes": args.episodes, "step": step, "max_steps": args.max_steps, "successes": sum(r["success"] for r in results), "status_text": "PAUSED"})
                    else:
                        print(f"[RESUMED]")

            if paused:
                if args.gui:
                    app_launcher.app.update()
                _time.sleep(0.05)
                continue

            action_np = policy_client.select_action(obs)
            action_np = np.clip(action_np, -100, 100)
            action_np[5] = np.clip(action_np[5], 0, 100)  # gripper [0, 100]

            obs, _, terminated, truncated, _ = env.step(action_np)
            step += 1
            pbar.update(1)

            # Update preview
            if args.preview:
                if "observation.images.top" in obs:
                    write_camera_to_shm("top", obs["observation.images.top"])
                if "observation.images.wrist" in obs:
                    write_camera_to_shm("wrist", obs["observation.images.wrist"])
                write_status_to_shm({"state": "EVAL", "episode": ep + 1, "total_episodes": args.episodes, "step": step, "max_steps": args.max_steps, "successes": sum(r["success"] for r in results), "status_text": "Running..."})

            # Success detection via sim reward milestones
            reward_details = env.get_reward_details()
            milestone = reward_details.get("reward/milestone_placed_weighted", 0.0)
            ep_reward += milestone

            pbar.set_postfix({"placed": f"{ep_reward:.0f}"})

            if ep_reward > 0 or terminated or truncated:
                break

        pbar.close()

        if restart_episode:
            reuse_seed = episode_seed
            continue
        if skip_episode:
            ep += 1
            continue
        if quit_requested:
            break

        success = ep_reward > 0
        results.append({"reward": ep_reward, "steps": step, "success": success, "seed": episode_seed})

        status = "SUCCESS" if success else "FAIL"
        print(f"  Episode {ep + 1}/{args.episodes}: {status} | reward={ep_reward:.1f} | steps={step}")

        if args.preview:
            write_status_to_shm({"state": "EVAL", "episode": ep + 1, "total_episodes": args.episodes, "step": step, "max_steps": args.max_steps, "success": success, "successes": sum(r["success"] for r in results), "status_text": status})

        ep += 1

    # Summary
    n_completed = len(results)
    if n_completed > 0:
        n_successes = sum(r["success"] for r in results)
        avg_reward = np.mean([r["reward"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])
        success_rate = round(n_successes / n_completed * 100, 2)
        success_steps = [r["steps"] for r in results if r["success"]]
        fail_steps = [r["steps"] for r in results if not r["success"]]

        print(f"\n{'='*50}")
        print(f"VLA Evaluation Summary")
        print(f"{'='*50}")
        print(f"  Checkpoint:   {args.checkpoint}")
        print(f"  Task:         {args.task}")
        print(f"  Seed:         {args.seed}")
        print(f"  Episodes:     {n_completed}/{args.episodes}")
        print(f"  Success rate: {n_successes}/{n_completed} ({success_rate:.0f}%)")
        print(f"  Avg reward:   {avg_reward:.2f}")
        print(f"  Avg steps:    {avg_steps:.0f}")
        if success_steps:
            print(f"  Avg steps (success): {np.mean(success_steps):.0f}")
        if fail_steps:
            print(f"  Avg steps (fail):    {np.mean(fail_steps):.0f}")
        print(f"{'='*50}")

        # Per-episode table with seeds
        print(f"\nPer-episode results:")
        print(f"  {'Ep':>3}  {'Seed':>12}  {'Result':>7}  {'Steps':>5}")
        for i, r in enumerate(results):
            res = "OK" if r["success"] else "FAIL"
            print(f"  {i+1:3d}  {r['seed']:12d}  {res:>7}  {r['steps']:5d}")

        # Save summary.json if --output specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "total_episodes": n_completed,
                "successes": n_successes,
                "success_rate": success_rate,
                "avg_steps": round(float(avg_steps), 1),
                "avg_steps_success": round(float(np.mean(success_steps)), 1) if success_steps else None,
                "avg_steps_fail": round(float(np.mean(fail_steps)), 1) if fail_steps else None,
                "avg_reward": round(float(avg_reward), 2),
                "checkpoint": args.checkpoint,
                "seed": args.seed,
                "env": args.env,
                "episodes": [{"seed": r["seed"], "success": r["success"], "steps": r["steps"]} for r in results],
            }
            summary_path = output_dir / "summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to {summary_path}")
    else:
        print("\nNo episodes completed.")

    # Cleanup
    policy_client.close()
    if server_proc is not None:
        server_proc.terminate()
        server_proc.wait()
    if args.preview:
        stop_viewer(viewer_proc)
        cleanup_shm()

    env.close()


if __name__ == "__main__":
    main(args)
    app_launcher.app.close()
