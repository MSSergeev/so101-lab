#!/usr/bin/env python3
"""Evaluate trained Diffusion Policy in Isaac Lab simulation.

This script runs a trained diffusion policy in simulation and collects statistics
on success rate, episode length, and optionally saves episode recordings.

Usage:
    # Basic evaluation (best checkpoint, 10 episodes)
    python scripts/eval/eval_diffusion_policy.py --checkpoint outputs/diffusion_v1

    # Evaluate specific checkpoint
    python scripts/eval/eval_diffusion_policy.py --checkpoint outputs/diffusion_v1 --step 15000

    # Evaluate with GUI and preview
    python scripts/eval/eval_diffusion_policy.py --checkpoint outputs/diffusion_v1 --gui --preview

    # Save failed episodes for analysis
    python scripts/eval/eval_diffusion_policy.py --checkpoint outputs/diffusion_v1 --save-episodes fail

    # Full evaluation run
    python scripts/eval/eval_diffusion_policy.py \\
        --checkpoint outputs/diffusion_v1 \\
        --episodes 100 \\
        --save-episodes all \\
        --output outputs/eval/run_001

Controls (Isaac Sim window or eval_viewer.py):
    Space  - Pause/Resume
    N      - Next episode (skip current)
    R      - Restart current episode
    Escape - Quit evaluation
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments before Isaac Sim launch
parser = argparse.ArgumentParser(description="Evaluate Diffusion policy in simulation")

# Checkpoint selection
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to training output directory",
)
parser.add_argument(
    "--use-best",
    action="store_true",
    default=True,
    help="Load best checkpoint (default)",
)
parser.add_argument(
    "--use-latest",
    action="store_true",
    help="Load latest checkpoint from root directory",
)
parser.add_argument(
    "--step",
    type=int,
    default=None,
    help="Load specific checkpoint step (e.g., --step 15000)",
)

# Environment
parser.add_argument(
    "--env",
    type=str,
    default="figure_shape_placement",
    help="Environment name (default: figure_shape_placement)",
)

# Evaluation parameters
parser.add_argument(
    "--episodes",
    type=int,
    default=10,
    help="Number of evaluation episodes (default: 10)",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=1000,
    help="Max steps per episode (default: 1000)",
)
parser.add_argument(
    "--timeout-s",
    type=float,
    default=None,
    help="Optional timeout in seconds (steps take priority)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (default: random)",
)
parser.add_argument(
    "--episode-seed",
    type=int,
    default=None,
    help="Run single episode with this exact seed (useful for reproducing specific scenarios)",
)

# Recording
parser.add_argument(
    "--save-episodes",
    type=str,
    choices=["all", "success", "fail", "none"],
    default="none",
    help="Which episodes to save (default: none)",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for evaluation results (auto-generated if not specified)",
)
parser.add_argument(
    "--crf",
    type=int,
    default=23,
    help="Video quality for saved episodes (0-51, lower=better, default: 23)",
)
parser.add_argument(
    "--gop",
    type=str,
    default="auto",
    help="GOP size / keyframe interval (default: auto, use integer for fixed value)",
)

# Display
parser.add_argument(
    "--gui",
    action="store_true",
    help="Run with GUI (default: headless)",
)
parser.add_argument(
    "--preview",
    action="store_true",
    help="Enable camera preview window",
)

# Frequencies (same defaults as record_episodes.py)
parser.add_argument("--physics-hz", type=int, default=120,
                    help="Physics simulation frequency")
parser.add_argument("--policy-hz", type=int, default=30,
                    help="Policy/control frequency")
parser.add_argument("--render-hz", type=int, default=30,
                    help="Rendering frequency")
parser.add_argument("--preview-hz", type=int, default=30,
                    help="Preview update frequency (0 = every step)")

# Policy parameters
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=None,
    help="Override n_action_steps (default: from model config)",
)
parser.add_argument(
    "--num-inference-steps",
    type=int,
    default=None,
    help="Override number of diffusion denoising steps (default: from model config)",
)

# Domain randomization
parser.add_argument("--randomize-light", action="store_true",
                    help="Randomize sun light intensity/color on each reset")

# Add AppLauncher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Handle checkpoint selection logic
if args.use_latest:
    args.use_best = False
if args.step is not None:
    args.use_best = False
    args.use_latest = False

# Set headless based on --gui flag
if not args.gui:
    args.headless = True

# Enable cameras always (policy uses images)
args.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Disable rate limiting for maximum simulation speed
from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

# ═══════════════════════════════════════════════════════════════════════════════
# Imports after Isaac Sim launch
# ═══════════════════════════════════════════════════════════════════════════════

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from so101_lab.tasks import get_task
from so101_lab.policies.diffusion import DiffusionInference
from so101_lab.utils.checkpoint import resolve_checkpoint_path
from so101_lab.utils.scene_state import extract_scene_state

from so101_lab.utils.shm_preview import (
    cleanup_command_file,
    cleanup_shm,
    launch_viewer,
    read_command,
    stop_viewer,
    write_camera_to_shm,
    write_status_to_shm,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def setup_keyboard_listener():
    """Setup keyboard listener for eval controls.

    Returns (pending_commands dict, resources tuple for cleanup).
    """
    import carb
    import omni

    pending = {"next": False, "restart": False, "quit": False, "pause": False}

    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()

    def on_keyboard_event(event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            key_name = event.input.name
            if key_name == "N":
                pending["next"] = True
            elif key_name == "R":
                pending["restart"] = True
            elif key_name == "ESCAPE":
                pending["quit"] = True
            elif key_name == "SPACE":
                pending["pause"] = True

    keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)

    return pending, (input_iface, keyboard, keyboard_sub)


def poll_command(pending_commands, use_preview):
    """Read command from preview viewer and/or Isaac Sim keyboard.

    Returns command string or None: "pause", "quit", "next", "restart".
    """
    cmd = read_command() if use_preview else None

    for key in ("quit", "next", "restart", "pause"):
        if pending_commands[key]:
            cmd = key
            pending_commands[key] = False
            break

    return cmd


def run_episode(env, policy, device, episode_seed, recorder, ep_num, prev_successes,
                pending_commands, preview_interval):
    """Run one evaluation episode.

    Returns dict with keys:
        outcome: "done" | "skip" | "restart" | "quit"
        success: bool
        steps: int
        elapsed: float
        initial_state: dict
        final_state: dict

    IMPORTANT: Scene state is snapshotted BEFORE env.step() because Isaac Lab
    auto-resets the env after done=True, which overwrites object positions.
    """
    obs, info = env.reset(seed=episode_seed)
    policy.reset()

    # Zero-action step to update cameras after reset
    zero_action = torch.zeros((1, 6), dtype=torch.float32, device=device)
    obs, _, _, _, _ = env.step(zero_action)

    initial_joint_pos = obs["policy"]["joint_pos"][0].cpu().numpy()

    if recorder is not None:
        recorder.set_task(f"eval_{args.env}")
        recorder.on_reset(obs)

    initial_state = extract_scene_state(env, joint_pos_override=initial_joint_pos)
    initial_state["env_seed"] = episode_seed

    done = False
    success = False
    step = 0
    start_time = time.time()
    last_scene_state = initial_state
    paused = False

    if args.preview:
        write_status_to_shm({"state": "EVAL", "episode": ep_num, "total_episodes": args.episodes, "step": 0, "max_steps": args.max_steps, "seed": episode_seed, "success": None, "successes": prev_successes, "status_text": "Running...", "timestamp": time.time()})

    while not done and step < args.max_steps:
        cmd = poll_command(pending_commands, args.preview)

        # --- Command handling ---
        if cmd == "pause":
            paused = not paused
            if paused:
                print(f"[PAUSED] Press Space to resume")
                if args.preview:
                    write_status_to_shm({"state": "EVAL", "episode": ep_num, "total_episodes": args.episodes, "step": step, "max_steps": args.max_steps, "seed": episode_seed, "success": None, "successes": prev_successes, "status_text": "PAUSED", "timestamp": time.time()})
            else:
                print(f"[RESUMED]")
            continue
        elif paused:
            if args.gui:
                simulation_app.update()
            time.sleep(0.05)
            continue
        elif cmd == "quit":
            print("\n[QUIT] Evaluation stopped by user")
            return {"outcome": "quit", "success": False, "steps": step, "elapsed": time.time() - start_time, "initial_state": initial_state, "final_state": last_scene_state}
        elif cmd == "next":
            print(f"[NEXT] Skipping episode {ep_num}")
            return {"outcome": "skip", "success": False, "steps": step, "elapsed": time.time() - start_time, "initial_state": initial_state, "final_state": last_scene_state}
        elif cmd == "restart":
            print(f"[RESTART] Restarting episode {ep_num}")
            return {"outcome": "restart", "success": False, "steps": step, "elapsed": time.time() - start_time, "initial_state": initial_state, "final_state": last_scene_state}

        if args.timeout_s and (time.time() - start_time) > args.timeout_s:
            break

        # --- Extract observations ---
        policy_data = obs["policy"]
        policy_obs = {
            "joint_pos": policy_data["joint_pos"].cpu().numpy()[0],
            "images": {},
        }

        if "top" in policy_data:
            img_top = policy_data["top"][0].cpu().numpy()
            policy_obs["images"]["top"] = img_top
            if args.preview and step % preview_interval == 0:
                write_camera_to_shm("top", img_top)
        if "wrist" in policy_data:
            img_wrist = policy_data["wrist"][0].cpu().numpy()
            policy_obs["images"]["wrist"] = img_wrist
            if args.preview and step % preview_interval == 0:
                write_camera_to_shm("wrist", img_wrist)

        # --- Get action ---
        action = policy.select_action(policy_obs)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)

        # Snapshot state BEFORE env.step — Isaac Lab auto-resets after done=True,
        # overwriting object positions. This is the last valid scene state.
        last_scene_state = extract_scene_state(
            env, joint_pos_override=policy_data["joint_pos"].cpu().numpy()[0]
        )

        if recorder is not None:
            recorder.on_step(obs, action_tensor)
        obs, reward, terminated, truncated, info = env.step(action_tensor)
        step += 1

        if terminated[0].item():
            success = True
            done = True
            print(f"  [SUCCESS] Task completed at step {step}")
        elif truncated[0].item():
            done = True

        if args.preview and step % preview_interval == 0:
            write_status_to_shm({"state": "EVAL", "episode": ep_num, "total_episodes": args.episodes, "step": step, "max_steps": args.max_steps, "seed": episode_seed, "success": None, "successes": prev_successes, "status_text": "Running...", "timestamp": time.time()})

    return {"outcome": "done", "success": success, "steps": step, "elapsed": time.time() - start_time, "initial_state": initial_state, "final_state": last_scene_state}


def save_eval_results(registry, successes, episode_lengths, episode_times, output_dir, quit_requested):
    """Compute summary, print results, save JSON files."""
    if not successes:
        print("\n[INFO] No episodes completed, nothing to save")
        return

    success_rate = sum(successes) / len(successes) * 100
    avg_steps = sum(episode_lengths) / len(episode_lengths)
    avg_time = sum(episode_times) / len(episode_times)
    avg_sim_time = avg_steps / args.policy_hz
    realtime_factor = avg_sim_time / avg_time if avg_time > 0 else 0

    success_steps = [l for l, s in zip(episode_lengths, successes) if s]
    fail_steps = [l for l, s in zip(episode_lengths, successes) if not s]

    summary = {
        "total_episodes": len(successes),
        "requested_episodes": args.episodes,
        "completed": not quit_requested,
        "successes": sum(successes),
        "failures": len(successes) - sum(successes),
        "success_rate": round(success_rate, 2),
        "avg_steps": round(avg_steps, 1),
        "avg_time_s": round(avg_time, 2),
        "avg_sim_time_s": round(avg_sim_time, 2),
        "realtime_factor": round(realtime_factor, 2),
        "avg_steps_success": round(sum(success_steps) / len(success_steps), 1) if success_steps else None,
        "avg_steps_fail": round(sum(fail_steps) / len(fail_steps), 1) if fail_steps else None,
    }
    registry["summary"] = summary

    print("\n" + "=" * 60)
    completed_text = f"{len(successes)}/{args.episodes}" if quit_requested else f"{args.episodes}"
    print(f"EVALUATION RESULTS ({completed_text} episodes)")
    print("=" * 60)
    print(f"Success rate: {success_rate:.1f}% ({sum(successes)}/{len(successes)})")
    print(f"Avg steps: {avg_steps:.1f}")
    print(f"Avg time: {avg_time:.2f}s (wall), {avg_sim_time:.2f}s (sim)")
    print(f"Realtime factor: {realtime_factor:.2f}x")
    if success_steps:
        print(f"Avg steps (success): {summary['avg_steps_success']}")
    if fail_steps:
        print(f"Avg steps (fail): {summary['avg_steps_fail']}")

    with open(output_dir / "registry.json", "w") as f:
        json.dump(registry, f, indent=2)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "config.json", "w") as f:
        json.dump(registry["config"], f, indent=2)

    print(f"\nResults saved to: {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Set seeds
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"Using random seed: {args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    episode_rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint, args.use_best, args.use_latest, args.step
    )
    print(f"Loading policy from: {checkpoint_path}")

    # Load policy
    policy = DiffusionInference(checkpoint_path=str(checkpoint_path), device=device)

    if args.n_action_steps is not None:
        old_val = policy.config.n_action_steps
        policy.config.n_action_steps = args.n_action_steps
        policy.policy.config.n_action_steps = args.n_action_steps
        policy.policy.reset()
        print(f"  n_action_steps: {old_val} -> {args.n_action_steps}")

    if args.num_inference_steps is not None:
        old_val = policy.policy.diffusion.num_inference_steps
        policy.policy.diffusion.num_inference_steps = args.num_inference_steps
        print(f"  num_inference_steps: {old_val} -> {args.num_inference_steps}")

    # Check if policy_hz matches training fps
    train_config_path = checkpoint_path / "train_config.json"
    if not train_config_path.exists():
        train_config_path = checkpoint_path.parent / "train_config.json"

    training_fps = None
    if train_config_path.exists():
        with open(train_config_path) as f:
            train_config = json.load(f)
        training_fps = train_config.get("fps")

    if training_fps is not None and training_fps != args.policy_hz:
        print(f"\n[WARNING] Policy trained at {training_fps} Hz, but eval running at {args.policy_hz} Hz")
        print(f"          Consider using --policy-hz {training_fps} for consistent behavior\n")

    # Create environment
    print(f"Creating environment: {args.env}")
    EnvClass, EnvCfgClass = get_task(args.env)

    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.dt = 1.0 / args.physics_hz
    env_cfg.decimation = args.physics_hz // args.policy_hz
    env_cfg.sim.render_interval = args.physics_hz // args.render_hz

    camera_period = 1.0 / args.render_hz
    if hasattr(env_cfg.scene, 'top'):
        env_cfg.scene.top.update_period = camera_period
    if hasattr(env_cfg.scene, 'wrist'):
        env_cfg.scene.wrist.update_period = camera_period

    if hasattr(env_cfg, 'randomize_light'):
        env_cfg.randomize_light = args.randomize_light

    env_cfg.terminate_on_success = True
    env_cfg.episode_length_s = args.max_steps / args.policy_hz

    render_mode = "human" if args.gui else None
    env = EnvClass(env_cfg, render_mode=render_mode)

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("outputs/eval") / f"{args.env}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Registry
    registry = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "env": args.env,
            "num_episodes": args.episodes,
            "max_steps": args.max_steps,
            "timeout_s": args.timeout_s,
            "initial_seed": args.seed,
            "physics_hz": args.physics_hz,
            "policy_hz": args.policy_hz,
            "training_fps": training_fps,
            "timestamp": datetime.now().isoformat()
        },
        "episodes": []
    }

    # Episode recording
    recorder = None
    if args.save_episodes != "none":
        from so101_lab.data.lerobot_dataset import LeRobotDatasetWriter
        from so101_lab.data.collector import RecordingManager

        gop_value = None if args.gop == "auto" else int(args.gop)
        episodes_dir = output_dir / "episodes"
        dataset_writer = LeRobotDatasetWriter(
            output_dir=episodes_dir, fps=args.policy_hz, crf=args.crf, gop=gop_value,
        )
        recorder = RecordingManager(dataset_writer, env)
        print(f"Episode recording enabled: {args.save_episodes} -> {episodes_dir}")

    # Preview
    viewer_proc = None
    preview_interval = 1
    if args.preview:
        print("Launching camera preview...")
        viewer_proc = launch_viewer("eval_viewer.py")
        cleanup_command_file()
        time.sleep(1.0)
        if args.preview_hz > 0:
            preview_interval = max(1, args.policy_hz // args.preview_hz)

    # Keyboard listener
    pending_commands, keyboard_resources = setup_keyboard_listener()

    print(f"\nRunning {args.episodes} evaluation episodes...")
    print("=" * 60)
    print("Controls: Space=Pause  N=Next  R=Restart  Esc=Quit")
    print("=" * 60)

    successes = []
    episode_lengths = []
    episode_times = []

    ep = 0
    quit_requested = False
    reuse_seed = None

    if args.episode_seed is not None:
        args.episodes = 1
        print(f"Running single episode with seed: {args.episode_seed}")

    try:
        while ep < args.episodes and not quit_requested:
            if args.episode_seed is not None:
                episode_seed = args.episode_seed
            elif reuse_seed is not None:
                episode_seed = reuse_seed
                reuse_seed = None
            else:
                episode_seed = int(episode_rng.integers(0, 2**31))

            result = run_episode(
                env, policy, device, episode_seed, recorder, ep + 1,
                sum(successes), pending_commands, preview_interval,
            )

            if result["outcome"] == "restart":
                if recorder is not None:
                    recorder.on_episode_end(success=False)
                reuse_seed = episode_seed
                continue

            if result["outcome"] == "skip":
                if recorder is not None:
                    recorder.on_episode_end(success=False)
                ep += 1
                continue

            if result["outcome"] == "quit":
                if recorder is not None:
                    recorder.on_episode_end(success=False)
                quit_requested = True
                break

            success = result["success"]
            successes.append(success)
            episode_lengths.append(result["steps"])
            episode_times.append(result["elapsed"])

            should_save = (
                args.save_episodes == "all" or
                (args.save_episodes == "success" and success) or
                (args.save_episodes == "fail" and not success)
            )
            if recorder is not None:
                recorder.on_episode_end(success=should_save)

            registry["episodes"].append({
                "id": ep,
                "success": success,
                "steps": result["steps"],
                "duration_s": round(result["elapsed"], 2),
                "initial_state": result["initial_state"],
                "final_state": result["final_state"],
                "saved": should_save,
            })

            status = "SUCCESS" if success else "FAIL"
            rate = sum(successes) / len(successes) * 100
            print(f"Episode {ep + 1:3d}/{args.episodes}: {status:7s} | steps: {result['steps']:4d} | time: {result['elapsed']:.1f}s | rate: {rate:.0f}%")

            if args.preview:
                write_status_to_shm({"state": "EVAL", "episode": ep + 1, "total_episodes": args.episodes, "step": result["steps"], "max_steps": args.max_steps, "seed": episode_seed, "success": success, "successes": sum(successes), "status_text": status, "timestamp": time.time()})

            ep += 1

        save_eval_results(registry, successes, episode_lengths, episode_times, output_dir, quit_requested)

    finally:
        input_iface, keyboard, keyboard_sub = keyboard_resources
        if keyboard_sub is not None:
            input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)

        if recorder is not None:
            recorder.dataset.close()
        if viewer_proc:
            stop_viewer(viewer_proc)
        cleanup_shm()
        if args.preview:
            cleanup_command_file()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
