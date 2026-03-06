# Runs in: isaaclab-env (Python 3.11)
# gRPC two-process split — client (isaaclab-env) keeps env loop, server (lerobot-env) owns policy + buffers

"""Train SAC policy with composite reward — gRPC split version.

Client (isaaclab-env, Python 3.11): env loop, HIL, preview, tracker.
Server (lerobot-env, Python 3.12): SACPolicy, replay buffers, reward models, SAC update.

Usage:
    # With auto-server:
    python scripts/train/train_sac_composite_grpc.py \
        --reward-mode composite --reward-model outputs/reward_classifier_v2/best \
        --num-steps 50000 --output outputs/sac_grpc_v1 \
        --auto-server --headless

    # Or start server manually first:
    # Terminal 1 (lerobot-env):
    #   python scripts/train/sac_server.py --port 8082
    # Terminal 2 (isaaclab-env):
    python scripts/train/train_sac_composite_grpc.py \
        --reward-mode composite --reward-model outputs/reward_classifier_v2/best \
        --num-steps 50000 --output outputs/sac_grpc_v1 --headless
"""

# Section 1: argparse + AppLauncher (before Isaac Sim)
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC policy with composite reward (gRPC)")

# Environment
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment name (default: figure_shape_placement)")
parser.add_argument("--seed", type=int, default=42)

# Reward mode (passed to server)
parser.add_argument(
    "--reward-mode",
    type=str,
    default="composite",
    choices=["composite", "sim_only", "classifier_only", "vip_only"],
    help="Reward mode: composite (sim+classifier), sim_only, classifier_only, vip_only",
)
parser.add_argument(
    "--reward-model",
    type=str,
    default="outputs/reward_classifier_v2/best",
    help="Path to trained reward classifier (for composite/classifier_only)",
)

# Reward weights
parser.add_argument("--w-classifier", type=float, default=1.0, help="Classifier reward weight")
parser.add_argument("--success-bonus", type=float, default=10.0, help="One-time terminal bonus on first classifier success")
parser.add_argument("--w-drop", type=float, default=-10.0, help="Drop penalty weight")
parser.add_argument("--w-jerky-motion", type=float, default=-0.5, dest="w_jerky_motion", help="Jerky motion penalty weight")
parser.add_argument("--w-smoothness", type=float, default=-0.1, help="Action smoothness penalty weight")
parser.add_argument("--w-time", type=float, default=-0.05, help="Time penalty weight")
parser.add_argument("--w-distance", type=float, default=0.5, help="Distance cube→slot reward weight")
parser.add_argument("--w-distance-gripper", type=float, default=0.5, help="Distance gripper→cube reward weight")
parser.add_argument("--w-milestone", type=float, default=150.0, help="Milestone reward weight")
parser.add_argument("--w-table-contact", type=float, default=-0.1, help="Table contact penalty weight")
parser.add_argument(
    "--sim-rewards",
    type=str,
    default="all",
    help="Sim rewards to enable: 'all', 'none', 'penalties', or comma-separated names",
)

# VIP reward (passed to server)
parser.add_argument("--vip-goal-dataset", type=str, default=None, help="LeRobot dataset path for VIP goal images")
parser.add_argument("--w-vip", type=float, default=1.0, help="VIP reward weight")
parser.add_argument("--vip-camera", type=str, default="observation.images.top", help="Camera key for VIP embedding")
parser.add_argument("--vip-goal-mode", type=str, default="mean", choices=["mean", "min"])
parser.add_argument("--vip-use-labeled", action="store_true")
parser.add_argument("--vip-label-dataset", type=str, default=None)
parser.add_argument("--vip-normalize", action="store_true")

# Offline data (passed to server)
parser.add_argument("--demo-dataset", type=str, default=None)

# Training (passed to server)
parser.add_argument("--num-steps", type=int, default=50000)
parser.add_argument("--batch-size", type=int, default=256)
parser.add_argument("--utd-ratio", type=int, default=2)
parser.add_argument("--discount", type=float, default=0.99)
parser.add_argument("--actor-lr", type=float, default=3e-4)
parser.add_argument("--critic-lr", type=float, default=3e-4)
parser.add_argument("--temperature-lr", type=float, default=3e-4)
parser.add_argument("--min-temperature", type=float, default=0.0)
parser.add_argument("--warmup-steps", type=int, default=500)
parser.add_argument("--policy-update-freq", type=int, default=1)

# Buffer (passed to server)
parser.add_argument("--online-capacity", type=int, default=30000)

# Logging & Checkpoints
parser.add_argument("--output", type=str, default="outputs/sac_composite_grpc_v1")
from so101_lab.utils.tracker import add_tracker_args
add_tracker_args(parser, default_project="so101-sac")
parser.add_argument("--log-freq", type=int, default=100)
parser.add_argument("--eval-freq", type=int, default=5000)
parser.add_argument("--eval-episodes", type=int, default=5)
parser.add_argument("--save-freq", type=int, default=10000)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to checkpoint to resume from (e.g. outputs/sac_v1/final/pretrained_model)",
)
parser.add_argument("--gui", action="store_true", help="Run with GUI visualization")
parser.add_argument(
    "--image-size",
    type=int,
    default=0,
    choices=[0, 128, 256],
    help="Resize images to NxN (0 = no resize, uses original 480x640)",
)
parser.add_argument(
    "--no-randomize-light",
    action="store_true",
    help="Disable light randomization",
)
parser.add_argument("--no-randomize-physics", action="store_true")
parser.add_argument("--no-camera-noise", action="store_true")
parser.add_argument("--no-distractors", action="store_true")
parser.add_argument("--no-domain-rand", action="store_true")
parser.add_argument("--no-online", action="store_true", help="Offline pretraining only")
parser.add_argument("--no-offline", action="store_true", help="Online RL only")
parser.add_argument(
    "--buffer-image-size",
    type=int,
    default=128,
    help="Image size for replay buffer storage (default 128)",
)
parser.add_argument("--num-workers", type=int, default=4)
parser.add_argument("--max-episodes", type=int, default=None)

# Performance (passed to server)
parser.add_argument("--torch-compile", action="store_true")

# HIL (Human-in-the-Loop)
parser.add_argument("--hil", action="store_true", help="Enable HIL mode (leader arm takeover)")
parser.add_argument("--hil-port", type=str, default="/dev/ttyACM0")
parser.add_argument("--hil-calibration", type=str, default=None)
parser.add_argument("--intervention-capacity", type=int, default=20000)
parser.add_argument("--preview", action="store_true", default=None)
parser.add_argument("--no-preview", action="store_true")

# gRPC
parser.add_argument("--server-host", type=str, default="127.0.0.1")
parser.add_argument("--server-port", type=int, default=8082)
parser.add_argument("--auto-server", action="store_true",
                    help="Auto-start sac_server.py in lerobot-env")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Validate reward mode
if args.reward_mode == "vip_only" and not args.vip_goal_dataset:
    print("ERROR: --reward-mode vip_only requires --vip-goal-dataset")
    sys.exit(1)

if args.reward_mode in ("composite", "classifier_only") and not args.reward_model:
    print(f"ERROR: --reward-mode {args.reward_mode} requires --reward-model")
    sys.exit(1)

# Resolve --preview default: True if --hil, False otherwise
if args.no_preview:
    args.preview = False
elif args.preview is None:
    args.preview = args.hil

# Validate: --hil needs at least one toggle method
if args.hil and not args.gui and not args.preview:
    print("ERROR: --hil requires --gui or --preview (need a way to toggle takeover)")
    sys.exit(1)

if not args.gui:
    args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)

# Disable rate limiting for maximum simulation speed
from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

# Section 2: All imports (after Isaac Sim init — no lerobot imports needed)
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv
from so101_lab.rl.domain_rand import apply_domain_rand_flags
from so101_lab.tasks import get_rl_task
from so101_lab.transport.sac_client import SACTrainingClient
from so101_lab.utils.shm_preview import (
    cleanup_command_file,
    cleanup_shm,
    launch_viewer,
    stop_viewer,
    write_camera_to_shm,
    write_status_to_shm,
)


# --- Sim reward config (client-side, same as original) ---

SIM_REWARD_MAP = {
    "drop": ("drop_penalty", "w_drop"),
    "jerky_motion": ("jerky_motion_penalty", "w_jerky_motion"),
    "smoothness": ("action_smoothness_penalty", "w_smoothness"),
    "time": ("time_penalty", "w_time"),
    "distance": ("distance_cube_to_slot", "w_distance"),
    "distance_gripper": ("distance_gripper_to_cube", "w_distance_gripper"),
    "milestone": (["milestone_picked", "milestone_placed"], "w_milestone"),
    "table_contact": ("table_contact_penalty", "w_table_contact"),
}

SIM_REWARD_PRESETS = {
    "penalties": {"drop", "jerky_motion", "smoothness", "time", "table_contact"},
}


def parse_sim_rewards(spec: str) -> set[str] | None:
    spec = spec.strip().lower()
    if spec == "all":
        return None
    if spec == "none":
        return set()
    if spec in SIM_REWARD_PRESETS:
        return SIM_REWARD_PRESETS[spec]
    names = {s.strip() for s in spec.split(",")}
    unknown = names - set(SIM_REWARD_MAP.keys())
    if unknown:
        valid = ", ".join(sorted(SIM_REWARD_MAP.keys()))
        raise ValueError(f"Unknown sim rewards: {unknown}. Valid: {valid}, or presets: all, none, penalties")
    return names


def apply_reward_weights(env_cfg, args) -> None:
    r = env_cfg.rewards
    enabled = parse_sim_rewards(args.sim_rewards)
    for short_name, (attrs, args_attr) in SIM_REWARD_MAP.items():
        weight = getattr(args, args_attr) if (enabled is None or short_name in enabled) else 0.0
        if isinstance(attrs, list):
            for a in attrs:
                getattr(r, a).weight = weight
        else:
            getattr(r, attrs).weight = weight


def zero_sim_weights(env_cfg) -> None:
    r = env_cfg.rewards
    r.time_out_penalty.weight = 0.0
    for attrs, _ in SIM_REWARD_MAP.values():
        if isinstance(attrs, list):
            for a in attrs:
                getattr(r, a).weight = 0.0
        else:
            getattr(r, attrs).weight = 0.0


@torch.no_grad()
def evaluate(
    env: IsaacLabGymEnv,
    client: SACTrainingClient,
    n_episodes: int,
    max_steps: int = 300,
) -> dict:
    """Evaluate policy via deterministic actions from server."""
    client.set_mode("eval")
    successes = 0
    total_reward = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        obs_raw, info = env.reset()
        ep_reward = 0.0
        ep_steps = 0

        for _ in range(max_steps):
            action_env = client.sample_action_deterministic(obs_raw)
            obs_raw, sim_reward, terminated, truncated, info = env.step(action_env)
            ep_reward += sim_reward
            ep_steps += 1

            if terminated or truncated:
                break

        if ep_reward > 0:
            successes += 1
        total_reward += ep_reward
        total_steps += ep_steps

    client.set_mode("train")
    return {
        "success_rate": successes / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_steps": total_steps / n_episodes,
    }


def main(args):
    if args.no_online and args.no_offline:
        raise ValueError("Cannot use both --no-online and --no-offline")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    if args.image_size == 0:
        print("\n" + "=" * 60)
        print("WARNING: --image-size 0 uses original 480x640 images.")
        print("This will likely cause OOM errors. Use --image-size 128.")
        print("=" * 60 + "\n")

    # Save config
    config_path = os.path.join(args.output, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # 1. Auto-start server if requested
    server_proc = None
    if args.auto_server:
        from so101_lab.utils.policy_server import start_sac_server
        server_proc = start_sac_server(port=args.server_port, host=args.server_host)

    # 2. Connect to server
    client = SACTrainingClient(host=args.server_host, port=args.server_port)
    try:
        client.connect()
        print(f"Connected to SAC server at {args.server_host}:{args.server_port}")
    except Exception as e:
        print(f"ERROR: Cannot connect to SAC server: {e}")
        if server_proc:
            server_proc.terminate()
        sys.exit(1)

    # 3. Init server (send config)
    init_config = {
        # Policy
        "resume": args.resume,
        "image_size": args.image_size,
        "torch_compile": args.torch_compile,
        # Training
        "discount": args.discount,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "temperature_lr": args.temperature_lr,
        "utd_ratio": args.utd_ratio,
        "policy_update_freq": args.policy_update_freq,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "min_temperature": args.min_temperature,
        # Reward
        "reward_mode": args.reward_mode,
        "reward_model": args.reward_model if args.reward_mode in ("composite", "classifier_only") else None,
        "w_classifier": args.w_classifier,
        "success_bonus": args.success_bonus,
        "w_vip": args.w_vip,
        "vip_goal_dataset": args.vip_goal_dataset,
        "vip_camera": args.vip_camera,
        "vip_goal_mode": args.vip_goal_mode,
        "vip_use_labeled": args.vip_use_labeled,
        "vip_label_dataset": args.vip_label_dataset,
        "vip_normalize": args.vip_normalize,
        # Buffers
        "no_online": args.no_online,
        "no_offline": args.no_offline,
        "online_capacity": args.online_capacity,
        "demo_dataset": args.demo_dataset,
        "buffer_image_size": args.buffer_image_size,
        "num_workers": args.num_workers,
        "max_episodes": args.max_episodes,
        # HIL
        "hil": args.hil,
        "intervention_capacity": args.intervention_capacity,
    }
    resume_meta = client.init(init_config)
    resumed_step = resume_meta["start_step"]
    resumed_ep = resume_meta["start_episode"]
    best_sr = resume_meta["best_sr"]

    # 4. Env (only if online)
    env = None
    if not args.no_online:
        RLEnvCfgClass = get_rl_task(args.env)
        env_cfg = RLEnvCfgClass()
        env_cfg.scene.num_envs = 1
        apply_domain_rand_flags(env_cfg, args)
        apply_reward_weights(env_cfg, args)
        if args.reward_mode in ("classifier_only", "vip_only"):
            zero_sim_weights(env_cfg)
        env = IsaacLabGymEnv(env_cfg=env_cfg)

    # 5. HIL setup
    hil_toggle = None
    hil_device = None
    if args.hil:
        if args.no_online:
            raise ValueError("--hil requires online mode (incompatible with --no-online)")
        from so101_lab.rl.hil_input import HILToggle
        from so101_lab.rl.hil_device import HILDeviceReader

        try:
            hil_device = HILDeviceReader(port=args.hil_port, calibration_path=args.hil_calibration)
        except (FileNotFoundError, OSError) as e:
            print(f"\n[HIL] ERROR: Cannot connect to leader arm: {e}")
            print("[HIL] Continuing without HIL mode.")
            args.hil = False

    if args.hil:
        hil_toggle = HILToggle(gui=args.gui)
        print(f"[HIL] Enabled: press Enter to toggle takeover")

    has_online = not args.no_online
    has_offline = not args.no_offline and args.demo_dataset is not None

    # 6. Warmup logic
    if args.no_online:
        effective_warmup = 0
    else:
        effective_warmup = resumed_step + args.warmup_steps

    # 7. Experiment tracker
    run_name = os.path.basename(os.path.normpath(args.output))
    from so101_lab.utils.tracker import setup_tracker, cleanup_tracker
    tracker, sys_monitor = setup_tracker(args, run_name)

    # 8. Preview
    viewer_proc = None
    preview_active = args.preview
    if preview_active:
        cleanup_command_file()
        viewer_proc = launch_viewer("hil_viewer.py")
        preview_active = viewer_proc is not None

    # 9. Main loop
    obs_raw = None
    if has_online:
        obs_raw, info = env.reset()
    opt_step = resumed_step
    ep_count = resumed_ep
    ep_reward = 0.0
    ep_steps = 0
    t_start = time.time()

    mode = "online + offline" if has_online and has_offline else ("offline only" if has_offline else "online only")
    if args.hil:
        mode += " + HIL"

    total_steps = resumed_step + args.num_steps
    print(f"\nStarting SAC training (gRPC): {args.num_steps} steps (total {total_steps})")
    print(f"  Mode: {mode}")
    print(f"  Reward mode: {args.reward_mode}")
    print(f"  Server: {args.server_host}:{args.server_port}")
    if args.resume:
        print(f"  Resumed from: {args.resume} (step {resumed_step})")
    print()

    pbar = tqdm(range(resumed_step, total_steps), desc="Training", unit="step", initial=resumed_step, total=total_steps)
    training_infos = {}

    for step in pbar:
        # a) Online exploration
        is_intervention = False

        if has_online:
            # Select action
            if hil_toggle is not None and hil_toggle.is_active:
                action_env = hil_device.read_action()
                is_intervention = True
            elif step < effective_warmup:
                action_env = env.action_space.sample()
            else:
                action_env = client.sample_action(obs_raw)

            # Step env
            next_obs_raw, sim_reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            # Send transition to server → get reward breakdown
            reward_info = client.send_step_result(
                obs=obs_raw,
                obs_next=next_obs_raw,
                action=action_env,
                sim_reward=sim_reward,
                done=done,
                truncated=truncated,
                is_intervention=is_intervention,
                image_size=args.image_size,
            )

            reward = reward_info["reward"]
            ep_reward += reward
            ep_steps += 1

            # Preview: write images to shm
            if preview_active:
                write_camera_to_shm("top", next_obs_raw["observation.images.top"])
                write_camera_to_shm("wrist", next_obs_raw["observation.images.wrist"])
                hil_on = hil_toggle is not None and hil_toggle.is_active
                write_status_to_shm({
                    "state": "HIL ON" if hil_on else "POLICY",
                    "teleop": hil_on,
                    "episode": int(ep_count),
                    "frame": int(ep_steps),
                    "status_text": f"step {step} | r {ep_reward:.1f}",
                    "timestamp": time.time(),
                })

            # Poll viewer/GUI commands
            if hil_toggle is not None:
                hil_cmd = hil_toggle.poll_commands()
                if hil_cmd == "reset":
                    tqdm.write(f"[RESET] Episode reset at step {step}")
                    obs_raw, info = env.reset()
                    ep_reward = 0.0
                    ep_steps = 0
                    continue

            # Detailed reward logging
            if tracker and step % args.log_freq == 0:
                detail = env.get_reward_details()
                detail["reward/classifier_pred"] = reward_info["classifier_pred"]
                detail["reward/vip_reward"] = reward_info["vip_reward"]
                detail["reward/bonus_given"] = 1 if reward_info["bonus_given"] else 0
                detail["reward/total"] = reward
                tracker.log(detail, step=step)

            # Episode tracking
            if done:
                ep_count += 1
                postfix = {
                    "ep": ep_count,
                    "r": f"{ep_reward:.1f}",
                }
                if hil_toggle is not None:
                    postfix["HIL"] = "ON" if hil_toggle.is_active else "off"
                pbar.set_postfix(postfix)
                if tracker:
                    tracker.log({"episode/reward": ep_reward, "episode/steps": ep_steps}, step=step)
                obs_raw, info = env.reset()
                ep_reward = 0.0
                ep_steps = 0
            else:
                obs_raw = next_obs_raw

        # b) Training update
        can_train = (args.no_online and has_offline) or \
                    (has_online and step >= effective_warmup)

        if can_train:
            update_config = {"opt_step": opt_step}
            metrics = client.run_sac_update(step, update_config)
            training_infos.update(metrics)
            opt_step += 1

            postfix = {
                "ep": ep_count,
                "r": f"{ep_reward:.1f}",
                "Lc": f"{metrics.get('loss_critic', 0):.3f}",
                "La": f"{metrics.get('loss_actor', 0):.3f}",
                "T": f"{metrics.get('temperature', 0):.3f}",
            }
            if hil_toggle is not None:
                postfix["HIL"] = "ON" if hil_toggle.is_active else "off"
            pbar.set_postfix(postfix)

            if step % args.log_freq == 0 and tracker:
                log_dict = {f"train/{k}": v for k, v in training_infos.items()}
                tracker.log(log_dict, step=step)

        # c) Eval + checkpoint
        if has_online and step > 0 and step % args.eval_freq == 0:
            tqdm.write(f"\n--- Evaluation at step {step} ---")
            eval_metrics = evaluate(env, client, args.eval_episodes)
            sr = eval_metrics["success_rate"]
            tqdm.write(
                f"  Success rate: {sr:.0%} | "
                f"Avg reward: {eval_metrics['avg_reward']:.2f} | "
                f"Avg steps: {eval_metrics['avg_steps']:.0f}"
            )
            if tracker:
                tracker.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=step)

            if sr > best_sr:
                best_sr = sr
                _ts = {"opt_step": opt_step, "ep_count": ep_count, "best_sr": best_sr}
                client.save_checkpoint(os.path.join(args.output, "best"), _ts)
                tqdm.write(f"  New best! SR={sr:.0%}")

            # Reset env after eval
            obs_raw, info = env.reset()
            ep_reward = 0.0
            ep_steps = 0

        if step > 0 and step % args.save_freq == 0:
            ckpt_name = f"step_{step}"
            ckpt_dir = os.path.join(args.output, ckpt_name)
            if os.path.exists(ckpt_dir):
                print(f"Skipping save: {ckpt_dir} already exists (resumed checkpoint)")
            else:
                _ts = {"opt_step": opt_step, "ep_count": ep_count, "best_sr": best_sr}
                client.save_checkpoint(os.path.join(args.output, ckpt_name), _ts)

    pbar.close()

    # Final save
    _ts = {"opt_step": opt_step, "ep_count": ep_count, "best_sr": best_sr}
    client.save_checkpoint(os.path.join(args.output, "final"), _ts)
    if has_online:
        print(f"\nTraining complete. Best success rate: {best_sr:.0%}")
    else:
        print(f"\nOffline pretraining complete.")
    print(f"Output: {args.output}")

    # Cleanup
    client.close()
    stop_viewer(viewer_proc)
    if preview_active:
        cleanup_shm()
        cleanup_command_file()
    if hil_toggle is not None:
        hil_toggle.stop()
    if hil_device is not None:
        hil_device.disconnect()
    if env is not None:
        env.close()
    cleanup_tracker(tracker, sys_monitor)
    if server_proc is not None:
        server_proc.terminate()
        server_proc.wait()


if __name__ == "__main__":
    main(args)
    app_launcher.app.close()
