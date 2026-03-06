"""PPO training for SmolVLA with Flow-Noise — gRPC split version.

Client (isaaclab-env): Isaac Sim environment only.
Server (lerobot-env):  FlowNoiseSmolVLA + VIPReward + PPO update.

Usage:
    # Start server in lerobot-env first (or use --auto-server):
    python scripts/train/ppo_server.py --port 8081

    # Then in isaaclab-env:
    python scripts/train/train_flow_noise_ppo_grpc.py \
        --checkpoint outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model \
        --iql-checkpoint outputs/iql_critics_v2_100k/final/critics.pt \
        --goal-dataset data/recordings/figure_shape_placement_v5 \
        --total-updates 2 --rollout-steps 16 \
        --output outputs/flow_noise_ppo_grpc_smoke --headless

    # With auto-server:
    python scripts/train/train_flow_noise_ppo_grpc.py \
        --checkpoint ... --goal-dataset ... \
        --auto-server --output outputs/flow_noise_ppo_grpc_v1 --headless
"""

# Section 1: argparse + AppLauncher (before Isaac Sim init)
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO training for SmolVLA (gRPC split)")

# Environment
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment name")

# Policy (passed to server)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to SmolVLA BC checkpoint")
parser.add_argument("--iql-checkpoint", type=str, default=None,
                    help="IQL critics checkpoint for value head warm start")
parser.add_argument("--task", type=str,
                    default="Place the cube into the matching slot on the platform")
parser.add_argument("--noise-prior", type=str, default=None,
                    help="Path to noise_prior.pt")

# Reward / VIP goals (passed to server)
parser.add_argument("--goal-dataset", type=str, required=True)
parser.add_argument("--vip-use-labeled", action="store_true")
parser.add_argument("--vip-label-dataset", type=str, default=None)
parser.add_argument("--vip-goal-mode", type=str, default="mean", choices=["mean", "min"])
parser.add_argument("--n-goal-frames", type=int, default=5)
parser.add_argument("--vip-weight", type=float, default=1.0)
parser.add_argument("--sim-weight", type=float, default=0.0)
parser.add_argument("--success-bonus", type=float, default=0.0)

# PPO (passed to server)
parser.add_argument("--total-updates", type=int, default=1000)
parser.add_argument("--rollout-steps", type=int, default=256)
parser.add_argument("--update-epochs", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--clip-ratio", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--actor-lr", type=float, default=3e-5)
parser.add_argument("--value-lr", type=float, default=1e-4)
parser.add_argument("--max-grad-norm", type=float, default=1.0)
parser.add_argument("--reeval-batch-size", type=int, default=1)
parser.add_argument("--kv-cache-device", type=str, default="cpu", choices=["cpu", "cuda"])
parser.add_argument("--warmup-value", type=int, default=0)
parser.add_argument("--normalize-rewards", action="store_true")

# Environment
parser.add_argument("--max-episode-steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)

# Logging & Checkpoints
parser.add_argument("--output", type=str, default="outputs/flow_noise_ppo_grpc_v1")
from so101_lab.utils.tracker import add_tracker_args
add_tracker_args(parser, default_project="so101-flow-ppo")
parser.add_argument("--log-freq", type=int, default=1)
parser.add_argument("--save-freq", type=int, default=50)
parser.add_argument("--eval-freq", type=int, default=50)
parser.add_argument("--eval-episodes", type=int, default=5)

# Resume
parser.add_argument("--resume", type=str, default=None)

# Domain randomization
parser.add_argument("--no-domain-rand", action="store_true")
parser.add_argument("--no-randomize-light", action="store_true")
parser.add_argument("--no-randomize-physics", action="store_true")
parser.add_argument("--no-camera-noise", action="store_true")
parser.add_argument("--no-distractors", action="store_true")

# gRPC
parser.add_argument("--server-host", type=str, default="127.0.0.1")
parser.add_argument("--server-port", type=int, default=8081)
parser.add_argument("--auto-server", action="store_true",
                    help="Auto-start ppo_server.py in lerobot-env")

# Display
parser.add_argument("--gui", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)

# Section 2: Imports after AppLauncher
from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm

from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv
from so101_lab.rl.domain_rand import apply_domain_rand_flags
from so101_lab.tasks import get_rl_task
from so101_lab.transport.ppo_client import PPOTrainingClient


def evaluate(
    client: PPOTrainingClient,
    env: IsaacLabGymEnv,
    episodes: int,
    max_steps: int,
) -> dict[str, float]:
    """Run deterministic evaluation episodes via gRPC."""
    client.set_mode("eval")
    client.reset_policy()

    successes = 0
    total_steps = 0
    total_reward = 0.0

    for ep in range(episodes):
        obs, _ = env.reset()
        client.reset_policy()
        ep_reward = 0.0

        for step in range(max_steps):
            action = client.sample_action_deterministic(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            details = env.get_reward_details()
            ep_reward += details.get("reward/milestone_placed_weighted", 0.0)
            total_steps += 1

            if terminated or truncated:
                break

        if ep_reward > 0:
            successes += 1
        total_reward += ep_reward

    client.set_mode("train")

    return {
        "success_rate": successes / episodes,
        "avg_reward": total_reward / episodes,
        "avg_steps": total_steps / episodes,
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Save config
    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # 1. Create environment
    RLEnvCfgClass = get_rl_task(args.env)
    env_cfg = RLEnvCfgClass()
    env_cfg.scene.num_envs = 1
    apply_domain_rand_flags(env_cfg, args)
    env = IsaacLabGymEnv(env_cfg=env_cfg)

    # 2. Start server if --auto-server
    server_proc = None
    if args.auto_server:
        from so101_lab.utils.policy_server import start_ppo_server
        server_proc = start_ppo_server(port=args.server_port, host=args.server_host)

    # 3. Connect to PPO server
    client = PPOTrainingClient(host=args.server_host, port=args.server_port)
    client.connect()

    # 4. Initialize server (send all config, server loads model + VIP)
    init_config = {
        "checkpoint": args.checkpoint,
        "iql_checkpoint": args.iql_checkpoint,
        "task": args.task,
        "noise_prior": args.noise_prior,
        "goal_dataset": args.goal_dataset,
        "vip_use_labeled": args.vip_use_labeled,
        "vip_label_dataset": args.vip_label_dataset,
        "vip_goal_mode": args.vip_goal_mode,
        "n_goal_frames": args.n_goal_frames,
        "vip_weight": args.vip_weight,
        "sim_weight": args.sim_weight,
        "success_bonus": args.success_bonus,
        "actor_lr": args.actor_lr,
        "value_lr": args.value_lr,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "update_epochs": args.update_epochs,
        "batch_size": args.batch_size,
        "clip_ratio": args.clip_ratio,
        "max_grad_norm": args.max_grad_norm,
        "reeval_batch_size": args.reeval_batch_size,
        "kv_cache_device": args.kv_cache_device,
        "normalize_rewards": args.normalize_rewards,
        "resume": args.resume,
    }
    resume_meta = client.init(init_config)
    start_update = resume_meta["start_update"]
    best_sr = resume_meta["best_sr"]
    ep_count = resume_meta["ep_count"]

    # 5. Tracker
    run_name = os.path.basename(os.path.normpath(args.output))
    from so101_lab.utils.tracker import setup_tracker, cleanup_tracker
    tracker, sys_monitor = setup_tracker(args, run_name)

    # 6. Training info
    total_updates = start_update + args.total_updates
    print(f"\nStarting PPO training (gRPC): {args.total_updates} updates (total {total_updates})")
    print(f"  Server: {args.server_host}:{args.server_port}")
    print(f"  Rollout steps: {args.rollout_steps}")
    print(f"  VIP weight: {args.vip_weight}, Sim weight: {args.sim_weight}")
    print()

    # 7. PPO loop
    obs, _ = env.reset()
    client.reset_policy()
    ep_reward_accum = 0.0
    ep_steps = 0
    t_start = time.time()

    pbar = tqdm(
        range(start_update, total_updates),
        desc="PPO", unit="update",
        initial=start_update, total=total_updates,
    )

    for update in pbar:
        update_t0 = time.time()

        # === Rollout ===
        for step in range(args.rollout_steps):
            # Send obs → server caches VIP emb, computes value, samples action
            action, log_prob, value = client.sample_action(obs)

            # Step environment
            obs_next, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Get sim reward details
            sim_rewards = {}
            if args.sim_weight > 0 or args.success_bonus > 0:
                sim_rewards = env.get_reward_details()

            # Send obs_next → server computes VIP reward, stores transition
            client.send_step_result(obs_next, sim_rewards, done)

            ep_reward_accum += 0  # actual reward tracked on server
            ep_steps += 1

            if done:
                ep_count += 1
                obs, _ = env.reset()
                client.reset_policy()
                ep_reward_accum = 0.0
                ep_steps = 0
            else:
                obs = obs_next

        # === PPO Update ===
        is_warmup = (update < start_update + args.warmup_value)
        update_config = {"is_warmup": is_warmup}
        metrics = client.run_ppo_update(obs, update_config)

        update_time = time.time() - update_t0

        # === Logging ===
        log_dict = {
            "rollout/reward_mean": metrics["reward_mean"],
            "rollout/reward_std": metrics["reward_std"],
            "rollout/advantage_mean": metrics["advantage_mean"],
            "rollout/log_prob_mean": metrics["log_prob_mean"],
            "rollout/value_mean": metrics["value_mean"],
            "rollout/episodes": ep_count,
            "train/actor_loss": metrics["actor_loss"],
            "train/value_loss": metrics["value_loss"],
            "train/ratio_mean": metrics["ratio_mean"],
            "train/ratio_std": metrics["ratio_std"],
            "train/ratio_max": metrics["ratio_max"],
            "train/ratio_min": metrics["ratio_min"],
            "train/update_time": update_time,
            "train/is_warmup": float(is_warmup),
        }

        pbar.set_postfix({
            "r": f"{metrics['reward_mean']:.3f}",
            "a_loss": f"{metrics['actor_loss']:.4f}",
            "v_loss": f"{metrics['value_loss']:.4f}",
            "ratio": f"{metrics['ratio_mean']:.3f}",
        })

        if update % args.log_freq == 0 and tracker:
            tracker.log(log_dict, step=update)

        # === Evaluation ===
        if update % args.eval_freq == 0 and update > start_update:
            eval_metrics = evaluate(
                client, env, args.eval_episodes, args.max_episode_steps,
            )
            print(f"\n  Eval (update {update}): SR={eval_metrics['success_rate']:.0%}"
                  f" reward={eval_metrics['avg_reward']:.2f}"
                  f" steps={eval_metrics['avg_steps']:.0f}")

            if tracker:
                tracker.log({f"eval/{k}": v for k, v in eval_metrics.items()}, step=update)

            if eval_metrics["success_rate"] > best_sr:
                best_sr = eval_metrics["success_rate"]
                meta = {"update": update + 1, "best_sr": best_sr, "ep_count": ep_count}
                client.save_checkpoint(os.path.join(args.output, "checkpoints", "best"), meta)
                print(f"  New best! SR={best_sr:.0%}")

            # Reset env after eval
            obs, _ = env.reset()
            client.reset_policy()
            ep_reward_accum = 0.0
            ep_steps = 0

        # === Checkpoint ===
        if update % args.save_freq == 0 and update > start_update:
            meta = {"update": update + 1, "best_sr": best_sr, "ep_count": ep_count}
            ckpt_dir = os.path.join(args.output, "checkpoints", f"{update:06d}")
            client.save_checkpoint(ckpt_dir, meta)

    pbar.close()

    # Final save
    meta = {"update": total_updates, "best_sr": best_sr, "ep_count": ep_count}
    client.save_checkpoint(os.path.join(args.output, "checkpoints", "last"), meta)

    elapsed = time.time() - t_start
    print(f"\nTraining complete. {args.total_updates} updates in {elapsed / 3600:.1f}h")
    print(f"  Best success rate: {best_sr:.0%}")
    print(f"  Output: {args.output}")

    cleanup_tracker(tracker, sys_monitor)
    client.close()
    env.close()

    if server_proc is not None:
        server_proc.terminate()
        server_proc.wait()


if __name__ == "__main__":
    main(args)
    app_launcher.app.close()
