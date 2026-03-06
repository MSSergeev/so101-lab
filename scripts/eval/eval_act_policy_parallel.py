#!/usr/bin/env python3
"""Evaluate trained ACT policy with parallel environments.

This script runs a trained policy across multiple parallel environments
for faster evaluation. Uses ParallelActionQueue for efficient batch inference.

Usage:
    # Basic parallel evaluation (4 envs, 100 episodes)
    python scripts/eval/eval_act_policy_parallel.py --checkpoint outputs/act_v1 --num-envs 4

    # More envs for faster evaluation
    python scripts/eval/eval_act_policy_parallel.py --checkpoint outputs/act_v1 --num-envs 8 --episodes 200

    # Override n_action_steps (use 1 to call model every step)
    python scripts/eval/eval_act_policy_parallel.py --checkpoint outputs/act_v1 --n-action-steps 1

    # Temporal ensembling (inference every step, smoother actions)
    python scripts/eval/eval_act_policy_parallel.py --checkpoint outputs/act_v1 --temporal-ensemble-coeff 0.01

Notes:
    - No preview, recording, or keyboard controls (use single-env script for those)
    - Isaac Lab auto-resets envs on done, so we track per-env state
    - Default mode: action queue (model called every n_action_steps)
    - Temporal ensemble mode: model called every step, actions blended across chunks
    - n_action_steps defaults to model config (same as training)
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# Parse arguments before Isaac Sim launch
parser = argparse.ArgumentParser(description="Evaluate ACT policy in parallel environments")

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
parser.add_argument(
    "--num-envs",
    type=int,
    default=4,
    help="Number of parallel environments (default: 4)",
)

# Evaluation parameters
parser.add_argument(
    "--episodes",
    type=int,
    default=100,
    help="Number of evaluation episodes (default: 100)",
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=1000,
    help="Max steps per episode (default: 1000)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility (default: random)",
)

# Output
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output directory for evaluation results (auto-generated if not specified)",
)

# Display
parser.add_argument(
    "--gui",
    action="store_true",
    help="Run with GUI (default: headless, recommended for speed)",
)

# Frequencies
parser.add_argument("--physics-hz", type=int, default=120,
                    help="Physics simulation frequency")
parser.add_argument("--policy-hz", type=int, default=30,
                    help="Policy/control frequency")
parser.add_argument("--render-hz", type=int, default=30,
                    help="Rendering frequency")

# Policy parameters
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=None,
    help="Override n_action_steps (default: from model config)",
)
parser.add_argument(
    "--temporal-ensemble-coeff",
    type=float,
    default=None,
    help="Enable temporal ensembling with given coefficient (e.g., 0.01). Incompatible with action queue.",
)
parser.add_argument(
    "--ensemble-interval",
    type=int,
    default=1,
    help="Run model every N steps when using temporal ensembling (default: 1, every step)",
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

# Validate ensemble-interval
if args.ensemble_interval != 1 and args.temporal_ensemble_coeff is None:
    parser.error("--ensemble-interval requires --temporal-ensemble-coeff")

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
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from so101_lab.tasks import get_task
from so101_lab.policies.act import ACTInference
from so101_lab.utils.checkpoint import resolve_checkpoint_path


# ═══════════════════════════════════════════════════════════════════════════════
# Parallel action queue for efficient batch inference
# ═══════════════════════════════════════════════════════════════════════════════

class ParallelActionQueue:
    """Action queue for parallel environments.

    Maintains separate queues for each env. When a queue is empty,
    runs batch inference only for envs that need new actions.
    This reduces forward passes by n_action_steps factor.
    """

    def __init__(self, num_envs: int, n_action_steps: int, policy: ACTInference):
        self.num_envs = num_envs
        self.n_action_steps = n_action_steps
        self.policy = policy

        # Per-env action queues (list of np.ndarray actions)
        self.queues: list[list[np.ndarray]] = [[] for _ in range(num_envs)]

        # Stats
        self.inference_calls = 0
        self.total_actions = 0

    def reset_env(self, env_id: int):
        """Clear queue for a specific env (call on episode reset)."""
        self.queues[env_id].clear()

    def select_actions(self, obs: dict, skip_envs: np.ndarray | None = None) -> np.ndarray:
        """Get actions for all envs, running inference only when needed.

        Args:
            obs: Batched observations dict with:
                - "joint_pos": (num_envs, 6) in radians
                - "images": {"top": (num_envs, H, W, 3), ...}
            skip_envs: Boolean mask of envs to skip (return zero action, no inference)

        Returns:
            actions: (num_envs, 6) in radians
        """
        action_dim = 6
        actions = np.zeros((self.num_envs, action_dim), dtype=np.float32)

        # Handle skipped envs (camera update step after reset)
        if skip_envs is not None and skip_envs.any():
            active_envs = ~skip_envs
        else:
            active_envs = np.ones(self.num_envs, dtype=bool)

        # Find active envs with empty queues
        need_inference = [i for i in range(self.num_envs)
                         if active_envs[i] and len(self.queues[i]) == 0]

        if need_inference:
            # Build subset batch for inference
            subset_obs = {
                "joint_pos": obs["joint_pos"][need_inference],
                "images": {cam: img[need_inference] for cam, img in obs.get("images", {}).items()},
            }

            # Run batch inference - get action chunk
            action_chunks = self.policy.select_action_chunk_batch(subset_obs, self.n_action_steps)

            # Fill queues
            for idx, env_id in enumerate(need_inference):
                for step in range(self.n_action_steps):
                    self.queues[env_id].append(action_chunks[idx, step])

            self.inference_calls += 1

        # Pop one action from each active env's queue
        for i in range(self.num_envs):
            if active_envs[i] and len(self.queues[i]) > 0:
                actions[i] = self.queues[i].pop(0)

        self.total_actions += self.num_envs
        return actions

    def get_stats(self) -> dict:
        """Return inference statistics."""
        return {
            "inference_calls": self.inference_calls,
            "total_actions": self.total_actions,
            "actions_per_inference": round(self.total_actions / max(1, self.inference_calls), 1),
        }




# ═══════════════════════════════════════════════════════════════════════════════
# Main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    num_envs = args.num_envs

    # Set seeds
    if args.seed is None:
        args.seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"Using random seed: {args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Parallel environments: {num_envs}")

    # Resolve checkpoint path
    checkpoint_path = resolve_checkpoint_path(
        args.checkpoint, args.use_best, args.use_latest, args.step
    )
    print(f"Loading policy from: {checkpoint_path}")

    # Load policy
    policy = ACTInference(
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    # Configure temporal ensembling or action queue mode
    use_temporal_ensemble = args.temporal_ensemble_coeff is not None

    ensemble_interval = args.ensemble_interval if use_temporal_ensemble else 1

    if use_temporal_ensemble:
        policy.config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        policy.policy.config.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        from so101_lab.policies.act.modeling_act import ACTTemporalEnsembler
        policy.policy.temporal_ensembler = ACTTemporalEnsembler(
            args.temporal_ensemble_coeff, policy.policy.config.chunk_size
        )
        policy.policy.reset()
        print(f"  temporal_ensemble_coeff: {args.temporal_ensemble_coeff}")
        print(f"  ensemble_interval: {ensemble_interval}")
        print(f"  mode: temporal ensembling (inference every {ensemble_interval} step(s))")
        n_action_steps = 1  # Not used with ensemble, but kept for stats
    else:
        n_action_steps = policy.config.n_action_steps
        if args.n_action_steps is not None:
            old_val = n_action_steps
            n_action_steps = args.n_action_steps
            print(f"  n_action_steps: {old_val} -> {n_action_steps} (override)")
        else:
            print(f"  n_action_steps: {n_action_steps} (from model config)")

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

    # Create environment with multiple envs
    print(f"Creating environment: {args.env} (num_envs={num_envs})")
    EnvClass, EnvCfgClass = get_task(args.env)

    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = num_envs

    # Configure frequencies
    env_cfg.sim.dt = 1.0 / args.physics_hz
    env_cfg.decimation = args.physics_hz // args.policy_hz
    env_cfg.sim.render_interval = args.physics_hz // args.render_hz

    # Camera update period
    camera_period = 1.0 / args.render_hz
    if hasattr(env_cfg.scene, 'top'):
        env_cfg.scene.top.update_period = camera_period
    if hasattr(env_cfg.scene, 'wrist'):
        env_cfg.scene.wrist.update_period = camera_period

    # Light randomization
    if hasattr(env_cfg, 'randomize_light'):
        env_cfg.randomize_light = args.randomize_light

    # Terminate on success (early exit instead of running to timeout)
    env_cfg.terminate_on_success = True

    # Episode length
    env_cfg.episode_length_s = args.max_steps / args.policy_hz

    # Create env
    render_mode = "human" if args.gui else None
    env = EnvClass(env_cfg, render_mode=render_mode)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("outputs/eval") / f"{args.env}_parallel_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize registry
    registry = {
        "config": {
            "checkpoint": str(checkpoint_path),
            "env": args.env,
            "num_envs": num_envs,
            "num_episodes": args.episodes,
            "max_steps": args.max_steps,
            "initial_seed": args.seed,
            "physics_hz": args.physics_hz,
            "policy_hz": args.policy_hz,
            "training_fps": training_fps,
            "timestamp": datetime.now().isoformat()
        },
        "episodes": []
    }

    # Per-env tracking state
    env_step_counts = np.zeros(num_envs, dtype=int)
    env_start_times = [time.time() for _ in range(num_envs)]
    # Track envs that need camera update after auto-reset (stale image fix)
    needs_camera_update = np.zeros(num_envs, dtype=bool)

    completed_episodes = []
    global_step = 0  # Global step counter for ensemble_interval

    # Create action queue (not used in temporal ensemble mode)
    action_queue = None if use_temporal_ensemble else ParallelActionQueue(num_envs, n_action_steps, policy)

    # Initial reset
    print(f"\nRunning {args.episodes} evaluation episodes across {num_envs} parallel envs...")
    if use_temporal_ensemble:
        print(f"Using temporal ensembling (coeff={args.temporal_ensemble_coeff}, interval={ensemble_interval})")
    else:
        print(f"Using action queue with n_action_steps={n_action_steps}")
    print("=" * 60)

    obs, info = env.reset()
    policy.reset()

    # Zero action step to ensure cameras are updated
    zero_action = torch.zeros((num_envs, 6), dtype=torch.float32, device=device)
    obs, _, _, _, _ = env.step(zero_action)

    eval_start_time = time.time()

    try:
        while len(completed_episodes) < args.episodes:
            # Extract observations for batch inference
            policy_data = obs["policy"]

            # Build batched observation dict
            policy_obs = {
                "joint_pos": policy_data["joint_pos"].cpu().numpy(),  # (num_envs, 6)
                "images": {}
            }

            if "top" in policy_data:
                policy_obs["images"]["top"] = policy_data["top"].cpu().numpy()  # (num_envs, H, W, 3)
            if "wrist" in policy_data:
                policy_obs["images"]["wrist"] = policy_data["wrist"].cpu().numpy()

            # Get actions: temporal ensemble or action queue
            if use_temporal_ensemble:
                stale_ids = list(np.where(needs_camera_update)[0]) if needs_camera_update.any() else []
                needs_camera_update[:] = False

                if global_step % ensemble_interval == 0:
                    # skip_env_ids: don't blend stale obs into ensembler
                    actions = policy.select_action_batch_ensemble(
                        policy_obs, skip_env_ids=stale_ids,
                    )
                else:
                    actions = policy.pop_ensemble_actions()

                # Stale envs: zero action for camera update, then reset
                # ensembler so next update() re-inits from fresh obs
                if stale_ids:
                    actions[stale_ids] = 0.0
                    policy.policy.temporal_ensembler.reset_envs(stale_ids)

                global_step += 1
            else:
                actions = action_queue.select_actions(policy_obs, skip_envs=needs_camera_update)
                if needs_camera_update.any():
                    needs_camera_update[:] = False

            # Convert to tensor for env.step
            action_tensor = torch.tensor(actions, dtype=torch.float32, device=device)

            # Step all envs
            obs, rewards, terminated, truncated, info = env.step(action_tensor)
            env_step_counts += 1

            # terminated = success (from _get_dones with terminate_on_success=True)
            terminated_np = terminated.cpu().numpy().flatten()
            done = (terminated | truncated).cpu().numpy().flatten()
            done_indices = np.where(done)[0]

            for env_id in done_indices:
                success = bool(terminated_np[env_id])

                elapsed = time.time() - env_start_times[env_id]
                steps = int(env_step_counts[env_id])

                # Record episode result
                episode_record = {
                    "id": len(completed_episodes),
                    "env_id": int(env_id),
                    "seed": "N/A (auto-reset)",
                    "success": success,
                    "steps": steps,
                    "duration_s": round(elapsed, 3),
                }
                completed_episodes.append(episode_record)
                registry["episodes"].append(episode_record)

                # Print progress
                rate = sum(1 for e in completed_episodes if e["success"]) / len(completed_episodes) * 100
                status = "SUCCESS" if success else "FAIL"
                print(f"Episode {len(completed_episodes):3d}/{args.episodes}: {status:7s} | "
                      f"env:{env_id} steps:{steps:4d} | rate: {rate:.0f}%")

                # Reset tracking for this env (Isaac Lab auto-resets the env)
                env_step_counts[env_id] = 0
                env_start_times[env_id] = time.time()

                # Reset policy state for this env
                # Temporal ensemble: handled via skip_env_ids + reset_envs
                # in the action step when needs_camera_update is processed
                if not use_temporal_ensemble:
                    action_queue.reset_env(env_id)

                # Mark env for camera update on next step (stale image fix)
                needs_camera_update[env_id] = True

                # Check if we've collected enough episodes
                if len(completed_episodes) >= args.episodes:
                    break

        eval_elapsed = time.time() - eval_start_time

        # Compute summary statistics
        successes = [e["success"] for e in completed_episodes]
        steps_list = [e["steps"] for e in completed_episodes]
        durations = [e["duration_s"] for e in completed_episodes]

        success_rate = sum(successes) / len(successes) * 100
        avg_steps = sum(steps_list) / len(steps_list)
        avg_duration = sum(durations) / len(durations)
        avg_sim_duration = avg_steps / args.policy_hz
        realtime_factor = avg_sim_duration / avg_duration if avg_duration > 0 else 0
        total_sim_throughput = realtime_factor * num_envs

        success_steps = [s for s, ok in zip(steps_list, successes) if ok]
        fail_steps = [s for s, ok in zip(steps_list, successes) if not ok]

        total_steps = sum(steps_list)
        actions_per_second = total_steps / eval_elapsed

        summary = {
            "total_episodes": len(completed_episodes),
            "requested_episodes": args.episodes,
            "num_envs": num_envs,
            "n_action_steps": n_action_steps,
            "temporal_ensemble_coeff": args.temporal_ensemble_coeff,
            "ensemble_interval": ensemble_interval if use_temporal_ensemble else None,
            "completed": True,
            "successes": sum(successes),
            "failures": len(successes) - sum(successes),
            "success_rate": round(success_rate, 2),
            "avg_steps": round(avg_steps, 1),
            "avg_episode_duration_s": round(avg_duration, 3),
            "avg_sim_duration_s": round(avg_sim_duration, 3),
            "realtime_factor": round(realtime_factor, 2),
            "total_sim_throughput": round(total_sim_throughput, 2),
            "avg_steps_success": round(sum(success_steps) / len(success_steps), 1) if success_steps else None,
            "avg_steps_fail": round(sum(fail_steps) / len(fail_steps), 1) if fail_steps else None,
            "total_eval_time_s": round(eval_elapsed, 2),
            "total_steps": total_steps,
            "episodes_per_second": round(len(completed_episodes) / eval_elapsed, 2),
            "actions_per_second": round(actions_per_second, 1),
        }

        # Add action queue stats if using queue mode
        if action_queue is not None:
            queue_stats = action_queue.get_stats()
            summary["inference_calls"] = queue_stats["inference_calls"]
            summary["actions_per_inference"] = queue_stats["actions_per_inference"]

        registry["summary"] = summary

        # Print summary
        print("\n" + "=" * 60)
        print(f"EVALUATION RESULTS ({args.episodes} episodes, {num_envs} parallel envs)")
        print("=" * 60)
        print(f"Success rate: {success_rate:.1f}% ({sum(successes)}/{len(successes)})")
        print(f"Avg steps: {avg_steps:.1f}")
        print(f"Avg episode duration: {avg_duration:.3f}s (wall), {avg_sim_duration:.3f}s (sim)")
        print(f"Realtime factor: {realtime_factor:.2f}x (per env), {total_sim_throughput:.2f}x (total)")
        if success_steps:
            print(f"Avg steps (success): {summary['avg_steps_success']}")
        if fail_steps:
            print(f"Avg steps (fail): {summary['avg_steps_fail']}")
        print(f"\nTotal eval time: {eval_elapsed:.1f}s")
        print(f"Throughput: {summary['episodes_per_second']:.2f} eps/sec, {actions_per_second:.1f} actions/sec")
        if action_queue is not None:
            queue_stats = action_queue.get_stats()
            print(f"Inference calls: {queue_stats['inference_calls']} ({queue_stats['actions_per_inference']:.1f} actions/call)")
        else:
            print(f"Inference calls: {total_steps} (temporal ensembling, 1 per step)")

        # Save outputs
        with open(output_dir / "registry.json", "w") as f:
            json.dump(registry, f, indent=2)

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(output_dir / "config.json", "w") as f:
            json.dump(registry["config"], f, indent=2)

        print(f"\nResults saved to: {output_dir}")

    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
