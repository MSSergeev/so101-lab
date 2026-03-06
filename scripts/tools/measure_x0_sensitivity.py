"""Measure how initial noise x_0 affects SmolVLA flow matching actions.

Runs N ODE passes with different x_0 for the same observation and measures
per-joint action variance. Also tests directional sensitivity by shifting
x_0[0,0,:joint] by a fixed magnitude.

If per-joint std is small → ODE "forgets" x_0 → noise prior is useless.
If std is significant → x_0 matters → noise prior can steer actions.

Usage:
    python scripts/tools/measure_x0_sensitivity.py \
        --checkpoint outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model \
        --num-scenes 1 --num-samples 10 --headless

    python scripts/tools/measure_x0_sensitivity.py \
        --checkpoint outputs/flow_noise_ppo_v2/best/pretrained_model \
        --num-scenes 5 --num-samples 100 \
        --output outputs/x0_sensitivity.json --headless
"""

# Section 1: argparse + AppLauncher (before Isaac Sim init)
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure x_0 sensitivity in SmolVLA ODE")

parser.add_argument("--checkpoint", type=str, required=True,
                    help="SmolVLA checkpoint (pretrained_model dir)")
parser.add_argument("--task", type=str,
                    default="Place the cube into the matching slot on the platform")

parser.add_argument("--num-scenes", type=int, default=5,
                    help="Number of different env resets")
parser.add_argument("--num-samples", type=int, default=100,
                    help="ODE runs per condition")
parser.add_argument("--mu-magnitude", type=float, default=1.0,
                    help="Shift magnitude for directional test")
parser.add_argument("--output", type=str, default=None,
                    help="Save results JSON to this path")
parser.add_argument("--seed", type=int, default=42)

# Domain randomization
parser.add_argument("--no-domain-rand", action="store_true")

# Display
parser.add_argument("--gui", action="store_true")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.gui:
    args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)

# Section 2: LeRobot sys.path
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

# Section 3: Imports after AppLauncher
import json
import os
import time

import numpy as np
import torch

from so101_lab.policies.rl.flow_noise_smolvla import FlowNoiseSmolVLA
from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv
from so101_lab.tasks.figure_shape_placement.rl.env_cfg import FigureShapePlacementRLEnvCfg

JOINT_NAMES = ["base", "shoulder", "elbow", "wrist_pitch", "wrist_roll", "gripper"]


def _apply_domain_rand_flags(env_cfg, args):
    if args.no_domain_rand:
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import SceneEntityCfg
        from so101_lab.tasks.figure_shape_placement.rl import mdp

        env_cfg.events.randomize_light = None
        env_cfg.events.randomize_cube_material = None
        env_cfg.events.randomize_platform_material = None
        env_cfg.events.randomize_cube_mass = None
        env_cfg.observations.policy.images_top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False},
        )
        env_cfg.observations.policy.images_wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )
        env_cfg.events.reset_distractors = None


def run_baseline(policy, obs, actions_shape, num_samples):
    """Run ODE with random x_0, return (num_samples, 6) array of first actions."""
    actions = []
    for _ in range(num_samples):
        x_0 = torch.randn(actions_shape, device="cuda")
        chunk = policy.run_ode_with_x0(obs, x_0)
        actions.append(chunk[0])
    return np.array(actions)


def run_shifted(policy, obs, actions_shape, num_samples, joint, magnitude):
    """Run ODE with mu-shifted x_0 on one joint, return (num_samples, 6) array."""
    actions = []
    mu = torch.zeros(6, device="cuda")
    mu[joint] = magnitude
    for _ in range(num_samples):
        x_0 = torch.randn(actions_shape, device="cuda")
        x_0[0, 0, :6] = mu + torch.randn(6, device="cuda")
        chunk = policy.run_ode_with_x0(obs, x_0)
        actions.append(chunk[0])
    return np.array(actions)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Environment
    env_cfg = FigureShapePlacementRLEnvCfg()
    env_cfg.scene.num_envs = 1
    _apply_domain_rand_flags(env_cfg, args)
    env = IsaacLabGymEnv(env_cfg=env_cfg)

    # 2. Policy
    print(f"\nLoading SmolVLA from {args.checkpoint}")
    policy = FlowNoiseSmolVLA(
        checkpoint_path=args.checkpoint,
        device="cuda",
        task_string=args.task,
    )
    for p in policy.policy.parameters():
        p.requires_grad = False

    model = policy.policy.model
    actions_shape = (1, model.config.chunk_size, model.config.max_action_dim)

    # 3. Run measurements
    all_results = []

    print(f"\n{'=' * 70}")
    print(f"x_0 Sensitivity Analysis")
    print(f"  scenes={args.num_scenes}, samples={args.num_samples}, "
          f"mu_magnitude={args.mu_magnitude}")
    print(f"{'=' * 70}")

    for scene_idx in range(args.num_scenes):
        scene_seed = args.seed + scene_idx
        obs, _ = env.reset(seed=scene_seed)
        policy.policy.reset()

        print(f"\n--- Scene {scene_idx} (seed={scene_seed}) ---")

        # Baseline
        t0 = time.time()
        baseline_actions = run_baseline(policy, obs, actions_shape, args.num_samples)
        t_baseline = time.time() - t0

        baseline_mean = baseline_actions.mean(axis=0)
        baseline_std = baseline_actions.std(axis=0)

        print(f"\nBaseline ({args.num_samples} random x_0, {t_baseline:.1f}s):")
        print(f"  {'Joint':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*46}")
        for j in range(6):
            print(f"  {JOINT_NAMES[j]:<14} {baseline_mean[j]:>8.2f} "
                  f"{baseline_std[j]:>8.2f} "
                  f"{baseline_actions[:, j].min():>8.2f} "
                  f"{baseline_actions[:, j].max():>8.2f}")

        # Directional sensitivity
        directional = {}
        t0 = time.time()
        for shift_joint in range(6):
            shifted_actions = run_shifted(
                policy, obs, actions_shape, args.num_samples,
                shift_joint, args.mu_magnitude,
            )
            shifted_mean = shifted_actions.mean(axis=0)
            shifted_std = shifted_actions.std(axis=0)
            delta = shifted_mean - baseline_mean
            directional[shift_joint] = {
                "mean": shifted_mean.tolist(),
                "std": shifted_std.tolist(),
                "delta": delta.tolist(),
            }
        t_dir = time.time() - t0

        print(f"\nDirectional sensitivity (mu_magnitude={args.mu_magnitude}, {t_dir:.1f}s):")
        print(f"  {'Shifted joint':<14} → {'delta[0]':>8} {'delta[1]':>8} "
              f"{'delta[2]':>8} {'delta[3]':>8} {'delta[4]':>8} {'delta[5]':>8}")
        print(f"  {'-'*70}")
        for shift_joint in range(6):
            d = directional[shift_joint]["delta"]
            print(f"  {JOINT_NAMES[shift_joint]:<14} → "
                  + " ".join(f"{v:>8.3f}" for v in d))

        scene_result = {
            "seed": scene_seed,
            "baseline_mean": baseline_mean.tolist(),
            "baseline_std": baseline_std.tolist(),
            "directional": directional,
        }
        all_results.append(scene_result)

    # 4. Aggregate
    print(f"\n{'=' * 70}")
    print(f"AGGREGATE ({args.num_scenes} scenes)")
    print(f"{'=' * 70}")

    all_stds = np.array([r["baseline_std"] for r in all_results])
    mean_std = all_stds.mean(axis=0)

    print(f"\nMean baseline std across scenes:")
    print(f"  {'Joint':<14} {'Mean std':>8}")
    print(f"  {'-'*24}")
    for j in range(6):
        print(f"  {JOINT_NAMES[j]:<14} {mean_std[j]:>8.3f}")

    # Directional: average absolute delta across scenes
    all_deltas = np.zeros((args.num_scenes, 6, 6))  # (scenes, shifted_joint, action_joint)
    for s, r in enumerate(all_results):
        for sj in range(6):
            all_deltas[s, sj] = r["directional"][sj]["delta"]

    mean_abs_delta = np.abs(all_deltas).mean(axis=0)  # (6, 6)
    diagonal_delta = np.array([mean_abs_delta[j, j] for j in range(6)])

    print(f"\nMean |delta| on diagonal (shift joint j → action joint j):")
    print(f"  {'Joint':<14} {'|delta|':>8}")
    print(f"  {'-'*24}")
    for j in range(6):
        print(f"  {JOINT_NAMES[j]:<14} {diagonal_delta[j]:>8.3f}")

    # Conclusion
    threshold = 1.0  # motor units
    significant = mean_std > threshold
    print(f"\nConclusion (threshold={threshold} motor units):")
    if significant.any():
        joints_str = ", ".join(JOINT_NAMES[j] for j in range(6) if significant[j])
        print(f"  x_0 has SIGNIFICANT influence on: {joints_str}")
        print(f"  → Noise prior CAN steer these joints")
    else:
        print(f"  x_0 has LOW influence (all std < {threshold})")
        print(f"  → Noise prior unlikely to help")

    # 5. Save JSON
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "checkpoint": args.checkpoint,
            "num_scenes": args.num_scenes,
            "num_samples": args.num_samples,
            "mu_magnitude": args.mu_magnitude,
            "seed": args.seed,
            "mean_baseline_std": mean_std.tolist(),
            "mean_diagonal_delta": diagonal_delta.tolist(),
            "scenes": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved results: {args.output}")

    env.close()


if __name__ == "__main__":
    main(args)
    app_launcher.app.close()
