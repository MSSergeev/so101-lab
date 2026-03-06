"""Collect scripted IK rollouts with domain randomization for dataset generation.

Phase 0 (Variant A): IK scripted policy → 300+ rollouts with DR.
Output: LeRobot v3 dataset with next.reward + sim_rewards.pt side-file.

Usage:
    # 1 env, 5 test episodes
    python scripts/eval/collect_rollouts.py \\
        --policy ik --episodes 5 \\
        --output-dataset /tmp/test_rollouts \\
        --headless --num_envs 1

    # Full collection (4 envs, 300 episodes, no GUI)
    python scripts/eval/collect_rollouts.py \\
        --policy ik --episodes 300 \\
        --output-dataset data/recordings/ik_dr_v1 \\
        --headless --num_envs 4

    # With GUI
    python scripts/eval/collect_rollouts.py \\
        --policy ik --episodes 20 \\
        --output-dataset /tmp/test_rollouts --gui
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect IK scripted rollouts")
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment name (default: figure_shape_placement)")
parser.add_argument("--policy", type=str, default="ik", choices=["ik"])
parser.add_argument("--episodes", type=int, default=300)
parser.add_argument("--output-dataset", type=str, required=True)
parser.add_argument("--episode-length", type=float, default=30.0, help="Max episode length (s)")
parser.add_argument("--physics-hz", type=int, default=120)
parser.add_argument("--policy-hz", type=int, default=30)
parser.add_argument(
    "--reward-mode", type=str, default="sim+success",
    choices=["success", "sim", "sim+success"],
)
parser.add_argument("--success-bonus", type=float, default=10.0)
parser.add_argument("--reward-weights", type=str, default="",
                    help="e.g. 'drop_penalty=-10,time_penalty=-0.05'")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--phase-timeout", type=int, default=None,
                    help="Max steps per IK phase (0=unlimited, default: from config yaml)")
parser.add_argument("--realtime", action="store_true", help="Rate-limit simulation to physics_hz")
parser.add_argument("--gui", action="store_true", help="Run with GUI")
parser.add_argument("--no-domain-rand", action="store_true")
parser.add_argument("--no-randomize-light", action="store_true")
parser.add_argument("--no-randomize-physics", action="store_true")
parser.add_argument("--no-camera-noise", action="store_true")
parser.add_argument("--no-distractors", action="store_true")
parser.add_argument("--success-only", action="store_true",
                    help="Only save episodes where cube was placed in slot at episode end")
parser.add_argument("--config", type=str, default="configs/collect_rollouts.yaml",
                    help="Path to collect_rollouts.yaml")
parser.add_argument("--crf", type=int, default=23)
parser.add_argument("--gop", type=str, default="auto")
parser.add_argument("--task", type=str, default="pick up cube and place in slot")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if not args.gui:
    args.headless = True
args.enable_cameras = True

app_launcher = AppLauncher(args)

# ── Post-launch imports ─────────────────────────────────────────────────────
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from so101_lab.utils import disable_rate_limiting

if args.realtime:
    try:
        import carb
        settings = carb.settings.get_settings()
        settings.set("/app/runLoops/main/rateLimitEnabled", True)
        settings.set("/app/runLoops/main/rateLimitFrequency", args.physics_hz)
    except Exception:
        pass
else:
    disable_rate_limiting()

from isaaclab.envs import ManagerBasedRLEnv

from so101_lab.data.converters import joint_rad_to_motor_normalized
from so101_lab.data.lerobot_dataset import LeRobotDatasetWriter
from so101_lab.policies.rl.ik_policy import IKScriptedPolicy
from so101_lab.rl.domain_rand import apply_domain_rand_flags
from so101_lab.tasks import get_rl_task

# Task-specific constants (lazy-loaded in main based on --env)
SUCCESS_THRESHOLD = None
ORIENTATION_THRESHOLD = None
TABLE_HEIGHT = None
slot_pos_w = None


def parse_reward_weights(weights_str: str) -> dict[str, float]:
    if not weights_str.strip():
        return {}
    out = {}
    for pair in weights_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            name, val = pair.split("=", 1)
            out[name.strip()] = float(val.strip())
    return out


def compute_reward_scalar(
    sim_metrics: dict[str, float],
    success: bool,
    reward_mode: str,
    weights: dict[str, float],
    success_bonus: float,
) -> float:
    if reward_mode == "success":
        return 1.0 if success else 0.0
    sim_r = sum(weights.get(n, 1.0) * v for n, v in sim_metrics.items())
    if reward_mode == "sim":
        return sim_r
    return sim_r + (success_bonus if success else 0.0)


def compute_success(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    z_tolerance: float,
    yaw_threshold_rad: float,
) -> torch.Tensor:
    """(N,) bool — XY + Z + yaw check with overrideable thresholds."""
    import math as _math
    platform = env.scene["platform"]
    cube = env.scene["cube"]

    platform_quat = platform.data.root_quat_w
    cube_pos = cube.data.root_pos_w
    cube_quat = cube.data.root_quat_w

    slot_local = slot_pos_w(env)
    slot_world = slot_local + env.scene.env_origins

    dist_xy = torch.norm(cube_pos[:, :2] - slot_world[:, :2], dim=1)
    pos_xy_ok = dist_xy < xy_threshold

    pos_z_ok = torch.abs(cube_pos[:, 2] - (TABLE_HEIGHT + 0.006)) < z_tolerance

    from isaaclab.utils.math import euler_xyz_from_quat
    _, _, platform_yaw = euler_xyz_from_quat(platform_quat)
    _, _, cube_yaw = euler_xyz_from_quat(cube_quat)
    angle_diff = (cube_yaw - platform_yaw) % (_math.pi / 2)
    rot_ok = (angle_diff < yaw_threshold_rad) | (angle_diff > (_math.pi / 2 - yaw_threshold_rad))

    return pos_xy_ok & pos_z_ok & rot_ok


def get_sim_metrics(env: ManagerBasedRLEnv, env_idx: int) -> dict[str, float]:
    """Per-term weighted reward values from reward manager for one env."""
    rm = env.reward_manager
    return {name: float(rm._step_reward[env_idx, i]) for i, name in enumerate(rm._term_names)}


def build_frame(obs_dict: dict, action: torch.Tensor, env_idx: int) -> dict:
    obs = obs_dict["policy"]
    joint_pos_norm = joint_rad_to_motor_normalized(obs["joint_pos"][env_idx].cpu().numpy())
    action_norm = joint_rad_to_motor_normalized(action[env_idx].cpu().numpy())
    img_top = obs["images_top"][env_idx].cpu().numpy().astype(np.uint8)
    img_wrist = obs["images_wrist"][env_idx].cpu().numpy().astype(np.uint8)
    return {
        "observation.state": joint_pos_norm.astype(np.float32),
        "action": action_norm.astype(np.float32),
        "observation.images.top": img_top,
        "observation.images.wrist": img_wrist,
        "next.reward": np.zeros((), dtype=np.float32),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import math as _math
    import yaml as _yaml

    # Load task-specific constants
    global SUCCESS_THRESHOLD, ORIENTATION_THRESHOLD, TABLE_HEIGHT, slot_pos_w
    if args.env in ("figure_shape_placement", "figure_shape_placement_easy"):
        from so101_lab.tasks.figure_shape_placement.env_cfg import (
            SUCCESS_THRESHOLD as _st, ORIENTATION_THRESHOLD as _ot, TABLE_HEIGHT as _th,
        )
        from so101_lab.tasks.figure_shape_placement.rl.mdp.observations import slot_pos_w as _spw
        SUCCESS_THRESHOLD, ORIENTATION_THRESHOLD, TABLE_HEIGHT, slot_pos_w = _st, _ot, _th, _spw
    else:
        raise ValueError(f"collect_rollouts.py: no success constants for task '{args.env}'")

    num_envs = args.num_envs
    reward_weights = parse_reward_weights(args.reward_weights)
    with open(args.config) as _f:
        _cfg = _yaml.safe_load(_f)
    _scfg = _cfg.get("success", {})
    success_xy = _scfg.get("xy_threshold", SUCCESS_THRESHOLD)
    success_z_tol = _scfg.get("z_tolerance", 0.003)
    success_yaw_rad = _math.radians(_scfg.get("yaw_threshold_deg", ORIENTATION_THRESHOLD))
    gop_value = None if args.gop == "auto" else int(args.gop)

    RLEnvCfgClass = get_rl_task(args.env)
    env_cfg = RLEnvCfgClass()
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = args.episode_length
    env_cfg.sim.dt = 1.0 / args.physics_hz
    env_cfg.decimation = args.physics_hz // args.policy_hz
    apply_domain_rand_flags(env_cfg, args)

    print(f"Creating RL env ({num_envs} envs) ...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    policy = IKScriptedPolicy(env)
    if args.phase_timeout is not None:
        policy.phase_timeout = args.phase_timeout

    dataset = LeRobotDatasetWriter(
        args.output_dataset,
        fps=args.policy_hz,
        task=args.task,
        crf=args.crf,
        gop=gop_value,
        extra_features={"next.reward": ((), np.float32)},
    )

    # Sim rewards side-file (accumulated across all episodes, load existing on resume)
    sim_rewards_all: dict[str, list] = {}
    sr_path = Path(args.output_dataset) / "sim_rewards.pt"
    if sr_path.exists():
        existing = torch.load(sr_path, weights_only=True)
        for k, v in existing.items():
            sim_rewards_all[k] = v.tolist()
        print(f"[RESUME] Loaded sim_rewards.pt ({len(existing['episode_index'])} frames)")

    # Per-env buffers
    env_frames: list[list[dict]] = [[] for _ in range(num_envs)]
    env_sim_step: list[list[dict]] = [[] for _ in range(num_envs)]
    # Track success across episode (updated before env.step to avoid stale post-reset state)
    env_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    saved_episodes = 0
    total_episodes = 0
    success_episodes = 0
    all_env_ids = torch.arange(num_envs, device=device)

    policy.reset(all_env_ids)
    obs_dict, _ = env.reset()

    from so101_lab.policies.rl.ik_policy import Phase as IKPhase

    pbar = tqdm(total=args.episodes, desc="Episodes collected")
    step_count = 0

    while saved_episodes < args.episodes:
        # Compute action for all envs
        action = policy.compute()  # (N, 6) radians

        # Record frame BEFORE step
        for i in range(num_envs):
            env_frames[i].append(build_frame(obs_dict, action, i))

        # Compute success before step — post-step value is stale for auto-reset envs
        pre_step_success = compute_success(env, success_xy, success_z_tol, success_yaw_rad)
        env_success |= pre_step_success

        # Step (ManagerBasedRLEnv auto-resets terminated/truncated envs internally)
        obs_dict, _, terminated, truncated, _ = env.step(action)
        step_count += 1

        # Post-step success (correct for non-terminated envs)
        post_step_success = compute_success(env, success_xy, success_z_tol, success_yaw_rad)
        # For terminated/truncated envs the env already reset → cube is in new position.
        # Use pre-step value instead (= state right after the last action, before reset).
        success_mask = torch.where(terminated | truncated, pre_step_success, post_step_success)

        # Per-step status
        if num_envs == 1:
            ee = env.scene["ee_frame"].data.target_pos_w[0, 0, :].cpu()
            ee_quat_dbg = env.scene["ee_frame"].data.target_quat_w[:, 0, :]
            tgt = policy._get_targets(ee_quat_dbg)[0, :3].cpu()
            dist = (ee - tgt).norm().item()
            phase_name = IKPhase(int(policy._phase[0])).name
            phase_step = int(policy._phase_steps[0])
            pbar.set_postfix({
                "phase": phase_name,
                "step": phase_step,
                "dist": f"{dist:.3f}",
                "frames": len(env_frames[0]),
            }, refresh=True)
            if step_count % 30 == 0:
                from isaaclab.utils.math import euler_xyz_from_quat
                ee_quat = env.scene["ee_frame"].data.target_quat_w[0, 0, :].cpu()
                roll, pitch, yaw = euler_xyz_from_quat(ee_quat.unsqueeze(0))
                rpy_deg = [round(float(v[0]) * 57.3, 1) for v in (roll, pitch, yaw)]
                print(f"  [dbg] step={step_count} phase={phase_name} step_in_phase={phase_step} "
                      f"ee={ee.tolist()} tgt={tgt.tolist()} dist={dist:.4f} "
                      f"ee_quat={ee_quat.tolist()} rpy_deg={rpy_deg}")

        for i in range(num_envs):
            if args.reward_mode in ("sim", "sim+success"):
                metrics = get_sim_metrics(env, i)
            else:
                metrics = {}
            env_sim_step[i].append(metrics)

            reward = compute_reward_scalar(
                metrics, bool(success_mask[i]),
                args.reward_mode, reward_weights, args.success_bonus,
            )
            env_frames[i][-1]["next.reward"] = np.array(reward, dtype=np.float32)

        # Detect done episodes
        ik_done = policy.is_done()
        done = ik_done | terminated | truncated

        for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
            if saved_episodes >= args.episodes:
                break

            frames = env_frames[i]
            is_success = bool(env_success[i])
            if frames:
                total_episodes += 1
                if is_success:
                    success_episodes += 1
            if frames and (not args.success_only or is_success):
                # Accumulate sim rewards into global table
                n = len(frames)
                ep_idx = dataset._current_episode_idx
                sim_rewards_all.setdefault("episode_index", []).extend([ep_idx] * n)
                for key in (env_sim_step[i][0] if env_sim_step[i] else {}):
                    sim_rewards_all.setdefault(key, []).extend(
                        m.get(key, 0.0) for m in env_sim_step[i]
                    )

                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                saved_episodes += 1
                pbar.update(1)

            env_frames[i].clear()
            env_sim_step[i].clear()
            env_success[i] = False

            # Reset policy; env auto-reset already happened inside step() for
            # terminated/truncated. For IK-done-only envs, trigger manual reset.
            env_id = torch.tensor([i], device=device)
            policy.reset(env_id)

            if ik_done[i] and not terminated[i] and not truncated[i]:
                # env didn't auto-reset — do it now
                env._reset_idx(env_id)
                # Refresh scene data (cube pos, ee_frame, etc.) after manual reset;
                # without this, scene buffers are stale until the next physics step.
                env.scene.update(dt=env.physics_dt)
                obs_dict = env.observation_manager.compute(update_history=True)

    pbar.close()
    dataset.close()

    # Save sim_rewards.pt
    if sim_rewards_all:
        sr_data = {}
        for k, v in sim_rewards_all.items():
            dtype = torch.int64 if k == "episode_index" else torch.float32
            sr_data[k] = torch.tensor(v, dtype=dtype)
        torch.save(sr_data, sr_path)
        print(f"Saved sim_rewards.pt → {sr_path}")

    env.close()
    sr = f"{success_episodes}/{total_episodes} successful ({100*success_episodes/max(total_episodes,1):.1f}%)"
    print(f"\nDone. saved={saved_episodes}  {sr} → {args.output_dataset}")


if __name__ == "__main__":
    main()
    app_launcher.app.close()
