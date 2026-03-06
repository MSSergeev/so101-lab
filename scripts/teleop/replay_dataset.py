#!/usr/bin/env python3
"""Replay actions from an existing dataset with updated cameras.

Loads a source dataset (parquet + episode_metadata.json), replays each
episode's actions in the simulator with the current camera setup, and
writes a new dataset. Object positions and light are restored from
the original initial_state metadata.

Usage:
    python scripts/teleop/replay_dataset.py \
        --source data/recordings/figure_shape_placement_v1 \
        --output data/recordings/figure_shape_placement_v1_newcam

    # Replay only specific episodes
    python scripts/teleop/replay_dataset.py \
        --source data/recordings/figure_shape_placement_v1 \
        --output data/recordings/figure_shape_placement_v1_newcam \
        --episodes 0,1,2
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Replay dataset with new cameras")
parser.add_argument("--source", type=str, required=True,
                    help="Path to source dataset")
parser.add_argument("--output", type=str, required=True,
                    help="Path for new dataset")
parser.add_argument("--env", type=str, default="figure_shape_placement",
                    help="Task environment")
parser.add_argument("--physics-hz", type=int, default=120,
                    help="Physics simulation frequency")
parser.add_argument("--policy-hz", type=int, default=30,
                    help="Policy/control frequency")
parser.add_argument("--render-hz", type=int, default=30,
                    help="Rendering frequency")
parser.add_argument("--episodes", type=str, default=None,
                    help="Comma-separated episode indices to replay (default: all)")
parser.add_argument("--crf", type=int, default=23,
                    help="Video quality (0-51, lower=better)")
parser.add_argument("--gop", type=str, default="2",
                    help="GOP size / keyframe interval")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from so101_lab.data.collector import RecordingManager
from so101_lab.data.converters import motor_normalized_to_joint_rad
from so101_lab.data.lerobot_dataset import LeRobotDatasetWriter
from so101_lab.tasks import get_task


def load_source_dataset(source_path: str, episode_indices: list[int] | None):
    """Load actions and metadata from source dataset.

    Returns:
        episodes: list of dicts with keys: index, seed, initial_state, actions, task
    """
    source = Path(source_path)

    # Load episode metadata (seeds + initial states)
    metadata_path = source / "meta" / "episode_metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found")
        sys.exit(1)
    with open(metadata_path) as f:
        episode_metadata = {int(k): v for k, v in json.load(f).items()}

    # Load info.json for fps and task
    info_path = source / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    total_episodes = info.get("total_episodes", len(episode_metadata))

    # Load task descriptions
    task = "manipulation task"
    tasks_path = source / "meta" / "tasks.parquet"
    if tasks_path.exists():
        tasks_df = pd.read_parquet(tasks_path)
        if len(tasks_df) > 0:
            task = tasks_df.iloc[0]["task"]

    # Load all data parquet files
    data_dir = source / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        print(f"ERROR: No parquet files found in {data_dir}")
        sys.exit(1)
    all_data = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

    # Determine which episodes to replay
    if episode_indices is not None:
        ep_list = episode_indices
    else:
        ep_list = list(range(total_episodes))

    episodes = []
    for ep_idx in ep_list:
        if ep_idx not in episode_metadata:
            print(f"WARNING: Episode {ep_idx} not in metadata, skipping")
            continue

        meta = episode_metadata[ep_idx]
        ep_data = all_data[all_data["episode_index"] == ep_idx]
        if len(ep_data) == 0:
            print(f"WARNING: Episode {ep_idx} has no frames, skipping")
            continue

        # Extract actions (normalized motor positions)
        if "action" not in ep_data.columns:
            print(f"WARNING: No 'action' column in episode {ep_idx}, skipping")
            continue

        actions = np.stack(ep_data["action"].values)

        episodes.append({
            "index": ep_idx,
            "seed": meta.get("seed"),
            "initial_state": meta.get("initial_state"),
            "actions": actions,
            "task": task,
        })

    return episodes


def main():
    # Parse episode indices
    episode_indices = None
    if args.episodes:
        episode_indices = [int(x.strip()) for x in args.episodes.split(",")]

    print(f"\n{'=' * 60}")
    print("Replay Dataset")
    print(f"{'=' * 60}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Env: {args.env}")
    print(f"Physics: {args.physics_hz} Hz, Policy: {args.policy_hz} Hz, Render: {args.render_hz} Hz")
    if episode_indices:
        print(f"Episodes: {episode_indices}")
    else:
        print("Episodes: all")
    print(f"{'=' * 60}\n")

    # Load source dataset
    print("[1/3] Loading source dataset...")
    episodes = load_source_dataset(args.source, episode_indices)
    if not episodes:
        print("ERROR: No episodes to replay")
        sys.exit(1)
    print(f"  Loaded {len(episodes)} episodes")

    # Setup environment
    print("[2/3] Setting up environment...")
    EnvClass, EnvCfgClass = get_task(args.env)
    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 600.0  # Long enough for any episode

    env_cfg.sim.dt = 1.0 / args.physics_hz
    env_cfg.decimation = args.physics_hz // args.policy_hz
    env_cfg.sim.render_interval = args.physics_hz // args.render_hz

    camera_update_period = 1.0 / args.render_hz
    if hasattr(env_cfg.scene, 'top'):
        env_cfg.scene.top.update_period = camera_update_period
    if hasattr(env_cfg.scene, 'wrist'):
        env_cfg.scene.wrist.update_period = camera_update_period

    env = EnvClass(cfg=env_cfg)

    # Create output dataset
    gop_value = None if args.gop == "auto" else int(args.gop)
    dataset = LeRobotDatasetWriter(
        args.output, fps=args.policy_hz, task=episodes[0]["task"],
        crf=args.crf, gop=gop_value
    )
    recorder = RecordingManager(dataset, env)

    # Initial reset
    env.reset()

    # Replay each episode
    print(f"[3/3] Replaying {len(episodes)} episodes...")
    for i, ep in enumerate(episodes):
        ep_idx = ep["index"]
        actions = ep["actions"]  # (N, 6) normalized motor positions
        n_frames = len(actions)

        print(f"\n  Episode {ep_idx} ({i+1}/{len(episodes)}): {n_frames} frames")

        # Reset to exact initial state
        if ep["initial_state"] is None:
            print(f"    WARNING: No initial_state, using random reset")
            obs_dict, _ = env.reset()
        else:
            obs_dict, _ = env.reset_to_state(ep["initial_state"])

        # Setup recording
        if ep["task"]:
            recorder.set_task(ep["task"])
        if ep["seed"] is not None:
            recorder.set_episode_seed(ep["seed"])
        if ep["initial_state"] is not None:
            recorder.set_episode_initial_state(ep["initial_state"])
        recorder.on_reset(obs_dict)

        # Replay actions
        for frame_idx in range(n_frames):
            action_normalized = actions[frame_idx]  # (6,) normalized

            # Convert normalized motor → radians for sim
            action_rad = motor_normalized_to_joint_rad(action_normalized)
            action_tensor = torch.tensor(
                action_rad, dtype=torch.float32, device=env.device
            ).unsqueeze(0)  # (1, 6)

            # Record (obs_before, action) pair
            recorder.on_step(obs_dict, action_tensor)

            # Step environment
            obs_dict, _, _, _, _ = env.step(action_tensor)

        # Save episode
        recorder.on_episode_end(success=True)
        print(f"    Saved ({n_frames} frames)")

    # Cleanup
    dataset.close()
    env.close()

    print(f"\n{'=' * 60}")
    print("Replay Summary")
    print(f"{'=' * 60}")
    print(f"Total episodes: {dataset.total_episodes}")
    print(f"Total frames: {dataset.total_frames}")
    print(f"Dataset saved to: {args.output}")
    print(f"{'=' * 60}\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
