#!/usr/bin/env python3
"""Prepare LeRobot dataset for SAC reward classifier training.

Adds `next.reward` column (1.0 after gripper release, 0.0 before) and patches
camera shapes in info.json to [3, 128, 128] for SpatialLearnedEmbeddings compatibility.

Run in lerobot-env:
    eval "$(./activate_lerobot.sh)"
    python scripts/train/prepare_classifier_dataset.py \
        --dataset data/recordings/figure_shape_placement_v4 \
        --output data/recordings/figure_shape_placement_v4_labeled \
        --gripper-threshold 29.5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def find_last_gripper_transition(
    gripper_values: np.ndarray, threshold: float
) -> int | None:
    """Find last frame where gripper crosses threshold upward.

    Returns frame index of the transition, or None if no transition found.
    """
    below = gripper_values < threshold
    above = gripper_values >= threshold

    # Transition: below[i] and above[i+1]
    transitions = below[:-1] & above[1:]
    indices = np.where(transitions)[0]

    if len(indices) == 0:
        return None

    # Return the frame after the last transition (where gripper is open)
    return int(indices[-1] + 1)


def label_episodes(
    dataset_path: Path,
    gripper_threshold: float,
    gripper_index: int,
) -> np.ndarray:
    """Compute per-frame reward labels based on gripper transitions."""
    # Read all parquet data files
    data_dir = dataset_path / "data"
    parquet_files = sorted(data_dir.glob("*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True).sort_values("index").reset_index(drop=True)

    total_frames = len(df)
    rewards = np.zeros(total_frames, dtype=np.float32)

    episodes_with_release = 0
    total_success_frames = 0

    for ep_idx in sorted(df["episode_index"].unique()):
        ep_mask = df["episode_index"] == ep_idx
        ep_df = df[ep_mask]

        # Extract gripper values from observation.state
        states = np.stack(ep_df["observation.state"].values)
        gripper_values = states[:, gripper_index]

        transition_frame = find_last_gripper_transition(gripper_values, gripper_threshold)

        if transition_frame is None:
            print(f"  Warning: episode {ep_idx} has no gripper transition, all frames = 0.0")
            continue

        episodes_with_release += 1

        # All frames from transition onward get reward=1.0
        ep_indices = ep_df["index"].values
        success_start = ep_indices[transition_frame]
        success_mask = ep_indices >= success_start
        rewards[ep_indices[success_mask]] = 1.0
        total_success_frames += success_mask.sum()

    total_episodes = df["episode_index"].nunique()
    print(f"\nLabeling statistics:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Episodes with gripper release: {episodes_with_release}")
    print(f"  Episodes without release: {total_episodes - episodes_with_release}")
    print(f"  Total frames: {total_frames}")
    print(f"  Success frames: {total_success_frames} ({100 * total_success_frames / total_frames:.1f}%)")
    print(f"  Fail frames: {total_frames - total_success_frames}")

    return rewards


def main():
    parser = argparse.ArgumentParser(description="Label dataset with gripper-based reward")
    parser.add_argument("--dataset", type=str, required=True, help="Path to LeRobot dataset")
    parser.add_argument("--output", type=str, required=True, help="Output path for labeled dataset")
    parser.add_argument("--gripper-threshold", type=float, default=29.5,
                        help="Gripper state threshold for release detection (default: 29.5)")
    parser.add_argument("--gripper-index", type=int, default=5,
                        help="Index of gripper in observation.state (default: 5)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).absolute()
    output_path = Path(args.output).absolute()

    print(f"Loading dataset: {dataset_path}")

    # Patch missing total_tasks for older datasets (pre v3.0 field)
    import json
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    if "total_tasks" not in info:
        tasks_path = dataset_path / "meta" / "tasks.parquet"
        if tasks_path.exists():
            info["total_tasks"] = len(pd.read_parquet(tasks_path))
        else:
            info["total_tasks"] = 1
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"  Patched info.json: added total_tasks={info['total_tasks']}")

    dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path)

    print(f"Labeling with gripper threshold={args.gripper_threshold}, index={args.gripper_index}")
    rewards = label_episodes(dataset_path, args.gripper_threshold, args.gripper_index)

    print(f"\nAdding next.reward feature to dataset...")
    print(f"Output: {output_path}")

    add_features(
        dataset=dataset,
        features={
            "next.reward": (
                rewards,
                {"dtype": "float32", "shape": (1,), "names": None},
            ),
        },
        output_dir=output_path,
        repo_id="local_dataset_labeled",
    )

    # Fix parquet column order to match info.json features order.
    # add_features writes columns in arbitrary order, but LeRobotDataset
    # with episodes= filter requires exact match with features.arrow_schema.
    import json as _json

    with open(output_path / "meta" / "info.json") as f:
        info = _json.load(f)
    expected_order = [k for k, v in info["features"].items() if v["dtype"] != "video"]

    for pq_file in sorted((output_path / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq_file)
        if list(df.columns) != expected_order:
            df[expected_order].to_parquet(pq_file, index=False)

    # Re-encode videos to 128x128 for SpatialLearnedEmbeddings (4x4 at stride 32).
    import subprocess
    import tempfile

    video_dir = output_path / "videos"
    mp4_files = sorted(video_dir.glob("**/*.mp4"))
    if mp4_files:
        print(f"\nRe-encoding {len(mp4_files)} videos to 128x128...")
        for mp4 in mp4_files:
            with tempfile.NamedTemporaryFile(suffix=".mp4", dir=mp4.parent, delete=False) as tmp:
                tmp_path = Path(tmp.name)
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", str(mp4), "-vf", "scale=128:128",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-an", str(tmp_path)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg failed for {mp4}: {result.stderr}")
            tmp_path.rename(mp4)
            print(f"  {mp4.relative_to(output_path)}")

    # Update info.json: camera shapes and video_info dimensions.
    patched_cameras = []
    for key, feat in info["features"].items():
        if feat.get("dtype") == "video":
            feat["shape"] = [3, 128, 128]
            vi = feat.get("video_info", {})
            vi["video.height"] = 128
            vi["video.width"] = 128
            patched_cameras.append(key)
    if patched_cameras:
        out_info_path = output_path / "meta" / "info.json"
        with open(out_info_path, "w") as f:
            _json.dump(info, f, indent=4)
        print(f"Patched camera metadata to 128x128: {patched_cameras}")

    print("Done!")


if __name__ == "__main__":
    main()
