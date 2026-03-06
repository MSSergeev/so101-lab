#!/usr/bin/env python3
"""Verify dataset compatibility with LeRobot v3.0 API."""

import argparse
import json
from pathlib import Path


def verify_structure(dataset_path: Path) -> bool:
    """Verify v3.0 directory structure."""
    print("\n=== Structure Check ===")

    required = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.parquet",
    ]

    ok = True
    for rel_path in required:
        path = dataset_path / rel_path
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {rel_path}")
        if not exists:
            ok = False

    # Check data directory
    data_files = list((dataset_path / "data").rglob("*.parquet"))
    print(f"  {'✓' if data_files else '✗'} data/*.parquet ({len(data_files)} files)")

    # Check videos directory
    video_files = list((dataset_path / "videos").rglob("*.mp4"))
    print(f"  {'✓' if video_files else '✗'} videos/*.mp4 ({len(video_files)} files)")

    # Check episodes metadata
    episodes_files = list((dataset_path / "meta" / "episodes").rglob("*.parquet"))
    print(f"  {'✓' if episodes_files else '✗'} meta/episodes/*.parquet ({len(episodes_files)} files)")

    return ok and data_files and video_files


def verify_metadata(dataset_path: Path) -> dict:
    """Load and display metadata."""
    print("\n=== Metadata ===")

    with open(dataset_path / "meta" / "info.json") as f:
        info = json.load(f)

    print(f"  Version: {info.get('codebase_version', 'unknown')}")
    print(f"  Robot: {info.get('robot_type', 'unknown')}")
    print(f"  FPS: {info.get('fps', 'unknown')}")
    print(f"  Episodes: {info.get('total_episodes', 'unknown')}")
    print(f"  Frames: {info.get('total_frames', 'unknown')}")

    return info


def verify_lerobot_load(dataset_path: Path) -> bool:
    """Try loading with LeRobot API."""
    print("\n=== LeRobot API Load ===")

    try:
        # lerobot >= 0.5: lerobot.datasets, < 0.5: lerobot.common.datasets
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Load local dataset by setting root to dataset path
        # repo_id is just a name, root overrides the actual path
        print(f"  Loading from: {dataset_path}")

        dataset = LeRobotDataset(
            repo_id="local_dataset",
            root=dataset_path,
            tolerance_s=1 / 30,  # Match FPS tolerance
        )

        print(f"  ✓ Loaded successfully!")
        print(f"  Length: {len(dataset)}")
        print(f"  Features: {list(dataset.features.keys())}")

        # Try to get a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  Sample keys: {list(sample.keys())}")

            # Check for images
            for key in sample:
                if "image" in key:
                    img = sample[key]
                    print(f"  {key}: shape={img.shape if hasattr(img, 'shape') else type(img)}")

        return True

    except ImportError:
        print("  ✗ LeRobot not installed in this environment")
        print("  Run in lerobot-env: act-lerobot")
        return False
    except Exception as e:
        print(f"  ✗ Load failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify LeRobot dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset")
    args = parser.parse_args()

    dataset_path = args.dataset_path.resolve()
    print(f"Dataset: {dataset_path}")

    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist")
        return 1

    verify_structure(dataset_path)
    verify_metadata(dataset_path)
    verify_lerobot_load(dataset_path)

    return 0


if __name__ == "__main__":
    exit(main())
