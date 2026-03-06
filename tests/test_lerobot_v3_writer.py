#!/usr/bin/env python3
"""Test script for LeRobot v3.0 dataset writer."""

import shutil
import tempfile
from pathlib import Path

import numpy as np


def test_v3_writer():
    """Test basic v3.0 writer functionality."""
    from so101_lab.data.lerobot_dataset import LeRobotDatasetWriter

    # Create temp directory
    output_dir = Path(tempfile.mkdtemp(prefix="lerobot_test_"))

    try:
        # Create writer
        writer = LeRobotDatasetWriter(
            output_dir=output_dir,
            fps=30,
            task="test task",
            codec="h264",
            crf=23,
        )

        # Create synthetic episodes
        for ep in range(3):
            num_frames = np.random.randint(50, 100)
            writer.set_task(f"test task {ep % 2}")

            for _ in range(num_frames):
                frame = {
                    "observation.state": np.random.randn(6).astype(np.float32),
                    "action": np.random.randn(6).astype(np.float32),
                    "observation.images.top": np.random.randint(
                        0, 255, (480, 640, 3), dtype=np.uint8
                    ),
                    "observation.images.wrist": np.random.randint(
                        0, 255, (480, 640, 3), dtype=np.uint8
                    ),
                }
                writer.add_frame(frame)

            ep_idx = writer.save_episode()
            print(f"Saved episode {ep_idx} with {num_frames} frames")

        writer.close()

        # Verify structure
        print("\nVerifying v3.0 structure:")

        # Check meta files
        assert (output_dir / "meta" / "info.json").exists(), "info.json missing"
        assert (output_dir / "meta" / "stats.json").exists(), "stats.json missing"
        assert (output_dir / "meta" / "tasks.parquet").exists(), "tasks.parquet missing"
        assert (
            output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ).exists(), "episodes metadata missing"
        print("  ✓ meta/ files present")

        # Check data parquet
        data_files = list((output_dir / "data").rglob("*.parquet"))
        assert len(data_files) > 0, "No data parquet files"
        print(f"  ✓ data/ has {len(data_files)} parquet file(s)")

        # Check videos
        video_files = list((output_dir / "videos").rglob("*.mp4"))
        assert len(video_files) >= 2, "Missing video files"
        print(f"  ✓ videos/ has {len(video_files)} video file(s)")

        # Check video directories
        assert (output_dir / "videos" / "observation.images.top").exists()
        assert (output_dir / "videos" / "observation.images.wrist").exists()
        print("  ✓ videos/ organized by camera")

        # Load and verify info.json
        import json
        with open(output_dir / "meta" / "info.json") as f:
            info = json.load(f)
        assert info["codebase_version"] == "v3.0"
        assert info["total_episodes"] == 3
        print(f"  ✓ info.json: {info['total_episodes']} episodes, {info['total_frames']} frames")

        # Load and verify tasks.parquet
        import pandas as pd
        tasks_df = pd.read_parquet(output_dir / "meta" / "tasks.parquet")
        assert len(tasks_df) == 2  # "test task 0" and "test task 1"
        print(f"  ✓ tasks.parquet: {len(tasks_df)} tasks")

        # Load and verify episodes metadata
        episodes_df = pd.read_parquet(
            output_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        )
        assert len(episodes_df) == 3
        assert "data/chunk_index" in episodes_df.columns
        assert "videos/observation_images_top/chunk_index" in episodes_df.columns
        print(f"  ✓ episodes metadata: {len(episodes_df)} episodes with chunk info")

        # Load and verify data parquet
        data_df = pd.read_parquet(data_files[0])
        assert "episode_index" in data_df.columns
        assert "observation.state" in data_df.columns
        assert "action" in data_df.columns
        print(f"  ✓ data parquet: {len(data_df)} frames")

        print("\n✓ All v3.0 structure checks passed!")

        # Print directory tree
        print("\nDirectory structure:")
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(output_dir)
                size = path.stat().st_size / 1024
                print(f"  {rel} ({size:.1f} KB)")

        # Test resume functionality
        print("\n--- Testing Resume ---")
        test_resume(output_dir, info["total_episodes"], info["total_frames"])

    finally:
        # Cleanup
        shutil.rmtree(output_dir, ignore_errors=True)


def test_resume(output_dir: Path, prev_episodes: int, prev_frames: int):
    """Test resuming from existing dataset."""
    from so101_lab.data.lerobot_dataset import LeRobotDatasetWriter

    # Reopen existing dataset
    writer = LeRobotDatasetWriter(
        output_dir=output_dir,
        fps=30,
        task="resume test",
    )

    assert writer.total_episodes == prev_episodes, f"Expected {prev_episodes} episodes, got {writer.total_episodes}"
    assert writer.total_frames == prev_frames, f"Expected {prev_frames} frames, got {writer.total_frames}"
    print(f"  ✓ Resumed with {writer.total_episodes} episodes, {writer.total_frames} frames")

    # Add another episode
    for _ in range(50):
        frame = {
            "observation.state": np.random.randn(6).astype(np.float32),
            "action": np.random.randn(6).astype(np.float32),
            "observation.images.top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "observation.images.wrist": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        writer.add_frame(frame)

    ep_idx = writer.save_episode()
    print(f"  ✓ Saved episode {ep_idx} after resume")

    writer.close()

    # Verify updated metadata
    import json
    with open(output_dir / "meta" / "info.json") as f:
        info = json.load(f)

    assert info["total_episodes"] == prev_episodes + 1
    assert info["total_frames"] == prev_frames + 50
    print(f"  ✓ Updated: {info['total_episodes']} episodes, {info['total_frames']} frames")
    print("\n✓ Resume test passed!")


if __name__ == "__main__":
    test_v3_writer()
