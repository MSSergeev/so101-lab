#!/usr/bin/env python3
"""Visualize LeRobot v3.0 dataset in Rerun viewer.

Supports native LeRobot v3.0 format with:
- Consolidated data files (many episodes per file)
- Concatenated video files
- Parquet metadata (tasks.parquet, episodes/*.parquet)

Usage:
    # View all episodes (up to 10 by default)
    python scripts/visualize_lerobot_rerun.py data/recordings/dataset

    # View specific episodes
    python scripts/visualize_lerobot_rerun.py data/recordings/dataset -e 0 2 5

    # Fast mode without images
    python scripts/visualize_lerobot_rerun.py data/recordings/dataset --skip-images

    # Load more episodes
    python scripts/visualize_lerobot_rerun.py data/recordings/dataset --max-episodes=50

    # Save to file instead of viewer
    python scripts/visualize_lerobot_rerun.py data/recordings/dataset --save output.rrd

    # No re-compression (raw RGB from video decode)
    python scripts/visualize_lerobot_rerun.py data/recordings/dataset --no-jpeg

    # Custom JPEG quality
    python scripts/visualize_lerobot_rerun.py data/recordings/dataset --jpeg-quality=95

    # Smaller batch for less RAM (~1GB instead of ~4.5GB)
    python scripts/visualize_lerobot_rerun.py data/recordings/dataset --batch-size=500
"""

import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rerun as rr
from PIL import Image

SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    """Encode image as JPEG."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class VideoReaderMultiFile:
    """Video reader supporting multiple files with batch prefetch."""

    def __init__(self, video_dir: Path, batch_size: int = 2500):
        self.video_dir = video_dir
        self.batch_size = batch_size
        self._current_file_idx = -1
        self._cap: cv2.VideoCapture | None = None
        self._buffer: dict[int, np.ndarray] = {}
        self._buffer_start = -1
        self._buffer_end = -1
        self._current_file_frames = 0

    def _open_file(self, file_idx: int):
        """Open a specific video file."""
        if self._cap is not None:
            self._cap.release()
        self._buffer.clear()
        self._buffer_start = -1
        self._buffer_end = -1

        video_path = self.video_dir / f"file-{file_idx:03d}.mp4"
        if not video_path.exists():
            self._cap = None
            self._current_file_idx = -1
            self._current_file_frames = 0
            return

        self._cap = cv2.VideoCapture(str(video_path))
        self._current_file_idx = file_idx
        self._current_file_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _load_batch(self, start_idx: int):
        """Load a batch of frames starting from start_idx."""
        if self._cap is None:
            return

        self._buffer.clear()
        self._buffer_start = start_idx
        self._buffer_end = min(start_idx + self.batch_size, self._current_file_frames)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        for idx in range(self._buffer_start, self._buffer_end):
            ret, frame = self._cap.read()
            if not ret:
                break
            self._buffer[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get_frame(self, file_idx: int, frame_idx: int) -> np.ndarray | None:
        """Get a specific frame by file index and frame index."""
        # Open new file if needed
        if file_idx != self._current_file_idx:
            self._open_file(file_idx)

        if self._cap is None:
            return None

        if frame_idx < 0 or frame_idx >= self._current_file_frames:
            return None

        # Check if in current buffer
        if frame_idx in self._buffer:
            return self._buffer[frame_idx]

        # Load new batch starting from requested frame
        self._load_batch(frame_idx)
        return self._buffer.get(frame_idx)

    def close(self):
        self._buffer.clear()
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def visualize_dataset(
    dataset_path: str,
    episodes: list[int] | None = None,
    max_episodes: int = 10,
    skip_images: bool = False,
    downsample: int = 1,
    save_path: str | None = None,
    jpeg: bool = True,
    jpeg_quality: int = 80,
    batch_size: int = 2500,
):
    """
    Visualize LeRobot dataset in Rerun.

    Args:
        dataset_path: Path to LeRobot dataset directory
        episodes: Specific episode indices to visualize (None = all up to max)
        max_episodes: Maximum episodes to load
        skip_images: Skip image data for faster loading
        downsample: Take every N-th frame
        save_path: Save to .rrd file instead of spawning viewer
        jpeg: Compress images as JPEG (reduces viewer memory ~10x)
        jpeg_quality: JPEG quality 1-100 (default: 80)
        batch_size: Number of video frames to prefetch at once (default: 2500)
    """
    dataset_path = Path(dataset_path)
    app_id = f"so101-lerobot-{dataset_path.name}"

    if save_path:
        rr.init(app_id, spawn=False)
        rr.save(save_path)
    else:
        rr.init(app_id, spawn=True)

    # Read info.json
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")

    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    fps = info["fps"]

    # Read tasks (v3.0 uses parquet, fallback to jsonl)
    tasks = {}
    tasks_parquet_path = dataset_path / "meta" / "tasks.parquet"
    tasks_jsonl_path = dataset_path / "meta" / "tasks.jsonl"

    if tasks_parquet_path.exists():
        tasks_df = pd.read_parquet(tasks_parquet_path)
        for idx, row in tasks_df.iterrows():
            # Support both formats: columns {task, task_index} and {task_index} with task as index
            if "task" in tasks_df.columns and "task_index" in tasks_df.columns:
                tasks[row["task_index"]] = row["task"]
            elif "task_index" in tasks_df.columns:
                tasks[row["task_index"]] = str(idx)
            else:
                tasks[idx] = str(row.iloc[0]) if len(row) > 0 else f"task_{idx}"
    elif tasks_jsonl_path.exists():
        with open(tasks_jsonl_path) as f:
            for line in f:
                task = json.loads(line)
                tasks[task["task_index"]] = task["task"]

    # Load episode metadata (v3.0 format)
    episodes_parquet_path = dataset_path / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    if not episodes_parquet_path.exists():
        raise FileNotFoundError(f"Episodes metadata not found at {episodes_parquet_path}")

    episodes_df = pd.read_parquet(episodes_parquet_path)

    # Filter episodes
    if episodes is not None:
        episodes_df = episodes_df[episodes_df["episode_index"].isin(episodes)]
    else:
        episodes_df = episodes_df.head(max_episodes)

    print(f"Loading {len(episodes_df)} episodes...")

    # Load ALL data files in chunk-000 (v3.0 multi-file support)
    data_dir = dataset_path / "data" / "chunk-000"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at {data_dir}")

    data_files = sorted(data_dir.glob("file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data files found in {data_dir}")

    # Load and concatenate all data files
    print(f"Loading {len(data_files)} data files...")
    data_dfs = [pd.read_parquet(f) for f in data_files]
    data_df = pd.concat(data_dfs, ignore_index=False).sort_values("index").reset_index(drop=True)

    # Open video readers (streaming - multi-file support)
    video_reader_top: VideoReaderMultiFile | None = None
    video_reader_wrist: VideoReaderMultiFile | None = None

    if not skip_images:
        for video_key in ["observation.images.top", "observation.images.wrist"]:
            video_dir = dataset_path / "videos" / video_key / "chunk-000"
            if not video_dir.exists():
                continue

            # Check if any video files exist
            video_files = list(video_dir.glob("file-*.mp4"))
            if not video_files:
                continue

            print(f"Opening {video_key} (batch_size={batch_size}, {len(video_files)} files)...")
            reader = VideoReaderMultiFile(video_dir, batch_size=batch_size)

            if video_key == "observation.images.top":
                video_reader_top = reader
            else:
                video_reader_wrist = reader

    global_frame = 0

    for _, ep_row in episodes_df.iterrows():
        ep_idx = int(ep_row["episode_index"])
        ep_name = f"episode_{ep_idx:06d}"

        # Extract episode data from main dataframe
        data_start_idx = int(ep_row["dataset_from_index"])
        data_end_idx = int(ep_row["dataset_to_index"])
        df = data_df.iloc[data_start_idx:data_end_idx]
        num_frames = len(df)
        frames_to_load = range(0, num_frames, downsample)

        # Get video file indices and frame ranges
        top_file_idx = int(ep_row.get("videos/observation.images.top/file_index", 0))
        top_from_frame = int(ep_row.get("videos/observation.images.top/from_frame", data_start_idx))
        wrist_file_idx = int(ep_row.get("videos/observation.images.wrist/file_index", 0))
        wrist_from_frame = int(ep_row.get("videos/observation.images.wrist/from_frame", data_start_idx))

        print(f"  {ep_name}: {len(frames_to_load)} frames (top: file-{top_file_idx:03d}@{top_from_frame}, wrist: file-{wrist_file_idx:03d}@{wrist_from_frame})")

        # Get task for this episode
        task_idx = int(ep_row["task_index"]) if "task_index" in ep_row else 0
        task_name = tasks.get(task_idx, "unknown")

        for i, local_frame in enumerate(frames_to_load):
            row = df.iloc[local_frame]

            # Set timelines
            rr.set_time("frame", sequence=global_frame)
            rr.set_time("episode_frame", sequence=local_frame)

            # Log episode index and task
            rr.log("episode/index", rr.Scalars(ep_idx))
            rr.log("episode/task", rr.TextDocument(task_name))

            # Log images (use video file index and frame offset)
            if not skip_images:
                if video_reader_top is not None:
                    video_frame_idx = top_from_frame + local_frame
                    img_top = video_reader_top.get_frame(top_file_idx, video_frame_idx)
                    if img_top is not None:
                        if jpeg:
                            rr.log("cameras/top/rgb", rr.EncodedImage(
                                contents=encode_jpeg(img_top, jpeg_quality),
                                media_type="image/jpeg"
                            ))
                        else:
                            rr.log("cameras/top/rgb", rr.Image(img_top))

                if video_reader_wrist is not None:
                    video_frame_idx = wrist_from_frame + local_frame
                    img_wrist = video_reader_wrist.get_frame(wrist_file_idx, video_frame_idx)
                    if img_wrist is not None:
                        if jpeg:
                            rr.log("cameras/wrist/rgb", rr.EncodedImage(
                                contents=encode_jpeg(img_wrist, jpeg_quality),
                                media_type="image/jpeg"
                            ))
                        else:
                            rr.log("cameras/wrist/rgb", rr.Image(img_wrist))

            # Log joint states (already in normalized motor positions)
            state = np.array(row["observation.state"])
            action = np.array(row["action"])

            for j, joint_name in enumerate(SO101_JOINT_NAMES):
                rr.log(f"robot/state/{joint_name}", rr.Scalars(state[j]))
                rr.log(f"robot/action/{joint_name}", rr.Scalars(action[j]))

            if "next.reward" in row.index:
                rr.log("reward", rr.Scalars(float(row["next.reward"])))

            global_frame += 1

    # Cleanup video readers
    if video_reader_top is not None:
        video_reader_top.close()
    if video_reader_wrist is not None:
        video_reader_wrist.close()

    print(f"Done. Total frames: {global_frame}")


def main():
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset in Rerun")
    parser.add_argument("dataset_path", type=str, help="Path to LeRobot dataset directory")
    parser.add_argument(
        "-e", "--episodes",
        type=int, nargs="+",
        help="Specific episode indices to visualize",
    )
    parser.add_argument(
        "--max-episodes",
        type=int, default=10,
        help="Maximum episodes to load (default: 10)",
    )
    parser.add_argument(
        "--save",
        type=str, metavar="PATH",
        help="Save to .rrd file instead of spawning viewer",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip images for faster loading",
    )
    parser.add_argument(
        "--downsample",
        type=int, default=1, metavar="N",
        help="Take every N-th frame (default: 1)",
    )
    parser.add_argument(
        "--jpeg",
        action="store_true", default=True,
        help="Compress images as JPEG (default: True, reduces memory ~10x)",
    )
    parser.add_argument(
        "--no-jpeg",
        action="store_true",
        help="Disable JPEG compression (raw images, ~10x more memory)",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int, default=80, metavar="Q",
        help="JPEG quality 1-100 (default: 80)",
    )
    parser.add_argument(
        "--batch-size",
        type=int, default=2500, metavar="N",
        help="Video frames to prefetch at once (default: 2500, ~4.5GB RAM)",
    )
    args = parser.parse_args()

    use_jpeg = args.jpeg and not args.no_jpeg

    visualize_dataset(
        dataset_path=args.dataset_path,
        episodes=args.episodes,
        max_episodes=args.max_episodes,
        skip_images=args.skip_images,
        downsample=args.downsample,
        save_path=args.save,
        jpeg=use_jpeg,
        jpeg_quality=args.jpeg_quality,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
