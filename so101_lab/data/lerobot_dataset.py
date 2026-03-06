# Adapted from: LeRobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: Simplified v3.0 writer for so101-lab recording workflow
"""LeRobot v3.0 dataset writer for direct recording (parquet + video format).

Output format based on LeRobot v3.0 specification:
https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3

Key differences from v2.x:
- Size-based file chunking (many episodes per file)
- Separate videos/ directory with per-camera organization
- Episode metadata in meta/episodes/ parquet files
- Tasks stored in tasks.parquet (not jsonl)
"""

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .video_utils import (
    concatenate_video_files,
    encode_images_to_video,
    get_file_size_mb,
    get_video_duration_s,
)


@dataclass
class ChunkFileState:
    """Tracks current chunk/file indices for size-based file management."""

    chunk_index: int = 0
    file_index: int = 0
    current_size_mb: float = 0.0


@dataclass
class VideoFileState:
    """Tracks video file state per camera."""

    chunk_index: int = 0
    file_index: int = 0
    current_size_mb: float = 0.0
    current_duration_s: float = 0.0
    current_frame_count: int = 0


@dataclass
class StatsComputer:
    """Compute statistics for normalization."""

    running_stats: dict = field(default_factory=dict)

    def add_episode(self, data: dict[str, np.ndarray]) -> dict[str, dict]:
        """Compute stats for a single episode. Returns episode stats dict."""
        episode_stats = {}
        for key, values in data.items():
            is_image = "images" in key and values.dtype == np.uint8

            if is_image:
                # Convert uint8 [0,255] -> float32 [0,1], HWC->CHW, per-channel stats
                vals_f = values.astype(np.float32) / 255.0
                # (N, H, W, C) -> (N, C, H, W)
                vals_f = vals_f.transpose(0, 3, 1, 2)
                # Per-channel stats: reduce over (N, H, W), keep C
                axes = (0, 2, 3)
                n_pixels = values.shape[0] * values.shape[1] * values.shape[2]

                ch_mean = vals_f.mean(axis=axes)
                ch_std = vals_f.std(axis=axes)
                ch_min = vals_f.min(axis=axes)
                ch_max = vals_f.max(axis=axes)

                episode_stats[key] = {
                    "mean": ch_mean.tolist(),
                    "std": ch_std.tolist(),
                    "min": ch_min.tolist(),
                    "max": ch_max.tolist(),
                    "count": len(values),
                }

                # Running stats for images: shape (C,)
                if key not in self.running_stats:
                    c = values.shape[3]
                    self.running_stats[key] = {
                        "sum": np.zeros(c, dtype=np.float64),
                        "sum_sq": np.zeros(c, dtype=np.float64),
                        "min": np.full(c, np.inf, dtype=np.float64),
                        "max": np.full(c, -np.inf, dtype=np.float64),
                        "count": 0,
                    }
                rs = self.running_stats[key]
                rs["sum"] += vals_f.sum(axis=axes).astype(np.float64)
                rs["sum_sq"] += (vals_f.astype(np.float64) ** 2).sum(axis=axes)
                rs["min"] = np.minimum(rs["min"], ch_min)
                rs["max"] = np.maximum(rs["max"], ch_max)
                rs["count"] += n_pixels

            elif values.dtype in (np.float32, np.float64):
                stats = {
                    "mean": values.mean(axis=0).tolist(),
                    "std": values.std(axis=0).tolist(),
                    "min": values.min(axis=0).tolist(),
                    "max": values.max(axis=0).tolist(),
                    "count": len(values),
                }
                episode_stats[key] = stats

                if key not in self.running_stats:
                    self.running_stats[key] = {
                        "sum": np.zeros_like(values[0], dtype=np.float64),
                        "sum_sq": np.zeros_like(values[0], dtype=np.float64),
                        "min": np.full_like(values[0], np.inf, dtype=np.float64),
                        "max": np.full_like(values[0], -np.inf, dtype=np.float64),
                        "count": 0,
                    }
                rs = self.running_stats[key]
                rs["sum"] += values.sum(axis=0).astype(np.float64)
                rs["sum_sq"] += (values.astype(np.float64) ** 2).sum(axis=0)
                rs["min"] = np.minimum(rs["min"], values.min(axis=0))
                rs["max"] = np.maximum(rs["max"], values.max(axis=0))
                rs["count"] += len(values)

        return episode_stats

    def get_aggregated_stats(self) -> dict[str, dict]:
        """Compute final aggregated statistics.

        Output format matches LeRobot v3:
        - image keys: mean/std/min/max as (C, 1, 1) nested lists
        - state/action keys: mean/std/min/max as (D,) flat lists
        - count: always [N] (single-element list)
        """
        stats = {}
        for key, rs in self.running_stats.items():
            mean = rs["sum"] / rs["count"]
            var = (rs["sum_sq"] / rs["count"]) - (mean ** 2)
            std = np.sqrt(np.maximum(var, 0))
            is_image = "images" in key
            if is_image:
                # (C,) -> (C, 1, 1) for LeRobot broadcast compatibility
                stats[key] = {
                    "mean": mean.reshape(-1, 1, 1).tolist(),
                    "std": std.reshape(-1, 1, 1).tolist(),
                    "min": rs["min"].reshape(-1, 1, 1).tolist(),
                    "max": rs["max"].reshape(-1, 1, 1).tolist(),
                    "count": [int(rs["count"])],
                }
            else:
                stats[key] = {
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "min": rs["min"].tolist(),
                    "max": rs["max"].tolist(),
                    "count": [int(rs["count"])],
                }
        return stats

    def load_prior_stats(self, stats: dict) -> None:
        """Restore running_stats from previously saved stats.json.

        Handles both LeRobot format (C,1,1) image stats and flat (C,) format.
        Keys without 'count' are silently skipped (backward compat).
        """
        for key, s in stats.items():
            if not isinstance(s, dict) or "count" not in s:
                continue
            count = s["count"]
            if isinstance(count, list):
                count = count[0]
            mean = np.array(s["mean"], dtype=np.float64).flatten()
            std = np.array(s["std"], dtype=np.float64).flatten()
            mn = np.array(s["min"], dtype=np.float64).flatten()
            mx = np.array(s["max"], dtype=np.float64).flatten()

            self.running_stats[key] = {
                "sum": mean * count,
                "sum_sq": (std ** 2 + mean ** 2) * count,
                "min": mn,
                "max": mx,
                "count": count,
            }


class LeRobotDatasetWriter:
    """
    Write episodes directly to LeRobot v3.0 format.

    v3.0 structure:
    ```
    dataset/
    ├── meta/
    │   ├── info.json
    │   ├── stats.json
    │   ├── tasks.parquet
    │   └── episodes/
    │       └── chunk-000/
    │           └── file-000.parquet
    ├── data/
    │   └── chunk-000/
    │       └── file-000.parquet
    └── videos/
        ├── observation.images.top/
        │   └── chunk-000/
        │       └── file-000.mp4
        └── observation.images.wrist/
            └── chunk-000/
                └── file-000.mp4
    ```

    API is unchanged from v2.x for compatibility.
    """

    FRAME_SPEC = {
        "observation.state": ((6,), np.float32),
        "action": ((6,), np.float32),
        "observation.images.top": ((480, 640, 3), np.uint8),
        "observation.images.wrist": ((480, 640, 3), np.uint8),
    }

    VIDEO_KEYS = ["observation.images.top", "observation.images.wrist"]

    def __init__(
        self,
        output_dir: str | Path,
        fps: int = 30,
        task: str = "default task",
        codec: str = "h264",
        crf: int = 23,
        gop: int | None = 2,
        # v3.0 chunking parameters
        chunks_size: int = 1000,
        data_files_size_mb: float = 100.0,
        video_files_size_mb: float = 200.0,
        extra_features: dict[str, tuple[tuple, type]] | None = None,
    ):
        """
        Initialize dataset writer.

        Args:
            output_dir: Path to output directory (LeRobot format)
            fps: Frames per second (default: 30)
            task: Default task description
            codec: Video codec (h264 or h265)
            crf: Video quality (lower = better, 23 is default)
            gop: GOP size / keyframe interval (2 = fast random access, None = auto)
            chunks_size: Max files per chunk directory
            data_files_size_mb: Target size for parquet files
            video_files_size_mb: Target size for video files
            extra_features: Additional frame features, e.g. {"next.reward": ((), np.float32)}
        """
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.task = task
        self.codec = codec
        self.crf = crf
        self.gop = gop
        self.chunks_size = chunks_size
        self.data_files_size_mb = data_files_size_mb
        self.video_files_size_mb = video_files_size_mb

        # Frame spec: base + extra features
        self._frame_spec = dict(self.FRAME_SPEC)
        if extra_features:
            self._frame_spec.update(extra_features)

        # Episode buffer
        self._episode_buffer: list[dict] = []
        self._current_episode_idx = 0
        self._total_frames = 0

        # Stats tracking
        self._stats_computer = StatsComputer()

        # Task management
        self._tasks: dict[str, int] = {}

        # v3.0 state tracking
        self._data_state = ChunkFileState()
        self._video_states: dict[str, VideoFileState] = {
            key: VideoFileState() for key in self.VIDEO_KEYS
        }

        # Episode metadata buffer (for meta/episodes/)
        self._episode_metadata_buffer: list[dict] = []
        self._metadata_buffer_size = 10

        # Episode metadata tracking (seed + initial_state per episode)
        self._episode_metadata: dict[int, dict] = {}
        self._current_episode_seed: int | None = None
        self._current_episode_initial_state: dict | None = None

        # Parquet writer state
        self._data_writer: pq.ParquetWriter | None = None
        self._data_schema: pa.Schema | None = None

        # Create directories
        self._meta_dir = self.output_dir / "meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)

        # Temp directory for video encoding
        self._temp_dir = Path(tempfile.mkdtemp(prefix="lerobot_"))

        # Try to resume from existing dataset
        self._load_resume_state()

    def _load_resume_state(self) -> None:
        """Load state from existing dataset for resume support."""
        info_path = self._meta_dir / "info.json"
        if not info_path.exists():
            return

        with open(info_path) as f:
            info = json.load(f)

        self._total_frames = info.get("total_frames", 0)
        self._current_episode_idx = info.get("total_episodes", 0)

        # Load tasks (task string as index, task_index as column — LeRobot v3 format)
        tasks_path = self._meta_dir / "tasks.parquet"
        if tasks_path.exists():
            tasks_df = pd.read_parquet(tasks_path)
            if "task" in tasks_df.columns:
                # Old format: task as column
                for _, row in tasks_df.iterrows():
                    self._tasks[row["task"]] = row["task_index"]
            else:
                # New format: task string as index
                for task_str, row in tasks_df.iterrows():
                    self._tasks[task_str] = row["task_index"]

        # Load episode metadata
        metadata_path = self._meta_dir / "episode_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
            self._episode_metadata = {int(k): v for k, v in data.items()}

        # Load stats for resumable aggregation
        stats_path = self._meta_dir / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                prior_stats = json.load(f)
            self._stats_computer.load_prior_stats(prior_stats)

        # Load episode metadata to find last file positions
        episodes_meta_dir = self._meta_dir / "episodes"
        if episodes_meta_dir.exists():
            # Find the latest episodes metadata file
            ep_files = sorted(episodes_meta_dir.rglob("*.parquet"))
            if ep_files:
                last_ep_df = pd.read_parquet(ep_files[-1])
                if len(last_ep_df) > 0:
                    last = last_ep_df.iloc[-1]
                    self._data_state.chunk_index = int(last["data/chunk_index"])
                    self._data_state.file_index = int(last["data/file_index"])

                    # Load video states from last episode
                    for video_key in self.VIDEO_KEYS:
                        self._video_states[video_key].chunk_index = int(
                            last.get(f"videos/{video_key}/chunk_index", 0)
                        )
                        self._video_states[video_key].file_index = int(
                            last.get(f"videos/{video_key}/file_index", 0)
                        )

        # Calculate current file sizes and create new files if needed
        data_path = self._get_data_path(self._data_state)
        if data_path.exists():
            self._data_state.current_size_mb = get_file_size_mb(data_path)
            # ParquetWriter doesn't support append - must create new file on resume
            self._data_state.file_index += 1
            self._data_state.current_size_mb = 0.0
            print(f"[RESUME] Creating new data file: chunk-{self._data_state.chunk_index:03d}/file-{self._data_state.file_index:03d}.parquet")

        for video_key in self.VIDEO_KEYS:
            state = self._video_states[video_key]
            video_path = self._get_video_path(video_key, state)
            if video_path.exists():
                state.current_size_mb = get_file_size_mb(video_path)
                state.current_duration_s = get_video_duration_s(video_path)
                # Video files can be concatenated, but start fresh for cleaner structure
                state.file_index += 1
                state.current_size_mb = 0.0
                state.current_duration_s = 0.0
                state.current_frame_count = 0  # Reset for new file
                print(f"[RESUME] Creating new video file: videos/{video_key}/chunk-{state.chunk_index:03d}/file-{state.file_index:03d}.mp4")

    def set_task(self, task: str) -> None:
        """Set task for next episode."""
        self.task = task

    def set_episode_seed(self, seed: int) -> None:
        """Set seed for current episode (before recording starts)."""
        self._current_episode_seed = seed

    def set_episode_initial_state(self, state: dict) -> None:
        """Set initial prim state for current episode (before recording starts)."""
        self._current_episode_initial_state = state

    def add_frame(self, frame: dict) -> None:
        """Add frame to current episode buffer."""
        # Fill defaults for extra features not yet set (e.g. next.reward)
        for key, (shape, dtype) in self._frame_spec.items():
            if key not in frame and key not in self.FRAME_SPEC:
                frame[key] = np.zeros(shape, dtype=dtype)
        self._validate_frame(frame)

        if "timestamp" not in frame:
            frame["timestamp"] = len(self._episode_buffer) / self.fps

        frame["episode_index"] = self._current_episode_idx
        frame["frame_index"] = len(self._episode_buffer)

        self._episode_buffer.append(frame)

    def save_episode(self) -> int:
        """
        Save current episode to dataset.

        Returns:
            Episode index that was saved
        """
        if not self._episode_buffer:
            return -1

        ep_idx = self._current_episode_idx
        num_frames = len(self._episode_buffer)

        # Get or create task index
        if self.task not in self._tasks:
            self._tasks[self.task] = len(self._tasks)
        task_idx = self._tasks[self.task]

        # Prepare episode data
        states = np.stack([f["observation.state"] for f in self._episode_buffer])
        actions = np.stack([f["action"] for f in self._episode_buffer])
        timestamps = np.array(
            [f["timestamp"] for f in self._episode_buffer], dtype=np.float32
        )

        # Collect extra feature arrays
        extra_arrays: dict[str, np.ndarray] = {}
        for key in self._frame_spec:
            if key in self.FRAME_SPEC or key in self.VIDEO_KEYS:
                continue
            extra_arrays[key] = np.array(
                [f.get(key, np.float32(0.0)) for f in self._episode_buffer],
                dtype=np.float32,
            )

        # Compute stats (include sampled images for normalization)
        stats_data = {"observation.state": states, "action": actions}
        for key, arr in extra_arrays.items():
            stats_data[key] = arr
        # Sample 10% of frames for image stats (every 10th frame)
        sample_step = 10
        for video_key in self.VIDEO_KEYS:
            sampled = np.stack(
                [self._episode_buffer[i][video_key] for i in range(0, num_frames, sample_step)]
            )
            stats_data[video_key] = sampled
        self._stats_computer.add_episode(stats_data)

        # Build episode metadata
        episode_meta = {
            "episode_index": ep_idx,
            "length": num_frames,
            "task_index": task_idx,
        }

        # Save video files first (to get positions)
        for video_key in self.VIDEO_KEYS:
            images = np.stack([f[video_key] for f in self._episode_buffer])
            video_meta = self._save_episode_video(video_key, images)
            episode_meta.update(video_meta)

        # Save data parquet
        data_meta = self._save_episode_data(
            ep_idx, task_idx, num_frames, states, actions, timestamps,
            extra_arrays=extra_arrays,
        )
        episode_meta.update(data_meta)

        # Buffer episode metadata
        self._episode_metadata_buffer.append(episode_meta)
        if len(self._episode_metadata_buffer) >= self._metadata_buffer_size:
            self._flush_episode_metadata()

        # Save episode metadata (seed + initial_state)
        ep_meta = {}
        if self._current_episode_seed is not None:
            ep_meta["seed"] = self._current_episode_seed
            self._current_episode_seed = None
        if self._current_episode_initial_state is not None:
            ep_meta["initial_state"] = self._current_episode_initial_state
            self._current_episode_initial_state = None
        if ep_meta:
            self._episode_metadata[ep_idx] = ep_meta

        # Update counters
        self._total_frames += num_frames
        self._current_episode_idx += 1
        self._episode_buffer.clear()

        return ep_idx

    def _save_episode_data(
        self,
        ep_idx: int,
        task_idx: int,
        num_frames: int,
        states: np.ndarray,
        actions: np.ndarray,
        timestamps: np.ndarray,
        extra_arrays: dict[str, np.ndarray] | None = None,
    ) -> dict:
        """Save episode data to parquet file."""
        # Estimate episode size (~0.5KB per frame for state/action)
        ep_size_mb = num_frames * 0.5 / 1024

        # Check if need to rotate file
        if (
            self._data_state.current_size_mb + ep_size_mb >= self.data_files_size_mb
            and self._data_writer is not None
        ):
            self._close_data_writer()
            self._rotate_data_file()

        # Prepare parquet data
        parquet_data = {
            "episode_index": np.full(num_frames, ep_idx, dtype=np.int64),
            "frame_index": np.arange(num_frames, dtype=np.int64),
            "index": np.arange(
                self._total_frames, self._total_frames + num_frames, dtype=np.int64
            ),
            "task_index": np.full(num_frames, task_idx, dtype=np.int64),
            "timestamp": timestamps,
            "observation.state": [row.tolist() for row in states],
            "action": [row.tolist() for row in actions],
        }
        if extra_arrays:
            for key, arr in extra_arrays.items():
                parquet_data[key] = arr.tolist()

        # Create writer if needed
        if self._data_writer is None:
            path = self._get_data_path(self._data_state)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Build schema
            schema_fields = [
                ("episode_index", pa.int64()),
                ("frame_index", pa.int64()),
                ("index", pa.int64()),
                ("task_index", pa.int64()),
                ("timestamp", pa.float32()),
                ("observation.state", pa.list_(pa.float32())),
                ("action", pa.list_(pa.float32())),
            ]
            if extra_arrays:
                for key in extra_arrays:
                    schema_fields.append((key, pa.float32()))
            self._data_schema = pa.schema(schema_fields)
            self._data_writer = pq.ParquetWriter(path, self._data_schema)

        # Write to parquet
        table = pa.Table.from_pydict(parquet_data, schema=self._data_schema)
        self._data_writer.write_table(table)
        self._data_state.current_size_mb += ep_size_mb

        return {
            "data/chunk_index": self._data_state.chunk_index,
            "data/file_index": self._data_state.file_index,
            "dataset_from_index": self._total_frames,
            "dataset_to_index": self._total_frames + num_frames,
        }

    def _save_episode_video(self, video_key: str, images: np.ndarray) -> dict:
        """Save episode video, concatenating with existing file if needed."""
        state = self._video_states[video_key]

        # Encode to temp file
        temp_path = self._temp_dir / f"{video_key.replace('.', '_')}_{self._current_episode_idx}.mp4"
        h, w = images.shape[1:3]
        encode_images_to_video(
            images, temp_path, self.fps, codec=self.codec, crf=self.crf, g=self.gop
        )

        ep_size_mb = get_file_size_mb(temp_path)
        ep_duration_s = get_video_duration_s(temp_path)
        ep_frames = len(images)

        # Check if need new file
        video_path = self._get_video_path(video_key, state)

        if state.current_size_mb + ep_size_mb >= self.video_files_size_mb and video_path.exists():
            # Rotate to new file
            self._rotate_video_file(video_key)
            state = self._video_states[video_key]
            video_path = self._get_video_path(video_key, state)

        # Save position info before updating state
        from_timestamp = state.current_duration_s
        from_frame = state.current_frame_count

        if video_path.exists():
            # Concatenate with existing: need temp output since input=output
            concat_output = video_path.with_suffix(".concat.mp4")
            concatenate_video_files([video_path, temp_path], concat_output)
            temp_path.unlink()
            video_path.unlink()
            concat_output.rename(video_path)
        else:
            # First episode in this file
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_path), str(video_path))

        # Update state
        state.current_duration_s += ep_duration_s
        state.current_size_mb = get_file_size_mb(video_path)
        state.current_frame_count += ep_frames

        # Use original key with dots for LeRobot compatibility
        return {
            f"videos/{video_key}/chunk_index": state.chunk_index,
            f"videos/{video_key}/file_index": state.file_index,
            f"videos/{video_key}/from_timestamp": from_timestamp,
            f"videos/{video_key}/to_timestamp": state.current_duration_s,
            f"videos/{video_key}/from_frame": from_frame,
            f"videos/{video_key}/to_frame": state.current_frame_count,
        }

    def _get_data_path(self, state: ChunkFileState) -> Path:
        """Get path to data parquet file."""
        return (
            self.output_dir
            / "data"
            / f"chunk-{state.chunk_index:03d}"
            / f"file-{state.file_index:03d}.parquet"
        )

    def _get_video_path(self, video_key: str, state: VideoFileState) -> Path:
        """Get path to video file."""
        return (
            self.output_dir
            / "videos"
            / video_key
            / f"chunk-{state.chunk_index:03d}"
            / f"file-{state.file_index:03d}.mp4"
        )

    def _get_episodes_meta_path(self) -> Path:
        """Get path to episodes metadata parquet file."""
        # For simplicity, use single file
        path = self._meta_dir / "episodes" / "chunk-000" / "file-000.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _rotate_data_file(self) -> None:
        """Move to next data file."""
        self._data_state.file_index += 1
        if self._data_state.file_index >= self.chunks_size:
            self._data_state.file_index = 0
            self._data_state.chunk_index += 1
        self._data_state.current_size_mb = 0.0

    def _rotate_video_file(self, video_key: str) -> None:
        """Move to next video file for a camera."""
        state = self._video_states[video_key]
        state.file_index += 1
        if state.file_index >= self.chunks_size:
            state.file_index = 0
            state.chunk_index += 1
        state.current_size_mb = 0.0
        state.current_duration_s = 0.0
        state.current_frame_count = 0

    def _close_data_writer(self) -> None:
        """Close current parquet writer."""
        if self._data_writer is not None:
            self._data_writer.close()
            self._data_writer = None

    def _flush_episode_metadata(self) -> None:
        """Write buffered episode metadata to parquet."""
        if not self._episode_metadata_buffer:
            return

        path = self._get_episodes_meta_path()

        # Load existing if present
        if path.exists():
            existing_df = pd.read_parquet(path)
            new_df = pd.DataFrame(self._episode_metadata_buffer)
            df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(self._episode_metadata_buffer)

        df.to_parquet(path, index=False)
        self._episode_metadata_buffer.clear()

    def clear_episode(self) -> None:
        """Discard current episode buffer without saving."""
        self._episode_buffer.clear()

    def close(self) -> None:
        """Finalize dataset: write all metadata files."""
        # Save any remaining buffered data
        if self._episode_buffer:
            self.save_episode()

        # Close data writer
        self._close_data_writer()

        # Flush remaining episode metadata
        self._flush_episode_metadata()

        if self._current_episode_idx == 0:
            # Cleanup temp dir
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            return

        # Write tasks.parquet (task string as index, task_index as column — LeRobot v3 format)
        task_strings = list(self._tasks.keys())
        task_indices = list(self._tasks.values())
        tasks_df = pd.DataFrame({"task_index": task_indices}, index=task_strings)
        tasks_df = tasks_df.sort_values("task_index")
        tasks_df.to_parquet(self._meta_dir / "tasks.parquet")

        # Write stats.json
        stats = self._stats_computer.get_aggregated_stats()
        with open(self._meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Write info.json
        info = self._build_info()
        with open(self._meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Write episode_metadata.json (seeds + initial states)
        if self._episode_metadata:
            with open(self._meta_dir / "episode_metadata.json", "w") as f:
                json.dump(self._episode_metadata, f, indent=2)

        # Cleanup temp dir
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def _build_info(self) -> dict:
        """Build info.json content."""
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [6],
                "names": [f"motor_{i}" for i in range(6)],
            },
            "action": {
                "dtype": "float32",
                "shape": [6],
                "names": [f"motor_{i}" for i in range(6)],
            },
            "observation.images.top": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": self.codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": float(self.fps),
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
                "video_info": {
                    "video.height": 480,
                    "video.width": 640,
                    "video.codec": self.codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": float(self.fps),
                    "video.channels": 3,
                    "has_audio": False,
                },
            },
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        }

        # Add extra features to info
        for key in self._frame_spec:
            if key not in self.FRAME_SPEC and key not in self.VIDEO_KEYS:
                shape, dtype = self._frame_spec[key]
                features[key] = {
                    "dtype": "float32",
                    "shape": list(shape) if shape else [1],
                    "names": None,
                }

        return {
            "codebase_version": "v3.0",
            "robot_type": "so101",
            "fps": self.fps,
            "total_episodes": self._current_episode_idx,
            "total_frames": self._total_frames,
            "total_tasks": len(self._tasks),
            "features": features,
            "splits": {"train": f"0:{self._current_episode_idx}"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "chunks_size": self.chunks_size,
        }

    def _validate_frame(self, frame: dict) -> None:
        """Validate frame structure against expected specification."""
        for key, (expected_shape, expected_dtype) in self._frame_spec.items():
            if key not in frame:
                raise ValueError(f"Missing required key: {key}")

            value = frame[key]
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Key '{key}' must be numpy array, got {type(value)}")

            if value.shape != expected_shape:
                raise ValueError(
                    f"Key '{key}' has wrong shape: expected {expected_shape}, got {value.shape}"
                )

            if value.dtype != expected_dtype:
                raise ValueError(
                    f"Key '{key}' has wrong dtype: expected {expected_dtype}, got {value.dtype}"
                )

    @property
    def total_episodes(self) -> int:
        """Get total number of episodes saved."""
        return self._current_episode_idx

    @property
    def total_frames(self) -> int:
        """Get total number of frames saved across all episodes."""
        return self._total_frames

    @property
    def current_episode_length(self) -> int:
        """Get number of frames in current episode buffer."""
        return len(self._episode_buffer)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor to ensure dataset is finalized."""
        if hasattr(self, "_episode_buffer") and self._episode_buffer:
            self.close()
