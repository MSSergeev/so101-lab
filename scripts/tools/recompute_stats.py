#!/usr/bin/env python3
"""Recompute stats for existing LeRobot datasets.

Reads parquet data files for state/action stats and video files for image stats.
Streams video frames in batches (constant memory).

Usage:
    python scripts/tools/recompute_stats.py data/recordings/figure_shape_placement_v1
    python scripts/tools/recompute_stats.py data/recordings/figure_shape_placement_v1 --sample-rate 0.1
    python scripts/tools/recompute_stats.py data/recordings/figure_shape_placement_v1 --max-frames 5000
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

BATCH_SIZE = 100


def compute_video_stats_streaming(
    video_path: Path, max_frames: int, batch_size: int = BATCH_SIZE
) -> tuple[dict, int]:
    """Compute per-channel image stats from a video via batched streaming.

    Reads frames in batches of `batch_size` for vectorized numpy ops
    while keeping memory bounded.

    Returns:
        (stats_dict with (3,) arrays, n_sampled_frames)
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError(f"Cannot read video: {video_path}")

    step = max(1, total // max_frames)
    sample_indices = set(range(0, total, step))

    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    ch_min = np.ones(3, dtype=np.float64)
    ch_max = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    n_sampled = 0

    batch = []
    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx not in sample_indices:
            continue

        batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if len(batch) >= batch_size:
            _accumulate_batch(batch, pixel_sum, pixel_sq_sum, ch_min, ch_max)
            H, W = batch[0].shape[:2]
            total_pixels += H * W * len(batch)
            n_sampled += len(batch)
            batch.clear()

    # Flush remaining
    if batch:
        _accumulate_batch(batch, pixel_sum, pixel_sq_sum, ch_min, ch_max)
        H, W = batch[0].shape[:2]
        total_pixels += H * W * len(batch)
        n_sampled += len(batch)

    cap.release()

    if n_sampled == 0:
        raise ValueError(f"No frames sampled from {video_path}")

    mean = pixel_sum / total_pixels
    std = np.sqrt(pixel_sq_sum / total_pixels - mean**2)

    return {"mean": mean, "std": std, "min": ch_min, "max": ch_max}, n_sampled


def _accumulate_batch(
    batch: list[np.ndarray],
    pixel_sum: np.ndarray,
    pixel_sq_sum: np.ndarray,
    ch_min: np.ndarray,
    ch_max: np.ndarray,
):
    """Accumulate running stats from a batch of uint8 HWC frames."""
    arr = np.stack(batch).astype(np.float64) / 255.0  # (B, H, W, 3)
    for c in range(3):
        ch = arr[:, :, :, c]
        pixel_sum[c] += ch.sum()
        pixel_sq_sum[c] += (ch**2).sum()
        ch_min[c] = min(ch_min[c], ch.min())
        ch_max[c] = max(ch_max[c], ch.max())


def main():
    parser = argparse.ArgumentParser(description="Recompute all stats for LeRobot dataset")
    parser.add_argument("dataset", type=Path, help="Path to dataset directory")
    parser.add_argument(
        "--sample-rate", type=float, default=0.05, help="Fraction of frames to sample (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Absolute max frames per camera (overrides --sample-rate)"
    )
    args = parser.parse_args()

    dataset = args.dataset
    stats_path = dataset / "meta" / "stats.json"

    if not stats_path.exists():
        print(f"ERROR: {stats_path} not found")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    # --- Recompute state/action stats from parquet ---
    data_dir = dataset / "data"
    if data_dir.exists():
        parquet_files = sorted(data_dir.rglob("*.parquet"))
        if parquet_files:
            df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
            for key in ("observation.state", "action"):
                if key not in df.columns:
                    continue
                vals = np.stack(df[key].values).astype(np.float64)
                count = len(vals)
                stats[key] = {
                    "mean": vals.mean(axis=0).tolist(),
                    "std": vals.std(axis=0).tolist(),
                    "min": vals.min(axis=0).tolist(),
                    "max": vals.max(axis=0).tolist(),
                    "count": [count],
                }
                print(f"  {key}: {count} frames -> mean={stats[key]['mean']}")
    else:
        print(f"  WARNING: {data_dir} not found, skipping state/action stats")

    # --- Recompute image stats from video ---
    videos_dir = dataset / "videos"
    if not videos_dir.exists():
        print(f"ERROR: {videos_dir} not found")
        return

    camera_dirs = sorted(d for d in videos_dir.iterdir() if d.is_dir())

    for cam_dir in camera_dirs:
        key = cam_dir.name
        video_files = sorted(cam_dir.rglob("*.mp4"))

        if not video_files:
            print(f"  {key}: no video files found, skipping")
            continue

        # Count total frames across all video files to compute sample budget
        total_video_frames = 0
        for vf in video_files:
            cap = cv2.VideoCapture(str(vf))
            total_video_frames += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        if args.max_frames is not None:
            budget = args.max_frames
        else:
            budget = max(50, int(total_video_frames * args.sample_rate))

        per_file = max(1, budget // len(video_files))

        # Aggregate across video files
        agg_pixel_sum = np.zeros(3, dtype=np.float64)
        agg_pixel_sq_sum = np.zeros(3, dtype=np.float64)
        agg_min = np.ones(3, dtype=np.float64)
        agg_max = np.zeros(3, dtype=np.float64)
        agg_pixels = 0
        agg_frames = 0

        for vf in video_files:
            try:
                file_stats, n_frames = compute_video_stats_streaming(vf, max_frames=per_file)
            except ValueError as e:
                print(f"  WARNING: {e}")
                continue

            # Recover raw sums from per-file stats for correct merging
            cap = cv2.VideoCapture(str(vf))
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue
            H, W = frame.shape[:2]
            n_pix = H * W * n_frames

            agg_pixel_sum += file_stats["mean"] * n_pix
            agg_pixel_sq_sum += (file_stats["std"] ** 2 + file_stats["mean"] ** 2) * n_pix
            agg_min = np.minimum(agg_min, file_stats["min"])
            agg_max = np.maximum(agg_max, file_stats["max"])
            agg_pixels += n_pix
            agg_frames += n_frames

        if agg_pixels == 0:
            continue

        mean = agg_pixel_sum / agg_pixels
        std = np.sqrt(agg_pixel_sq_sum / agg_pixels - mean**2)

        image_stats = {
            "mean": mean.reshape(-1, 1, 1).tolist(),
            "std": std.reshape(-1, 1, 1).tolist(),
            "min": agg_min.reshape(-1, 1, 1).tolist(),
            "max": agg_max.reshape(-1, 1, 1).tolist(),
            "count": [agg_pixels],
        }
        stats[key] = image_stats
        pct = agg_frames / total_video_frames * 100 if total_video_frames else 0
        print(f"  {key}: {agg_frames}/{total_video_frames} frames ({pct:.1f}%) -> mean={image_stats['mean']}")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nUpdated {stats_path}")


if __name__ == "__main__":
    main()
