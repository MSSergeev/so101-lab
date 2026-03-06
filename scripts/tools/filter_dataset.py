#!/usr/bin/env python3
"""Filter LeRobot dataset by excluding specific episodes.

Creates a new dataset with only the desired episodes, renumbered sequentially.

Usage:
    # Exclude episodes 3, 7, 12
    python scripts/tools/filter_dataset.py \
        --input data/recordings/v1 \
        --output data/recordings/v1_filtered \
        --exclude-episodes 3 7 12

    # Keep only episodes 0, 1, 5
    python scripts/tools/filter_dataset.py \
        --input data/recordings/v1 \
        --output data/recordings/v1_filtered \
        --keep-episodes 0 1 5

    # Delete original after filtering
    python scripts/tools/filter_dataset.py \
        --input data/recordings/v1 \
        --output data/recordings/v1_filtered \
        --exclude-episodes 3 7 \
        --delete-original
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def get_video_duration_s(path: Path) -> float:
    """Get video duration using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path)
        ],
        capture_output=True, text=True
    )
    return float(result.stdout.strip()) if result.stdout.strip() else 0.0


def extract_video_segment(input_path: Path, output_path: Path, start: float, end: float):
    """Extract video segment using ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_path),
        "-ss", str(start), "-to", str(end),
        "-c", "copy",
        str(output_path)
    ], capture_output=True, check=True)


def concatenate_videos(input_paths: list[Path], output_path: Path):
    """Concatenate video files using ffmpeg."""
    if len(input_paths) == 1:
        shutil.copy(input_paths[0], output_path)
        return

    # Create concat file
    concat_file = output_path.parent / ".concat_list.txt"
    with open(concat_file, "w") as f:
        for p in input_paths:
            f.write(f"file '{p.absolute()}'\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path)
    ], capture_output=True, check=True)

    concat_file.unlink()


def filter_dataset(
    input_dir: Path,
    output_dir: Path,
    keep_episodes: list[int] | None = None,
    exclude_episodes: list[int] | None = None,
):
    """Filter dataset, keeping only specified episodes with renumbering."""

    print(f"\n{'='*60}")
    print("LeRobot Dataset Filter")
    print(f"{'='*60}")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Load info
    info_path = input_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    fps = info["fps"]

    # Determine which episodes to keep
    all_episodes = set(range(total_episodes))

    if keep_episodes is not None:
        episodes_to_keep = sorted(set(keep_episodes) & all_episodes)
    elif exclude_episodes is not None:
        episodes_to_keep = sorted(all_episodes - set(exclude_episodes))
    else:
        episodes_to_keep = sorted(all_episodes)

    episodes_to_remove = sorted(all_episodes - set(episodes_to_keep))

    print(f"\nTotal episodes: {total_episodes}")
    print(f"Episodes to keep: {len(episodes_to_keep)} {episodes_to_keep}")
    print(f"Episodes to remove: {len(episodes_to_remove)} {episodes_to_remove}")

    if not episodes_to_keep:
        print("\nERROR: No episodes to keep!")
        return False

    if len(episodes_to_keep) == total_episodes:
        print("\nNothing to filter, all episodes kept.")
        if input_dir != output_dir:
            print(f"Copying dataset to {output_dir}...")
            shutil.copytree(input_dir, output_dir)
        return True

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create episode index mapping (old -> new)
    old_to_new = {old: new for new, old in enumerate(episodes_to_keep)}

    # ─── 1. Filter data parquet ────────────────────────────────────
    print("\n[1/5] Filtering data parquet...")

    data_dir = input_dir / "data"
    out_data_dir = output_dir / "data"

    new_frames = []
    frame_offset = 0

    for parquet_file in sorted(data_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        # Filter by episode
        df_filtered = df[df["episode_index"].isin(episodes_to_keep)].copy()

        if len(df_filtered) == 0:
            continue

        # Renumber episodes
        df_filtered["episode_index"] = df_filtered["episode_index"].map(old_to_new)

        new_frames.append(df_filtered)

    if new_frames:
        combined = pd.concat(new_frames, ignore_index=True)
        # Renumber frame indices per episode
        combined = combined.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        combined["index"] = range(len(combined))

        # Recalculate frame_index per episode
        combined["frame_index"] = combined.groupby("episode_index").cumcount()

        # Save to single file
        out_path = out_data_dir / "chunk-000" / "file-000.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out_path, index=False)

        total_frames = len(combined)
        print(f"    Saved {total_frames} frames")
    else:
        print("    ERROR: No data frames!")
        return False

    # ─── 2. Filter episode metadata ────────────────────────────────
    print("[2/5] Filtering episode metadata...")

    episodes_meta_dir = input_dir / "meta" / "episodes"
    out_episodes_meta_dir = output_dir / "meta" / "episodes"

    ep_dfs = []
    for parquet_file in sorted(episodes_meta_dir.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file)
        df_filtered = df[df["episode_index"].isin(episodes_to_keep)].copy()
        if len(df_filtered) > 0:
            ep_dfs.append(df_filtered)

    if ep_dfs:
        ep_combined = pd.concat(ep_dfs, ignore_index=True)
        ep_combined["episode_index"] = ep_combined["episode_index"].map(old_to_new)
        ep_combined = ep_combined.sort_values("episode_index").reset_index(drop=True)

        # Recalculate dataset indices
        frame_counts = combined.groupby("episode_index").size().to_dict()
        dataset_from = 0
        for idx, row in ep_combined.iterrows():
            ep_idx = row["episode_index"]
            ep_len = frame_counts.get(ep_idx, row["length"])
            ep_combined.at[idx, "length"] = ep_len
            ep_combined.at[idx, "dataset_from_index"] = dataset_from
            ep_combined.at[idx, "dataset_to_index"] = dataset_from + ep_len
            dataset_from += ep_len

        # Reset data chunk/file indices
        ep_combined["data/chunk_index"] = 0
        ep_combined["data/file_index"] = 0

    # ─── 3. Process videos ─────────────────────────────────────────
    print("[3/5] Processing videos...")

    videos_dir = input_dir / "videos"
    out_videos_dir = output_dir / "videos"

    video_keys = ["observation.images.top", "observation.images.wrist"]
    temp_dir = output_dir / ".temp_videos"
    temp_dir.mkdir(exist_ok=True)

    for video_key in video_keys:
        print(f"    Processing {video_key}...")

        video_subdir = videos_dir / video_key
        if not video_subdir.exists():
            continue

        # Collect all video files
        video_files = sorted(video_subdir.rglob("*.mp4"))
        if not video_files:
            continue

        # Get episode video info from metadata
        segments_to_extract = []
        current_timestamp = 0.0

        for old_ep_idx in episodes_to_keep:
            # Find episode metadata row
            ep_row = ep_dfs[0][ep_dfs[0]["episode_index"] == old_ep_idx].iloc[0] if ep_dfs else None
            if ep_row is None:
                continue

            from_ts = ep_row.get(f"videos/{video_key}/from_timestamp", 0)
            to_ts = ep_row.get(f"videos/{video_key}/to_timestamp", 0)
            chunk_idx = int(ep_row.get(f"videos/{video_key}/chunk_index", 0))
            file_idx = int(ep_row.get(f"videos/{video_key}/file_index", 0))

            # Find source video file
            src_video = video_subdir / f"chunk-{chunk_idx:03d}" / f"file-{file_idx:03d}.mp4"
            if not src_video.exists():
                # Try first available
                src_video = video_files[0]

            new_ep_idx = old_to_new[old_ep_idx]
            temp_segment = temp_dir / f"{video_key.replace('.', '_')}_{new_ep_idx:04d}.mp4"

            segments_to_extract.append({
                "src": src_video,
                "dst": temp_segment,
                "from": from_ts,
                "to": to_ts,
                "new_ep_idx": new_ep_idx,
                "duration": to_ts - from_ts,
            })

        # Extract segments
        for seg in segments_to_extract:
            try:
                extract_video_segment(seg["src"], seg["dst"], seg["from"], seg["to"])
            except subprocess.CalledProcessError as e:
                print(f"    WARNING: Failed to extract segment: {e}")

        # Concatenate all segments
        segment_files = [seg["dst"] for seg in segments_to_extract if seg["dst"].exists()]
        if segment_files:
            out_video_path = out_videos_dir / video_key / "chunk-000" / "file-000.mp4"
            out_video_path.parent.mkdir(parents=True, exist_ok=True)
            concatenate_videos(segment_files, out_video_path)

            # Update episode metadata with new timestamps
            current_ts = 0.0
            current_frame = 0
            for seg in segments_to_extract:
                new_ep_idx = seg["new_ep_idx"]
                duration = seg["duration"]
                ep_frames = frame_counts.get(new_ep_idx, int(duration * fps))

                mask = ep_combined["episode_index"] == new_ep_idx
                ep_combined.loc[mask, f"videos/{video_key}/chunk_index"] = 0
                ep_combined.loc[mask, f"videos/{video_key}/file_index"] = 0
                ep_combined.loc[mask, f"videos/{video_key}/from_timestamp"] = current_ts
                ep_combined.loc[mask, f"videos/{video_key}/to_timestamp"] = current_ts + duration
                ep_combined.loc[mask, f"videos/{video_key}/from_frame"] = current_frame
                ep_combined.loc[mask, f"videos/{video_key}/to_frame"] = current_frame + ep_frames

                current_ts += duration
                current_frame += ep_frames

            print(f"    Saved {len(segment_files)} episodes to {out_video_path}")

    # Cleanup temp
    shutil.rmtree(temp_dir, ignore_errors=True)

    # Save episode metadata
    out_ep_path = out_episodes_meta_dir / "chunk-000" / "file-000.parquet"
    out_ep_path.parent.mkdir(parents=True, exist_ok=True)
    ep_combined.to_parquet(out_ep_path, index=False)

    # ─── 4. Recalculate stats ──────────────────────────────────────
    print("[4/5] Recalculating statistics...")

    stats = {}
    for col in ["observation.state", "action"]:
        if col in combined.columns:
            values = np.stack(combined[col].values)
            stats[col] = {
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
                "min": values.min(axis=0).tolist(),
                "max": values.max(axis=0).tolist(),
            }

    stats_path = output_dir / "meta" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # ─── 5. Update info.json and copy tasks ────────────────────────
    print("[5/5] Updating metadata...")

    # Update info
    info["total_episodes"] = len(episodes_to_keep)
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{len(episodes_to_keep)}"}

    info_out_path = output_dir / "meta" / "info.json"
    with open(info_out_path, "w") as f:
        json.dump(info, f, indent=2)

    # Copy tasks.parquet
    tasks_src = input_dir / "meta" / "tasks.parquet"
    tasks_dst = output_dir / "meta" / "tasks.parquet"
    if tasks_src.exists():
        shutil.copy(tasks_src, tasks_dst)

    # Filter episode_metadata.json
    metadata_src = input_dir / "meta" / "episode_metadata.json"
    if metadata_src.exists():
        with open(metadata_src) as f:
            metadata = json.load(f)
        new_metadata = {}
        for old_idx, meta in metadata.items():
            old_idx_int = int(old_idx)
            if old_idx_int in old_to_new:
                new_metadata[str(old_to_new[old_idx_int])] = meta
        with open(output_dir / "meta" / "episode_metadata.json", "w") as f:
            json.dump(new_metadata, f, indent=2)

    print(f"\n{'='*60}")
    print("Filtering complete!")
    print(f"{'='*60}")
    print(f"Episodes: {total_episodes} -> {len(episodes_to_keep)}")
    print(f"Frames: {info['total_frames']} -> {total_frames}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Filter LeRobot dataset by episodes")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input dataset directory")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Output dataset directory")
    parser.add_argument("--keep-episodes", "-k", type=int, nargs="+",
                        help="Episodes to keep (others will be removed)")
    parser.add_argument("--exclude-episodes", "-e", type=int, nargs="+",
                        help="Episodes to exclude (others will be kept)")
    parser.add_argument("--delete-original", "-d", action="store_true",
                        help="Delete original dataset after successful filtering")

    args = parser.parse_args()

    if args.keep_episodes and args.exclude_episodes:
        print("ERROR: Use either --keep-episodes or --exclude-episodes, not both")
        sys.exit(1)

    if not args.keep_episodes and not args.exclude_episodes:
        print("ERROR: Specify --keep-episodes or --exclude-episodes")
        sys.exit(1)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if output_dir.exists():
        print(f"ERROR: Output directory already exists: {output_dir}")
        print("Please remove it first or choose a different output path.")
        sys.exit(1)

    if input_dir.resolve() == output_dir.resolve():
        print("ERROR: Input and output directories must be different")
        sys.exit(1)

    # Run filter
    success = filter_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        keep_episodes=args.keep_episodes,
        exclude_episodes=args.exclude_episodes,
    )

    if not success:
        sys.exit(1)

    # Delete original if requested
    if args.delete_original:
        print(f"\n{'!'*60}")
        print("WARNING: You requested to delete the original dataset!")
        print(f"{'!'*60}")
        print(f"Path: {input_dir}")
        print()
        confirm = input("Type 'DELETE' to confirm deletion: ").strip()

        if confirm == "DELETE":
            print(f"\nDeleting {input_dir}...")
            shutil.rmtree(input_dir)
            print("Original dataset deleted.")
        else:
            print("\nDeletion cancelled. Original dataset preserved.")


if __name__ == "__main__":
    main()
