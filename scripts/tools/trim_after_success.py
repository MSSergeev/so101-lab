#!/usr/bin/env python3
"""Trim episodes to N frames after first success.

Modifies dataset in-place: removes trailing frames after success,
updates parquet, episode metadata, sim_rewards, and stats.

Video modes:
    (default)      - videos untouched, only metadata updated
    --reencode     - re-encode videos (precise cut, slight quality loss)
    --keyframe-cut - cut at nearest keyframe via -c copy (no quality loss,
                     trim point snaps to closest keyframe after first success)

Usage:
    python scripts/tools/trim_after_success.py data/recordings/ds --keep-after 10 --dry-run
    python scripts/tools/trim_after_success.py data/recordings/ds --keep-after 10
    python scripts/tools/trim_after_success.py data/recordings/ds --keep-after 10 --reencode
    python scripts/tools/trim_after_success.py data/recordings/ds --keep-after 10 --keyframe-cut
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def decode_video(video_path: Path) -> np.ndarray:
    """Decode video to numpy array (N, H, W, 3) uint8."""
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height",
         "-print_format", "json", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(probe.stdout)["streams"][0]
    w, h = int(stream["width"]), int(stream["height"])

    proc = subprocess.run(
        ["ffmpeg", "-i", str(video_path),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-v", "quiet", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w, 3)


def encode_video(frames: np.ndarray, output_path: Path, fps: int, crf: int, gop: int | None = None):
    """Encode numpy frames to video."""
    n, h, w, _ = frames.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "rgb24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", str(crf),
    ]
    if gop is not None:
        cmd.extend(["-g", str(gop)])
    cmd.extend(["-preset", "medium", str(output_path)])
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = proc.communicate(input=frames.tobytes())
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg encode failed: {stderr.decode()}")


def get_keyframe_frames(video_path: Path, fps: int) -> list[int]:
    """Get keyframe positions as frame numbers."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0", "-skip_frame", "nokey",
        "-show_entries", "frame=best_effort_timestamp_time",
        "-of", "csv=p=0", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    frames = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            frames.append(round(float(line) * fps))
    return sorted(frames)


def find_nearest_keyframe(keyframes: list[int], target: int, minimum: int) -> int | None:
    """Find keyframe closest to target but >= minimum.

    Returns frame number or None if no valid keyframe exists.
    """
    candidates = [kf for kf in keyframes if kf >= minimum]
    if not candidates:
        return None
    return min(candidates, key=lambda kf: abs(kf - target))


def extract_segment_copy(video_path: Path, output_path: Path, start_time: float, end_time: float):
    """Extract video segment using -c copy (no re-encoding)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", str(video_path),
        "-ss", f"{start_time:.6f}",
        "-to", f"{end_time:.6f}",
        "-c", "copy", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg extract failed: {result.stderr}")


def concat_videos_copy(segment_paths: list[Path], output_path: Path):
    """Concatenate video segments using -c copy."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in segment_paths:
            escaped = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        list_path = f.name

    try:
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-f", "concat", "-safe", "0",
            "-i", list_path, "-c", "copy", str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")
    finally:
        Path(list_path).unlink(missing_ok=True)


def get_video_frame_count(video_path: Path) -> int:
    """Get actual frame count from video file."""
    cmd = [
        "ffprobe", "-v", "quiet", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-print_format", "json", str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return int(data["streams"][0]["nb_read_frames"])


def main():
    parser = argparse.ArgumentParser(description="Trim episodes after success")
    parser.add_argument("dataset", type=str, help="Path to dataset directory")
    parser.add_argument("--keep-after", type=int, default=10,
                        help="Frames to keep after first success frame (default: 10)")
    parser.add_argument("--reencode", action="store_true",
                        help="Re-encode videos (precise cut)")
    parser.add_argument("--keyframe-cut", action="store_true",
                        help="Cut at nearest keyframe via -c copy (no quality loss)")
    parser.add_argument("--crf", type=int, default=23,
                        help="CRF for --reencode (default: 23)")
    parser.add_argument("--gop", type=str, default="auto",
                        help="GOP for --reencode (default: auto)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be trimmed without modifying files")
    args = parser.parse_args()

    if args.reencode and args.keyframe_cut:
        print("ERROR: --reencode and --keyframe-cut are mutually exclusive")
        return

    ds = Path(args.dataset)

    # Load data
    data_files = sorted(ds.glob("data/**/*.parquet"))
    data_df = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)

    ep_meta_path = ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep_df = pd.read_parquet(ep_meta_path)

    with open(ds / "meta" / "info.json") as f:
        info = json.load(f)

    sr_path = ds / "sim_rewards.pt"
    sim_rewards = torch.load(sr_path, weights_only=True) if sr_path.exists() else None

    fps = info["fps"]
    total_before = len(data_df)
    cams = ["observation.images.top", "observation.images.wrist"]

    # --- For keyframe-cut: get keyframe positions first ---
    # Map: (cam, chunk, file) -> list of keyframe frame numbers
    keyframes_cache: dict[tuple[str, int, int], list[int]] = {}
    if args.keyframe_cut and not args.dry_run:
        for cam in cams:
            for _, row in ep_df.iterrows():
                chunk = int(row[f"videos/{cam}/chunk_index"])
                findex = int(row[f"videos/{cam}/file_index"])
                cache_key = (cam, chunk, findex)
                if cache_key not in keyframes_cache:
                    video_path = ds / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{findex:03d}.mp4"
                    if video_path.exists():
                        keyframes_cache[cache_key] = get_keyframe_frames(video_path, fps)

    # --- Compute first_success per episode ---
    ep_first_success: dict[int, int] = {}  # ep_idx -> first success frame (episode-local)
    for ep_idx in sorted(data_df["episode_index"].unique()):
        ep_mask = data_df["episode_index"] == ep_idx
        rewards = data_df[ep_mask]["next.reward"].values
        success_indices = np.where(rewards > 0)[0]
        if len(success_indices) > 0:
            ep_first_success[int(ep_idx)] = int(success_indices[0])

    # --- For keyframe-cut: adjust trim points to keyframe boundaries ---
    # ep_idx -> actual trim length (may differ per camera, take minimum)
    if args.keyframe_cut and not args.dry_run:
        kf_trim_info: dict[int, int] = {}
        for ep_idx, first_success in ep_first_success.items():
            desired_trim = first_success + args.keep_after
            ep_row = ep_df[ep_df["episode_index"] == ep_idx].iloc[0]
            old_len = int(ep_row["length"])

            if desired_trim >= old_len:
                continue

            # Find best keyframe across both cameras (use most conservative = smallest)
            best_trim = desired_trim
            for cam in cams:
                chunk = int(ep_row[f"videos/{cam}/chunk_index"])
                findex = int(ep_row[f"videos/{cam}/file_index"])
                from_f = int(ep_row[f"videos/{cam}/from_frame"])
                cache_key = (cam, chunk, findex)
                kfs = keyframes_cache.get(cache_key, [])

                # Convert to episode-local frame numbers
                local_kfs = [kf - from_f for kf in kfs if from_f <= kf < from_f + old_len]

                # Find nearest keyframe to desired_trim, but >= first_success
                kf = find_nearest_keyframe(local_kfs, desired_trim, first_success)
                if kf is not None:
                    best_trim = min(best_trim, kf) if best_trim != desired_trim else kf

            if best_trim < old_len:
                kf_trim_info[ep_idx] = best_trim

    # --- Build keep mask and trim_info ---
    keep_mask = np.ones(total_before, dtype=bool)
    trim_info: dict[int, int] = {}

    print(f"Dataset: {ds}")
    print(f"Episodes: {info['total_episodes']}, Frames: {total_before}")
    print(f"Keep after success: {args.keep_after} frames")
    if args.reencode:
        gop_label = args.gop if args.gop != "auto" else "auto"
        print(f"Video mode: re-encode (CRF={args.crf}, GOP={gop_label})")
    elif args.keyframe_cut:
        print("Video mode: keyframe cut (-c copy)")
    else:
        print("Video mode: metadata only")
    print("-" * 60)

    for ep_idx in sorted(data_df["episode_index"].unique()):
        ep_idx = int(ep_idx)
        ep_mask = data_df["episode_index"] == ep_idx
        old_len = int(ep_mask.sum())

        if ep_idx not in ep_first_success:
            print(f"  Ep {ep_idx}: {old_len} frames, no success — skip")
            continue

        first_success = ep_first_success[ep_idx]

        # Determine trim point
        if args.keyframe_cut and not args.dry_run and ep_idx in kf_trim_info:
            trim_at = kf_trim_info[ep_idx]
            label = f"keyframe@{trim_at}"
        else:
            trim_at = first_success + args.keep_after
            label = f"exact@{trim_at}"

        if trim_at >= old_len:
            print(f"  Ep {ep_idx}: {old_len} frames, success@{first_success} — no trim needed")
            continue

        global_indices = np.where(ep_mask)[0]
        keep_mask[global_indices[trim_at:]] = False
        trim_info[ep_idx] = trim_at
        removed = old_len - trim_at
        print(f"  Ep {ep_idx}: {old_len} -> {trim_at} frames (success@{first_success}, {label}, -{removed})")

    total_after = int(keep_mask.sum())
    print("-" * 60)
    print(f"Total: {total_before} -> {total_after} frames (-{total_before - total_after})")

    if args.dry_run:
        print("\n[DRY RUN] No files modified.")
        return

    if total_after == total_before:
        print("\nNothing to trim.")
        return

    # --- Apply trim to data parquet ---
    trimmed_df = data_df[keep_mask].copy()
    trimmed_df["index"] = np.arange(len(trimmed_df), dtype=np.int64)
    trimmed_df["frame_index"] = trimmed_df.groupby("episode_index").cumcount().astype(np.int64)

    for f in data_files:
        f.unlink()
    out_data_path = ds / "data" / "chunk-000" / "file-000.parquet"
    out_data_path.parent.mkdir(parents=True, exist_ok=True)
    trimmed_df.to_parquet(out_data_path, index=False)
    print(f"\n[WRITE] {out_data_path}")

    # --- Process videos ---
    gop_val = None if args.gop == "auto" else int(args.gop)

    if args.reencode:
        with tempfile.TemporaryDirectory(prefix="reencode_") as tmp_dir:
            tmp = Path(tmp_dir)
            for cam in cams:
                video_files: dict[tuple[int, int], list[int]] = {}
                for _, erow in ep_df.iterrows():
                    chunk = int(erow[f"videos/{cam}/chunk_index"])
                    findex = int(erow[f"videos/{cam}/file_index"])
                    video_files.setdefault((chunk, findex), []).append(int(erow["episode_index"]))

                for (chunk, findex), ep_indices in video_files.items():
                    video_path = ds / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{findex:03d}.mp4"
                    if not video_path.exists():
                        continue

                    print(f"[REENCODE] {video_path} ({len(ep_indices)} episodes)")

                    # Get video dimensions once
                    probe = subprocess.run(
                        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
                         "-show_entries", "stream=width,height",
                         "-print_format", "json", str(video_path)],
                        capture_output=True, text=True, check=True,
                    )
                    stream = json.loads(probe.stdout)["streams"][0]
                    w, h = int(stream["width"]), int(stream["height"])

                    # Process episode-by-episode to avoid OOM
                    segments = []
                    total_in = 0
                    total_out = 0
                    for eidx in ep_indices:
                        erow = ep_df[ep_df["episode_index"] == eidx].iloc[0]
                        from_f = int(erow[f"videos/{cam}/from_frame"])
                        to_f = int(erow[f"videos/{cam}/to_frame"])
                        ep_len = to_f - from_f
                        total_in += ep_len

                        # Decode single episode via ffmpeg seek
                        start_t = from_f / fps
                        proc = subprocess.run(
                            ["ffmpeg", "-ss", f"{start_t:.6f}",
                             "-i", str(video_path),
                             "-frames:v", str(ep_len),
                             "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-v", "quiet", "-"],
                            capture_output=True, check=True,
                        )
                        frames = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(-1, h, w, 3)

                        if eidx in trim_info:
                            frames = frames[:trim_info[eidx]]
                        total_out += len(frames)

                        seg_path = tmp / f"{cam.replace('.', '_')}_{eidx}.mp4"
                        encode_video(frames, seg_path, fps, args.crf, gop=gop_val)
                        segments.append(seg_path)
                        del frames

                    # Concatenate segments via -c copy (lossless, same codec)
                    print(f"  {cam}: {total_in} -> {total_out} frames")
                    concat_out = tmp / f"{cam.replace('.', '_')}_final.mp4"
                    concat_videos_copy(segments, concat_out)

                    import shutil
                    shutil.move(str(concat_out), str(video_path))
                    for seg in segments:
                        seg.unlink(missing_ok=True)
                    print(f"  [WRITE] {video_path}")

    elif args.keyframe_cut:
        with tempfile.TemporaryDirectory(prefix="trim_") as tmp_dir:
            tmp = Path(tmp_dir)
            for cam in cams:
                video_files: dict[tuple[int, int], list[int]] = {}
                for _, row in ep_df.iterrows():
                    chunk = int(row[f"videos/{cam}/chunk_index"])
                    findex = int(row[f"videos/{cam}/file_index"])
                    video_files.setdefault((chunk, findex), []).append(int(row["episode_index"]))

                for (chunk, findex), ep_indices in video_files.items():
                    video_path = ds / "videos" / cam / f"chunk-{chunk:03d}" / f"file-{findex:03d}.mp4"
                    if not video_path.exists():
                        continue

                    print(f"[KEYFRAME-CUT] {video_path}")
                    segments = []
                    for eidx in ep_indices:
                        row = ep_df[ep_df["episode_index"] == eidx].iloc[0]
                        from_f = int(row[f"videos/{cam}/from_frame"])
                        to_f = int(row[f"videos/{cam}/to_frame"])

                        if eidx in trim_info:
                            end_f = from_f + trim_info[eidx]
                        else:
                            end_f = to_f

                        start_t = from_f / fps
                        end_t = end_f / fps
                        seg_path = tmp / f"{cam.replace('.','_')}_{eidx}.mp4"
                        extract_segment_copy(video_path, seg_path, start_t, end_t)
                        segments.append(seg_path)

                    # Concatenate segments
                    concat_out = tmp / f"{cam.replace('.','_')}_concat.mp4"
                    concat_videos_copy(segments, concat_out)

                    # Verify frame count
                    actual_count = get_video_frame_count(concat_out)
                    expected = sum(
                        trim_info.get(eidx, int(ep_df[ep_df["episode_index"] == eidx].iloc[0]["length"]))
                        for eidx in ep_indices
                    )
                    print(f"  {cam}: expected {expected} frames, got {actual_count}")

                    # Replace original
                    import shutil
                    shutil.move(str(concat_out), str(video_path))
                    print(f"  [WRITE] {video_path}")

    # --- Update episode metadata ---
    new_ep_rows = []
    # Track frame offsets per (cam, chunk, file) — each video file starts at frame 0
    video_frame_offsets: dict[tuple[str, int, int], int] = {}

    for _, row in ep_df.iterrows():
        ep_idx = int(row["episode_index"])
        ep_trimmed = trimmed_df[trimmed_df["episode_index"] == ep_idx]
        new_len = len(ep_trimmed)
        old_len = int(row["length"])

        new_row = row.copy()
        new_row["length"] = new_len
        new_row["dataset_from_index"] = int(ep_trimmed["index"].iloc[0])
        new_row["dataset_to_index"] = int(ep_trimmed["index"].iloc[-1]) + 1
        new_row["data/chunk_index"] = 0
        new_row["data/file_index"] = 0

        for cam in cams:
            if args.reencode or args.keyframe_cut:
                chunk = int(row[f"videos/{cam}/chunk_index"])
                findex = int(row[f"videos/{cam}/file_index"])
                key = (cam, chunk, findex)
                offset = video_frame_offsets.get(key, 0)
                new_row[f"videos/{cam}/from_frame"] = offset
                new_row[f"videos/{cam}/to_frame"] = offset + new_len
                new_row[f"videos/{cam}/from_timestamp"] = offset / fps
                new_row[f"videos/{cam}/to_timestamp"] = (offset + new_len) / fps
                video_frame_offsets[key] = offset + new_len
            else:
                frames_removed = old_len - new_len
                to_frame_key = f"videos/{cam}/to_frame"
                if to_frame_key in row:
                    new_row[to_frame_key] = int(row[to_frame_key]) - frames_removed
                    from_ts = float(row[f"videos/{cam}/from_timestamp"])
                    new_row[f"videos/{cam}/to_timestamp"] = from_ts + new_len / fps

        new_ep_rows.append(new_row)

    new_ep_df = pd.DataFrame(new_ep_rows)
    # Restore int64 dtypes (row.copy() + int assignment causes float64 promotion)
    for col in new_ep_df.columns:
        if col in ep_df.columns and ep_df[col].dtype == "int64":
            new_ep_df[col] = new_ep_df[col].astype("int64")
    new_ep_df.to_parquet(ep_meta_path, index=False)
    print(f"[WRITE] {ep_meta_path}")

    # --- Update info.json ---
    info["total_frames"] = total_after
    with open(ds / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[WRITE] meta/info.json")

    # --- Update stats.json ---
    stats_path = ds / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        for key in ["observation.state", "action", "next.reward"]:
            if key in trimmed_df.columns:
                values = trimmed_df[key]
                if key in ["observation.state", "action"]:
                    arr = np.stack(values.values)
                else:
                    arr = values.values.astype(np.float32).reshape(-1, 1)
                stats[key] = {
                    "mean": np.mean(arr, axis=0).tolist(),
                    "std": np.std(arr, axis=0).tolist(),
                    "min": np.min(arr, axis=0).tolist(),
                    "max": np.max(arr, axis=0).tolist(),
                    "count": [len(arr)],
                }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[WRITE] meta/stats.json")

    # --- Update sim_rewards.pt ---
    if sim_rewards is not None and "episode_index" in sim_rewards:
        sr_ei = sim_rewards["episode_index"]
        # Build per-episode frame counts in sim_rewards
        sr_ep_counts = {}
        for ep in sr_ei.unique().tolist():
            sr_ep_counts[ep] = (sr_ei == ep).sum().item()
        # Build keep mask per tensor length
        sr_mask = torch.ones(len(sr_ei), dtype=torch.bool)
        sr_offset = 0
        for ep in sorted(sr_ep_counts.keys()):
            ep_len = sr_ep_counts[ep]
            if ep in trim_info:
                # Trim same relative amount, but clamp to sr episode length
                trim_at = min(trim_info[ep], ep_len)
                if trim_at < ep_len:
                    sr_mask[sr_offset + trim_at : sr_offset + ep_len] = False
            sr_offset += ep_len
        new_sr = {}
        for k, v in sim_rewards.items():
            if v.shape[0] == len(sr_ei):
                new_sr[k] = v[sr_mask]
            else:
                # Different length tensor (e.g. collision_penalty) — skip trimming
                new_sr[k] = v
        torch.save(new_sr, sr_path)
        sr_len = new_sr["episode_index"].shape[0]
        print(f"[WRITE] sim_rewards.pt ({sr_len} frames)")

    if args.reencode:
        gop_label = args.gop if args.gop != "auto" else "auto"
        print(f"\nDone. Videos: re-encoded (CRF={args.crf}, GOP={gop_label}).")
    elif args.keyframe_cut:
        print("\nDone. Videos: keyframe cut (-c copy, lossless).")
    else:
        print("\nDone. Videos: unchanged (metadata only).")


if __name__ == "__main__":
    main()
