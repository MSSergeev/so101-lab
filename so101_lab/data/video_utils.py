# Adapted from: LeRobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: FFmpeg subprocess utilities for video concatenation and metadata
"""Video utility functions for LeRobot v3.0 dataset writer.

Uses FFmpeg subprocess calls for simplicity and compatibility.
"""

import json
import subprocess
import tempfile
from pathlib import Path


def get_video_duration_s(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def concatenate_video_files(input_paths: list[Path], output_path: Path) -> None:
    """Concatenate videos using FFmpeg concat demuxer (no re-encoding).

    Uses concat demuxer for fast concatenation without quality loss:
    ffmpeg -f concat -safe 0 -i concat.txt -c copy output.mp4

    Args:
        input_paths: List of video files to concatenate (in order)
        output_path: Output path for concatenated video
    """
    import shutil

    if not input_paths:
        return

    if len(input_paths) == 1:
        # Single file, just copy/move
        if input_paths[0] != output_path:
            shutil.copy2(input_paths[0], output_path)
        return

    # Create concat file list with absolute paths
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for path in input_paths:
            abs_path = Path(path).resolve()
            # Escape single quotes in path
            escaped = str(abs_path).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
        concat_list_path = f.name

    try:
        # Use temp output to avoid overwriting input
        temp_output = output_path.with_suffix(".tmp.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            str(temp_output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg concat failed: {result.stderr}")

        # Move temp to final output
        temp_output.rename(output_path)
    finally:
        Path(concat_list_path).unlink(missing_ok=True)


def encode_images_to_video(
    images,  # np.ndarray (N, H, W, 3) uint8
    output_path: Path,
    fps: int,
    codec: str = "h264",
    crf: int = 23,
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
) -> None:
    """Encode numpy images to MP4 video using FFmpeg.

    Args:
        images: Numpy array of shape (N, H, W, 3) with uint8 RGB values
        output_path: Path for output video
        fps: Frames per second
        codec: Video codec (h264 or h265)
        crf: Quality (lower = better, 23 is default)
        pix_fmt: Pixel format
        g: GOP size (keyframe interval). g=2 for fast random access during training.
           None = FFmpeg default (auto).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_frames, height, width, _ = images.shape
    codec_map = {"h264": "libx264", "h265": "libx265"}
    ffmpeg_codec = codec_map.get(codec, "libx264")

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", ffmpeg_codec,
        "-pix_fmt", pix_fmt,
        "-crf", str(crf),
    ]
    if g is not None:
        cmd.extend(["-g", str(g)])
    cmd.extend(["-preset", "medium", str(output_path)])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _, stderr = proc.communicate(input=images.tobytes())

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg encoding failed: {stderr.decode()}")


def get_video_frame_count(video_path: Path) -> int:
    """Get number of frames in video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-print_format", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return int(data["streams"][0]["nb_read_frames"])
