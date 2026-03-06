# visualize_lerobot_rerun.py

Visualize LeRobot v3.0 dataset episodes in Rerun viewer. `rerun` venv.

---

## Overview

Loads a LeRobot v3.0 dataset (parquet + multi-file MP4), decodes video frames, and streams everything to Rerun: camera images, per-joint state and action, rewards. Supports JPEG re-compression to reduce viewer memory (~10x).

---

## Setup

```bash
uv venv venvs/rerun
source venvs/rerun/bin/activate
uv pip install rerun-sdk opencv-python numpy pandas pillow
```

---

## Usage

```bash
source venvs/rerun/bin/activate

# View first 10 episodes (default)
python scripts/visualize_lerobot_rerun.py data/recordings/figure_shape_placement_v4

# Specific episodes
python scripts/visualize_lerobot_rerun.py data/recordings/dataset -e 0 2 5

# Load more episodes
python scripts/visualize_lerobot_rerun.py data/recordings/dataset --max-episodes 50

# Skip images (fast, state/action only)
python scripts/visualize_lerobot_rerun.py data/recordings/dataset --skip-images

# Save to .rrd file instead of spawning viewer
python scripts/visualize_lerobot_rerun.py data/recordings/dataset --save output.rrd

# Reduce RAM (~1 GB instead of ~4.5 GB)
python scripts/visualize_lerobot_rerun.py data/recordings/dataset --batch-size 500

# Raw RGB (no JPEG re-compression, ~10x more viewer memory)
python scripts/visualize_lerobot_rerun.py data/recordings/dataset --no-jpeg

# Higher quality JPEG
python scripts/visualize_lerobot_rerun.py data/recordings/dataset --jpeg-quality 95

# Downsample frames (every 3rd frame)
python scripts/visualize_lerobot_rerun.py data/recordings/dataset --downsample 3
```

---

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `dataset_path` | required | Path to LeRobot v3.0 dataset directory |
| `-e` / `--episodes` | — | Specific episode indices (space-separated) |
| `--max-episodes` | 10 | Cap on episodes when `-e` not given |
| `--skip-images` | false | Skip image decoding (state/action/reward only) |
| `--downsample N` | 1 | Log every N-th frame |
| `--save PATH` | — | Save to `.rrd` instead of spawning viewer |
| `--jpeg` / `--no-jpeg` | jpeg=true | JPEG re-compression of decoded frames |
| `--jpeg-quality` | 80 | JPEG quality 1–100 |
| `--batch-size` | 2500 | Video frames to prefetch per batch (affects RAM) |

---

## Rerun data layout

```
episode/index       — scalar: current episode index
episode/task        — text: task string from tasks.parquet
cameras/top/rgb     — image: top camera frame
cameras/wrist/rgb   — image: wrist camera frame
robot/state/<joint> — scalar per joint (normalized motor units)
robot/action/<joint>— scalar per joint (normalized motor units)
reward              — scalar: next.reward (if present in dataset)
```

Two timelines: `frame` (global across all episodes) and `episode_frame` (local within episode).

Joint names: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`.

---

## Architecture: VideoReaderMultiFile

LeRobot v3.0 stores video in multiple files per camera: `videos/<key>/chunk-000/file-000.mp4`, `file-001.mp4`, etc. Each file contains frames from multiple episodes.

`VideoReaderMultiFile` handles this:

1. Opens a file on demand (`cv2.VideoCapture`)
2. Prefetches a batch of frames into an in-memory dict (`frame_idx → np.ndarray`)
3. On cache miss — loads the next batch starting from the requested frame
4. On file switch — releases previous `VideoCapture`, clears buffer

Per-episode frame range is read from `meta/episodes/chunk-000/file-000.parquet`:
- `videos/observation.images.top/file_index` — which MP4 file
- `videos/observation.images.top/from_frame` — frame offset within that file

Frame lookup: `video_frame_idx = from_frame + local_frame_within_episode`.

---

## Memory and performance

| Mode | RAM (10 episodes, 2 cameras) | Notes |
|------|-------------------------------|-------|
| `--jpeg` (default) | ~0.5 GB | JPEG at quality 80 sent to Rerun |
| `--no-jpeg` | ~4.5 GB | Raw RGB arrays sent to Rerun |
| `--batch-size 500` | ~1 GB | Smaller prefetch buffer |
| `--skip-images` | <50 MB | No video decoding at all |

Default `--batch-size 2500` prefetches ~2500 frames per camera into RAM. For large datasets or many episodes, reduce with `--batch-size 500`.

JPEG compression is done with Pillow (`Image.save(..., format="JPEG")`) before passing to `rr.EncodedImage`. Rerun stores the compressed bytes, so viewer memory scales with JPEG size, not raw frame size.

---

## Notes

- `tasks.parquet` format: supports both `{task, task_index}` columns and task-as-index formats (v3.0 compatibility).
- All data files in `chunk-000/` are loaded and concatenated at startup; episode slices are extracted by `dataset_from_index` / `dataset_to_index` from the episodes parquet.
- `--save` uses `rr.save()` — the `.rrd` file can be opened later with `rerun output.rrd`.
- If a camera directory is missing (e.g. dataset has only top camera), that camera is silently skipped.
- `next.reward` column is optional — logged only if present in the data parquet.
