# lerobot_dataset.md

`LeRobotDatasetWriter` — writes teleoperation episodes directly to LeRobot v3.0 format (parquet + H.264 video). lerobot-env for training; Isaac Lab env for recording.

---

Runs in **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

Writes multi-episode datasets in LeRobot v3.0 format with size-based file chunking (many episodes per parquet/video file). Fully compatible with `LeRobotDataset` API for training.

Key design decisions:
- **FFmpeg subprocess** (not PyAV) — simpler, no extra deps, concat demuxer avoids re-encoding
- **Size-based chunking** — many episodes per file, scales to large datasets
- **Resume** creates new files (ParquetWriter has no append mode); LeRobot API auto-merges all files in a chunk

## Dataset structure

```
dataset/
├── meta/
│   ├── info.json                        # schema, fps, robot_type, totals
│   ├── stats.json                       # mean/std/min/max for normalization
│   ├── tasks.parquet                    # task_index → task string
│   └── episodes/chunk-000/
│       └── file-000.parquet             # episode boundaries and file refs
├── data/chunk-000/
│   ├── file-000.parquet                 # ~100 MB, many episodes
│   └── file-001.parquet                 # on resume or size limit
└── videos/
    ├── observation.images.top/chunk-000/
    │   ├── file-000.mp4                 # ~200 MB, concatenated episodes
    │   └── file-001.mp4
    └── observation.images.wrist/chunk-000/
        ├── file-000.mp4
        └── file-001.mp4
```

## Frame format

```python
{
    "observation.state":        np.ndarray,  # (6,) float32, normalized [-100, 100]
    "action":                   np.ndarray,  # (6,) float32, normalized [-100, 100]
    "observation.images.top":   np.ndarray,  # (480, 640, 3) uint8 RGB
    "observation.images.wrist": np.ndarray,  # (480, 640, 3) uint8 RGB
    "timestamp":                float,       # seconds from episode start (auto if missing)
}
```

## API

```python
from so101_lab.data import LeRobotDatasetWriter

writer = LeRobotDatasetWriter(
    output_dir="data/recordings/my_task",
    fps=30,
    task="pick cube",
    codec="h264",          # or "h265" (~50% smaller, slower)
    crf=23,                # 18=near-lossless, 23=default, 30=max compress
    chunks_size=1000,      # max files per chunk directory
    data_files_size_mb=100.0,
    video_files_size_mb=200.0,
)

for frame in episode_frames:
    writer.add_frame(frame)

episode_idx = writer.save_episode()
writer.close()  # writes info.json, stats.json, tasks.parquet
```

**Resume:**
```python
writer = LeRobotDatasetWriter(output_dir="data/recordings/my_task", fps=30, task="pick cube")
# [RESUME] Creating new data file: chunk-000/file-001.parquet
# [RESUME] Creating new video file: videos/.../chunk-000/file-001.mp4
print(writer.total_episodes)  # continues from last episode
```

**Load with LeRobot API:**
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(repo_id="my_dataset", root="/path/to/dataset", tolerance_s=1/30)
sample = ds[0]
# sample["observation.images.top"]: torch.Size([3, 480, 640])
```

## Verification

```bash
act-lerobot  # activate lerobot-env
python scripts/verify_lerobot_dataset.py data/recordings/my_task
```

Unit tests: `tests/test_lerobot_v3_writer.py` (lerobot-env)

## Data flow and conversions

Recording pipeline: `record_episodes.py` calls `recorder.on_step(obs, action)` **before** `env.step()` — records `(obs_t, action_t)` pairs for correct IL semantics.

`RecordingManager._build_frame()` (`data/collector.py`) converts:

| Step | Input | Output | Loss |
|------|-------|--------|------|
| GPU tensor → numpy | torch.float32 | float64 | ~1e-6 |
| rad → normalized | `joint_rad_to_motor_normalized()` | float32 | ~1e-6 |
| image float→uint8 | float32 [0,1] | uint8 [0,255] | ~0.004 (quantization) |

`joint_rad_to_motor_normalized()` (`data/converters.py`): linear interpolation from `SO101_JOINT_LIMITS_RAD` → `SO101_MOTOR_LIMITS_NORMALIZED`. Note: `wrist_roll` has asymmetric limits [-157°, +162°] → non-linear mapping at extremes.

## Episode metadata schema

`meta/episodes/chunk-000/file-000.parquet` columns:

```
episode_index, length, task_index,
data/chunk_index, data/file_index,
dataset_from_index, dataset_to_index,
videos/observation.images.top/chunk_index,
videos/observation.images.top/file_index,
videos/observation.images.top/from_timestamp,
videos/observation.images.top/to_timestamp,
videos/observation.images.top/from_frame,
videos/observation.images.top/to_frame,
# same for observation.images.wrist
```

`dataset_from_index` / `dataset_to_index` — global frame indices for parquet lookup. `from_timestamp` / `to_timestamp` — used by LeRobot API to seek into the concatenated video file. `file_index` tells which `file-00N.*` contains this episode (changes on resume or size rotation).

## stats.json schema

```json
{
  "observation.state": {
    "mean": [v0, v1, v2, v3, v4, v5],
    "std":  [v0, v1, v2, v3, v4, v5],
    "min":  [v0, v1, v2, v3, v4, v5],
    "max":  [v0, v1, v2, v3, v4, v5],
    "count": [N_frames]
  },
  "observation.images.top": {
    "mean": [[[r]], [[g]], [[b]]],
    "std":  [[[r]], [[g]], [[b]]],
    "min":  [[[r]], [[g]], [[b]]],
    "max":  [[[r]], [[g]], [[b]]],
    "count": [N_pixels]
  }
}
```

State/action: shape `(D,)`. Images: shape `(C, 1, 1)` — broadcast-compatible with `(C, H, W)`. `count` is used for correct weighted aggregation across resume sessions.

## Video sizes

480×640 RGB @ 30 fps, 1 minute of recording:

| Codec | CRF | Size | Notes |
|-------|-----|------|-------|
| h265 | 18 | ~400 MB | near-lossless |
| h264 | 23 | ~260 MB | **default** |
| h265 | 30 | ~80 MB | max compression |

Uncompressed: ~2.6 GB. GOP size `g=2` — fast random access during training (keyframe every 2 frames).

## Visualization and publishing

```bash
# Visualize in Rerun
source venvs/rerun/bin/activate
python scripts/visualize_lerobot_rerun.py data/recordings/my_task

# Push to HuggingFace Hub
act-lerobot  # activate lerobot-env
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id='local', root='data/recordings/my_task')
ds.push_to_hub(repo_id='username/my-task')
"
```

## Notes

- `stats.json` uses weighted running sums (via `count`) — correctly aggregates across resume sessions
- Video concatenation: FFmpeg concat demuxer (`-c copy`), no re-encoding, no quality loss
- `recompute_stats.py` — recomputes stats if recorded incorrectly
- `filter_dataset.py` — filter/renumber episodes (re-encodes video segments)

## Troubleshooting

**`tolerance_s violated`** — LeRobot can't match video frame timestamp. Fix: pass `tolerance_s=1/30` to `LeRobotDataset`.

**FFmpeg concat failed** — episodes encoded with different codec/resolution/fps. All episodes in one dataset must use identical encoding params (enforced by `LeRobotDatasetWriter` if you don't change params between sessions).
