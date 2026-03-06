# trim_after_success.py

Remove trailing frames after the first success event from a LeRobot dataset. lerobot-env venv.

Modifies dataset in-place: parquet, episode metadata, stats, and optionally videos.

---

## Overview

After teleop recording with `--reward-mode success`, all frames following the first `is_success()` receive `reward=1`. These "hold position" frames dilute the RL reward signal. Trimming leaves only the meaningful frames up to and slightly after success.

---

## CLI

```bash
# Preview — show what would be trimmed, no changes
python scripts/tools/trim_after_success.py data/recordings/ds --keep-after 10 --dry-run

# Metadata only (default) — parquet + episode meta, videos untouched
python scripts/tools/trim_after_success.py data/recordings/ds --keep-after 10

# Re-encode videos (precise frame-accurate cut)
python scripts/tools/trim_after_success.py data/recordings/ds --keep-after 10 --reencode --crf 18

# Keyframe cut — lossless -c copy, trim snaps to nearest keyframe
python scripts/tools/trim_after_success.py data/recordings/ds --keep-after 10 --keyframe-cut
```

| Flag | Default | Description |
|------|---------|-------------|
| `dataset` | required | Path to dataset directory |
| `--keep-after` | 10 | Frames to keep after the first success frame |
| `--reencode` | false | Re-encode videos (precise cut, slight quality loss) |
| `--keyframe-cut` | false | Cut at nearest keyframe via `-c copy` (lossless) |
| `--crf` | 23 | CRF quality for `--reencode` (lower = better quality) |
| `--gop` | auto | GOP size for `--reencode` (auto = ffmpeg default) |
| `--dry-run` | false | Print plan without modifying any files |

`--reencode` and `--keyframe-cut` are mutually exclusive.

---

## Video Modes

| Mode | Quality | Precision | Speed |
|------|---------|-----------|-------|
| default | 100% (untouched) | Trailing frames left in mp4 file | Instant |
| `--reencode` | ~100% (one transcode) | Frame-accurate | Slow |
| `--keyframe-cut` | 100% (lossless) | ±GOP frames from success | Fast |

**Tip:** if you know you'll trim later, record with `--gop 2`. Then `--keyframe-cut` will be accurate to ±1 frame without re-encoding.

---

## What Gets Updated

| File | Change |
|------|--------|
| `data/chunk-000/file-000.parquet` | Rows removed, `index` and `frame_index` re-numbered |
| `meta/episodes/chunk-000/file-000.parquet` | `length`, `dataset_from/to_index`, `to_frame`, `to_timestamp` |
| `meta/info.json` | `total_frames` |
| `meta/stats.json` | Recomputed for `observation.state`, `action`, `next.reward` (image stats not recomputed) |
| `sim_rewards.pt` | Filtered by same frame mask (if file exists) |
| Videos | Depends on mode (see above) |

---

## Notes

- Episodes with no success frame (`next.reward` never > 0) are skipped with a warning.
- `--keyframe-cut` finds the keyframe nearest to `first_success + keep_after` but no earlier than `first_success`, independently per camera. The more conservative (earlier) cut is used across both cameras.
- Default mode (metadata only) leaves extra frames in the mp4 file — the video file is longer than the metadata says. This is harmless for LeRobot dataset loading (it uses `from_frame`/`to_frame` to seek), but wastes disk space.
- `sim_rewards.pt` trimming uses the same relative trim points; if the tensor has a different length than parquet (rare), it is left untouched.
