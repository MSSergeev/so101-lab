# Viewer Tools

OpenCV-based camera preview tools for monitoring simulation in real time. `viewer` venv (isolated from Isaac Lab).

---

## Overview

All viewer scripts follow the same pattern:

```
Isaac Lab process          viewer process (venvs/viewer)
──────────────────         ──────────────────────────────
write_camera_to_shm()  →   cv2.imread(/dev/shm/so101_camera_*.jpg)
write_status_to_shm()  →   read_status() → overlay on frame
                       ←   write_command() → read_command()
```

The Isaac Lab process writes camera frames as JPEG to `/dev/shm` (shared memory filesystem) and reads command JSON from `/dev/shm/so101_command.json`. The viewer reads frames, draws an overlay, and writes commands back. Atomic rename (`tmp → final`) prevents torn reads.

**Why a separate venv?** The viewer only needs `opencv-python` and `numpy`. Isaac Lab's `PYTHONPATH` would conflict, so `launch_viewer()` strips Isaac-related env vars before spawning the subprocess.

---

## Setup

```bash
uv venv venvs/viewer
source venvs/viewer/bin/activate
uv pip install opencv-python numpy
```

One-time setup. The viewer venv is shared by all four scripts.

---

## Shared infrastructure: `shm_preview.py`

`so101_lab/utils/shm_preview.py` — server side, used by Isaac Lab scripts.

```python
from so101_lab.utils.shm_preview import (
    write_camera_to_shm,   # write top/wrist frame to /dev/shm
    write_status_to_shm,   # write status dict to /dev/shm
    read_command,          # read latest command from viewer
    launch_viewer,         # spawn viewer subprocess (strips Isaac env vars)
    stop_viewer,           # SIGTERM viewer subprocess
    cleanup_shm,           # remove /dev/shm files on exit
)
```

`launch_viewer(script="camera_viewer.py")` — pass the script name to select which viewer to launch. Returns a `subprocess.Popen` handle or `None` if venv not found (continues without preview).

Status dict keys used by each viewer:

| Key | Used by |
|-----|---------|
| `state`, `recording`, `teleop`, `episode`, `frame`, `status_text`, `timestamp` | camera_viewer |
| `episode`, `total_episodes`, `step`, `max_steps`, `seed`, `success`, `successes`, `status_text` | eval_viewer |
| `state`, `teleop`, `episode`, `frame`, `status_text`, `timestamp` | hil_viewer |

---

## camera_viewer.py

For `record_episodes.py`. Displays both cameras side-by-side with recording state overlay.

```bash
source venvs/viewer/bin/activate
python scripts/tools/camera_viewer.py
python scripts/tools/camera_viewer.py --scale 0.5
python scripts/tools/camera_viewer.py --fps 60 --no-hints
```

**Overlay:** IDLE (gray) / TELEOP (green circle) / REC (red circle) + episode/frame counters + preview FPS (computed from `status.timestamp` deltas).

**Controls:**

| Key | Command sent | Effect in record_episodes.py |
|-----|-------------|------------------------------|
| Space | `teleop` | IDLE → TELEOP |
| N | `record` | IDLE/TELEOP → RECORDING |
| F | `save` | Save episode + reset |
| R | `rerecord` | Restart same scene |
| X | `discard` | Reset with new scene |
| Escape | `quit` | Stop recording |

---

## eval_viewer.py

For `eval_act_policy.py` (and other eval scripts with `--preview`). Displays episode/step progress and live success rate.

```bash
source venvs/viewer/bin/activate
python scripts/tools/eval_viewer.py
python scripts/tools/eval_viewer.py --scale 1.0
```

**Overlay:** episode counter, step counter, seed, success rate with color (green ≥50%, orange ≥25%, red <25%).

**Controls:**

| Key | Command sent | Effect in eval script |
|-----|-------------|----------------------|
| Space | `pause` | Pause/Resume |
| N | `next` | Skip current episode |
| R | `restart` | Restart current episode (same seed) |
| Escape | `quit` | Stop evaluation |

---

## hil_viewer.py

For `train_sac.py --hil`. Displays HIL takeover state.

```bash
source venvs/viewer/bin/activate
python scripts/tools/hil_viewer.py
```

**Overlay:** POLICY (orange circle) / HIL ON (green circle) + episode/frame + preview FPS.

**Controls:**

| Key | Command sent | Effect in train_sac.py |
|-----|-------------|------------------------|
| Space | `teleop` | Toggle HIL takeover ON/OFF |
| X | `discard` | Reset episode |
| Escape | `quit` | Stop training |

---

## CLI flags (all viewers)

| Flag | Default | Description |
|------|---------|-------------|
| `--scale` | 1.5 | Window scale factor (base: 640×480 per camera) |
| `--fps` | 30 | Target display FPS (rate-limited via sleep) |
| `--no-hints` | false | Hide keyboard hint overlay (camera_viewer only) |

---

## Notes

- If `/dev/shm/so101_camera_*.jpg` is absent, the viewer shows "Waiting for data..." and polls until Isaac Lab starts writing.
- The viewer does not block Isaac Lab — it runs in a separate process. Frame rate mismatch is normal (viewer may display the same frame twice or skip frames).
- If `venvs/viewer` does not exist, `launch_viewer()` prints a warning and returns `None`. Isaac Lab continues without preview.
- Viewer is launched with `start_new_session=True` — it survives `Ctrl+C` in the main process. `stop_viewer()` sends `SIGTERM` to the process group.
