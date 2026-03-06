# Local Virtual Environments

Lightweight tool environments for the project. Created locally, not tracked in git.

## venvs/rerun — Rerun visualizer

For visualizing recorded episodes.

```bash
python3 -m venv venvs/rerun
source venvs/rerun/bin/activate
pip install rerun-sdk pandas pyarrow opencv-python numpy
```

**Usage:**
```bash
source venvs/rerun/bin/activate
python scripts/visualize_lerobot_rerun.py data/recordings/easy_task_v1 -e 0 5 10
```

## venvs/viewer — OpenCV camera preview

For live camera preview during teleoperation recording.

```bash
python3 -m venv venvs/viewer
source venvs/viewer/bin/activate
pip install opencv-python numpy
```

**Usage (two terminals):**
```bash
# Terminal 1: recording (Isaac Lab env)
source /path/to/env_isaaclab/bin/activate
python scripts/teleop/record_episodes.py --teleop-device=so101leader ...

# Terminal 2: live preview
source venvs/viewer/bin/activate
python scripts/tools/camera_viewer.py
```

The recording script writes camera frames to `/dev/shm/` (RAM disk):
- `/dev/shm/so101_camera_top.jpg`
- `/dev/shm/so101_camera_wrist.jpg`
- `/dev/shm/so101_status.json`

`camera_viewer.py` reads these and displays them in OpenCV windows.

## Structure

```
venvs/
├── README.md     # This file (tracked in git)
├── rerun/        # Rerun SDK (git-ignored)
└── viewer/       # OpenCV GUI (git-ignored)
```
