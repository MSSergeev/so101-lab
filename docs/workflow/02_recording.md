# Recording Episodes

Teleoperation setup, dataset recording, post-processing, and visualization.

---

## 0. Leader arm calibration (first time only)

Calibration is required before using the SO-101 leader arm. It writes homing offsets and range limits to a JSON file in `so101_lab/devices/lerobot/.cache/`.

```bash
act-lerobot  # activate lerobot-env
python scripts/calibrate_leader.py --port /dev/ttyACM2 --name leader_1
```

Interactive steps:
1. Move arm to **middle** of range → press Enter
2. Move all joints through **full range of motion** → press Enter

The file is saved as `so101_lab/devices/lerobot/.cache/leader_1.json` and referenced via `--calibration-file=leader_1.json` during recording.

See [reference/teleop_devices.md](../reference/teleop_devices.md) for details.

---

## 1. Before recording

Make sure the env spawn looks correct:

```bash
act-isaac  # activate Isaac Lab env
python scripts/eval/test_env_spawn.py --gui --env figure_shape_placement_easy --resets 20
```

Check that the cube appears in the expected zone and does not intersect the platform.
See [01_env_design.md](01_env_design.md) for spawn zone details.

---

## 2. Recording commands

### Basic recording (without rewards)

```bash
act-isaac  # activate Isaac Lab env

python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --headless \
    --teleop-device=so101leader \
    --calibration-file=leader_1.json \
    --port=/dev/ttyACM0 \
    --crf 18
```

### Recording with sim rewards

```bash
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --headless \
    --teleop-device=so101leader \
    --calibration-file=leader_1.json \
    --port=/dev/ttyACM0 \
    --crf 18 \
    --reward-mode success
```

`--reward-mode success` writes a binary `next.reward` column (0/1) to parquet. Shaped sim rewards are saved to `sim_rewards.pt` alongside the dataset.

> **`--task`** — text description written to `meta/tasks.parquet`. If the env has `get_task_description()` it takes priority and `--task` is ignored. Optional, useful as a fallback for envs without a description.

### Key CLI options

| Option | Description |
|--------|-------------|
| `--crf 18` | H.264 quality (lower = better, 18 is near-lossless) |
| `--headless` | Run without GUI (faster) |
| `--reward-mode success` | Binary reward from episode outcome |
| `--diversity-keys cube_x,cube_y` | Enable spawn diversity tracking |
| `--diversity-ratio 1.2` | Bias resampling toward underrepresented zones |
| `--port` | Leader arm serial port |
| `--calibration-file` | Leader arm calibration JSON filename |

---

## 3. Spawn diversity

For larger datasets (50+ episodes), use spawn diversity to ensure even coverage of the spawn zone. The diversity checker requires at least 10 existing episodes to estimate the bounding box — it auto-disables with fewer. Record the first batch without diversity, then enable it for subsequent batches:

```bash
# First batch (50 episodes) — no diversity
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --headless --teleop-device=so101leader \
    --calibration-file=leader_1.json --port=/dev/ttyACM0 \
    --crf 18 --reward-mode success

# Subsequent batches — with diversity
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --headless --teleop-device=so101leader \
    --calibration-file=leader_1.json --port=/dev/ttyACM0 \
    --crf 18 --reward-mode success \
    --diversity-keys cube_x,cube_y \
    --diversity-ratio 1.2
```

Each subsequent batch appends to the same output directory. The diversity tracker reads existing episodes and biases new spawns toward underrepresented positions.

Check diversity of an existing dataset:

```bash
act-lerobot  # activate lerobot-env
python -c "
from so101_lab.utils.spawn_diversity import SpawnDiversityChecker
c = SpawnDiversityChecker('data/recordings/my_task_v1', ['cube_x', 'cube_y'])
print(c.report())
"
```

---

## 4. Live camera preview

Camera preview is enabled by default — `record_episodes.py` auto-launches `camera_viewer.py` as a subprocess using `venvs/viewer`. No separate terminal needed.

To disable: `--no-preview`. To run the viewer manually in a separate terminal:

```bash
source venvs/viewer/bin/activate
python scripts/tools/camera_viewer.py
```

Images are exchanged via `/dev/shm/` (RAM disk).

---

## 5. Post-processing

### Save a backup before modifying

```bash
cp -r data/recordings/my_task_v1 data/recordings/my_task_v1_raw
```

### Trim trailing frames

Episodes often have extra frames after the success event where the robot holds position. Trim them:

```bash
act-lerobot  # activate lerobot-env

python scripts/tools/trim_after_success.py \
    data/recordings/my_task_v1 --keep-after 10 --reencode --crf 18
```

`--keep-after 10` keeps 10 frames after the last `reward=1` frame. `--reencode` re-encodes the video after trimming.

### Verify dataset integrity

```bash
python scripts/verify_lerobot_dataset.py data/recordings/my_task_v1
```

---

## 6. Visualization

Inspect recorded episodes in Rerun:

```bash
source venvs/rerun/bin/activate
python scripts/visualize_lerobot_rerun.py data/recordings/my_task_v1 -e 0 5 10 50
```

`-e` selects specific episode indices to visualize. Shows both camera feeds, joint positions, and reward signal.

---

## Dataset format

Recordings are saved in LeRobot v3.0 format:

```
data/recordings/my_task_v1/
├── meta/
│   ├── info.json                          # features, fps, robot_type
│   ├── stats.json                         # per-feature statistics
│   ├── tasks.parquet                      # task description strings
│   └── episodes/chunk-000/file-000.parquet  # per-episode metadata
├── data/chunk-000/file-000.parquet        # observations, actions, rewards
└── videos/
    ├── observation.images.top/chunk-000/file-000.mp4
    └── observation.images.wrist/chunk-000/file-000.mp4
```

Frame format:

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `observation.state` | (6,) | float32 | Joint positions, normalized [-100, 100] |
| `action` | (6,) | float32 | Target joint positions, same space |
| `observation.images.top` | — | video | Top-down camera, 640×480, H.264 |
| `observation.images.wrist` | — | video | Wrist camera, 640×480, H.264 |
| `next.reward` | scalar | float32 | Binary success (0/1) if `--reward-mode success` |
