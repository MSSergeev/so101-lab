# Adapting the Project

How to fork the project for a different robot, task, or scene.

The pipeline is split into two independent halves: the **simulator side** (Isaac Lab env, tasks, scene objects) and the **training side** (lerobot policies, training scripts). You can change either independently. A new task does not require changing any training script; a different policy does not require changing any environment.

---

## Adding a new task

The fastest path is copying an existing task. Use `figure_shape_placement_easy` as the template — it's the simplest variant.

### 1. Copy and rename

```bash
cp -r so101_lab/tasks/figure_shape_placement_easy so101_lab/tasks/<new_task>
```

Rename all classes inside (find/replace `FigureShapePlacementEasy` → `NewTask`):
- `env.py`: `FigureShapePlacementEasyEnv` → `NewTaskEnv`
- `env_cfg.py`: `FigureShapePlacementEasySceneCfg`, `FigureShapePlacementEasyEnvCfg`
- `rl/env_cfg.py`: `FigureShapePlacementEasyRLSceneCfg`, `FigureShapePlacementEasyRLEnvCfg`
- `rl/__init__.py`: gym ID `"SO101-FigureShapePlacementEasy-RL-v0"` → `"SO101-NewTask-RL-v0"`

Update the import paths inside each file: `from so101_lab.tasks.figure_shape_placement_easy.*` → `from so101_lab.tasks.<new_task>.*`

### 2. Edit scene constants in `env_cfg.py`

All task-specific geometry lives at the top of `env_cfg.py` as module-level constants:

```python
# Spawn zone — where the cube appears at each reset
CUBE_SPAWN_X_MIN = 0.314    # world frame, metres
CUBE_SPAWN_X_MAX = 0.354
CUBE_SPAWN_Y_MIN = 0.058
CUBE_SPAWN_Y_MAX = 0.098

# Cube yaw at spawn — discrete values in degrees
CUBE_YAW_VALUES_DEG = tuple(float(d) for d in range(0, 81, 10))

# Platform position — fixed per reset
PLATFORM_FIXED_X = 0.154
PLATFORM_FIXED_Y = 0.048
PLATFORM_YAW_VALUES_DEG = (90.0,)  # one value = fixed orientation

# Success condition
SUCCESS_THRESHOLD = 0.030   # XY distance from cube to slot (m)
ORIENTATION_THRESHOLD = 30.0  # allowed yaw error (degrees)

# Robot position (used for reachability checks on distractors)
ROBOT_X, ROBOT_Y = 0.254, 0.278
MIN_REACH = 0.105   # metres from robot pivot
MAX_REACH = 0.40
```

The RL env_cfg (`rl/env_cfg.py`) imports all these constants from `env_cfg.py` — you only need to change them in one place.

**The slot offset** (`SLOT_OFFSET`) defines where success is measured relative to the platform center. Change this if your target object has a different geometry.

### 3. Swap scene objects (USD files)

Object USD paths are set in `env_cfg.py`:

```python
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "props"
PLATFORM_USD_PATH = ASSETS_DIR / "platfrom_full.usd"
CUBE_USD_PATH = ASSETS_DIR / "light_blue_cube.usd"
DISTRACTOR_USD_PATHS = [...]
```

Drop your USD files into `assets/props/` and update these paths. The physics properties (friction, mass, solver iterations) are set in `FigureShapePlacementSceneCfg` inside `env_cfg.py`.

### 4. Set episode timing in RL env_cfg

In `rl/env_cfg.py`, `__post_init__`:

```python
self.decimation = 4          # physics steps per policy step (120 Hz / 4 = 30 Hz policy)
self.episode_length_s = 20.0 # episode timeout in seconds
self.sim.dt = 1.0 / 120.0   # physics timestep
```

The BC env uses `episode_length_s = 25.0` (set in `TemplateEnvCfg`). If you change the RL episode length, consider whether the teleop recording duration still makes sense.

### 5. Register the task

In `so101_lab/tasks/__init__.py`, add to both registry functions:

```python
# In get_task_registry():
from so101_lab.tasks.<new_task>.env import NewTaskEnv
from so101_lab.tasks.<new_task>.env_cfg import NewTaskEnvCfg
import so101_lab.tasks.<new_task>.rl  # gym registration
# ...
"<new_task>": (NewTaskEnv, NewTaskEnvCfg),

# In _get_rl_task_registry():
from so101_lab.tasks.<new_task>.rl.env_cfg import NewTaskRLEnvCfg
# ...
"<new_task>": NewTaskRLEnvCfg,
```

### 6. Verify

```bash
python tests/sim/test_env_spawn.py --gui --env <new_task> --resets 20
```

Check that objects appear in the right zone and don't overlap.

---

## Changing the scene (table, lighting, workbench)

The workbench scene (table, ceiling lamp, window light, dome light) is a single USD file referenced from `so101_lab/assets/scenes/workbench.py`:

```python
WORKBENCH_CLEAN_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Scene",
    spawn=sim_utils.UsdFileCfg(usd_path=".../workbench_v1.usd"),
)
```

To use a different scene, swap the USD file or add a new `AssetBaseCfg` in `workbench.py` and reference it in `TemplateSceneCfg`.

**Lighting** is embedded in the workbench USD. Light parameters used for domain randomization are set in `rl/mdp/events.py` (`randomize_light`). If you change the light names in the USD, update `events.py` accordingly.

---

## Changing the camera setup

Camera parameters are in `so101_lab/tasks/template/env_cfg.py` (`TemplateSceneCfg`). Both cameras are defined there and inherited by all tasks:

```python
top: TiledCameraCfg = TiledCameraCfg(
    prim_path="{ENV_REGEX_NS}/top_camera",
    offset=TiledCameraCfg.OffsetCfg(
        pos=(-0.309, -0.457, 1.217),
        rot=(0.84493, 0.44012, -0.17551, -0.24817),
        convention="opengl",
    ),
    spawn=sim_utils.PinholeCameraCfg(
        focal_length=23.5,       # affects FOV
        horizontal_aperture=38.11,
    ),
    width=640, height=480,
    update_period=1 / 30.0,      # 30 Hz
)
```

Individual tasks can override the top camera (e.g. `figure_shape_placement` moves it closer). To do that, add a `top: TiledCameraCfg = ...` field to your task's `SceneCfg`.

**If you change resolution or FPS**, also update:
- `so101_lab/data/lerobot_dataset.py` — video encoding parameters
- The dataset's `meta/info.json` keys match the camera names (`observation.images.top`, `observation.images.wrist`)
- Reward classifiers and VIP models trained on the old resolution won't transfer

---

## What to keep

These parts of the pipeline are task-agnostic and do not need changes:

- **All training scripts** (`train_act.py`, `train_diffusion.py`, etc.) — they read from a LeRobot dataset, regardless of task
- **Eval scripts** — take `--env <task_name>` as a flag
- **Data format** — LeRobot v3.0 parquet + H.264 is fixed; all tooling expects it
- **Policy inference wrappers** (`ACTInference`, `DiffusionInference`) — task-independent
- **RL infrastructure** (`isaac_lab_gym_env.py`, `domain_rand.py`) — task-independent

The only task-specific code is in `so101_lab/tasks/<name>/` and the constants at the top of `env_cfg.py`.

---

## Don't forget checklist

When creating a new task, these are the places where you need to be consistent:

- [ ] `env_cfg.py` `SUCCESS_THRESHOLD` and `ORIENTATION_THRESHOLD` match what you want
- [ ] `rl/env_cfg.py` `TerminationsCfg` uses the same thresholds (they import from `env_cfg.py` — verify the import)
- [ ] `rl/__init__.py` has a unique gym ID — duplicate IDs cause silent registration failures
- [ ] Task registered in `tasks/__init__.py` in **both** `get_task_registry()` and `_get_rl_task_registry()`
- [ ] If inheriting rewards from a parent task, add a re-export in `rl/mdp/rewards.py` (see [01_env_design.md](01_env_design.md#inherited-tasks-need-a-rewards-re-export))
- [ ] Camera noise parameters in `rl/mdp/events.py` match the new camera setup
- [ ] `configs/collect_rollouts.yaml` IK heights and thresholds updated if object dimensions changed
- [ ] Reward classifier retrained if camera view changed significantly
