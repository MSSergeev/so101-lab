# Environment Design

How task environments are structured in SO-101 Lab. **Isaac Lab env.**

For spawn zones, sim rewards, and domain randomization in practice, see [workflow/01_env_design.md](../workflow/01_env_design.md).

---

## DirectRLEnv vs ManagerBasedRLEnv

Two Isaac Lab base classes, used for different purposes:

| | `DirectRLEnv` | `ManagerBasedRLEnv` |
|-|---------------|---------------------|
| **Used for** | Teleoperation, BC eval, recording | RL training (SAC, PPO) |
| **Registered as** | `get_task(name)` | `get_rl_task(name)` (gym ID) |
| **Location** | `tasks/<name>/env.py` | `tasks/<name>/rl/` |
| **Reset/reward** | Manual methods in `env.py` | Declarative via `rl/mdp/` manager components |
| **Multi-env** | No | Yes (vectorized) |

`DirectRLEnv` is simpler — used for anything that doesn't need the full RL manager stack.
`ManagerBasedRLEnv` supports vectorized envs, DR at reset, and the reward/termination manager system.

### Task registry

All tasks registered in `so101_lab/tasks/__init__.py`. Scripts accept `--env <name>` and look up at runtime:

```python
from so101_lab.tasks import get_task, get_rl_task

EnvClass, EnvCfgClass = get_task("figure_shape_placement_easy")   # DirectRLEnv
RLEnvCfgClass = get_rl_task("figure_shape_placement_easy")         # ManagerBasedRLEnv
```

Adding a new task does not require changing any script — only the registry.

---

## Task file structure

Each task lives in `so101_lab/tasks/<name>/`:

```
tasks/<name>/
├── __init__.py          # Lazy imports (omni.physics deferred until runtime)
├── env.py               # DirectRLEnv — teleop, BC eval
├── env_cfg.py           # Scene constants, spawn bounds, camera config
└── rl/
    ├── __init__.py      # gym.register("SO101-<Name>-RL-v0")
    ├── env_cfg.py       # ManagerBasedRLEnvCfg
    └── mdp/
        ├── __init__.py     # Re-export isaaclab.envs.mdp + local functions
        ├── events.py       # Reset functions: robot, platform, cube, lights
        ├── observations.py # Extra obs: cube_pos_w, slot_pos_w, platform_yaw
        ├── rewards.py      # Sim reward components
        └── terminations.py # cube_out_of_bounds, time_out
```

Geometry constants (spawn bounds, success thresholds, table height) live in `env_cfg.py` and are imported by both `env.py` and `rl/env_cfg.py` — single source of truth for IL and RL.

---

## Scene components

Both env types share the same scene: robot + workbench scene + cameras.

**Template scene** (`so101_lab/tasks/template/env_cfg.py`, `TemplateSceneCfg`):

| Component | Type | Description |
|-----------|------|-------------|
| `scene` | `AssetBaseCfg` | `workbench_v1.usd` — table, lighting, no manipulation objects |
| `robot` | `ArticulationCfg` | SO-101 (6-DOF arm + gripper) |
| `ee_frame` | `FrameTransformerCfg` | End-effector position tracking |
| `top` | `TiledCameraCfg` | Fixed top-down camera, 640×480 RGB |
| `wrist` | `TiledCameraCfg` | Camera on gripper link, 640×480 RGB |

Task-specific scenes inherit `TemplateSceneCfg` and add manipulation objects (`RigidObjectCfg`).

All prim paths use `{ENV_REGEX_NS}` for multi-env support.

---

## Observation and action spaces

### DirectRLEnv (IL / teleop)

```python
action_space = 6  # absolute joint position targets, normalized [-100, 100]

observations = {
    "joint_pos":        (N, 6),          # current joint positions
    "actions":          (N, 6),          # previous actions
    "joint_pos_target": (N, 6),          # current targets
    "top":              (N, 480, 640, 3), # top camera RGB uint8
    "wrist":            (N, 480, 640, 3), # wrist camera RGB uint8
}
```

### ManagerBasedRLEnv (RL training)

```python
# Policy observation group (concatenate_terms=False → dict, not concat)
observations["policy"] = {
    "joint_pos":    (N, 6),           # float32
    "joint_vel":    (N, 6),           # float32
    "images_top":   (N, 480, 640, 3), # uint8, normalize=False
    "images_wrist": (N, 480, 640, 3), # uint8
}

# Actions: JointPositionActionCfg — 6 joints
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Extra observations available via mdp (not in policy group, used by reward/debug):
`cube_pos_w` (N,3), `cube_quat_w` (N,4), `slot_pos_w` (N,3), `platform_yaw` (N,1).

---

## RL env configuration

```python
decimation = 4          # policy at 30 Hz (120 Hz physics / 4)
episode_length_s = 15.0
sim.dt = 1/120          # 120 Hz physics
sim.render_interval = 4 # render every policy step
```

Gym ID: `SO101-FigureShapePlacement-RL-v0` (registered in `rl/__init__.py`).

### Using the RL env

```python
import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
import so101_lab.tasks.figure_shape_placement.rl  # registers gym ID

env_cfg = parse_env_cfg("SO101-FigureShapePlacement-RL-v0", device="cuda:0", num_envs=4)
env = gym.make("SO101-FigureShapePlacement-RL-v0", cfg=env_cfg).unwrapped

obs, info = env.reset()
# obs["policy"] is a dict: joint_pos, joint_vel, images_top, images_wrist
```

`parse_env_cfg` is needed because `gym.register` stores the config class as a string; `ManagerBasedRLEnv.__init__` expects an instantiated config object.

---

## Reset events (RL)

Defined in `mdp/events.py`, run at every reset:

1. `reset_scene_to_default` — built-in Isaac Lab reset (robot joints + all rigid bodies)
2. `reset_platform` — fixed XY position, discrete random yaw from {80°, 85°, 90°, 95°, 100°}
3. `reset_cube_polar` — rejection sampling in polar coordinates, collision check against platform
4. `randomize_light` — random intensity (0–200k lux), color temperature (3500–6500K) via USD API

---

## figure_shape_placement_easy task

Simplified variant for the easy benchmark: small spawn zone (4×4 cm), fixed platform yaw (90°).

**Success criteria:**
- Cube center within ±2 mm of slot center (XY)
- Cube orientation aligned with platform (±5°, accounting for 90° symmetry)

**Success check** (not automatic termination — called by recording/eval scripts):
```python
success = env.is_success()  # → (N,) bool tensor
```

**Task description** (stored per-episode in `tasks.parquet`):
```python
env.get_task_description()  # → "Place the cube into the matching slot on the platform"
```

---

## Adding a new task (checklist)

1. Copy `tasks/figure_shape_placement/` → `tasks/<new_task>/`
2. Rename all classes: `FigureShapePlacement*` → `NewTask*`
3. Update internal imports to `so101_lab.tasks.<new_task>.*`
4. Edit constants in `env_cfg.py` (spawn bounds, success thresholds, etc.)
5. `rl/__init__.py`: new gym ID `"SO101-NewTask-RL-v0"`
6. Register in `tasks/__init__.py`:
   - `get_task_registry()` — BC env
   - `_get_rl_task_registry()` — RL env
7. Verify: `python scripts/eval/test_env_spawn.py --gui --env <new_task>`

## Lazy imports

`tasks/<name>/__init__.py` uses `__getattr__` for lazy loading because `env.py` imports `omni.physics` which requires the Isaac Sim runtime. Direct imports at module load time would break any non-Isaac-Sim process that imports `so101_lab`.

```python
def __getattr__(name):
    if name in ("FigureShapePlacementEnv", "FigureShapePlacementEnvCfg"):
        from .env import FigureShapePlacementEnv
        from .env_cfg import FigureShapePlacementEnvCfg
        ...
```
