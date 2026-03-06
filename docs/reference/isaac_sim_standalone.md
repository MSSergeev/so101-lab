# isaac_sim_standalone.md

Test SO-101 joint movements using pure Isaac Sim API — no Isaac Lab required.

Scripts: `scripts/isaac_sim_standalone/`

---

Runs in **Isaac Lab env** (`act-isaac`). See [setup](../workflow/00_setup.md).

## Overview

Alternative to `scripts/test_robot.py` (which uses Isaac Lab). Useful when Isaac Lab is not
installed or for understanding the low-level Isaac Sim API directly.

**Key difference from `scripts/test_robot.py`:**

| | `isaac_sim_standalone/test_robot.py` | `scripts/test_robot.py` |
|--|--------------------------------------|-------------------------|
| API | Pure Isaac Sim (`isaacsim.core`) | Isaac Lab (`TemplateEnv`) |
| Data types | `numpy` | `torch.Tensor` |
| Scene setup | Manual (`add_reference_to_stage`) | Automatic via `TemplateEnvCfg` |
| Robot position | **Origin (0,0,0)** — known limitation | Correct position on table |
| Use case | Low-level debugging, Isaac Sim learning | Production testing |

**Known limitation:** robot spawns at world origin (0,0,0), not on the table. This is because
`fix_root_link=True` in `so101.py` prevents `set_world_poses()` from working — it causes
"Invalid PhysX transform" errors. For joint range testing this is acceptable; for visual
validation use `scripts/test_robot.py`.

---

## File structure

```
scripts/isaac_sim_standalone/
├── test_robot.py          # Entry point: scene setup + mode dispatch
├── configs.py             # Joint limits, PD gains, preset positions
└── modes/
    ├── __init__.py
    ├── interactive.py     # InteractiveController (keyboard, carb.input)
    └── auto_tests.py      # test_individual_joints, test_joint_limits, test_preset_positions
```

---

## Usage

```bash
# Requires Isaac Sim to be installed and on PATH (not Isaac Lab venv)
# Activate Isaac Sim python environment, e.g.:
act-isaac  # activate Isaac Lab env

cd /path/to/so101-lab

python scripts/isaac_sim_standalone/test_robot.py --mode interactive  # GUI, keyboard control
python scripts/isaac_sim_standalone/test_robot.py --mode preset        # cycle through presets
python scripts/isaac_sim_standalone/test_robot.py --mode test --headless  # automated tests
python scripts/isaac_sim_standalone/test_robot.py --mode all           # preset → test → interactive
```

`--mode interactive` requires GUI (incompatible with `--headless`).

---

## Modes

### interactive

Manual keyboard control. Keys:
- `1`–`6` / `Numpad 1`–`6` — select joint (shoulder_pan … gripper)
- `↑` / `W` — increase angle (+0.02 rad ≈ 1°), hold for continuous
- `↓` / `S` — decrease angle
- `H` — HOME, `P` — PICK_READY, `L` — PLACE_READY, `M` — JOINT_MID, `R` — REST
- `G` — toggle gripper (open/close)
- `ESC` / `Q` — quit

Status printed every 100 steps: selected joint, target, actual, error.

### preset

Cycles through all 7 preset positions (250 steps each, 1s pause between):
HOME → PICK_READY → PLACE_READY → GRIPPER_OPEN → GRIPPER_CLOSE → JOINT_MID → REST.
Prints position error at each step.

### test

Three automated tests:
1. **Individual joints** — moves each joint to mid-range, checks error < 0.05 rad (✓/✗)
2. **Joint limits** — tests upper and lower limits with 5% safety margin
3. **Preset positions** — moves to each preset, checks error < 0.05 rad

### all

Runs: preset → test → interactive (if not headless).

---

## Configuration (`configs.py`)

### Joint limits

Read from `so101.usd` (USD stores in degrees, converted to radians):

| Joint | Lower (rad) | Upper (rad) |
|-------|------------|------------|
| shoulder_pan | -1.920 (-110°) | 1.920 (110°) |
| shoulder_lift | -1.745 (-100°) | 1.745 (100°) |
| elbow_flex | -1.690 (-96.8°) | 1.536 (88.0°) |
| wrist_flex | -1.658 (-95°) | 1.658 (95°) |
| wrist_roll | -2.740 (-157°) | 2.827 (162°) |
| gripper | -0.175 (-10°) | 1.745 (100°) |

Note: limits here differ slightly from `scripts/test_robot.py` which reads them from the
running sim at runtime. These are hardcoded from USD inspection.

### PD gains

`stiffness=17.8, damping=0.60` — set via `UsdPhysics.DriveAPI` directly on joint prims
before articulation is created.

### Physics

`PHYSICS_DT = 1/360` s (360 Hz) for stable simulation.

---

## How scene setup works

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # MUST be first

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage

world = World(stage_units_in_meters=1.0, physics_dt=1/360)
world.scene.add_default_ground_plane()

add_reference_to_stage(usd_path="assets/scenes/workbench_clean.usd", prim_path="/World/Scene")
add_reference_to_stage(usd_path="assets/robots/so101/usd/so101_w_cam.usd", prim_path="/World/Robot")

# Configure PD BEFORE creating Articulation
stage = omni.usd.get_context().get_stage()
drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath("/World/Robot/joints/shoulder_pan"), "angular")
drive.GetStiffnessAttr().Set(17.8)
drive.GetDampingAttr().Set(0.60)

robot = Articulation(prim_paths_expr="/World/Robot", name="so101")
world.scene.add(robot)
world.reset()  # initializes physics

# Control loop
while simulation_app.is_running():
    world.step(render=True)
    if world.is_playing():
        robot.set_joint_positions(target_positions)  # numpy array (6,)
```

**Critical:** `SimulationApp` must be created before any `omni.*` or `isaacsim.*` imports.
