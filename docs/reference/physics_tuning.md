# physics_tuning.md

PhysX solver, actuator, and friction settings for SO-101 Lab. Isaac Lab env.

---

## Overview

Three layers of physics configuration:
1. **Global** — `SimulationCfg.physx` in `template/env_cfg.py`: sets bounds for all objects
2. **Per-articulation** — `ArticulationRootPropertiesCfg` in `assets/robots/so101.py`: robot-wide
3. **Per-rigid-body** — `RigidBodyPropertiesCfg` in `tasks/*/env_cfg.py`: per-object

Per-object values are clamped to global bounds. SO-101 Lab uses defaults (1–255) so per-object
values are never clamped in practice.

**Important:** cannot set different iterations for different links within one articulation —
PhysX solves it as a single system. The robot articulation has uniform iterations across all links.

---

## PhysX Solver Iterations

Iterative constraint solver: more iterations = more accurate, slower.

- **position_iterations** — positional constraints (penetration, joint positions)
- **velocity_iterations** — velocity constraints (friction, damping)

**Key rule:** at contact between two objects, PhysX uses the **maximum** iterations of the pair:
```
gripper ↔ cube: position_iterations = max(robot=8, cube=32) = 32
robot ↔ table:  position_iterations = max(robot=8, table=4)  = 8
```
Iterations apply per-contact pair, not to the whole articulation — so high values only where needed.

### SO-101 Lab settings

| Object | position | velocity | Where |
|--------|---------|----------|-------|
| SO-101 robot | 8 | 4 | `assets/robots/so101.py` |
| Cube | 32 | 4 | `tasks/figure_shape_placement/env_cfg.py` |
| Platform (static) | 4 | 0 | `tasks/figure_shape_placement/env_cfg.py` |

```python
# so101.py — robot
articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    solver_position_iteration_count=8,
    solver_velocity_iteration_count=4,
    fix_root_link=True,
    enabled_self_collisions=True,
)

# env_cfg.py — cube
rigid_props=sim_utils.RigidBodyPropertiesCfg(
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=4,
)
```

### When to change

- Robot vibrates/unstable → increase position iterations
- Objects penetrate each other → increase position iterations
- Gripper slip during grasp → increase velocity iterations for cube
- Slow simulation → decrease iterations

---

## solve_articulation_contact_last

```python
# template/env_cfg.py
sim: SimulationCfg = SimulationCfg(
    physx=PhysxCfg(solve_articulation_contact_last=True)
)
```

Solves articulation contacts last instead of first → more stable gripper behavior (object
doesn't pop out of grasp). Available in Isaac Sim 5.1+ / Isaac Lab 2.3+.

**Side effect:** contact force readings from `ContactSensor` may be less accurate.

---

## Actuators (PD controller)

`ImplicitActuatorCfg` implements a PD controller per joint:

```
τ_desired = stiffness × (target_pos - current_pos) - damping × current_vel
τ = clamp(τ_desired, -effort_limit, +effort_limit)
```

### SO-101 settings (arm + gripper, same params)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `stiffness` | 10.8 Nm/rad | Spring stiffness. Higher → faster, can vibrate |
| `damping` | 0.9 Nm·s/rad | Damping. Higher → slower, smoother |
| `effort_limit_sim` | 5 Nm | Max torque. Too high → crushes objects |
| `velocity_limit_sim` | 10 rad/s | Max joint speed (~0.3s for 60° move) |

```python
# so101.py
actuators={
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        effort_limit_sim=5, velocity_limit_sim=10, stiffness=10.8, damping=0.9,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["gripper"],
        effort_limit_sim=5, velocity_limit_sim=10, stiffness=10.8, damping=0.9,
    ),
}
```

### Common problems

| Problem | Cause | Fix |
|---------|-------|-----|
| Robot vibrates | High stiffness, low damping | Reduce stiffness or increase damping |
| Gripper crushes objects | High effort_limit | Reduce to 3–5 Nm |
| Slow movement | Low velocity_limit | Increase to 10–15 rad/s |
| Doesn't reach target | Low stiffness or effort | Increase stiffness |

---

## Friction

### In Python config

```python
physics_material=sim_utils.RigidBodyMaterialCfg(
    static_friction=0.8,
    dynamic_friction=0.8,
    friction_combine_mode="max",  # effective = max(object, gripper)
)
```

Combine modes at contact: `"average"`, `"min"`, `"max"`, `"multiply"`.
Use `"max"` for grasped objects to ensure high friction even if gripper material is low.

### In USD file (cube and platform)

`UsdFileCfg` does not support overriding `physics_material` for existing USD assets.
Friction must be set directly in the USD:

```python
from pxr import Usd, Sdf

stage = Usd.Stage.Open("path/to/object.usd")
mat_prim = stage.GetPrimAtPath("/object/_materials/PhysicsMaterial")
mat_prim.GetAttribute("physics:staticFriction").Set(0.8)
mat_prim.GetAttribute("physics:dynamicFriction").Set(0.8)
mat_prim.GetAttribute("physxMaterial:frictionCombineMode").Set("max")
stage.GetRootLayer().Save()
```

**SO-101 Lab cube:** `static=0.8, dynamic=0.8, combine=max` set in USD file.

---

## ContactSensor (for contact-based rewards)

```python
from isaaclab.sensors import ContactSensorCfg

scene["gripper_contact"] = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
    update_period=0.0,
    history_length=1,
    filter_prim_paths_expr=[
        "{ENV_REGEX_NS}/Table",
        "{ENV_REGEX_NS}/Platform",
    ],
)
```

In reward function:
```python
F = scene["gripper_contact"].data.net_forces_w   # (num_envs, 3)
contact_mag = torch.linalg.norm(F, dim=-1)        # (num_envs,)
touch = contact_mag > 0.1                          # tune threshold
reward -= w_contact * touch.float()
```

**Notes:**
- Sensor returns net force on the attached body — doesn't identify which surface was touched
- For different penalties per surface: use separate sensors with different `filter_prim_paths_expr`
- SDF mesh colliders work with ContactSensor but are expensive (prefer convex hulls for RL)
- `solve_articulation_contact_last=True` may reduce ContactSensor accuracy (forces may be stale)

**Checklist for contact rewards to work:**
- Gripper link has a collider (not disabled)
- Table and platform have colliders
- `filter_prim_paths_expr` matches actual prim paths
- Threshold tuned above numerical noise (~0.01–0.1 N)

---

## Debugging

**Check current values in GUI:** select object → Property Panel → Physics → Solver Iteration Count.

**Enable contact visualization:**
```python
env_cfg.sim.physx.enable_debug_visualization = True
```

**FPS increases during gripper contact** (observed behavior): stable contact converges faster
(PhysX solver caching), and static objects enter sleep state. Normal — not a bug.
