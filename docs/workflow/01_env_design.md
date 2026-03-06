# Env Design

How to create a task environment: spawn zones, sim rewards, domain randomization.

The main task is `figure_shape_placement` — polar spawn zone, random cube yaw, 5 platform orientations.
`figure_shape_placement_easy` is a simplified variant (small fixed spawn zone) useful for pipeline validation.

---

## 1. File structure

Each task lives in `so101_lab/tasks/<name>/`:

```
tasks/<name>/
├── __init__.py            # Lazy imports
├── env.py                 # DirectRLEnv (teleop, BC eval)
├── env_cfg.py             # Scene + spawn constants
└── rl/
    ├── __init__.py        # gym.register("SO101-<Name>-RL-v0")
    ├── env_cfg.py         # ManagerBasedRLEnvCfg (RL training, VLA eval)
    └── mdp/
        ├── __init__.py    # re-exports isaaclab.envs.mdp + local
        ├── events.py      # reset functions (spawn logic)
        ├── observations.py
        ├── rewards.py
        └── terminations.py
```

The easiest way to create a new task is to copy `figure_shape_placement/` and modify constants.

**Checklist:**
1. Copy `tasks/figure_shape_placement/` → `tasks/<new_task>/`
2. Rename classes: `FigureShapePlacement*` → `NewTask*`
3. Update internal imports to `so101_lab.tasks.<new_task>.*`
4. Change constants in `env_cfg.py` (spawn params, thresholds, etc.)
5. `rl/__init__.py`: set a new gym ID `"SO101-NewTask-RL-v0"`
6. Register in `so101_lab/tasks/__init__.py`:
   - `get_task_registry()` — for teleop / BC eval
   - `_get_rl_task_registry()` — for RL training / VLA eval
7. Verify: `python scripts/eval/test_env_spawn.py --gui --env <new_task>`

---

## 2. Spawn zones

Spawn logic lives in `rl/mdp/events.py` and is called at each episode reset.

### Example: easy task vs original

| Parameter | `figure_shape_placement` | `figure_shape_placement_easy` |
|-----------|--------------------------|-------------------------------|
| Cube spawn | Polar coordinates, quarter-circle arc | Rectangle 4×4 cm |
| Cube zone | X∈[0.13, 0.45], Y∈[-0.05, 0.28] | X∈[0.314, 0.354], Y∈[0.058, 0.098] |
| Cube yaw | Uniform [−π, π] | Discrete: 0°, 10°, …, 80° |
| Platform pos | (0.154, 0.038) | (0.154, 0.048) — 1 cm closer |
| Platform yaw | 5 values: 80°–100° | Fixed: 90° |

The easy variant narrows the spawn to a small rectangle in front of the robot. This reduces task difficulty and is useful for validating the training pipeline before moving to a harder distribution.

### Verifying spawn visually

```bash
python scripts/eval/test_env_spawn.py --gui --env figure_shape_placement --resets 20
python scripts/eval/test_env_spawn.py --gui --env figure_shape_placement_easy --resets 20
```

The script runs N resets, prints object positions, and holds the GUI open. Check:
- Cube appears in the expected zone
- Platform is at the correct position and orientation
- Cube and platform do not intersect

### Spawn diversity during recording

Use `--diversity-keys` and `--diversity-ratio` to ensure even coverage of the spawn zone. The keys are the state variable names printed by `test_env_spawn.py` (e.g. `cube_x`, `cube_y`):

```bash
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement \
    --diversity-keys cube_x,cube_y \
    --diversity-ratio 1.2 \
    ...
```

`--diversity-ratio 1.2` means underrepresented spawn positions are 1.2× more likely to be selected. The tracking is done internally — no manual setup needed.

---

## 3. Sim rewards

Sim rewards are computed inside the simulator and can be saved alongside the dataset for use in offline RL.

### Reward functions

Reward functions live in `rl/mdp/rewards.py` and follow Isaac Lab's `RewardTermCfg` pattern. Each function receives the `ManagerBasedRLEnv` and returns a tensor of shape `(N,)` — one value per environment.

Available rewards in `figure_shape_placement`:

| Function | Description |
|----------|-------------|
| `distance_cube_to_slot` | Negative XY distance from cube to slot |
| `distance_gripper_to_cube` | Negative 3D distance from gripper to cube |
| `MilestonePicked` | One-time bonus when cube is lifted (stateful) |
| `MilestonePlaced` | One-time bonus when cube is placed in slot (stateful) |
| `drop_penalty` | Penalty if cube falls off the table |
| `jerky_motion_penalty` | Penalty for excess joint velocity |
| `action_smoothness_penalty` | Penalty for large action changes |
| `table_contact_penalty` | Penalty for gripper contact with table |
| `time_penalty` | Constant penalty per step |

### Recording sim rewards

Two reward modes are available via `--reward-mode`:

- `success` — `next.reward` in parquet is 0/1 based on episode outcome.
- `sim` — `next.reward` is the sum of all sim reward terms.
- `sim+success` — `next.reward` is sim reward sum + success bonus.

`sim_rewards.pt` (per-frame shaped rewards) is always saved when `rl/mdp/rewards.py` is available — regardless of `--reward-mode`. Required for offline RL (IQL critics).

```bash
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement \
    --reward-mode sim+success \
    ...
```

`--reward-mode sim` automatically discovers reward functions from `rl/mdp/rewards.py`. All public functions (no `_` prefix) are included.

### Inherited tasks need a rewards re-export

If your task inherits from another, add a `rewards.py` that re-exports the parent rewards — otherwise `discover_reward_functions()` won't find them:

```python
# so101_lab/tasks/<new_task>/rl/mdp/rewards.py
from so101_lab.tasks.<base_task>.rl.mdp.rewards import *  # noqa: F401,F403
```

---

## 4. Domain randomization

RL environments support domain randomization at each episode reset. All DR is off by default during teleop; enabled automatically during RL training.

### What gets randomized

| Component | Range |
|-----------|-------|
| Light intensity | 0–200K |
| Light color temperature | 3500–6500K (all 3 lamps) |
| Cube/platform friction | 0.3–1.5 |
| Cube mass | 0.8–1.5× nominal |
| Camera brightness | ±20 |
| Camera contrast | 0.8–1.2 |
| Camera Gaussian noise | std = 3–12 |
| Distractors | 0–3 random objects on table |

### CLI flags

All RL scripts accept DR flags via `apply_domain_rand_flags()` from `so101_lab/rl/domain_rand.py`:

```bash
--no-domain-rand          # disable all DR
--no-randomize-physics    # keep friction and mass fixed
--no-randomize-light      # keep lighting fixed
--no-camera-noise         # clean camera images
--no-distractors          # no distractor objects
```

Use `--no-domain-rand` during debugging or when comparing methods on identical scenes.
