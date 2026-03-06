# Project Structure

Where things live, why they're separated, and how to adapt the project.

Full script and module index: [reference/index.md](index.md)

---

## Top-level layout

```
so101-lab-clean/
├── so101_lab/        # Library — importable modules
├── scripts/          # Executable scripts, grouped by stage
├── configs/          # Hydra YAML configs (training hyperparams, IK params)
├── tests/            # Unit tests (no Isaac Sim required)
├── docs/             # Documentation
└── data/             # Datasets (not in git)
```

---

## Library: `so101_lab/`

### The rl/ boundary

The most important thing to understand:

- **`so101_lab/rl/`** — runs inside Isaac Sim (Isaac Lab env). Uses `omni.usd`, `pxr`, Isaac Lab APIs. **Cannot import lerobot.**
- **`so101_lab/policies/rl/`** — runs in lerobot-env (Python 3.12). Uses PyTorch, lerobot internals. **Cannot import Isaac Lab.**

These two subtrees have the same name fragment by coincidence. They do not share imports and must not be mixed.

All other modules (`tasks/`, `devices/`, `data/`, `assets/`) are Isaac Lab only — they require the `il` venv.

### Module purposes

**`assets/`** — robot and scene USD/URDF configs for Isaac Lab.
- `robots/so101.py`: `SO101_CFG` (ArticulationCfg) — change this if you swap the robot
- `scenes/workbench.py`: workbench and ground plane — change this if you change the scene geometry

**`tasks/`** — Isaac Lab environments (DirectRLEnv for BC/teleop, ManagerBasedRLEnv for RL).

Each task is a self-contained folder:
```
tasks/<task_name>/
├── env.py          # DirectRLEnv subclass
├── env_cfg.py      # Scene setup, camera positions, spawn params
└── rl/
    ├── env_cfg.py  # ManagerBasedRLEnv config
    ├── __init__.py # Gym ID registration
    └── mdp/
        ├── rewards.py
        ├── observations.py
        ├── terminations.py
        └── events.py
```

Tasks are registered in `tasks/__init__.py`. Scripts never import tasks directly — they use `get_task("name")` or `get_rl_task("name")`, so you can add a task without changing any script.

**`devices/`** — teleoperation input: keyboard, gamepad, SO-101 leader arm. Used by `teleop_agent.py` and `record_episodes.py`. Not needed for training or eval.

**`data/`** — LeRobot v3.0 dataset writing (`LeRobotDatasetWriter`) and recording orchestration (`RecordingManager`). Also handles rad↔normalized conversion.

**`policies/`** — inference wrappers for trained checkpoints. The wrappers (`ACTInference`, `DiffusionInference`) handle checkpoint loading and normalization — you don't call lerobot directly from eval scripts.

**`rewards/`** — reward models: binary classifier (`classifier.py`) and VIP goal-conditioned reward (`vip_reward.py`). Used by RL training scripts, not by BC training.

**`rl/`** — Isaac Lab RL infrastructure: Gymnasium wrapper (`isaac_lab_gym_env.py`) and domain randomization helper (`domain_rand.py`). Used by all RL training and eval scripts that run in sim.

**`utils/`** — shared utilities used across scripts: checkpoint resolution, scene state extraction, experiment tracking, spawn diversity, camera shared memory.

---

## Scripts: `scripts/`

Grouped by workflow stage:

| Folder | Stage | Venv |
|--------|-------|------|
| `teleop/` | Record demonstrations | il |
| `eval/` | Evaluate policies in sim | il (+ lr for smolvla_server) |
| `train/` | Train policies and reward models | il or lr (see index) |
| `tools/` | Dataset utilities, diagnostics | lr or il |
| `test/` | Functional sim tests | il |
| `isaac_sim_standalone/` | Pure Isaac Sim (no Isaac Lab) | il |

Key cross-venv pattern: `eval_vla_policy.py` (il) launches `smolvla_server.py` (lr) as a subprocess and communicates via gRPC. This is the solution to the Python 3.11/3.12 boundary.

---

## Configs: `configs/`

```
configs/
├── collect_rollouts.yaml        # IK state machine: waypoints, thresholds, speeds
└── policy/
    ├── act/
    │   ├── baseline.yaml        # chunk_size=25, kl_weight=10, steps=60k
    │   ├── high_kl.yaml         # kl_weight variant
    │   └── low_kl.yaml          # kl_weight variant
    └── diffusion/
        └── baseline.yaml        # diffusion steps, n_obs_steps, etc.
```

Policy configs are passed with `--config configs/policy/act/baseline.yaml`. They override CLI defaults. Parameters not in the config file fall back to script defaults.

`collect_rollouts.yaml` controls the IK state machine geometry (heights, thresholds, gripper angles). These are task-specific — edit them when changing the scene or object dimensions.

There is no global config file. Task-specific parameters (spawn zone, success thresholds, reward scales) live in `tasks/<name>/env_cfg.py` and `tasks/<name>/rl/env_cfg.py`, not in YAML.

---

## Adapting to a different robot or task

### Changing the robot

1. Replace or edit `so101_lab/assets/robots/so101.py` with your robot's URDF/USD
2. Update joint count and names in `tasks/template/env_cfg.py` (observation/action space is `(6,)` for SO-101)
3. Update normalization range in `data/converters.py` if your joint limits differ
4. Re-calibrate if using a physical leader arm

### Adding a new task

See [workflow/01_env_design.md](../workflow/01_env_design.md) for the full checklist. Short version:

1. Copy `tasks/figure_shape_placement_easy/` → `tasks/<new_task>/`
2. Edit `env_cfg.py`: spawn zone bounds, platform position, success thresholds
3. Edit `rl/env_cfg.py`: same params for the RL variant
4. Register in `tasks/__init__.py` (both `get_task_registry` and `_get_rl_task_registry`)
5. Register gym ID in `tasks/<new_task>/rl/__init__.py`
6. Verify: `python scripts/eval/test_env_spawn.py --gui --env <new_task>`

Don't forget:
- Success thresholds must match between BC env and RL env — they're defined separately in each `env_cfg.py`
- Domain randomization ranges are in the RL env_cfg; DR is not applied during BC recording
- If you change camera positions, retrain the reward classifier (it's camera-view-dependent)

### Changing camera setup

Camera intrinsics and poses are in `tasks/template/env_cfg.py` (top and wrist cameras). Both BC and RL envs inherit from template. The image resolution (`480×640`) is baked into the observation space and the dataset format — changing it requires updating both.

### Using a different policy architecture

The eval scripts use `ACTInference` / `DiffusionInference` wrappers. To add a new policy type:
1. Add a wrapper in `so101_lab/policies/<your_policy>/`
2. Implement `reset()` and `select_action(obs)` matching the existing interface
3. Write an eval script following the pattern of `eval_act_policy.py`

The training side (lerobot-env) is independent — use whatever lerobot supports.
