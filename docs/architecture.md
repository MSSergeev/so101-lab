# Architecture

System design notes for SO-101 Lab.

---

## Virtual environments

Four separate Python environments:

| Env | Python | Contents | Used for |
|-----|--------|----------|----------|
| **isaaclab-env** | 3.11 | Isaac Sim + IsaacLab + project | teleop, eval, RL env loop |
| **lerobot-env** | 3.12 | LeRobot + project | BC/VLA training, IQL critics, offline sampler |
| **venvs/rerun** | any | rerun-sdk, pandas, pyarrow | episode visualization |
| **venvs/viewer** | any | opencv-python | live camera preview during recording |

### Why the split is mandatory

**Isaac Sim pins Python 3.11.** Its native C++ extensions (`_carb.cpython-311`, PhysX, RTX renderer) are compiled for 3.11 only. There is no 3.12 build. You cannot import `omni.isaac.core` or run any simulation outside this version.

**LeRobot 0.5.1 requires Python 3.12.** It uses PEP 695 syntax (`class Backtrackable[T]:`) that raises `SyntaxError` on 3.11. BC/VLA training, dataset tools, and the SmolVLA model all live here.

**Mixing them breaks both.** Installing LeRobot into isaaclab-env fails on syntax errors. Installing Isaac Sim extensions into lerobot-env fails on missing C++ libraries. A `sys.path` hack (importing lerobot from isaaclab-env) worked with older lerobot versions but breaks with 0.5.1.

### What runs where

| isaaclab-env (3.11) | lerobot-env (3.12) | Both (gRPC) |
|----------------------|--------------------|-------------|
| `record_episodes.py` | `lerobot train` (SmolVLA BC) | `train_flow_noise_ppo_grpc.py` |
| `eval_act_policy.py` | `train_act.py` | `train_sac_composite_grpc.py` |
| `eval_diffusion_policy.py` | `train_diffusion.py` | client (isaaclab) ↔ server (lerobot) |
| `test_env_spawn.py` | `train_iql_pretrain.py` | |
| domain randomization | `train_noise_sampler_offline.py` | |
| sim reward extraction | `train_vla_weighted_bc.py` | |
| | `train_reward_classifier_v2.py` | |

**Online RL (PPO, SAC)** needs both: the env loop runs in isaaclab-env, the model + RL update runs in lerobot-env. They communicate via gRPC over localhost. See [reference/train_flow_noise_ppo_grpc.md](reference/train_flow_noise_ppo_grpc.md) (PPO). SAC uses the same pattern: `train_sac_composite_grpc.py` (client) + `sac_server.py` (server).

**VLA eval** uses the same gRPC pattern: isaaclab-env runs the env, lerobot-env runs SmolVLA inference via `smolvla_server.py`.

### Tool envs

**venvs/rerun** and **venvs/viewer** are isolated because:
- rerun-sdk pulls heavy dependencies (arrow, protobuf versions) that conflict with both main envs
- opencv-python for the viewer is a single lightweight package that doesn't need torch or Isaac Sim
- Neither tool needs GPU compute — they're pure visualization

These are created locally in `venvs/` (git-ignored) and are optional.

### Known incompatibilities

**trackio** does not work in isaaclab-env (Python 3.11). It imports gradio, which imports typer, which uses `click.Choice[T]` — a Python 3.12+ syntax. Use `--tracker wandb` or `--tracker none` for scripts running in isaaclab-env. trackio works in lerobot-env.

**transformers ≥5.3** breaks custom HuggingFace models (e.g. `helper2424/resnet10` used by SACPolicy) that don't call `super().__init__()` in the expected order — missing `all_tied_weights_keys` attribute. Fix: call `so101_lab.utils.compat.patch_hf_custom_models()` before any SACPolicy creation. Already applied in `sac_server.py` and `train_iql_pretrain.py`.

### Configuration

Copy `.env.example` → `.env`, set `ISAACLAB_ENV` and `LEROBOT_ENV` paths. Activate with `act-isaac` / `act-lerobot` aliases — see [workflow/00_setup.md](workflow/00_setup.md). Local tool envs: see `venvs/README.md`.

---

## Env types and task structure

See [docs/reference/env_design.md](reference/env_design.md) for the full breakdown: DirectRLEnv vs ManagerBasedRLEnv, task file structure, scene components, obs/action spaces, RL env config, and how to add a new task.

### Code boundary

- **`so101_lab/rl/`** — Isaac Lab side only (gym wrapper, domain randomization, HIL). Depends on `ManagerBasedRLEnv`. Runs inside Isaac Sim.
- **`so101_lab/policies/rl/`** — policy/inference side (critics, flow noise head, noise prior, neural IK). Runs in lerobot-env or as inference components.

This split prevents Isaac Sim dependencies from leaking into policy code.

---

## Timing and frequencies

The recording loop uses a synchronous architecture — physics, rendering, and recording all run in the same thread.

### Current configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `sim.dt` | 1/60 s | PhysX timestep |
| `decimation` | 1 | Physics steps per policy step |
| `render_interval` | 1 | Render every physics step |
| `camera.update_period` | 1/30 s | Camera capture period |
| Policy / recording rate | 30 Hz | One `env.step()` per 1/30 s |

At 30 Hz, `env.step()` takes ~5 ms real time; the remaining ~28 ms is sleep. Simulation runs at ~5.5× real time when sleep is removed (headless RL training mode).

### Frequency hierarchy

```
Physics (60 Hz)
    └─ env.step() / recording (30 Hz)   ← decimation=1
           └─ camera capture (30 Hz)    ← update_period=1/30
```

**Rule:** `recording_hz == policy_hz`. Camera update period must not exceed render interval, otherwise duplicate frames appear in the dataset.

### decimation

`decimation` sets how many physics steps happen inside one `env.step()`:

```
decimation=1, dt=1/60 → policy at 60 Hz
decimation=2, dt=1/60 → policy at 30 Hz (action held for 2 physics steps)
decimation=4, dt=1/60 → policy at 15 Hz
```

The project uses `decimation=1` with the main loop throttled to 30 Hz via sleep.

### Recording semantics

Frames are recorded as `(obs_before_step, action)` — the observation seen by the policy before the action is applied. This matches standard MDP convention and is required for correct imitation learning.

---

## Data format

Recordings are stored in LeRobot v3.0 format (parquet + H.264 video):

```
data/recordings/<task>/
├── meta/
│   ├── info.json                              # features, fps, robot_type
│   ├── stats.json                             # per-feature statistics
│   ├── tasks.parquet                          # task description strings
│   └── episodes/chunk-000/file-000.parquet   # per-episode metadata
├── data/chunk-000/file-000.parquet            # observations, actions, rewards
└── videos/
    ├── observation.images.top/chunk-000/file-000.mp4
    └── observation.images.wrist/chunk-000/file-000.mp4
```

| Key | Shape | Type | Description |
|-----|-------|------|-------------|
| `observation.state` | (6,) | float32 | Joint positions, normalized [-100, 100] |
| `action` | (6,) | float32 | Target joint positions, same space |
| `observation.images.top` | — | video | Top-down camera, 640×480, H.264 |
| `observation.images.wrist` | — | video | Wrist camera, 640×480, H.264 |
| `next.reward` | scalar | float32 | Binary success (0/1) if `--reward-mode success` |

### Joint normalization

Isaac Lab works in radians. The normalized space [-100, 100] used by LeRobot is:

```
normalized = (joint_rad - offset) / scale × 100
```

Conversion is handled automatically in `RecordingManager`. Policy inference receives and returns normalized values. The real robot's motors also operate in normalized space, so no conversion is needed at deployment.

---

## Domain randomization

Applied at every env reset in RL envs (`ManagerBasedRLEnv`). Shared helper: `so101_lab/rl/domain_rand.py` → `apply_domain_rand_flags(env_cfg, args)`.

| Category | Range |
|----------|-------|
| **Light** | Intensity 0–200K lux, color temperature 3500–6500K, all 3 lamps |
| **Physics** | Cube/platform friction 0.3–1.5×, cube mass 0.8–1.5× |
| **Camera** | Brightness ±20, contrast 0.8–1.2, Gaussian noise std=3–12 |
| **Distractors** | 0–3 random objects on the table |

CLI flags to disable DR: `--no-domain-rand` (all), `--no-randomize-physics`, `--no-randomize-light`, `--no-camera-noise`, `--no-distractors`.

DR is disabled during BC eval to get stable numbers (`--no-domain-rand` in sweep scripts).

---

## Reward models

Two reward sources are available for offline RL:

**VIP (Visual Imitation through Pre-training)** — dense float reward from the negative distance in a ResNet50 feature space pre-trained on Ego4D video. Weights: `~/.vip/resnet50/` (~98 MB, downloaded on first use). Used as the reward signal in IQL (#8, best result 86–88%).

**Binary sim reward** — `next.reward = 1` when the task success condition is met (e.g., cube placed in slot). Recorded during teleoperation via `--reward-mode success`. Used as the reward signal in IQL ablations (#8b, #8c, both ≈ baseline).

For online RL (Flow-Noise PPO), VIP rewards are computed on-the-fly from rollout frames.
