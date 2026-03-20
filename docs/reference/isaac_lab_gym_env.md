# IsaacLabGymEnv

Gymnasium wrapper around `ManagerBasedRLEnv` for LeRobot SAC/PPO compatibility. Isaac Lab venv.

---

## Overview

`ManagerBasedRLEnv` (Isaac Lab) and LeRobot SAC have incompatible interfaces:

| Aspect | ManagerBasedRLEnv | LeRobot SAC |
|--------|-------------------|-------------|
| Data format | torch tensors (GPU) | numpy arrays (CPU) |
| Batch dim | Always present `(1, D)` | None `(D,)` |
| Joint space | Radians | Normalized `[-100, 100]` |
| Obs keys | `joint_pos`, `images_top` | `observation.state`, `observation.images.top` |
| Reward | Internal (RewardManager terms) | External (reward model) |

`IsaacLabGymEnv` is a thin adapter that handles all conversions.

---

## Data Flow

```
LeRobot SAC / PPO
  action: numpy (6,), normalized [-100, 100]
  obs: {"observation.state": (6,), "observation.images.top": (480, 640, 3), ...}
  reward: float (from external reward model, not from env)
        │                          ▲
        ▼                          │
  IsaacLabGymEnv
    step(): motor_normalized_to_joint_rad → unsqueeze(0) → torch tensor
    _convert_obs(): squeeze(0) → cpu().numpy() → joint_rad_to_motor_normalized
        │                          ▲
        ▼                          │
  ManagerBasedRLEnv (Isaac Lab)
    action: torch (1, 6), radians
    obs["policy"]: {"joint_pos": (1,6), "images_top": (1,480,640,3), ...}
    PhysX at 120 Hz, cameras top+wrist RGB
```

---

## API

### Constructor

```python
IsaacLabGymEnv(env_cfg, randomize_light: bool = True)
```

- `env_cfg` — an already-instantiated `ManagerBasedRLEnvCfg` object (use `parse_env_cfg()` from isaaclab to get it from the gym registry)
- `randomize_light=False` — disables light randomization event (useful when training a classifier on fixed-lighting data)

**AppLauncher must be initialized before creating this env.**

### Observation space

```python
gymnasium.spaces.Dict({
    "observation.state":        Box(shape=(6,),          low=-100, high=100, dtype=float32),
    "observation.images.top":   Box(shape=(480, 640, 3), low=0,    high=255, dtype=uint8),
    "observation.images.wrist": Box(shape=(480, 640, 3), low=0,    high=255, dtype=uint8),
})
```

### Action space

```python
Box(shape=(6,), low=-100.0, high=100.0, dtype=float32)  # normalized motors
```

### Methods

| Method | Returns | Notes |
|--------|---------|-------|
| `reset(seed, options)` | `(obs, info)` | `info["ground_truth"]` contains sim state |
| `step(action)` | `(obs, reward, terminated, truncated, info)` | reward comes from sim RewardManager |
| `close()` | — | Closes Isaac Lab env |
| `render()` | — | Delegates to inner env |
| `get_reward_details()` | `dict[str, float]` | Per-term weighted rewards + total |
| `step_dt` (property) | `float` | Env step duration in seconds |

### Reward behavior

`step()` returns the sim reward from `ManagerBasedRLEnv` (sum of all RewardManager terms). In SAC/PPO training loops, this reward is typically **replaced** by an external reward model (VIP, classifier). Use `get_reward_details()` for debugging sim reward breakdown.

### `info["ground_truth"]`

Populated on every `reset()` and `step()`. Contains raw sim state for reward computation:

```python
{
    "cube_pos":    np.ndarray (3,),  # cube position relative to env origin
    "cube_quat":   np.ndarray (4,),  # cube orientation (w, x, y, z)
    "slot_pos":    np.ndarray (3,),  # slot position (computed from platform pose + yaw)
    "gripper_pos": np.ndarray (3,),  # end-effector position
    "joint_vel":   np.ndarray (6,),  # joint velocities rad/s
}
```

Returns `{}` for tasks that don't have `cube`, `platform`, or `ee_frame` scene entities.

---

## Internals

### Observation conversion (`_convert_obs`)

Isaac Lab obs keys → LeRobot keys:

| Isaac Lab key | LeRobot key | Transform |
|---------------|-------------|-----------|
| `joint_pos` `(1, 6)` radians | `observation.state` `(6,)` | squeeze + `joint_rad_to_motor_normalized()` |
| `images_top` `(1, 480, 640, 3)` uint8 | `observation.images.top` `(480, 640, 3)` | squeeze + `.astype(uint8)` |
| `images_wrist` `(1, 480, 640, 3)` uint8 | `observation.images.wrist` `(480, 640, 3)` | squeeze + `.astype(uint8)` |
| `joint_vel` | — | Not exposed (not used by LeRobot format) |

### Action conversion (`step`)

```
numpy (6,) normalized → motor_normalized_to_joint_rad() → numpy (6,) rad
→ torch.unsqueeze(0) → torch (1, 6) → ManagerBasedRLEnv.step()
```

### Env creation

The wrapper takes a pre-instantiated `env_cfg` object. The typical calling pattern:

```python
from isaaclab_tasks.utils import parse_env_cfg
env_cfg = parse_env_cfg("SO101-FigureShapePlacementEasy-RL-v0", num_envs=1)
env = IsaacLabGymEnv(env_cfg)
```

`parse_env_cfg()` resolves the gym registry entry, imports the config class, and instantiates it. The wrapper does not call `gymnasium.make()` because Isaac Lab's gym entries pass `env_cfg_entry_point` as a string that `ManagerBasedRLEnv.__init__` cannot accept directly.

---

## Usage Example

```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
app = app_launcher.app

from isaaclab_tasks.utils import parse_env_cfg
from so101_lab.rl.isaac_lab_gym_env import IsaacLabGymEnv
import numpy as np

env_cfg = parse_env_cfg("SO101-FigureShapePlacementEasy-RL-v0", num_envs=1)
env = IsaacLabGymEnv(env_cfg)

obs, info = env.reset()
# obs["observation.state"]:        (6,) float32
# obs["observation.images.top"]:   (480, 640, 3) uint8
# info["ground_truth"]["cube_pos"]: (3,) float32

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)

print(env.get_reward_details())
env.close()
app.close()
```

---

## Notes

- `num_envs=1` only — LeRobot SAC runs single-process; the batch dim in Isaac Lab is always 1.
- `randomize_light=False` is used in SAC training when the reward classifier was trained on fixed-lighting data and light randomization degrades classifier accuracy.
- `terminated` and `truncated` come from Isaac Lab's termination conditions (timeout 15 s, cube out of bounds). In PPO, `truncated=True` triggers bootstrap; in SAC, episodes are collected until either flag is set.
- The `get_reward_details()` method accesses `reward_manager._step_reward` and `_term_names` which are private Isaac Lab internals — may break on Isaac Lab version upgrades.

---

## Files

```
so101_lab/rl/
├── isaac_lab_gym_env.py   # IsaacLabGymEnv
└── domain_rand.py         # apply_domain_rand_flags() — DR CLI flags for all RL scripts

tests/sim/
└── test_rl_gym_wrapper.py  # functional test (requires Isaac Sim)
```

Converters: `so101_lab/data/converters.py` — `joint_rad_to_motor_normalized`, `motor_normalized_to_joint_rad`
