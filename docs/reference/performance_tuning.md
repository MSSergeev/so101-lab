# performance_tuning.md

Isaac Sim rendering performance for recording and RL training. Isaac Lab env.

---

## Bottleneck profile

Profiling of `record_episodes.py` loop:

| Component | Time | % |
|-----------|------|---|
| `env.step()` (camera render + physics) | 23–28 ms | ~90% |
| shared-memory preview write | 0.6 ms | ~2% |
| image copy from GPU | 0.2 ms | <1% |
| other | 1.9 ms | ~7% |

**Main bottleneck is camera rendering.** Physics (PhysX) takes ~0.15 ms — negligible.

Scene render breakdown:
| Object | Time (ms) | % |
|--------|-----------|---|
| Platform | 17.9 | 52% |
| Cube | 14.5 | 42% |
| Robot | 0.3 | 1% |

Robot renders fast despite complexity (URDF meshes are typically optimized).
Platform and cube with PBR materials/textures dominate.

---

## Rate limiting

Isaac Sim applies a rate limit by default (~17 FPS actual throughput despite 120 Hz target).

**Disable via code** (required for headless; GUI preferences don't affect headless):
```python
import carb.settings

settings = carb.settings.get_settings()
settings.set("/app/runLoops/main/rateLimitEnabled", False)
settings.set("/app/runLoops/rendering_0/rateLimitEnabled", False)
settings.set("/app/renderer/skipWhileMinimized", False)
settings.set("/app/renderer/sleepMsOnFocus", 0)
settings.set("/app/renderer/sleepMsOutOfFocus", 0)
```

**Disable via GUI** (only affects GUI mode): `Edit → Preferences → Developer → Throttle Rendering` → uncheck "Enable UI FPS Limit" or select "No Pacing".

### Results

| Mode | Before | After | Gain |
|------|--------|-------|------|
| GUI | ~17 FPS | ~33 FPS | +94% |
| Headless | ~17 FPS | ~39 FPS | +129% |

---

## Speed-up options

### 1. Headless mode

Always use `--headless` for production recording:
```bash
python scripts/teleop/record_episodes.py --headless --no-preview
```

### 2. Lower render frequency

```bash
python scripts/teleop/record_episodes.py --render-hz 15
```
Cameras update less often; data is still recorded at policy frequency.

### 3. Reduce camera resolution

In `env_cfg.py`:
```python
env_cfg.scene.top.width = 320
env_cfg.scene.top.height = 240
```

### 4. Disable viewport (headless only)

```python
SimulationApp({"headless": True, "disable_viewport_updates": True})
```
Camera sensors continue working — only viewport is disabled.

### 5. Single camera

For tasks where one camera suffices, disable the second entirely.

---

## Profiling

### Built-in timing

`scripts/teleop/record_episodes_profiler.py` prints timing breakdown every 100 frames:

```
[TIMING] Last 100 frames:
  Total:        25.66 ms/frame (39.0 FPS)
  get_images:    0.19 ms
  preview_shm:   0.64 ms
  env_step:     22.99 ms
  idle_render:   0.00 ms
  other:         1.85 ms
```

### Omniverse Profiler (GUI)

1. `Window → Extensions` → enable `omni.kit.profiler.window`
2. `Window → Profiler` to open

Or via code:
```python
import omni.kit.app
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled("omni.kit.profiler.window", True)
```

### NVIDIA Nsight Systems (GPU-level)

```bash
nsys profile -o report python scripts/teleop/record_episodes.py --gui
```

---

## Omniverse event system overhead

~14 ms is consumed by `dispatch_event("update")` — notifies all extension subscribers on each
frame. Isaac Lab loads 50+ extensions; each can register update observers.

**No easy fix:** disabling extensions risks breaking functionality. For fastest headless
training, use a minimal `.kit` file — but this requires Isaac Lab internals knowledge and is
not supported by default.

---

## Recommended settings

### Teleoperation (quality)

```bash
python scripts/teleop/record_episodes.py \
    --gui --physics-hz 120 --policy-hz 30 --render-hz 30
```

### Batch recording (speed)

```bash
python scripts/teleop/record_episodes.py \
    --headless --no-preview --physics-hz 120 --policy-hz 30 --render-hz 30
```
With rate limit disabled: ~39 FPS = 1.3× realtime for 30 Hz policy.

---

## Camera setup tool

`scripts/utils/interactive_camera_setup.py` — GUI tool for positioning cameras:

1. Launch with `python scripts/utils/interactive_camera_setup.py`
2. Select camera in Stage panel → adjust Transform in Property panel
3. Run code from terminal output in Script Editor → copies transform values
4. Paste `pos` and `rot` into `tasks/template/env_cfg.py`

**Convention:** Isaac Sim GUI shows transforms in `opengl` convention. Use `convention="opengl"`
in `TiledCameraCfg.OffsetCfg`. Quaternion format: **(w, x, y, z)**.

Current camera config (from `template/env_cfg.py`):
```python
# Top camera (world-fixed)
pos=(-0.309, -0.457, 1.217), rot=(0.84493, 0.44012, -0.17551, -0.24817), convention="opengl"

# Wrist camera (attached to gripper)
pos=(0.008, -0.001, 0.016), rot=(0.0, 0.0, 0.0, 1.0), convention="opengl"
```

Both cameras: 640×480, H.264 video. Top: focal_length=28.7. Wrist: focal_length=36.5.
