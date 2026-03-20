# Setup

Prerequisites and environment setup for SO-101 Lab.

![Pipeline overview](../diagrams/pipeline.png)

This project uses **four separate Python environments** because Isaac Sim (Python 3.11) and LeRobot 0.5.1 (Python 3.12) cannot coexist. For the full rationale: [architecture.md — Virtual environments](../architecture.md#virtual-environments).

---

## Prerequisites

| Component | Version |
|-----------|---------|
| NVIDIA Isaac Sim | 5.1 |
| Isaac Lab | 2.3.0 |
| LeRobot | 0.5.1 |
| CUDA | 13.0 (cu128) |

- CUDA-capable GPU with **≥8 GB VRAM** (16 GB recommended; all default configs fit in 8–16 GB)
- Isaac Sim 5.1 installed (see [Isaac Sim docs](https://docs.isaacsim.omniverse.nvidia.com/))
- `uv` installed: `pip install uv` or see [uv docs](https://docs.astral.sh/uv/)

---

## 1. Install Isaac Lab

> **⚠️ Run outside any venv.** Create and activate a dedicated venv first, then run `./isaaclab.sh --install`.

Before installing, make sure cmake is installed via apt (not snap):

```bash
which cmake   # should be /usr/bin/cmake, not /snap/bin/cmake
# if snap: sudo apt install cmake
```

```bash
git clone https://github.com/isaac-sim/IsaacLab.git /path/to/IsaacLab
cd /path/to/IsaacLab
ln -s /path/to/isaacsim _isaac_sim   # link Isaac Sim

# create venv with Python 3.11 (required by Isaac Lab)
uv venv --python 3.11 /path/to/IsaacLab/env_isaaclab
source /path/to/IsaacLab/env_isaaclab/bin/activate
./isaaclab.sh --install
```

`isaaclab.sh --install` does not patch the venv's `activate` script — add Isaac Sim paths manually (replace `/path/to/IsaacLab` with your actual path):

```bash
cat >> /path/to/IsaacLab/env_isaaclab/bin/activate << 'EOF'

export ISAACLAB_PATH="/path/to/IsaacLab"
alias isaaclab="/path/to/IsaacLab/isaaclab.sh"
export RESOURCE_NAME="IsaacSim"

if [ -f "/path/to/IsaacLab/_isaac_sim/setup_conda_env.sh" ]; then
    . "/path/to/IsaacLab/_isaac_sim/setup_conda_env.sh"
fi
EOF
```

After installing, verify lerobot-env was not damaged (`isaaclab.sh` downgrades setuptools):

```bash
source /path/to/lerobot-env/bin/activate
python -c "import setuptools; print(setuptools.__version__)"
# if < 82, restore: uv pip install "setuptools>=82"
```

---

## 2. Install LeRobot

```bash
git clone https://github.com/huggingface/lerobot.git /path/to/lerobot
cd /path/to/lerobot
uv venv --python 3.12 /path/to/lerobot-env
source /path/to/lerobot-env/bin/activate
uv pip install -e .
```

If you already have LeRobot installed, check the version:

```bash
python -c "import lerobot; print(lerobot.__version__)"
```

If it's older than 0.5.1, upgrade:

```bash
cd /path/to/lerobot
git pull
uv pip install -e .
```

---

## 3. Clone and install the project

```bash
git clone <repo-url> so101-lab
cd so101-lab

cp .env.example .env
# Edit .env: set ISAACLAB_ENV and LEROBOT_ENV paths
```

Add shell aliases to `~/.bashrc` (adjust path to your project):

```bash
alias act-isaac='eval "$(~/path/to/so101-lab/activate_isaaclab.sh)"'
alias act-lerobot='eval "$(~/path/to/so101-lab/activate_lerobot.sh)"'
```

Then `source ~/.bashrc`.

### Isaac Lab env

Isaac Lab venv uses `uv` — use `uv pip` for all installs.

```bash
act-isaac  # activate Isaac Lab env
uv pip install -e .

# Additional packages (first time only):
uv pip install diffusers                                          # Diffusion Policy eval
uv pip install pygame                                            # gamepad support
uv pip install transformers draccus av datasets huggingface_hub accelerate  # VLA eval / SAC
uv pip install num2words safetensors                             # VLA eval
uv pip install cma                                               # Noise Prior CMA-ES
uv pip install nvidia-ml-py                                     # GPU metrics for experiment tracking
uv pip install pyserial deepdiff feetech-servo-sdk               # SO-101 leader arm (teleop)
```

> **Note:** `trackio` does not work in isaaclab-env (Python 3.11) — it depends on gradio/typer which require 3.12+. Use `--tracker wandb` or `--tracker none` for scripts running here. trackio works in lerobot-env.

> **VIP reward weights** (~98 MB) are downloaded automatically to `~/.vip/resnet50/` on first use. No manual download needed.

### LeRobot env

```bash
act-lerobot  # activate lerobot-env
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # install torch first
uv pip install -e .                  # install so101-lab itself
uv pip install -e ".[hardware]"      # SO-101 leader arm support
uv pip install trackio nvidia-ml-py  # experiment tracking
```

> **Blackwell GPUs (RTX 50xx):** use cu128 index — torch 2.10.0+cu128 works on RTX 50xx out of the box.

### SmolVLA fine-tuning dependencies (lerobot-env, first time only)

Must be run from the LeRobot repo root, not from so101-lab:

```bash
cd /path/to/lerobot
uv pip install -e ".[smolvla]"   # SmolVLA fine-tuning
uv pip install -e ".[async]"     # gRPC server (smolvla_server.py)
cd /path/to/so101-lab
```

> **⚠️ `train_expert_only=true` (the default) overrides `freeze_vision_encoder=false`.**
> To unfreeze the vision encoder you must explicitly set `--policy.train_expert_only=false`.
> See [workflow/03_bc_training.md](03_bc_training.md) for SmolVLA training options.

### Local tool envs

```bash
# Rerun — episode visualization
uv venv venvs/rerun
source venvs/rerun/bin/activate
uv pip install rerun-sdk pandas pyarrow opencv-python numpy

# Viewer — live camera preview during recording
uv venv venvs/viewer
source venvs/viewer/bin/activate
uv pip install opencv-python numpy
```

See `venvs/README.md` for usage.

---

## 4. Hardware setup (SO-101 robot)

### Ports

| Device | Default port |
|--------|-------------|
| Follower arm | `/dev/ttyACM0` |
| Leader arm | `/dev/ttyACM2` |
| Top camera | `/dev/video0` |
| Wrist camera | `/dev/video2` |

### Calibration

Calibration writes homing offsets to motor registers (STS3215 register 31).

```bash
act-lerobot  # activate lerobot-env
python scripts/calibrate_leader.py --port /dev/ttyACM2 --name leader_1
```

> **Important:** do not calibrate the same motor in both so101-lab and LeRobot without syncing the calibration files — offsets will conflict.
>
> To sync after calibrating in LeRobot:
> ```bash
> cp so101_lab/devices/lerobot/.cache/leader_1.json \
>    ~/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/leader_arm_1.json
> ```

---

## 5. Experiment tracking

All training scripts support `--tracker`:

```bash
--tracker trackio    # local SQLite tracker (lerobot-env only)
--tracker wandb      # cloud W&B (both envs)
--tracker none       # no tracking

--system-stats       # log CPU/RAM metrics (requires trackio)
```

> **trackio** only works in lerobot-env (Python 3.12). For scripts in isaaclab-env (PPO client, eval), use `--tracker wandb` or `--tracker none`.

View local runs:

```bash
trackio show
```

GPU metrics are logged automatically when `nvidia-ml-py` is installed.

---

## 6. Verify setup

```bash
act-isaac  # activate Isaac Lab env
python tests/sim/test_env_spawn.py --gui --env figure_shape_placement_easy
```

Isaac Sim window should open, robot should appear on the table, 10 resets with printed positions.

```bash
# Verify real robot connection
act-lerobot  # activate lerobot-env
python tests/sim/test_robot.py
```

If something goes wrong: [docs/troubleshooting.md](../troubleshooting.md).
