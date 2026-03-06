# experiment_tracking.md

Experiment tracking with trackio (local SQLite) or wandb (cloud). Supported by all training scripts.

---

## Overview

All training scripts use `so101_lab/utils/tracker.py` for unified tracker setup:

```python
from so101_lab.utils.tracker import add_tracker_args, setup_tracker, cleanup_tracker

add_tracker_args(parser, default_project="so101-act")  # in argparse
tracker, sys_monitor = setup_tracker(args, run_name)   # in main()
cleanup_tracker(tracker, sys_monitor)                   # at exit
```

CLI flags: `--tracker <backend>`, `--tracker-project <name>`, `--system-stats`, `--wandb` (deprecated).

| Value | Backend | Storage |
|-------|---------|---------|
| `trackio` | Local SQLite | `~/.cache/huggingface/trackio/` |
| `wandb` | Weights & Biases cloud | wandb.ai (requires `wandb login`) |
| `none` | Disabled | — |

Run name is taken from `os.path.basename(--output)` (e.g. `sac_v1`).
When resuming with the same `--output`, trackio appends metrics to the existing run.

---

## Wandb

```bash
python scripts/train/train_act.py --tracker wandb --tracker-project so101-act ...
```

Requires `wandb login` before first use. Project name defaults are listed in the support matrix.

---

## Trackio

### View

```bash
trackio show                            # all projects, web dashboard
trackio show --project so101-sac        # specific project
trackio list runs --project so101-sac
trackio list metrics --project so101-sac --run sac_v1
```

### Cleanup

Interactive script to delete old runs:

```bash
python scripts/tools/trackio_cleanup.py
```

Navigate: select project → select runs by number or `a` (all) → confirm delete.
Shows mode, step count, and metrics before deleting. Removes empty projects on request.

### Storage location

```
~/.cache/huggingface/trackio/
├── so101-sac.db
├── so101-sac.db-wal
└── so101-sac.lock
```

---

## GPU metrics (automatic)

Logged every 10 seconds when `auto_log_gpu=True` (enabled by default in all scripts).
Requires `nvidia-ml-py`: `pip install nvidia-ml-py` (il) or `uv pip install nvidia-ml-py` (lr).

| Metric | Description |
|--------|-------------|
| `gpu/0/utilization` | GPU utilization (%) |
| `gpu/0/memory_utilization` | VRAM utilization (%) |
| `gpu/0/allocated_memory` | Used VRAM (GiB) |
| `gpu/0/total_memory` | Total VRAM (GiB) |
| `gpu/0/power` | Power draw (W) |
| `gpu/0/temp` | Temperature (°C) |
| `gpu/0/sm_clock` | SM clock speed (MHz) |
| `gpu/0/memory_clock` | Memory clock (MHz) |
| `gpu/0/fan_speed` | Fan speed (%) |
| `gpu/0/throttle_thermal` | Thermal throttle flag |
| `gpu/0/throttle_power` | Power throttle flag |
| `gpu/0/energy_consumed` | Energy since run start (J) |
| `gpu/0/pcie_tx` / `pcie_rx` | PCIe throughput (KB/s) |

---

## CPU/RAM metrics (`--system-stats`)

Logged every 10 seconds. Uses `psutil` (already installed). Implemented in
`so101_lab/utils/system_monitor.py` (background thread + `trackio.log_system()`).

```bash
python scripts/train/train_act.py --dataset ... --tracker trackio --system-stats
```

| Metric | Description |
|--------|-------------|
| `system/cpu_percent` | CPU utilization (%) |
| `system/ram_percent` | RAM utilization (%) |
| `system/ram_used_gb` | Used RAM (GiB) |
| `system/ram_available_gb` | Available RAM (GiB) |
| `process/rss_gb` | Process RSS memory (GiB) |
| `process/cpu_percent` | Process CPU utilization (%) |

---

## Typical workflow

```bash
# IQL pretrain → separate run (loss_v, loss_q, loss_actor)
python scripts/train/train_iql_pretrain.py \
    --demo-dataset data/recordings/... \
    --output outputs/iql_v1 --tracker trackio

# SAC stages → one run, continuous step numbering (resume same --output)
python scripts/train/train_sac_composite_grpc.py \
    --resume outputs/iql_v1/final/pretrained_model \
    --demo-dataset data/recordings/... \
    --output outputs/sac_v1 --auto-server --headless

python scripts/train/train_sac_composite_grpc.py \
    --resume outputs/sac_v1/final/pretrained_model \
    --demo-dataset data/recordings/... \
    --output outputs/sac_v1 --auto-server --headless
```

In trackio: run `sac_bc_v1` (loss_bc) and run `sac_v1` (critic, actor, eval/* metrics).

---

## Per-script metrics

### `train_act.py`

| Metric | Description |
|--------|-------------|
| `loss` | Total loss |
| `l1_loss` | L1 reconstruction loss |
| `kl_loss` | KL divergence (action space regularization) |

Default project: `so101-act`

### `train_diffusion.py`

| Metric | Description |
|--------|-------------|
| `loss` | Diffusion noise prediction MSE |
| `lr` | Current learning rate |

Default project: `so101-diffusion`

### `train_reward_classifier_v2.py`

| Metric | Description |
|--------|-------------|
| `train_loss` / `val_loss` | BCE loss |
| `train_accuracy` / `val_accuracy` | Binary accuracy |

Default project: `so101-reward-classifier`

### `train_iql_critics.py`

| Metric | Description |
|--------|-------------|
| `train/loss_v` | Expectile V-loss |
| `train/loss_q` / `loss_q1` / `loss_q2` | TD Q-loss |
| `train/v_mean` / `v_std` | Value network output stats |
| `train/q1_mean` / `q2_mean` | Q-network output stats |
| `train/advantage_mean` / `advantage_std` | A = Q - V stats |
| `train/reward_mean` | Reward in batch |
| `train/td_target_mean` | TD target stats |

Default project: `so101-iql-critics`

### `train_iql_pretrain.py`

| Metric | Description |
|--------|-------------|
| `train/loss_v` | Expectile V-loss |
| `train/loss_q` | TD Q-loss |
| `train/loss_actor` | Actor BC loss |
| `train/v_mean` | Value network mean output |
| `train/q_mean` | Q-network mean output |
| `train/advantage_mean` | A = Q - V stats |

Default project: `so101-iql`

### `train_sac_composite_grpc.py` / `sac_server.py`

| Metric | Description |
|--------|-------------|
| `train/loss_critic` | SAC critic TD loss |
| `train/loss_actor` | SAC actor policy loss |
| `train/loss_temperature` | Temperature (α) loss |
| `train/temperature` | Current α |
| `train/q_actor_mean` | Mean Q for actor actions (extrapolation diagnostic) |
| `train/entropy` | Actor entropy (exploration diagnostic) |
| `train/batch_reward_mean/min/max` | Reward stats in training batch |
| `reward/vip_*` | VIP reward stats (if VIP mode) |
| `episode/reward` | Cumulative episode reward |
| `episode/steps` | Episode length |
| `eval/success_rate` | Evaluation success rate |
| `eval/avg_reward` | Evaluation average reward |
| `hil/active` | HIL intervention toggle state (1/0) |
| `hil/intervention_buffer_size` | Intervention replay buffer size |
| `train/loss_bc` | BC pretrain loss (during `--bc-pretrain` phase) |

Default project: `so101-sac`

### `train_flow_noise_ppo.py`

| Metric | Description |
|--------|-------------|
| `train/actor_loss` | PPO clipped surrogate loss |
| `train/value_loss` | Value MSE loss |
| `train/ratio_mean` | `exp(new_log_prob - old_log_prob)` mean |
| `train/reward_mean` | VIP reward per rollout |
| `train/value_mean` | Value head output mean |
| `eval/success_rate` | Evaluation success rate |

---

## Script support matrix

All scripts below use `add_tracker_args()` from `so101_lab/utils/tracker.py`, providing `--tracker`, `--system-stats`, and `auto_log_gpu` uniformly.

| Script | Default project |
|--------|:----------------|
| `train_act.py` | `so101-act` |
| `train_diffusion.py` | `so101-diffusion` |
| `train_iql_critics.py` | `so101-iql-critics` |
| `train_sac_composite_grpc.py` | `so101-sac` |
| `train_reward_classifier_v2.py` | `so101-reward-classifier` |
| `train_flow_noise_ppo.py` | `so101-flow-ppo` |
| `train_noise_sampler_online.py` | `so101-noise-sampler` |
| `train_iql_pretrain.py` | `so101-iql` |
