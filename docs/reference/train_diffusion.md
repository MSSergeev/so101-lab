# train_diffusion.md

Train Diffusion Policy on a LeRobot dataset. lerobot-env for training; Isaac Lab env for eval (`uv pip install diffusers` required in Isaac Lab venv).

---

Runs in **Isaac Lab env** (`act-isaac`) and **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

`scripts/train/train_diffusion.py` uses the LeRobot Diffusion Policy implementation. Same split-venv design as ACT: training in lerobot-env, inference via standalone `so101_lab/policies/diffusion/` in Isaac Lab.

## Key differences from ACT

| | ACT | Diffusion Policy |
|--|-----|-----------------|
| Normalization | MEAN_STD | **MIN_MAX → [-1, 1]** |
| Observation history | 1 step | **2 steps** (`n_obs_steps=2`) |
| Horizon (chunk) | 100 | **16** |
| n_action_steps | 100 | **8** |
| Learning rate | 1e-5 | **1e-4** |
| Loss | MSE + KL | MSE on noise (epsilon) |
| Inference | 1 forward pass | **100 denoising steps** |
| Vision | ResNet18, FrozenBatchNorm | **ResNet18, GroupNorm, random crop 84×84** |
| LR scheduler | None | **Cosine with 500-step warmup** |
| Temporal ensemble | Supported | Not supported |

## CLI

```bash
act-lerobot  # activate lerobot-env

# Basic
python scripts/train/train_diffusion.py --dataset data/recordings/figure_shape_placement_v1

# With config file
python scripts/train/train_diffusion.py \
    --dataset data/recordings/figure_shape_placement_v1 \
    --config configs/policy/diffusion/baseline.yaml

# Config + CLI override
python scripts/train/train_diffusion.py \
    --dataset data/recordings/figure_shape_placement_v1 \
    --config configs/policy/diffusion/baseline.yaml \
    --lr 5e-5 --steps 100000

# Resume
python scripts/train/train_diffusion.py \
    --dataset data/recordings/figure_shape_placement_v1 \
    --steps 100000 \
    --resume outputs/diffusion_figure_shape_placement_v1/checkpoint_30000
```

## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | **required** | Path to local LeRobot dataset |
| `--config` | — | YAML config file (CLI overrides) |
| `--output` | `outputs/` | Output root directory |
| `--name` | `diffusion_<dataset>` | Run name |
| `--steps` | 50000 | Training steps |
| `--batch-size` | 32 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--horizon` | 16 | Action prediction length |
| `--n-obs-steps` | 2 | Observation history steps |
| `--n-action-steps` | 8 | Actions executed per chunk |
| `--num-train-timesteps` | 100 | Diffusion training timesteps |
| `--num-inference-steps` | same as train | Denoising steps at inference |
| `--tracker` | `none` | `trackio`, `wandb`, or `none` |
| `--tracker-project` | `so101-diffusion` | Project name |
| `--system-stats` | false | Log CPU/RAM to trackio |
| `--seed` | 42 | Random seed |
| `--resume` | — | Resume from checkpoint directory |
| `--checkpoint-interval` | 5000 | Save checkpoint every N steps |
| `--best-check-interval` | 100 | Check best model every N steps |

## Architecture

**Observation history:** `n_obs_steps=2` — current + previous frame. First step of episode duplicates the observation (lerobot default).

**Horizon / action execution:**
```
timestep:    t-1    t    t+1  ...  t+7  ...  t+13  t+14  t+15
obs used:    YES   YES   NO        NO        NO
action gen:  YES   YES   YES  ...  YES  ...  YES    YES   YES
action used: NO    YES   YES  ...  YES   NO
```

**Denoising process:** sample random noise (B, horizon, action_dim) → denoise through 100 DDPM steps → UNet predicts noise at each step → clean action trajectory.

**Vision encoder:** ResNet18 with GroupNorm (not BatchNorm), random crop 84×84 during training / center crop at eval, SpatialSoftmax (32 keypoints) → feature vector. Shared encoder for all cameras.

**LR scheduler:** cosine with 500-step linear warmup (LR 0 → target over 500 steps, then cosine decay to 0).

Logged metrics: `loss` (MSE on noise), `lr`, `step`.

## Checkpoint structure

```
outputs/diffusion_figure_shape_placement_v1/
├── config.json                   # Diffusion architecture config
├── model.safetensors             # Final weights
├── train_config.json             # Training hyperparams + dataset path + fps
├── train_state.pt                # Optimizer + LR scheduler state
├── preprocessor_config.json
├── policy_preprocessor_step_3_normalizer_processor.safetensors  # MIN_MAX stats
├── postprocessor_config.json
├── best/                         # Best model (min loss)
│   ├── model.safetensors
│   ├── train_state.pt
│   └── ...
└── checkpoint_5000/
    └── ...
```

Resume restores model weights, optimizer state, **and LR scheduler state** (unlike ACT which has no scheduler).

## Steps guidelines

| Episodes | Recommended steps | Notes |
|----------|-------------------|-------|
| 10–20 | 30k–50k | |
| 50–100 | 50k–100k | |
| 200+ | 100k–200k | Diffusion typically needs more steps than ACT |

## Evaluation

```bash
act-isaac  # activate Isaac Lab env
uv pip install diffusers  # first time only

# Single-env
python scripts/eval/eval_diffusion_policy.py \
    --checkpoint outputs/diffusion_figure_shape_placement_v1 \
    --episodes 10

# Faster inference (20 denoising steps instead of 100, ~5x speedup)
python scripts/eval/eval_diffusion_policy.py \
    --checkpoint outputs/diffusion_figure_shape_placement_v1 \
    --num-inference-steps 20

# Sweep multiple checkpoints (recommended)
python scripts/eval/sweep_diffusion_eval_single.py \
    --checkpoint outputs/diffusion_figure_shape_placement_v1 \
    --all --episodes 50 --max-steps 500 \
    --num-inference-steps 10
```

Same controls and output format as `eval_act_policy.py` (see `reference/eval_act_policy.md`). Key difference: `--num-inference-steps` to reduce denoising steps.

> **Note:** `sweep_diffusion_eval_single.py` runs single-env eval (slower but supports `--save-episodes`). `sweep_diffusion_eval.py` uses parallel envs for faster sweeps.

## Troubleshooting

**`ImportError: diffusers not installed`** (Isaac Lab venv):
```bash
act-isaac  # activate Isaac Lab env
uv pip install diffusers
```

**Blackwell GPU (RTX 50xx):** LeRobot 0.5.1 ships torch 2.10.0+cu128 — no reinstall needed.

**Loss not decreasing:** check demo quality, increase episode count (diffusion needs diversity), try lower `--lr 5e-5`, try `--horizon 32` for longer tasks.

**Slow inference:** use `--num-inference-steps 20` (DDPM with fewer steps, small quality trade-off).

**CUDA OOM:** reduce `--batch-size` or `--num-envs`.

## Image normalization (important for correctness)

Diffusion Policy uses `MEAN_STD` normalization for images. The dataset must have per-channel
image stats in `stats.json` (shape (3,) per channel, values in [0, 1]):

```bash
# Verify stats.json has image stats
python3 -c "
import json
stats = json.load(open('data/recordings/my_dataset/meta/stats.json'))
for k in stats: print(f'{k}: shape={len(stats[k][\"mean\"])}')
"
# Expected: observation.images.top shape=3, observation.images.wrist shape=3
```

If missing (older datasets recorded before the fix), recompute:
```bash
python scripts/tools/recompute_stats.py data/recordings/my_dataset
```

**What went wrong before the fix:** `LeRobotDatasetWriter` did not include images in stats
computation → `stats.json` had no image entries → LeRobot normalizer applied IDENTITY to images
(left them as float32 [0, 1]) → eval inference normalized via `images / 127.5 - 1.0` → [-1, 1]
mismatch → erratic arm behavior despite low training loss (0.004).

**Current behavior:** dataset writer samples every 10th frame per episode, computes per-channel
mean/std in [0, 1], stores in `stats.json`. Eval inference applies `(x/255 - mean) / std`.
Old checkpoints fall back to [0, 1] passthrough (backward compatible).
