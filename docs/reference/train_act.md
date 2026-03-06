# train_act.md

Train ACT (Action Chunking with Transformers) policy on a LeRobot dataset. lerobot-env.

---

Runs in **Isaac Lab env** (`act-isaac`) and **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

`scripts/train/train_act.py` uses the LeRobot ACT implementation directly. Outputs a checkpoint directory loadable by `ACTInference` in Isaac Lab without lerobot dependency.

Split-venv design:
- **Training:** lerobot-env → `scripts/train/train_act.py`
- **Inference:** Isaac Lab → `so101_lab/policies/act/` (standalone, no lerobot)

## CLI

```bash
act-lerobot  # activate lerobot-env

# Basic
python scripts/train/train_act.py --dataset data/recordings/figure_shape_placement_v1

# With config file
python scripts/train/train_act.py \
    --dataset data/recordings/figure_shape_placement_v1 \
    --config configs/policy/act.yaml

# Config + CLI override (CLI wins)
python scripts/train/train_act.py \
    --dataset data/recordings/figure_shape_placement_v1 \
    --config configs/policy/act.yaml \
    --kl-weight 0.1 \
    --steps 120000

# Resume from checkpoint
python scripts/train/train_act.py \
    --dataset data/recordings/figure_shape_placement_v1 \
    --steps 100000 \
    --resume outputs/act_figure_shape_placement_v1/checkpoint_30000
```

## Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | **required** | Path to local LeRobot dataset |
| `--config` | — | YAML config file (CLI args override) |
| `--output` | `outputs/` | Output root directory |
| `--name` | `act_<dataset>` | Run name (subfolder under output) |
| `--steps` | 50000 | Training steps |
| `--batch-size` | 16 | Batch size |
| `--lr` | 1e-5 | Learning rate |
| `--chunk-size` | 100 | Action chunk size |
| `--n-action-steps` | 100 | Actions executed per chunk |
| `--kl-weight` | 10.0 | KL divergence weight (see below) |
| `--tracker` | `none` | `trackio`, `wandb`, or `none` |
| `--tracker-project` | `so101-act` | Project name |
| `--system-stats` | false | Log CPU/RAM to trackio |
| `--seed` | 42 | Random seed |
| `--num-workers` | 4 | DataLoader workers |
| `--resume` | — | Resume from checkpoint directory |
| `--checkpoint-interval` | 5000 | Save checkpoint every N steps |
| `--best-check-interval` | 100 | Check for best model every N steps |

## KL weight tuning

ACT uses a VAE to encode demonstration variability. `kl_weight` controls the reconstruction vs KL divergence trade-off.

**Posterior collapse symptom:** `kld_loss: 0.0000` in logs — model ignores the latent space, becomes deterministic.

| kl_weight | Effect |
|-----------|--------|
| 10.0 | Default; may cause posterior collapse |
| 1.0 | Compromise |
| 0.1 | More expressive latent space; recommended if collapse observed |

Healthy `kld_loss` range: 0.01–0.5.

## Steps guidelines

| Episodes | Recommended steps |
|----------|-------------------|
| 10–20 | 30k–50k |
| 50–100 | 50k–100k |
| 200+ | 100k–200k |

## Checkpoint structure

```
outputs/act_figure_shape_placement_v1/
├── config.json                   # ACT architecture (lerobot format)
├── model.safetensors             # Final weights (~200 MB)
├── train_config.json             # Training hyperparams + dataset path + fps
├── train_state.pt                # Optimizer state + step
├── preprocessor_config.json
├── postprocessor_config.json
├── best/                         # Best model (min loss over training)
│   ├── model.safetensors
│   ├── train_state.pt            # Includes loss value
│   └── ...
├── checkpoint_5000/
│   ├── model.safetensors
│   ├── train_state.pt
│   └── ...
└── checkpoint_10000/
    └── ...
```

`best/` = lowest loss checkpoint. Root = final step (may be worse than best). Use `best/` for eval by default.

## ACTInference module

`so101_lab/policies/act/` — standalone inference, no lerobot dependency:

- `configuration_act.py` — `ACTConfig.from_dict()` loads from `config.json`
- `normalization.py` — stats-based normalizer from checkpoint
- `modeling_act.py` — ACT network (adapted from lerobot, training code removed)
- `__init__.py` — `ACTInference` wrapper

```python
from so101_lab.policies.act import ACTInference

policy = ACTInference(checkpoint_path="outputs/act_v1", device="cuda")
policy.reset()  # call on episode start

obs = {
    "joint_pos": np.zeros(6),   # radians
    "images": {
        "top":   np.zeros((480, 640, 3), dtype=np.uint8),
        "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    }
}
action = policy.select_action(obs)  # (6,) radians
```

## Model architecture

Default ACT config (from `configs/policy/act.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dim_model` | 512 | Transformer hidden dim |
| `n_heads` | 8 | Attention heads |
| `n_encoder_layers` | 4 | Encoder depth |
| `n_decoder_layers` | 1 | Decoder depth |
| `dim_feedforward` | 3200 | FFN hidden dim |
| `vision_backbone` | resnet18 | Image encoder |
| `pretrained_backbone_weights` | ImageNet1K | Pretrained weights |
| `use_vae` | true | VAE for action variability |
| `latent_dim` | 32 | VAE latent size |
| `n_vae_encoder_layers` | 4 | VAE encoder depth |
| `dropout` | 0.1 | |
| `kl_weight` | 10.0 | KL loss weight |

`action_delta_indices`: which future timesteps to predict (default `range(chunk_size)`). `observation_delta_indices`: which past observation frames to use (default `[0]` = current only). Both loaded from `config.json` at training and inference.

## Normalization flow

```
Recording:   joint_pos [rad] → joint_rad_to_motor_normalized() → [-100,100] → dataset
Training:    dataset → ACT preprocessor (MEAN_STD) → model
Inference:   joint_pos [rad] → normalized motor → ACT normalizer → model
                             → denormalize → normalized motor → [rad]
```

Double conversion (rad → normalized → model-normalized) because LeRobot stores data in motor-normalized space and Isaac Lab operates in radians. `so101_lab/data/converters.py` handles the conversion.

## Troubleshooting

**Blackwell GPU (RTX 50xx) — CUDA not compatible:**
```bash
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --reinstall torchcodec huggingface-hub transformers
```

**Loss not decreasing:** Check demo quality (visualize with Rerun), increase episode count, try lower `--lr 1e-6`.

**kld_loss = 0.0:** Posterior collapse — use `--kl-weight 0.1` or `--config configs/policy/act/low_kl.yaml`.

**CUDA OOM:** Reduce `--batch-size`.

**Stats not found:** Pass `--dataset` explicitly or verify `train_config.json` has correct path.
