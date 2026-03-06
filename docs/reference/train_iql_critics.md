# train_iql_critics.md

Train IQL Q/V critics for advantage-weighted BC. Runs in `lerobot-env`.

---

Runs in **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

Trains standalone Q(s,a) and V(s) networks on a LeRobot dataset with pre-computed rewards
(`next.reward` column). The resulting critics are used to compute per-frame advantage weights
`w = exp(A/β)` for weighted BC fine-tuning of SmolVLA.

This is step 2 of the IQL pipeline:

```
1. prepare_reward_dataset.py  → copy dataset + write VIP rewards to next.reward
2. train_iql_critics.py       → train Q and V (this script)
3. compute_iql_weights.py     → A(s,a) = Q(s,a) - V(s) → w = exp(A/β) → parquet
4. train_vla_weighted_bc.py   → weighted BC: loss *= iql_weight per sample
```

**Key design choices:**
- Image encoder runs once and caches embeddings to `{encoder}_embeddings.pt` (~7 min for 194k frames). Subsequent runs skip this phase and go straight to training (~5 min for 50k steps).
- VIP or ImageNet ResNet50 both produce 1024-dim embeddings per camera; they are concatenated to 2048-dim before the MLP.
- Backbone is **not** loaded during training — only the cached embeddings are used.
- Twin Q-networks with EMA target for pessimistic Q estimates.

**Results:** VIP dense rewards give +12–16% SR over BC baseline (86–88% vs 70–76%). Sim sparse rewards (0/1) give no improvement with either encoder.

---

## Architecture

### Networks

Both networks share the same MLP structure (Linear → LayerNorm → SiLU, repeated):

**V(s):** `[emb(2048), state(6)] → Linear(2054,256) → LN → SiLU → Linear(256,256) → LN → SiLU → Linear(256,1)`

**Q(s,a):** `[emb(2048), state(6), action(6)] → Linear(2060,256) → LN → SiLU → Linear(256,256) → LN → SiLU → Linear(256,1)`

Twin Q: two independent Q-heads, `min(Q1, Q2)` for pessimistic estimate and TD targets.

| Network | Params |
|---------|--------|
| V-net | 593,153 |
| Q-nets (twin) | 1,189,378 (2×) |

### Image Encoder (`--encoder`)

| Value | Model | Output |
|-------|-------|--------|
| `vip` (default) | VIP ResNet50 (Ego4D time-contrastive) | 1024-dim per camera |
| `imagenet` | ResNet50 IMAGENET1K_V2 + Linear(2048→1024) | 1024-dim per camera |

Both produce 2048-dim total (top + wrist concatenated). Cache file: `{encoder}_embeddings.pt`
in the dataset directory. `compute_iql_weights.py` reads encoder type from checkpoint config.

### IQL math

**V-loss** (expectile regression, τ):
```
L_V = E[ L_τ(Q_target(s,a) - V(s)) ]
L_τ(u) = |τ - 1(u < 0)| · u²
```
V approximates the upper τ-quantile of Q-values (achievable maximum).

**Q-loss** (TD with V bootstrap):
```
L_Q = E[ (r + γ · V(s') - Q(s,a))² ]
```
Bootstrapping via V(s') avoids OOD actions (unlike max_a Q(s',a')).

---

## CLI

```bash
act-lerobot  # activate lerobot-env
cd /path/to/so101-lab

python scripts/train/train_iql_critics.py \
    --dataset data/recordings/figure_shape_placement_v5_vip \
    --output outputs/iql_critics_v1 \
    --num-steps 50000 --batch-size 256 \
    --expectile 0.7 --discount 0.99 \
    --reward-normalize \
    --tracker trackio
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | required | LeRobot dataset path (must have `next.reward` column) |
| `--output` | `outputs/iql_critics_v1` | Output directory |
| `--num-steps` | 50000 | Training steps |
| `--batch-size` | 256 | Batch size |
| `--expectile` | 0.7 | τ for V-loss (>0.5 = optimistic upper quantile) |
| `--discount` | 0.99 | γ for TD target |
| `--critic-lr` | 3e-4 | Learning rate for Q-networks |
| `--value-lr` | 3e-4 | Learning rate for V-network |
| `--target-tau` | 0.005 | EMA rate for target Q |
| `--hidden-dims` | `256 256` | MLP hidden layer sizes |
| `--reward-normalize` | false | Normalize rewards to ~[-1, 0] via p5 percentile |
| `--reward-clip` | `-2.0 0.0` | Clip range after normalization |
| `--encoder` | `vip` | Image encoder: `vip` or `imagenet` |
| `--tracker` | `none` | `trackio`, `wandb`, or `none` |
| `--log-freq` | 100 | Logging interval (steps) |
| `--save-freq` | 10000 | Checkpoint interval (steps) |

### Reward normalization

Use `--reward-normalize` for VIP rewards (typical range ~[-20, -1]). It scales by the p5
percentile to map to ~[-1, 0], then clips to `--reward-clip`. For sim sparse rewards (0/1),
normalization is not needed.

Real dataset stats (v5_vip, 194,215 frames):
- Raw VIP: `[-17.04, -1.16]`, mean=`-9.48`
- Norm scale (p5): `15.19` → after norm: `[-1.12, -0.08]`, mean=`-0.62`

---

## Output

```
{output}/
├── train_config.json              # CLI args
├── step_10000/critics.pt          # periodic checkpoint
├── step_20000/critics.pt
├── ...
└── final/critics.pt               # final checkpoint (always saved)
```

**`critics.pt` format:**
```python
{
    "v_net": state_dict,
    "q1_net": state_dict,
    "q2_net": state_dict,
    "q1_target": state_dict,
    "q2_target": state_dict,
    "config": {
        "hidden_dims": [256, 256],
        "state_dim": 6,
        "action_dim": 6,
        "emb_dim": 2048,
        "expectile": 0.7,
        "discount": 0.99,
        "encoder_type": "vip",
        "image_keys": ["observation.images.top", "observation.images.wrist"],
        "reward_normalize": true,
        "reward_scale": 15.19,
        "reward_clip": [-2.0, 0.0],
    }
}
```

---

## Notes

**Embedding cache:** first run saves `vip_embeddings.pt` (or `imagenet_embeddings.pt`) in the
dataset directory. This is ~1.5 GB for 194k frames (2048 × float32). Subsequent runs detect the
cache and skip encoding entirely.

**Training speed:** with cached embeddings, 50k steps takes ~5 min (no video decoding).
Embedding cache for 194k frames takes ~7 min.

**Logged metrics:** `train/loss_v`, `train/loss_q`, `train/v_mean`, `train/q1_mean`,
`train/advantage_mean`, `train/advantage_std`, `train/reward_mean`, `train/td_target_mean`.

**State/action normalization:** states and actions are normalized from env space
`[-100, 100]` (joints) / `[0, 100]` (gripper) to `[-1, 1]` before being fed to Q/V.

**Episode boundaries:** `done` flag is set when `episode_index[i] != episode_index[i+1]`.
Last frame of each episode uses `V(s') * (1 - done) = 0`.
