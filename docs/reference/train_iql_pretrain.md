# train_iql_pretrain.py

IQL offline pretraining for SAC actor + critic warm-start. lerobot-env.

Produces a checkpoint compatible with `train_sac_composite_grpc.py --resume`.

---

Runs in **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

Trains a SAC policy (actor + critic ensemble) offline on a demonstration dataset using Implicit Q-Learning (Kostrikov et al. 2022). The goal is to warm-start the actor and critic before online SAC fine-tuning, so the policy starts near the offline data distribution rather than from random initialization.

This is **SAC-side IQL** — it trains the full SACPolicy architecture (ResNet-10 encoder + actor + critic ensemble). For IQL critics used with VLA weighted BC, see [reference/train_iql_critics.md](train_iql_critics.md).

**When to use:** Before `train_sac_composite_grpc.py` when you have demonstration data with rewards. Typical workflow:

```
prepare_reward_dataset.py        → write rewards to next.reward
train_iql_pretrain.py            → warm-start actor + critic (this script)
train_sac_composite_grpc.py --resume outputs/iql_v1/final/pretrained_model --no-offline
```

---

## IQL Algorithm

Three losses computed each step:

**V-loss** (expectile regression):
```
L_V = E[ L_τ(Q_target(s,a) - V(s)) ]
L_τ(u) = |τ - 1(u < 0)| · u²
```
V approximates the upper τ-quantile of Q (achievable maximum given the dataset).

**Q-loss** (TD with V bootstrap):
```
L_Q = E[ (r + γ · V(s') - Q(s,a))² ]
```
Bootstrapping via V(s') avoids OOD actions (unlike max_a Q(s',a')).

**Actor loss** (advantage-weighted behavior cloning):
```
A(s,a) = Q_target(s,a) - V(s)
w = exp(β · A(s,a)).clamp(max=100)
L_actor = E[ w · ‖π(s) - a‖² ]
```
Actions from the dataset with positive advantage get up-weighted. β controls sharpness (higher β = closer to greedy).

---

## CLI

```bash
act-isaac  # activate Isaac Lab env

python scripts/train/train_iql_pretrain.py \
    --demo-dataset data/recordings/figure_shape_placement_v5_vip_128 \
    --num-steps 50000 --output outputs/iql_v1 --tracker trackio
```

| Flag | Default | Description |
|------|---------|-------------|
| `--demo-dataset` | required | LeRobot dataset path (must have `next.reward` column) |
| `--output` | `outputs/iql_v1` | Output directory |
| `--num-steps` | 50000 | Training steps |
| `--batch-size` | 256 | Batch size (sampled from offline buffer) |
| `--expectile` | 0.7 | τ for V-loss — upper quantile approximation |
| `--beta` | 3.0 | Advantage temperature for actor loss |
| `--discount` | 0.99 | γ for TD target |
| `--actor-lr` | 3e-4 | Actor optimizer LR |
| `--critic-lr` | 3e-4 | Critic ensemble optimizer LR |
| `--value-lr` | 3e-4 | V-network optimizer LR |
| `--image-size` | 128 | Resize images before storing in replay buffer |
| `--success-bonus` | 10.0 | One-time bonus added to first successful transition per episode |
| `--max-episodes` | None | Limit number of episodes loaded from dataset |
| `--num-workers` | 4 | DataLoader workers for video decoding |
| `--vip-normalize` | false | Normalize VIP rewards to ~[-1, 0] via p5 percentile (use if `train_sac_composite_grpc.py --vip-normalize`) |
| `--seed` | 42 | Random seed |
| `--log-freq` | 100 | Steps between metric logs |
| `--save-freq` | 10000 | Steps between checkpoint saves |
| `--tracker` | `none` | `trackio` / `wandb` / `none` |
| `--tracker-project` | `so101-iql` | Tracker project name |
| `--system-stats` | false | Log CPU/RAM metrics (trackio only) |

---

## Architecture

Uses the same `SACPolicy` as `train_sac_composite_grpc.py`:
- **Encoder:** ResNet-10 (`helper2424/resnet10`), frozen, shared between actor and critic
- **Actor:** Gaussian policy head
- **Critic:** 2-critic ensemble with EMA target network

Additionally trains a standalone **ValueNetwork** (V(s)) that is only used during pretraining:
```
encoder_features → Linear(encoder_dim, 256) → SiLU → Linear(256, 256) → SiLU → Linear(256, 1)
```
The V-network is saved alongside the SAC policy checkpoint (`v_network.pt`) but is not used during online SAC training.

**Image normalization:** ImageNet mean/std applied inside SACPolicy preprocessing.
**State/action normalization:** env space `[-100, 100]` (joints) / `[0, 100]` (gripper) → policy space `[-1, 1]`.

---

## Dataset Requirements

The dataset must have a `next.reward` column. Use `prepare_reward_dataset.py` to add it:

```bash
# VIP rewards
python scripts/train/prepare_reward_dataset.py \
    --dataset data/recordings/figure_shape_placement_v4 \
    --output  data/recordings/figure_shape_placement_v5_vip \
    --reward-type vip

# Then resize to 128×128 for IQL pretraining
python scripts/train/prepare_reward_dataset.py \
    --dataset data/recordings/figure_shape_placement_v5_vip \
    --output  data/recordings/figure_shape_placement_v5_vip_128 \
    --image-size 128
```

---

## VIP Reward Normalization

When using VIP rewards (typical range ~[-20, -1]), use `--vip-normalize` to map to ~[-1, 0]:

```bash
python scripts/train/train_iql_pretrain.py \
    --demo-dataset data/recordings/figure_shape_placement_v5_vip_128 \
    --vip-normalize \
    --num-steps 50000 --output outputs/iql_vip_v1 --tracker trackio
```

The scale is computed as `max(-p5(rewards), 1.0)`. Rewards are divided by this scale and clipped to `[-2.0, 0.0]`. Use the same `--vip-normalize` flag in `train_sac_composite_grpc.py --resume` to ensure consistent reward scaling.

---

## Output

```
outputs/iql_v1/
├── train_config.json
├── step_10000/
│   ├── pretrained_model/        # SACPolicy (SAC-compatible checkpoint)
│   │   ├── config.json
│   │   └── model.safetensors
│   └── v_network.pt             # V-network weights (not used in online SAC)
├── step_20000/
├── ...
└── final/
    ├── pretrained_model/
    └── v_network.pt
```

Pass `final/pretrained_model` (or any step checkpoint) to `train_sac_composite_grpc.py --resume`.

---

## Notes

- **Dataset loading:** the entire dataset is loaded into a `ReplayBuffer` in CPU memory before training. For large datasets this can take several minutes and significant RAM.
- **Target network:** updated via EMA (`policy.update_target_networks()`) every step.
- **Grad clipping:** applied to actor and critic using `policy.config.grad_clip_norm`.
- **Actor optimizer:** only updates non-encoder parameters (encoder is frozen and shared).
- **Episode boundaries:** `done` flag set when `episode_index[i] != episode_index[i+1]`. Last frame of each episode: `V(s') * (1 - done) = 0`.
- **`success_bonus`:** added once per episode on the first transition where `reward > 0.5`. Mimics the online SAC terminal bonus.
