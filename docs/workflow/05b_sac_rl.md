# SAC Online RL

SAC (Soft Actor-Critic) training for the manipulation task.

Script: **`train_sac_composite_grpc.py`** — gRPC two-process split (isaaclab-env + lerobot-env). Works with lerobot 0.5.1.

Results: [results/easy_task.md](../results/easy_task.md).

---

## Overview

SAC is an off-policy, maximum-entropy RL algorithm. It learns from both online env interaction and offline demo data simultaneously (RLPD pattern). Unlike the VLA fine-tuning approach in [05_online_rl.md](05_online_rl.md), SAC trains a compact ResNet-10 + MLP actor from scratch — not a pretrained language-vision model.

**When to use SAC vs PPO on VLA:**
- SAC: faster per-step training, works without a pretrained VLA, composable reward from multiple sources
- PPO on VLA: leverages language-vision pretraining, better on visually complex tasks

---

## SAC Architecture

### Encoder (shared between actor and critic)

```
observation.images.top   (3×H×W) ─┐
                                   ├─→ ResNet-10 (frozen, ImageNet pretrained)
observation.images.wrist (3×H×W) ─┘       ↓ feature maps per camera
                                   SpatialLearnedEmbeddings → Linear → LayerNorm → Tanh
                                   → (256,) per camera

observation.state (6,) ──────────→ Linear(6, 256) → LayerNorm → Tanh → (256,)

concat → (768,)  ← encoder output
```

ResNet-10 is **frozen** throughout training. The SpatialLearnedEmbeddings and post-encoders are trainable and updated through critic gradients only.

Why freeze ResNet-10: the encoder is shared between actor and critic. The actor detaches encoder output (its gradients don't flow back), while the critic does not. If CNN weights were updated via critic gradients, actor would see a shifting feature space without being able to adapt → instability. Freezing also enables caching CNN features within each training step.

### Actor

```
encoder output (768,) → MLP [256, 256, SiLU+LayerNorm]
  → mean_layer → μ (6,)
  → std_layer  → σ (6,), clamped [1e-5, 10]
→ TanhNormal(μ, σ) → action ∈ (-1, 1)
```

Outputs a distribution, not a single action. Actions are re-normalized to [-100, 100] / [0, 100] before env.step(). No action chunking — one action per step.

### Critics (clipped double-Q)

```
encoder output (768,) + action (6,) → concat (774,)
  → MLP [256, 256] → Linear → Q (scalar)
  × 2 independent heads (Q1, Q2)
```

Two Q-networks trained in parallel. TD-target uses `min(Q1_target, Q2_target)` to prevent overestimation. Target network = slow EMA copy: `θ_target ← 0.005·θ + 0.995·θ_target`.

### Temperature (α)

Learned scalar controlling entropy weight in the objective:
```
objective = reward + α · entropy
```
Auto-tuned to keep actor entropy near target `H = -dim(action)/2 = -3.0`. High α → more exploration; low α → more exploitation.

---

## Training Loop

One env step = full update cycle:

```
1. env.step(action) → transition (s, a, r, s', done) → online buffer
2. sample batch (128 online + 128 offline) → critic update ×2  (UTD=2)
3. sample batch (128 online + 128 offline) → critic + actor + temperature update
```

UTD=2 means the critic is updated twice per env step on independent random batches. Actor and temperature update once per step.

### TD-target

```
a' ~ π(·|s')
Q_target = min(Q1_target(s', a'), Q2_target(s', a')) - α·log π(a'|s')
y = r + γ·(1 - done)·Q_target
```

`done=True` when episode terminates (cube falls off table) or truncates (15 s timeout). Success does **not** terminate the episode — the policy continues until timeout.

### Losses

**Critic:** `MSE(Q_i(s,a), y)` for each head, one `backward()`.

**Actor:** `mean(α·log π(a|s) - min(Q1, Q2))` — maximize Q while keeping entropy.

**Temperature:** `mean(-α·(log π(a|s) + H_target))` — auto-adjust α.

---

## Reward

Four reward modes:

| Mode | Signal |
|------|--------|
| `sim_only` | Dense sim shaping (distance + milestones + penalties) |
| `classifier_only` | Binary 0/1 from image classifier |
| `composite` | sim shaping + classifier + optional VIP |
| `vip_only` | VIP embedding distance to goal |

Sim rewards are dt-scaled (continuous). Classifier and success bonus are not (discrete events).

---

## Full Workflow

### Step 1 — Prepare offline dataset

```bash
act-lerobot  # activate lerobot-env

python scripts/train/prepare_reward_dataset.py \
    --dataset     data/recordings/figure_shape_placement_v4 \
    --output      data/recordings/figure_shape_placement_v4_clf_128 \
    --reward-model outputs/reward_classifier_v2/best \
    --image-size  128
```

Dataset must have `next.reward`. See [reference/prepare_reward_dataset.md](../reference/prepare_reward_dataset.md).

### Step 2 — (Optional) IQL pretraining

Solves the asymmetric init problem: standard BC pretrain gives a good actor but random critic. When online SAC starts, random Q-values pull the actor off the BC solution within hundreds of steps.

IQL trains both actor and critic offline without OOD-action queries (uses V(s') instead of Q(s', a'~π)):

```
# Three losses per step:
diff = Q_target_min(s,a) - V(s)
loss_v     = mean(τ·diff² if diff>0 else (1-τ)·diff²)   # expectile regression, τ=0.7
loss_q     = MSE(Q_i(s,a), r + γ·V(s'))                  # TD with V bootstrap
loss_actor = mean(exp(β·advantage).clamp(100) · ‖tanh(μ) - a‖²)  # advantage-weighted BC
```

`τ=0.7` → V(s) converges to ~70th percentile of Q-values (optimistic but stable).
`β=3.0` → actions with positive advantage are upweighted; negative advantage is suppressed.

```bash
act-lerobot  # activate lerobot-env

python scripts/train/train_iql_pretrain.py \
    --demo-dataset data/recordings/figure_shape_placement_v4_clf_128 \
    --num-steps 50000 \
    --output outputs/iql_v1 \
    --tracker trackio
```

Outputs a standard SAC checkpoint compatible with `--resume`. V-network saved separately as `v_network.pt` (not needed for online SAC).

**IQL CLI flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--demo-dataset` | required | Dataset with `next.reward` |
| `--num-steps` | 50000 | Training steps |
| `--expectile` | 0.7 | τ for V-loss (0.5=mean, 0.7=optimistic) |
| `--beta` | 3.0 | Advantage temperature (0=BC, 3=selective, ∞=best-only) |
| `--discount` | 0.99 | γ |
| `--success-bonus` | 10.0 | One-time terminal bonus for success frames |
| `--image-size` | 128 | Dataset image size |
| `--vip-normalize` | false | Normalize VIP rewards to [-1, 0] (match online SAC flag) |

**IQL metrics to watch:**

| Metric | Expected behavior |
|--------|-------------------|
| `train/loss_v` | Decreasing |
| `train/loss_q` | Decreasing |
| `train/loss_actor` | Decreasing |
| `train/v_mean` | Increasing over time |
| `train/weights_max` | If often = 100 (clamp hit), β is too high |

### Step 3 — Online SAC training

Uses `train_sac_composite_grpc.py` (gRPC split). `--auto-server` starts `sac_server.py` in lerobot-env automatically.

```bash
act-isaac  # activate Isaac Lab env

# From IQL pretrain (recommended)
python scripts/train/train_sac_composite_grpc.py \
    --resume outputs/iql_v1/final/pretrained_model \
    --reward-mode sim_only \
    --no-offline \
    --num-steps 100000 --output outputs/sac_iql_v1 \
    --auto-server --no-randomize-light --torch-compile --headless

# From scratch (sim_only, no pretrain)
python scripts/train/train_sac_composite_grpc.py \
    --reward-mode sim_only \
    --num-steps 100000 --output outputs/sac_v1 \
    --auto-server --headless

# RLPD: online + offline demo data (50/50 per batch)
python scripts/train/train_sac_composite_grpc.py \
    --resume outputs/iql_v1/final/pretrained_model \
    --reward-mode composite \
    --reward-model outputs/reward_classifier_v2/best \
    --demo-dataset data/recordings/figure_shape_placement_v4_clf_128 \
    --num-steps 100000 --output outputs/sac_rlpd_v1 \
    --auto-server --image-size 128 --no-randomize-light --headless
```

Or start the server manually (useful for debugging):

```bash
# Terminal 1 (lerobot-env):
act-lerobot
python scripts/train/sac_server.py --port 8082

# Terminal 2 (isaaclab-env):
act-isaac
python scripts/train/train_sac_composite_grpc.py \
    --reward-mode sim_only \
    --server-port 8082 \
    --num-steps 100000 --output outputs/sac_v1 --headless
```

**Speed:** ~4 step/s (training), ~35 step/s (warmup). `--torch-compile` gives ~30% speedup after JIT warmup (~500 steps).

**Key hyperparameters:**
- `--discount 0.999` recommended for sparse reward (success only at end of episode)
- `--no-randomize-light` if classifier trained on fixed lighting
- `--warmup-steps 500` random actions fill online buffer before learning starts

### Step 4 — Evaluate

```bash
python scripts/eval/eval_sac_policy.py \
    --checkpoint outputs/sac_iql_v1/best/pretrained_model \
    --reward-model outputs/reward_classifier_v2/best \
    --episodes 20 \
    --image-size 128
```

---

## HIL-SERL (Human-in-the-Loop)

Operator takeover during online training via SO-101 leader arm. Provides high-quality transitions when the policy is stuck.

### How it works

1. `--hil` enables mode: connects leader arm, creates intervention buffer
2. Toggle takeover ON/OFF:
   - **Headless + preview** (recommended): Space in `hil_viewer.py`
   - **GUI mode** (`--gui`): Enter in Isaac Sim window
   - Both work in parallel
3. Reset episode: X in viewer or Isaac Sim window (works regardless of takeover state)
4. During takeover: action comes from leader arm current position (instead of policy/random)
5. Intervention transitions go into **both** online buffer and intervention buffer
6. Training sampling is 3-way: online + offline + intervention (equal thirds)

### Buffer sampling

| Active buffers | Sampling split (batch=256) |
|----------------|---------------------------|
| online + offline + intervention | 85 + 85 + 86 |
| online + offline | 128 + 128 |
| online + intervention | 128 + 128 |
| online only | 256 |

Intervention buffer doubles the weight of human transitions: they appear in both online and intervention buckets.

### Usage

```bash
# Headless + preview (recommended — best performance)
python scripts/train/train_sac_composite_grpc.py \
    --reward-mode sim_only \
    --hil --hil-port /dev/ttyACM2 \
    --num-steps 50000 --output outputs/sac_hil_v1 \
    --auto-server --image-size 128

# GUI + preview (both toggle methods work)
python scripts/train/train_sac_composite_grpc.py \
    --reward-mode sim_only \
    --hil --gui \
    --num-steps 50000 --output outputs/sac_hil_v1 \
    --auto-server --image-size 128

# Custom calibration file
python scripts/train/train_sac_composite_grpc.py \
    --hil --hil-port /dev/ttyACM2 \
    --hil-calibration path/to/leader.json \
    --intervention-capacity 100000 \
    --num-steps 50000 --output outputs/sac_hil_v1 \
    --auto-server
```

### HIL flags

| Flag | Default | Description |
|------|---------|-------------|
| `--hil` | false | Enable HIL mode |
| `--hil-port` | `/dev/ttyACM0` | Leader arm serial port |
| `--hil-calibration` | auto | Calibration JSON path (default: `devices/lerobot/.cache/so101_leader.json`) |
| `--intervention-capacity` | 20000 | Intervention buffer size |
| `--preview` | auto | Camera preview (default: on when `--hil`) |
| `--no-preview` | false | Disable preview |

### Notes

- `--hil` requires at least `--gui` or preview (need a toggle mechanism)
- Intervention buffer is **not saved** to checkpoint — starts empty after resume
- Leader arm must be calibrated; see [reference/teleop_devices.md](../reference/teleop_devices.md)

---

## Checkpoint Format

```
outputs/sac_v1/
├── train_config.json
├── best/                       # Best by eval success rate
│   ├── pretrained_model/
│   │   ├── config.json
│   │   └── model.safetensors   # ~33 MB (actor, critics, post-encoders, state encoder)
│   └── training_state.pt       # ~121 MB (optimizer states + counters)
├── step_10000/
└── final/
```

ResNet-10 encoder (~20 MB) is not saved — frozen, reloaded from HuggingFace on startup.

`training_state.pt`: optimizer state_dicts (actor/critic/temperature) + `opt_step`, `ep_count`, `best_sr`. Critic optimizer dominates (~117 MB).

### Resuming

```bash
python scripts/train/train_sac_composite_grpc.py \
    --resume outputs/sac_v1/final/pretrained_model \
    --num-steps 50000 --output outputs/sac_v1 \
    --auto-server --headless
```

Step numbering continues. Online buffer is not persisted — warmup re-runs after resume.

---

## Checklist

1. Dataset has `next.reward` column (`prepare_reward_dataset.py`)
2. `--image-size` matches dataset image size
3. `--no-randomize-light` if classifier trained on fixed lighting
4. Smoke test: `train_sac_composite_grpc.py --auto-server --num-steps 100 --warmup-steps 10 --no-offline --headless`
5. Critic loss decreasing after ~1000 steps
6. Temperature not collapsing to 0 or diverging
7. Success rate > 0 after 10–50k steps
