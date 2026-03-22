# Online RL

Online fine-tuning of SmolVLA: Learned Noise Sampler, Flow-Noise PPO.

All training runs in **Isaac Lab env** (online RL) or **lerobot-env** (offline sampler training).
Eval in **Isaac Lab env**.

Results for easy task: [results/easy_task.md](../results/easy_task.md).
Results for medium task: [results/medium_task.md](../results/medium_task.md).

---

## Background: flow matching noise and x_0

SmolVLA uses flow matching for action generation. Inference starts from `x_0 ~ N(0, I)` and runs an ODE to produce an action chunk. The starting point `x_0` matters — the ODE does not forget it.

The **Learned Noise Sampler** replaces N(0,I) with a state-conditional MLP: N(mu(obs), sigma(obs)).

---

## Learned Noise Sampler

State-conditional MLP replaces N(0,I) for x_0. Architecture:

```
VIPBackbone(frozen ResNet50) → (2048,) + state(6) → MLP(2054→256→12) → mu(6), log_sigma(6)
```

~529k trainable params. Zero initialization → starts as N(0,1) = baseline. VLA is fully frozen.

### Key sampler parameters

**`--noise-dims`** controls how many dimensions of x_0 the sampler learns to predict. SmolVLA's full noise tensor is `(50 tokens × 32 dims)`. The sampler replaces a subset with learned N(mu(obs), sigma(obs)):

| Value | What it controls | MLP params | Notes |
|-------|-----------------|------------|-------|
| **6** | First 6 dims of token 0 (joint actions) | ~34k | Minimal, fastest training |
| **32** | All 32 dims of token 0 (full first action token) | ~67k | **Recommended default** |
| **1600** | All 50×32 dims (entire x_0) | ~5.3M | Maximum expressiveness, slower |

Remaining dims stay N(0,I). Start with **32** — it controls the full first token while keeping training fast.

**`--n-action-steps`** — how many steps from the action chunk the policy executes before re-querying. Must match eval: if you eval with `--n-action-steps 50`, train with `--n-action-steps 50`. This also sets the default for `--offline-chunk-len` (how many actions are included in the MSE loss).

**`--offline-chunk-len`** — number of predicted actions to compare against expert in MSE loss. Defaults to `--n-action-steps`. Only the first N actions matter at inference, so supervising beyond that adds noise.

**`--batch-size`** — limited by GPU memory (ODE backprop is memory-intensive). Benchmarks on RTX 5070 Ti 16GB (5000 steps, noise-dims=32, n-action-steps=50):

| Batch size | VRAM | Time |
|-----------|------|------|
| 4 | 4 GB | ~2h |
| 8 | 8 GB | ~3h50m |
| 16 | 12 GB | ~7h |

Batch 4 is the sweet spot for wall time. However, total data seen = batch_size × steps, so batch 16 × 1250 steps ≈ same wall time as batch 4 × 5000 (~2h) with more stable gradients. Try larger batch + fewer steps if val_loss is noisy.

### Offline training (backprop through ODE)

Train by minimizing MSE between the VLA output with the learned x_0 and expert actions:

```bash
act-lerobot  # activate lerobot-env

python scripts/train/train_noise_sampler_offline.py \
    --checkpoint outputs/smolvla_vlm_v1/checkpoints/030000/pretrained_model \
    --dataset data/recordings/my_task_v1 \
    --output outputs/learned_sampler_offline_v1 \
    --noise-dims 32 --batch-size 4 \
    --offline-steps 5000 --n-action-steps 15
```

### Online training (REINFORCE in sim)

```bash
act-isaac  # activate Isaac Lab env

python scripts/train/train_noise_sampler_online.py \
    --checkpoint outputs/smolvla_vlm_v1/checkpoints/030000/pretrained_model \
    --output outputs/learned_sampler_v1 \
    --noise-dims 32 \
    --episodes-per-batch 10 --total-batches 50 \
    --eval-freq 10 --eval-episodes 50 \
    --max-episode-steps 500 \
    --env figure_shape_placement_easy --no-domain-rand \
    --n-action-steps 15
```

Episode return = success + 0.1 × speed_bonus. Loss uses REINFORCE with EMA baseline.

### Eval with sampler

```bash
act-isaac  # activate Isaac Lab env

python scripts/eval/eval_vla_policy.py \
    --checkpoint outputs/smolvla_vlm_v1/checkpoints/030000/pretrained_model \
    --noise-prior outputs/learned_sampler_v1/noise_prior.pt \
    --env figure_shape_placement_easy \
    --episodes 50 --max-steps 500 --no-domain-rand \
    --n-action-steps 15
```

`--noise-prior` accepts a learned sampler checkpoint (`noise_prior.pt`).

### Easy task results (offline sampler)

| Variant | Success rate |
|---------|-------------|
| BC baseline | 70–76% |
| BC + sampler (32d) | 80% |
| IQL + sampler (sampler trained on BC) | 82% |
| IQL only (#8) | 86–88% |

The sampler gives a marginal gain over BC (+4–10%), but hurts when combined with IQL
(82% vs 86–88%). Likely reason: the sampler was trained on the BC velocity field — after IQL
fine-tuning the field changes, so the sampler is sub-optimal for IQL.

> On the easy task the BC baseline is already strong, so the sampler has little room to improve.
> It is likely more useful on harder tasks with more diverse object positions.

---

## Flow-Noise PPO

Online PPO fine-tuning of SmolVLA. Adds a stochastic noise head to the denoising ODE, enabling tractable log π(a|s) for PPO. Only 3 components are trained:

| Component | Params |
|-----------|--------|
| `action_out_proj` | ~23k |
| `noise_log_std_head` | ~23k |
| Value MLP | ~593k |
| **Total** | **~639k** |

VLA backbone and vision encoder are frozen.

### Architecture: gRPC two-process split

The original `train_flow_noise_ppo.py` runs everything in one process but requires a `sys.path` hack for lerobot imports — breaks with lerobot 0.5.1 (Python 3.12 vs Isaac Sim's Python 3.11).

`train_flow_noise_ppo_grpc.py` splits into two processes:

| Process | Env | Role |
|---------|-----|------|
| Client (`train_flow_noise_ppo_grpc.py`) | isaaclab-env (3.11) | Isaac Sim env only |
| Server (`ppo_server.py`) | lerobot-env (3.12) | SmolVLA + VIP + PPO update |

Trajectory data (~3 GB/rollout) stays on server. Only obs (~1.8 MB) and actions cross the wire per step. `--auto-server` starts the server automatically.

### Command (gRPC version)

Two variants depending on whether you have an IQL checkpoint:

**Variant A — BC + sampler, no IQL warm start (recommended)**

Use when IQL critics were trained on a different VIP reward distribution than PPO (e.g. IQL used full-dataset goals, PPO uses `--vip-use-labeled`). IQL value scale (~-634) conflicts with normalized PPO rewards → huge v_loss. Start value head from scratch instead.

```bash
act-isaac  # activate Isaac Lab env

python scripts/train/train_flow_noise_ppo_grpc.py \
    --checkpoint outputs/smolvla_bc/checkpoints/040000/pretrained_model \
    --noise-prior outputs/learned_sampler/noise_prior.pt \
    --goal-dataset data/recordings/my_task_v1 \
    --vip-use-labeled \
    --env figure_shape_placement \
    --auto-server \
    --total-updates 200 --rollout-steps 256 \
    --normalize-rewards \
    --value-lr 3e-5 \
    --warmup-value 30 \
    --eval-freq 999 --save-freq 20 \
    --output outputs/flow_noise_ppo_v2 \
    --tracker none --headless
```

**Variant B — BC + IQL warm start (use only if IQL trained with same VIP config as PPO)**

IQL critics must be trained with the same `--vip-use-labeled` / goal dataset configuration as PPO, so value scales match.

```bash
act-isaac  # activate Isaac Lab env

python scripts/train/train_flow_noise_ppo_grpc.py \
    --checkpoint outputs/smolvla_bc/checkpoints/040000/pretrained_model \
    --iql-checkpoint outputs/iql_critics/final/critics.pt \
    --noise-prior outputs/learned_sampler/noise_prior.pt \
    --goal-dataset data/recordings/my_task_v1 \
    --vip-use-labeled \
    --env figure_shape_placement \
    --auto-server \
    --total-updates 200 --rollout-steps 256 \
    --normalize-rewards \
    --value-lr 3e-5 \
    --warmup-value 30 \
    --eval-freq 999 --save-freq 20 \
    --output outputs/flow_noise_ppo_v2 \
    --tracker none --headless
```

To resume from a checkpoint (e.g. after 200 updates, train 400 more):

```bash
act-isaac  # activate Isaac Lab env

python scripts/train/train_flow_noise_ppo_grpc.py \
    --checkpoint outputs/smolvla_bc/checkpoints/040000/pretrained_model \
    --noise-prior outputs/learned_sampler/noise_prior.pt \
    --goal-dataset data/recordings/my_task_v1 \
    --vip-use-labeled \
    --env figure_shape_placement \
    --auto-server \
    --total-updates 400 --rollout-steps 256 \
    --normalize-rewards \
    --value-lr 3e-5 \
    --warmup-value 0 \
    --eval-freq 999 --save-freq 40 \
    --output outputs/flow_noise_ppo_v2 \
    --resume outputs/flow_noise_ppo_v2/checkpoints/last \
    --tracker none --headless
```

Notes:
- `--resume` loads weights + optimizer state from `last/` checkpoint
- `--total-updates N` means N **additional** updates on top of resumed step
- `--warmup-value 0` — warmup already done, skip it on resume

Or start the server manually (useful for debugging):

```bash
# Terminal 1 (lerobot-env):
act-lerobot
python scripts/train/ppo_server.py --port 8081

# Terminal 2 (isaaclab-env):
act-isaac
python scripts/train/train_flow_noise_ppo_grpc.py \
    --checkpoint ... --goal-dataset ... \
    --server-port 8081 \
    --output outputs/flow_noise_ppo_v1 --headless
```

### Easy task results (original single-process script)

#### Exp A: IQL checkpoint → PPO

| Checkpoint | Success rate |
|------------|-------------|
| update_50 | 64% |
| update_100 | 64% |
| update_150 | 68% |
| **final (200)** | **74%** |

PPO degraded IQL (86–88% → 64%) but recovers to 74% by update_200. Value head did not converge in 200 updates (v_loss 25k→19k). Not recommended as a starting point for PPO.

#### Exp B: BC + learned sampler → PPO (recommended)

| Checkpoint | Success rate |
|------------|-------------|
| **update_50** | **90%** |
| update_100 | 82% |
| update_150 | 78% |
| final (200) | 86% |

BC + sampler + PPO (update_50) = **90%** — best result on the easy task benchmark.

Degradation after update_50 is due to the value head not converging (v_loss 23k→17k). Use `update_50` checkpoint.

### Eval for PPO

Single checkpoint:

```bash
act-isaac  # activate Isaac Lab env

python scripts/eval/eval_vla_policy.py \
    --checkpoint outputs/flow_noise_ppo_v1/<ckpt>/pretrained_model \
    --noise-prior outputs/learned_sampler/noise_prior.pt \
    --env figure_shape_placement \
    --episodes 100 --max-steps 600 \
    --n-action-steps 50 --seed 1988042740 \
    --auto-server --headless
```

Sweep all PPO checkpoints with `sweep_vla_eval.py` (PPO saves `NNNNNN/pretrained_model` format, compatible with sweep script):

```bash
act-isaac  # activate Isaac Lab env

# Quick screening sweep (50 ep) across all checkpoints
python scripts/eval/sweep_vla_eval.py \
    --checkpoint outputs/flow_noise_ppo_v2/checkpoints \
    --all --last \
    --noise-prior outputs/learned_sampler/noise_prior.pt \
    --env figure_shape_placement \
    --episodes 50 --max-steps 600 \
    --n-action-steps 50 --seed 1988042740 \
    --no-domain-rand

# Confirmation run (100 ep) on top candidates from screening
python scripts/eval/sweep_vla_eval.py \
    --checkpoint outputs/flow_noise_ppo_v2/checkpoints \
    --checkpoints 60 160 180 --last \
    --noise-prior outputs/learned_sampler/noise_prior.pt \
    --env figure_shape_placement \
    --episodes 100 --max-steps 600 \
    --n-action-steps 50 --seed 1988042740 \
    --no-domain-rand
```

Flags:
- `--all` — evaluate all `NNNNNN` checkpoints
- `--last` — also evaluate `last/` checkpoint
- `--checkpoints 60 160 180` — evaluate specific step numbers only

### Key PPO flags

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | required | SmolVLA checkpoint to fine-tune |
| `--iql-checkpoint` | — | IQL critics for value head warm start |
| `--noise-prior` | — | Learned sampler checkpoint (optional) |
| `--goal-dataset` | required | Dataset for VIP reward goal frames |
| `--vip-use-labeled` | false | Use only success frames (`next.reward > 0.5`) as VIP goal. Goal embeddings are computed once and cached to `<goal-dataset>/vip_goal_cache_<hash>.pt` — reused on subsequent runs. |
| `--total-updates` | 1000 | PPO update steps |
| `--rollout-steps` | 256 | Environment steps per rollout |
| `--reeval-batch-size` | 1 | Batch size for suffix re-eval (higher = faster, more VRAM) |
| `--warmup-value` | 0 | First N updates: train only value head, freeze actor |
| `--normalize-rewards` | false | Normalize rewards with running mean/std |
| `--auto-server` | false | Auto-start ppo_server.py in lerobot-env |
| `--server-port` | 8081 | gRPC server port |
| `--tracker` | trackio | `trackio`, `wandb`, or `none` |

---

## Summary: online RL on easy task

| Method | Starting point | Best SR | Best checkpoint |
|--------|---------------|---------|----------------|
| BC + learned sampler | BC 30k | 80% | offline best |
| IQL + sampler (BC sampler) | IQL 10k | 82% | — |
| IQL + PPO | IQL 10k | 74% | update_200 |
| **BC + sampler + PPO** | BC 30k | **90%** | **update_50** |

Key findings:
- **PPO on top of IQL degrades first** — IQL policy is already well-tuned; PPO with an unconverged value head overshoots
- **PPO on top of BC + sampler works** — weaker starting point gives PPO more room to improve
- **Value head convergence is the bottleneck** — v_loss 23k→17k over 200 updates; consider `--warmup-value` for future runs
- **Best pipeline**: BC (variant B, 30k) → offline sampler → PPO (200 updates, take update_50)

---

## Summary: online RL on medium task (figure_shape_placement, 4× spawn area)

> **2026-03-22:** Results corrected after fixing gRPC eval bug ([`7cdae2d`](https://github.com/MSSergeev/so101-lab/commit/7cdae2d)) — task string was not passed to SmolVLA. Previous results: BC 22%, IQL 32%, PPO 24%.

BC baseline: **45%** (600 ep, no DR). IQL: **48%**.

| Method | SR | vs BC | Notes |
|--------|----|-------|-------|
| BC + Learned Sampler | ~~20%~~ | — | Invalidated (gRPC task bug), not re-evaluated |
| BC + sampler + PPO v2 (600 updates) | 30% | −15 p.p. | Trained with DR, eval without DR — see note |

Key findings:
- **IQL is faster, not clearly better on SR** — IQL 48% vs BC 45% is not statistically significant (z=1.04, p≈0.30), but IQL solves episodes significantly faster (278 vs 300 steps, p<0.001). IQL was also consistently slightly ahead across prior evaluations.
- **PPO below BC on no-DR eval** — PPO was trained with domain randomization enabled, while eval ran without DR. The policy may perform better under DR conditions. This needs testing.
- **IQL warm start mismatch:** using IQL checkpoint with `--vip-use-labeled` PPO causes v_loss=32k+ (scale conflict). Fix: omit `--iql-checkpoint` and use `--normalize-rewards --warmup-value 30`
