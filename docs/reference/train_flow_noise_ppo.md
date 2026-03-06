# train_flow_noise_ppo.md

PPO online RL fine-tuning of SmolVLA via stochastic flow-noise head. Runs in Isaac Lab (`il`).

---

Runs in **Isaac Lab env** (`act-isaac`). See [setup](../workflow/00_setup.md).

## Overview

Fine-tunes SmolVLA in simulation using PPO with VIP dense reward. The challenge: SmolVLA uses
flow matching (deterministic ODE) which has no tractable `log π(a|s)`. Solution: add a learnable
`noise_log_std_head` to each ODE denoising step, making velocity sampling stochastic.

**Trainable parameters (~639k total):**
- `action_out_proj` (Linear 720→32): 23,072 — was already in SmolVLA, now also receives grads
- `noise_log_std_head` (Linear 720→32): 23,072 — new head added to each denoising step
- `value_mlp` (2054→256→256→1): ~593k — VIP-based value network

Everything else (SmolVLA backbone ~450M, VIP ResNet50 ~25M) is frozen.

**Results:**
- v1: 100 updates, rollout=256, IQL warm start → reward -27 → -24
- v2/v3: 200+ updates, rollout=500, random init → see `docs/results/easy_task.md`

**Key insight:** Use `--normalize-rewards --warmup-value 10` (random init value head).
IQL warm start for value head does not work well because IQL uses normalized rewards (~[-1,0])
while PPO uses raw VIP rewards (~-22/step); the mismatch wastes ~50 updates on rescaling.

---

## Architecture

### FlowNoiseSmolVLA

```
SmolVLA BC checkpoint (frozen ~450M)
  ├── SigLIP vision encoder (frozen)
  ├── SmolVLM2-500M text model (frozen)
  ├── Expert transformer (frozen) → suffix_out (720-dim)
  ├── action_out_proj: Linear(720 → 32)   ← TRAINABLE
  └── [NEW] noise_log_std_head: Linear(720 → 32)  ← TRAINABLE

Value head:
  ├── VIPBackbone: ResNet50 Ego4D (frozen, 2× 1024 = 2048-dim)
  └── value_mlp: Linear(2054→256)→LN→SiLU→Linear(256→256)→LN→SiLU→Linear(256→1)  ← TRAINABLE
```

### Stochastic denoising step

At each of 10 ODE steps:
```
suffix_out = frozen_expert_transformer(x_t, time, KV_cache)   # (B, 50, 720)
v_mean = action_out_proj(suffix_out)                           # trainable, (B, 50, 32)
v_log_std = noise_log_std_head(suffix_out)                     # trainable, (B, 50, 32)
v_std = exp(clamp(v_log_std, log_std_min, log_std_max))
eps ~ N(0, I)
v_t = v_mean + v_std * eps                                     # stochastic velocity
log_prob += log N(v_t | v_mean, v_std)                         # accumulated
```

**Log-prob slice:** only first chunk step and first 6 action dims are used (6 × 10 = 60 terms).
Full sum over 50 × 32 × 10 = 16000 terms causes ratio explosion.

**Noise bounds:**

| Mode | log_std range | std range | Behavior |
|------|--------------|-----------|----------|
| Train | [-5, 2] | [0.007, 7.4] | Stochastic for exploration |
| Eval | [-20, -2] | [2e-9, 0.14] | Near-deterministic |

### PPO re-evaluation

During rollout: save per-step `x_t` and `v_sampled` (10 ODE steps, CPU tensors ~6 MB/obs).
During re-eval: feed saved `x_t` → frozen transformer → updated heads → new `v_mean/v_std`
→ `log N(saved_v_sampled | new_v_mean, new_v_std)`. No ODE re-run needed.

### One PPO update

```
ROLLOUT (rollout-steps env steps, ~68s at 500 steps)
  for step:
    obs → SmolVLA forward (prefix + 10 ODE steps) → action, log_prob, trajectory
    action → env.step() → obs_next, reward (VIP)
    cache: obs, trajectory, vip_emb, state, log_prob, value, reward, done

GAE
  rewards + values + dones → advantages (normalized), returns

PPO UPDATE (update-epochs × mini-batches, ~19s)
  for epoch:
    for batch:
      reeval_log_prob(obs, saved_trajectory) → new log_prob
      ratio = exp(new_log_prob - old_log_prob)
      actor_loss = -min(ratio × adv, clip(ratio, 1±clip_ratio) × adv)
      value_loss = (V(s) - returns)²
      backward + step
```

---

## CLI

```bash
act-isaac  # activate Isaac Lab env
cd /path/to/so101-lab

# Recommended run (~88 sec/update)
python scripts/train/train_flow_noise_ppo.py \
    --checkpoint outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model \
    --goal-dataset data/recordings/figure_shape_placement_v5 \
    --vip-use-labeled \
    --total-updates 200 --rollout-steps 500 \
    --reeval-batch-size 16 \
    --normalize-rewards --warmup-value 10 \
    --output outputs/flow_noise_ppo_v3 \
    --tracker trackio --headless

# Resume
python scripts/train/train_flow_noise_ppo.py \
    --checkpoint outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model \
    --goal-dataset data/recordings/figure_shape_placement_v5 \
    --vip-use-labeled \
    --total-updates 100 --rollout-steps 500 \
    --resume outputs/flow_noise_ppo_v3/final \
    --output outputs/flow_noise_ppo_v3 \
    --tracker trackio --headless

# Smoke test (2 updates)
python scripts/train/train_flow_noise_ppo.py \
    --checkpoint outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model \
    --goal-dataset data/recordings/figure_shape_placement_v5 \
    --vip-use-labeled \
    --total-updates 2 --rollout-steps 16 \
    --output outputs/flow_noise_ppo_smoke --headless

# With noise prior (learned sampler)
python scripts/train/train_flow_noise_ppo.py \
    --checkpoint outputs/easy_smolvla_iql_v1/checkpoints/010000/pretrained_model \
    --noise-prior outputs/easy_learned_sampler_offline_v1/noise_prior.pt \
    --goal-dataset data/recordings/easy_task_v1 \
    --vip-use-labeled \
    --total-updates 200 --rollout-steps 500 \
    --reeval-batch-size 16 \
    --output outputs/flow_noise_ppo_sampler_v1 --tracker trackio --headless

# Eval (standard script)
python scripts/eval/eval_vla_policy.py \
    --checkpoint outputs/flow_noise_ppo_v3/best/pretrained_model \
    --episodes 50 --headless
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | SmolVLA BC checkpoint (pretrained_model dir) |
| `--iql-checkpoint` | None | IQL critics for value head warm start (not recommended) |
| `--noise-prior` | None | Path to `noise_prior.pt` — learned sampler or fixed mu shift |
| `--resume` | None | Checkpoint dir to continue training from |
| `--goal-dataset` | required | LeRobot dataset for VIP goal embeddings |
| `--vip-use-labeled` | false | Use frames with `next.reward > 0.5` as VIP goals |
| `--vip-label-dataset` | None | Separate dataset with labels (default: = goal-dataset) |
| `--vip-goal-mode` | `mean` | `mean` — one embedding, `min` — closest of N |
| `--n-goal-frames` | 5 | Final frames per episode (if not `--vip-use-labeled`) |
| `--total-updates` | 1000 | PPO updates (when resuming: additional updates) |
| `--rollout-steps` | 256 | Env steps per rollout (500 = 1 episode) |
| `--update-epochs` | 4 | PPO epochs per update |
| `--batch-size` | 64 | Mini-batch size for PPO update |
| `--reeval-batch-size` | 1 | Suffix re-eval batch size (recommend 8–16) |
| `--clip-ratio` | 0.2 | PPO clip range ε |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE λ |
| `--actor-lr` | 3e-5 | LR for action_out_proj + noise_log_std_head |
| `--value-lr` | 1e-4 | LR for value_mlp |
| `--vip-weight` | 1.0 | VIP reward weight |
| `--sim-weight` | 0.0 | Sim reward weight |
| `--success-bonus` | 0.0 | Milestone placed bonus |
| `--warmup-value` | 0 | First N updates: train value head only (actor frozen) |
| `--normalize-rewards` | false | Running mean/std reward normalization (Welford) |
| `--gui` | false | Isaac Sim GUI |
| DR flags | — | `--no-domain-rand`, `--no-randomize-light`, `--no-randomize-physics`, `--no-camera-noise`, `--no-distractors` |

---

## Output

```
{output}/{name}/
├── flow_noise_ppo.pt          # overlay for resume training
│   ├── action_out_proj        # Linear(720→32) state_dict
│   ├── noise_log_std_head     # Linear(720→32) state_dict
│   ├── value_mlp              # Sequential state_dict
│   ├── optimizers             # actor + value optimizer states
│   └── metadata               # {update, best_sr}
└── pretrained_model/          # standalone SmolVLA for eval
    ├── config.json
    ├── model.safetensors      # with updated action_out_proj
    ├── policy_preprocessor.json
    └── policy_postprocessor.json
```

`pretrained_model/` is a complete SmolVLA checkpoint usable with `eval_vla_policy.py` directly.

Saves: every `--save-freq` updates, at `best/` whenever SR improves, at `final/`.

---

## Performance

Profile (RTX 5070 Ti, rollout=500, reeval-batch=16, KV cache on CPU):

| Section | Time | % |
|---------|------|---|
| Rollout (500 forwards + env.step) | 68s | 78% |
| Re-eval (4 epochs × 500 obs, batched suffix) | 19s | 21% |
| Backward + optimizer | 0.3s | <1% |
| **Total** | **~88s** | |

Rollout is the bottleneck (sequential: action depends on obs). KV cache + batched re-eval gives 3.2× speedup vs naive implementation:

| Config | Update time | Speedup |
|--------|------------|---------|
| Baseline | 280s | 1× |
| + KV cache | 208s | 1.35× |
| + KV cache + batch=8 | 94s | 3× |
| + KV cache + batch=16 | 88s | 3.2× |

**KV cache:** prefix (450M params) is computed once during rollout and stored on CPU
(6 MB/obs, 500 obs = 2.9 GB). Re-eval uses cached KV instead of recomputing prefix.
`--kv-cache-device cuda` moves cache to GPU (+2.9 GB VRAM, no speedup — not the bottleneck).

**Batched re-eval:** `reeval_log_prob_batched()` stacks N trajectories, runs one batched suffix
forward instead of N sequential calls. `--reeval-batch-size 16` recommended (GPU saturated at 8).

---

## VIP Goal Cache

First run: VIP encodes up to 1000 goal frames (~30s), saves to:
```
data/recordings/{dataset}/vip_goal_cache_{hash}.pt
```
Hash includes: image_key, use_labeled, label_path, n_goal_frames. Subsequent runs: instant load.

---

## Notes

**`--rollout-steps 500`** — one full episode (500 env steps). Shorter rollouts (256) give high
variance advantages because an episode may be split mid-trajectory across updates.

**`--noise-prior`** patches `model.sample_noise()` — both rollout and eval use the same x_0 the
sampler generates. Supports two formats: `LearnedNoiseSampler` (state-conditional) and
`NoisePrior` (fixed 6-dim mu shift). Format is auto-detected from checkpoint.

**Memory per update** (rollout=500):
- `rollout_obs`: 500 obs dicts (numpy), ~900 MB
- `rollout_trajectories`: 500 × (10 × `x_t` + `v_sampled`), ~6 MB
- `rollout_vip_embs`: 500 × 2048, ~4 MB
- Total: ~910 MB, recreated each update.
