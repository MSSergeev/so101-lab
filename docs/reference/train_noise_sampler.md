# train_noise_sampler.md

Learned Noise Sampler: state-conditional x_0 for SmolVLA flow matching.

Two training modes: offline (lerobot-env, `lr`) and online (Isaac Lab, `il`).

---

Runs in **Isaac Lab env** (`act-isaac`) and **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

SmolVLA flow matching starts from `x_0 ~ N(0, I)` (shape 1×50×32). The Learned Noise Sampler
replaces this with `x_0 ~ N(mu(s), sigma(s))` where mu and sigma are predicted by a small MLP
conditioned on VIP image embeddings + joint state.

Inspired by Green-VLA (arxiv 2602.00919): "a small separate actor network that generates noise
fed into the base model."

**Why state-conditional over fixed mu (CMA-ES):** fixed mu optimizes a single global shift,
which cannot adapt to the current cube/arm position. The learned sampler produces a different
x_0 bias per observation, steering the ODE towards better actions in a state-dependent way.

**CMA-ES conclusion:** tested on easy task (baseline 100%), CMA-ES returned mu=[0,...,0] —
baseline too strong for gradient signal. Fixed mu approach abandoned. Use LearnedNoiseSampler.

**`--noise-dims`** controls how many dims of x_0 the sampler generates:
- `6`: x_0[:, 0, :6] — original joint action dims of token 0 (~34k MLP params)
- `32`: x_0[:, 0, :32] — all dims of first token (~67k params)
- `1600`: x_0[:, :50, :32] — full noise tensor (~5.3M params)

Note: `action_in_proj` projects all 32 dims of each of 50 tokens, and transformer attention
mixes all tokens. Shifting only 6 dims is a heuristic — cross-talk exists but diagonal dominates
(measured in `measure_x0_sensitivity.py`).

---

## Architecture

```
obs → VIPBackbone (frozen ResNet50 Ego4D) → concat(top, wrist) → (2048,)
obs → state (6,) ────────────────────────────────────────────────────────┐
                                                                          ├→ MLP → mu(noise_dims)
                                                                          └→ MLP → log_sigma(noise_dims)
```

- VIP backbone: frozen, shared with critics (if loaded together)
- MLP hidden dim: 256 for noise_dims=6, 512 for noise_dims=32, 1024 for noise_dims=1600
- Output: `clamp(log_sigma, -2.0, 2.0)`
- Init: zeros on last layer → at start, sampler outputs N(0,1) = baseline SmolVLA behavior

Eval mode: sigma → 0 (log_sigma clamped to -20), output is deterministic mu(s).

---

## Files

| File | Description |
|------|-------------|
| `so101_lab/policies/rl/learned_noise_sampler.py` | `LearnedNoiseSampler`: VIP+state → (mu, sigma), `patch_model()`, `sample_x0()`, `log_prob()` |
| `scripts/train/train_noise_sampler_offline.py` | Offline training: backprop through ODE, lerobot-env |
| `scripts/train/train_noise_sampler_online.py` | Online training: REINFORCE in sim, Isaac Lab |
| `scripts/eval/eval_vla_policy.py` | Eval with `--noise-prior` (auto-detects format) |

Related:
- `so101_lab/policies/rl/noise_prior.py` — fixed mu shift (6-dim, no MLP, CMA-ES approach)
- `scripts/train/train_noise_prior.py` — human-in-the-loop fixed mu training ([reference/train_noise_prior.md](train_noise_prior.md))
- `scripts/train/train_noise_prior_cmaes.py` — CMA-ES fixed mu optimization (concluded: abandoned)

---

## Training modes

### Offline (lerobot-env, no sim)

Backprop through the frozen SmolVLA ODE on expert demonstrations:

1. LeRobotDataset → frame (images, state, expert_action)
2. `LearnedNoiseSampler(VIP+state)` → mu, log_sigma → sample x_0 (reparameterized)
3. x_0 → `model.sample_actions()` via frozen ODE → predicted action chunk
4. Expert actions normalized via dataset stats (MEAN_STD)
5. Loss = MSE(predicted, expert) over first `--offline-chunk-len` steps
6. Backprop: loss → Euler steps → x_0 → sampler MLP

VIP embeddings are cached to disk on first run. No simulator needed.
Checkpoint selected by best validation loss (held-out 100 frames).

```bash
act-lerobot  # activate lerobot-env
cd /path/to/so101-lab

python scripts/train/train_noise_sampler_offline.py \
    --checkpoint outputs/easy_smolvla_vlm_pretrained_v1/checkpoints/030000/pretrained_model \
    --dataset data/recordings/easy_task_v1 \
    --output outputs/easy_learned_sampler_offline_v1 \
    --noise-dims 32 --batch-size 4 \
    --offline-steps 5000 --offline-chunk-len 15 --n-action-steps 15
```

### Online (REINFORCE, Isaac Lab)

REINFORCE with episode-level sim reward:

1. Sampler(obs) → mu, sigma → sample x_0 → frozen VLA ODE → action chunk
2. Execute chunk in sim for `n_action_steps` steps
3. Episode return = success + speed_bonus
4. REINFORCE: `loss = -log_prob(x_0) × advantage`, EMA baseline for variance reduction
5. Update sampler weights

```bash
act-isaac  # activate Isaac Lab env
cd /path/to/so101-lab

python scripts/train/train_noise_sampler_online.py \
    --checkpoint outputs/easy_smolvla_vlm_pretrained_v1/checkpoints/030000/pretrained_model \
    --output outputs/easy_learned_sampler_online_v1 \
    --noise-dims 32 \
    --episodes-per-batch 10 --total-batches 50 \
    --eval-freq 10 --eval-episodes 50 \
    --max-episode-steps 500 \
    --env figure_shape_placement_easy --no-domain-rand \
    --n-action-steps 15
```

---

## CLI — offline

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | SmolVLA BC checkpoint (pretrained_model dir) |
| `--dataset` | required | LeRobot dataset path |
| `--noise-dims` | 6 | Noise dims: `6`, `32`, or `1600` |
| `--offline-steps` | 5000 | Training steps |
| `--offline-chunk-len` | None | Actions to compare (default: n_action_steps) |
| `--batch-size` | 4 | Batch size (limited by GPU memory for ODE backprop) |
| `--lr` | 1e-4 | Learning rate |
| `--n-action-steps` | None | Override n_action_steps from model config |
| `--output` | required | Output directory |

## CLI — online

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | SmolVLA BC checkpoint |
| `--env` | `figure_shape_placement` | Task name |
| `--noise-dims` | 6 | Noise dims: `6`, `32`, or `1600` |
| `--episodes-per-batch` | 10 | Episodes per REINFORCE batch |
| `--total-batches` | 100 | Training batches |
| `--lr` | 1e-4 | Learning rate |
| `--max-episode-steps` | 500 | Max steps per episode |
| `--eval-freq` | 10 | Evaluate every N batches |
| `--eval-episodes` | 20 | Episodes for evaluation |
| `--n-action-steps` | None | Override n_action_steps |
| `--resume` | None | Path to `noise_prior.pt` for resume |
| `--output` | required | Output directory |
| DR flags | — | `--no-domain-rand`, etc. |

---

## Eval

```bash
# With learned sampler (eval_vla_policy.py auto-detects format)
python scripts/eval/eval_vla_policy.py \
    --checkpoint outputs/easy_smolvla_vlm_pretrained_v1/checkpoints/030000/pretrained_model \
    --noise-prior outputs/easy_learned_sampler_offline_v1/noise_prior.pt \
    --env figure_shape_placement_easy \
    --episodes 50 --max-steps 500 --no-domain-rand \
    --n-action-steps 15
```

`eval_vla_policy.py` reads the `noise_prior.type` field from checkpoint to distinguish
`LearnedNoiseSampler` from `NoisePrior` (fixed mu). Both use the same `--noise-prior` flag.

---

## Output (checkpoint format)

```python
{
    "noise_prior": {
        "type": "learned",          # distinguishes from fixed mu
        "noise_dims": 32,
        "mlp": state_dict,
        "log_sigma_min": -2.0,
        "log_sigma_max": 2.0,
    },
    "eval_sr": float,               # or "val_loss" for offline
}
```

Offline: saves `noise_prior.pt` + `train_config.json` at best val_loss checkpoint.
Online: saves `noise_prior.pt` + `train_config.json` + per-batch log at best eval SR.

---

## x_0 Sensitivity (measured on BC checkpoint 16k, 5 scenes × 100 samples)

| Joint | Baseline std (motor units) | Delta from mu[j]=1.0 |
|-------|---------------------------|----------------------|
| base | 0.97 | 0.77 |
| shoulder | 2.54 | 1.82 |
| elbow | 3.02 | 1.36 |
| wrist_pitch | 1.95 | 2.16 |
| wrist_roll | 2.18 | 1.71 |
| gripper | 3.81 | 4.86 |

ODE retains memory of starting point. Shift mu=1 moves corresponding action by 1–5 motor units
(out of [-100, 100] range). Cross-talk exists but diagonal dominates.

Measured with `scripts/tools/measure_x0_sensitivity.py`.

---

## Notes

**Batch size for offline training:** ODE backprop stores activations for all 10 denoising steps
× batch_size. With batch=4, ~10 GB VRAM. Increase if GPU allows; larger batch is more stable.

**`--offline-chunk-len`** should match `--n-action-steps` — only the first N actions of the 50-step
chunk are executed in sim, so supervising on more than N steps adds noise.

**Online vs offline:** offline is faster to experiment with (no sim startup) and can find a good
sampler from existing data. Online (REINFORCE) can improve beyond what offline data shows, but
requires more episodes for stable gradient estimates. Typical flow: offline first, then online
fine-tuning if needed.
