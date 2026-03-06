# flow_noise_smolvla.md

SmolVLA wrapper with stochastic noise head for PPO training.

File: `so101_lab/policies/rl/flow_noise_smolvla.py`. Used by `train_flow_noise_ppo.py`.

---

## Overview

`FlowNoiseSmolVLA` wraps a SmolVLA BC checkpoint and adds:
1. **Stochastic denoising**: a `noise_log_std_head` (Linear 720→32) predicts per-step velocity
   variance, making the ODE stochastic → tractable `log π(a|s)` for PPO.
2. **Value head**: VIPBackbone + small MLP predicts V(s) for advantage estimation.

All SmolVLA backbone params (~450M) and VIP ResNet50 (~25M) are frozen.

**Trainable params (~639k):**
- `action_out_proj` (Linear 720→32): 23,072
- `noise_log_std_head` (Linear 720→32): 23,072
- `value_mlp` (2054→256→256→1): ~593k

`expert_hidden_size` (720 for our fine-tuned checkpoint) is read automatically from
`model.vlm_with_expert.expert_hidden_size`.

---

## Key methods

```python
from so101_lab.policies.rl.flow_noise_smolvla import FlowNoiseSmolVLA

model = FlowNoiseSmolVLA(
    checkpoint_path="outputs/smolvla_teleop_only/checkpoints/016000/pretrained_model",
    device="cuda",
    iql_checkpoint=None,           # optional: warm start value head from IQL critics
    task_string="Place the cube...",
)

# Rollout: returns action + log_prob + cached trajectory
action, log_prob, trajectory = model.sample_actions_with_log_prob(obs)

# Re-evaluation: compute log_prob of saved actions under updated heads
new_log_prob = model.reeval_log_prob(obs, trajectory)
new_log_prob = model.reeval_log_prob_batched(obs_list, traj_list)  # batched version

# Value estimate
value = model.get_value(obs)   # scalar

# Deterministic ODE with given x_0 (for noise prior use)
action_chunk = model.run_ode_with_x0(x_0, obs)  # returns full chunk (1, 50, 6)

# Checkpointing
model.save("outputs/flow_noise_ppo_v3/final")
model.load_overlay("outputs/flow_noise_ppo_v3/final/flow_noise_ppo.pt")
```

### Noise bounds (train vs eval)

Set via `model.set_train_mode()` / `model.set_eval_mode()`:

| Mode | log_std clamp | std range |
|------|--------------|-----------|
| Train | [-5, 2] | [0.007, 7.4] |
| Eval | [-20, -2] | [2e-9, 0.14] |

### Log-prob computation

Only first chunk step + first 6 action dims per ODE step (6 × 10 = 60 terms total):
```python
v_mean_s = v_mean[:, 0, :6]   # (B, 6)
v_std_s  = v_std[:, 0, :6]
log_prob = (-0.5 * ((v_sampled_s - v_mean_s)**2 / var_s) - v_log_std_s).sum(dim=-1)
```
Summing all 50×32×10 = 16000 terms causes PPO ratio explosion.

### Trajectory cache (for re-evaluation)

`sample_actions_with_log_prob()` returns a `trajectory` dict containing per-ODE-step tensors
stored on CPU:
- `x_t_list`: 10× (1, 50, 32) — intermediate noise state
- `v_sampled_list`: 10× (1, 50, 32) — sampled velocities
- `prefix_pad_masks`: attention masks for KV cache
- `past_key_values`: KV cache (16 layers, prefix_len=305, 6 MB/obs total)

Re-eval feeds saved `x_t` → frozen transformer → updated heads → `log N(saved_v | new params)`.
No ODE re-run needed; this is valid because saved `x_t` depends only on previous (fixed) `v_sampled`.

### Value head

Architecture: `VIPBackbone(frozen) → [emb(2048), state(6)] → Linear(2054,256) → LN → SiLU → Linear(256,256) → LN → SiLU → Linear(256,1)`

Optional warm start from IQL critics checkpoint (`--iql-checkpoint`):
- IQL V-net has identical architecture but was trained on normalized rewards (~[-1,0])
- PPO uses raw VIP rewards (~-22/step) — scale mismatch wastes ~50 updates
- **Recommendation: random init** (`--iql-checkpoint None`). Value MLP (~593k) adapts in 5–10 updates.

Use `--normalize-rewards --warmup-value 10` to stabilize training from random init.

---

## Output files

```
{output}/{name}/
├── flow_noise_ppo.pt           # resume overlay
│   ├── action_out_proj         # Linear(720→32) weights
│   ├── noise_log_std_head      # Linear(720→32) weights
│   ├── value_mlp               # value MLP weights
│   ├── optimizers              # actor_opt + value_opt state dicts
│   └── metadata                # {update: int, best_sr: float}
└── pretrained_model/           # full SmolVLA checkpoint for eval
    ├── config.json
    ├── model.safetensors       # SmolVLA weights with updated action_out_proj
    ├── policy_preprocessor.json
    └── policy_postprocessor.json
```

`pretrained_model/` can be used directly with `eval_vla_policy.py` without any PPO infrastructure.

---

## Gradient flow

Prefix (SigLIP + SmolVLM text + state encoding) is computed in `torch.no_grad()`. The frozen
expert transformer's `suffix_out` is detached and re-attached with `requires_grad=True` so that
`action_out_proj` and `noise_log_std_head` receive gradients, but frozen backbone does not
accumulate activations.

This means ~450M frozen params are not in the compute graph during backward — memory stays bounded.

---

## Notes

**`run_ode_with_x0()`** — deterministic ODE (no stochastic sampling). Used by noise prior
training scripts to measure how x_0 shift affects output actions. Takes x_0 tensor directly,
bypasses `sample_noise()`.

**KV cache device:** default is CPU (to avoid OOM with 500 obs × 6 MB = 2.9 GB).
`--kv-cache-device cuda` moves to GPU for slightly faster transfer but adds 2.9 GB VRAM
with no measured speedup (bottleneck is not transfer).

**`reeval_log_prob_batched()`** stacks N trajectories along batch dim and runs one batched
suffix forward. `--reeval-batch-size 16` is the sweet spot (GPU saturated at 8).
