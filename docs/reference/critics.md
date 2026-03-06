# critics.md

IQL critic networks and Learned Noise Sampler library modules.

Files: `so101_lab/policies/rl/critics.py`, `so101_lab/policies/rl/learned_noise_sampler.py`

---

## critics.py

### Overview

Provides Q/V networks for offline IQL (Implicit Q-Learning). Two usage patterns:

1. **Training** (`train_iql_critics.py`): uses lightweight `VHead`/`QHead` classes that operate
   on pre-cached embeddings (no VIP backbone loaded at training time).
2. **Inference** (`compute_iql_weights.py`, `flow_noise_smolvla.py`): uses `VIPBackbone` +
   `VNetwork`/`QNetwork`/`TwinQ` which encode images on-the-fly.

### Classes

**`VIPBackbone`**

Frozen VIP ResNet50 (Ego4D pretrained) for two cameras. Takes raw images, preprocesses
(Resize 224×224, ImageNet normalize), returns concatenated embeddings.

```python
backbone = VIPBackbone(device="cuda")
emb = backbone(images_top, images_wrist)  # (B, 2048)
```

Input: `(B, H, W, 3)` uint8 or `(B, 3, H, W)` float. HWC format is auto-detected and permuted.

**`VNetwork`** / **`QNetwork`**

Full networks with embedded VIPBackbone:

```python
v_net = VNetwork(device="cuda")
v = v_net(images_top, images_wrist, state)              # (B, 1)

q_net = QNetwork(device="cuda")
q = q_net(images_top, images_wrist, state, action)      # (B, 1)
```

Both accept optional `backbone=` to share VIPBackbone between networks.

**`TwinQ`**

Two Q-networks sharing one VIPBackbone. Returns `(q1, q2)`, each `(B, 1)`.

```python
twin = TwinQ(device="cuda")
q1, q2 = twin(images_top, images_wrist, state, action)
q_min = twin.min_q(images_top, images_wrist, state, action)  # (B, 1)
```

### MLP architecture (all networks)

`Linear(input_dim, h) → LayerNorm(h) → SiLU`, repeated for each hidden dim, then `Linear(h, 1)`.

Default hidden: `[256, 256]`. Input dims:
- V: 2048 + 6 = 2054
- Q: 2048 + 6 + 6 = 2060

### Checkpoint format (critics.pt)

Saved by `train_iql_critics.py` (uses `VHead`/`QHead`, not `VNetwork`/`QNetwork`):

```python
{
    "v_net": VHead.state_dict(),
    "q1_net": QHead.state_dict(),
    "q2_net": QHead.state_dict(),
    "q1_target": QHead.state_dict(),
    "q2_target": QHead.state_dict(),
    "config": {
        "hidden_dims": [256, 256],
        "encoder_type": "vip",          # or "imagenet"
        "reward_scale": float,
        ...
    }
}
```

`compute_iql_weights.py` imports `VHead`/`QHead` from `train_iql_critics.py` directly to load
these checkpoints (not from `critics.py`).

`flow_noise_smolvla.py` uses `VIPBackbone` from `critics.py` for the value head backbone.

---

## learned_noise_sampler.py

### Overview

State-conditional noise sampler that replaces `x_0 ~ N(0, I)` in SmolVLA flow matching with
`x_0 ~ N(mu(s), sigma(s))`. See `reference/train_noise_sampler.md` for training details.

### `LearnedNoiseSampler`

```python
from so101_lab.policies.rl.learned_noise_sampler import LearnedNoiseSampler

sampler = LearnedNoiseSampler(device="cuda", noise_dims=32)

# Forward: returns (mu, log_sigma)
mu, log_sigma = sampler(images_top, images_wrist, state)  # each (B, noise_dims)

# Sample x_0 (reparameterized, differentiable)
x_0 = sampler.sample_x0(images_top, images_wrist, state)  # (B, 1, 50, 32)

# Log probability of a sample
lp = sampler.log_prob(x_0, images_top, images_wrist, state)  # (B,)

# Patch SmolVLA model to use this sampler
sampler.patch_model(model)  # replaces model.sample_noise()
```

**Parameters by noise_dims:**

| noise_dims | Dims patched | MLP hidden | Params |
|------------|-------------|-----------|--------|
| 6 | x_0[:, 0, :6] | 256 | ~34k |
| 32 | x_0[:, 0, :32] | 512 | ~67k |
| 1600 | x_0[:, :50, :32] (full) | 1024 | ~5.3M |

**Init:** zeros on final layer → outputs mu=0, log_sigma=0 → `N(0, e^0=1)` = baseline behavior.

**Eval mode:** `eval()` clamps log_sigma to -20 → sigma ≈ 0, output is deterministic mu(s).

### Checkpoint format (noise_prior.pt)

```python
{
    "noise_prior": {
        "type": "learned",          # distinguishes from NoisePrior (fixed mu)
        "noise_dims": 32,
        "mlp": state_dict,
        "log_sigma_min": -2.0,
        "log_sigma_max": 2.0,
    },
    "eval_sr": float,               # or "val_loss" for offline training
}
```

`eval_vla_policy.py --noise-prior` reads `noise_prior.type` to choose between
`LearnedNoiseSampler` and `NoisePrior`.

---

## noise_prior.py

### `NoisePrior`

Fixed (non-neural) mu shift — 6 scalar parameters for the first 6 action dims of x_0.

```python
from so101_lab.policies.rl.noise_prior import NoisePrior

prior = NoisePrior(lr=0.1)
prior.patch_model(model)       # replaces model.sample_noise()
prior.update("better", ep_mean_x0)  # human-in-the-loop update
prior.save("outputs/noise_prior_v1/noise_prior.pt")
```

When mu=0: behavior identical to standard SmolVLA (recovers N(0,I)).
Trained via `train_noise_prior.py` (human B/W/N/Q feedback) or `train_noise_prior_cmaes.py` (CMA-ES).

Checkpoint format:
```python
{
    "noise_prior": {
        "type": "fixed",
        "mu": tensor(6,),
        "lr": float,
    }
}
```
