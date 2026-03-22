# Easy Task Results

Task: `figure_shape_placement_easy` — place a cube into a slot on a platform.
Spawn zone: 4×4 cm, platform yaw fixed at 90°. Dataset: 200 teleoperated episodes.

**Eval protocol:** 50 episodes, seed 38531708, `--n-action-steps 15`, `--no-domain-rand`.

Statistical note: ±7% confidence interval for 50 episodes — differences smaller than this are within noise.

---

## Full results table

| # | Method | SR | Δ vs BC baseline | Notes |
|---|--------|----|-----------------|-------|
| #5.1 | ACT (chunk=20) | 56% | — | Different architecture |
| #6 | Diffusion Policy | ~0% | — | Cause not investigated |
| #7 | SmolVLA BC (robotics pretrain) | 0% | — | Possible cause: Open X-Embodiment weights conflict with sim |
| **#7.1** | **SmolVLA BC (VLM pretrained)** | **70–76%** | **baseline** | SmolVLM2 vision+language, random action expert |
| #7.1f | SmolVLA BC (frozen VLM) | 70% | ~0% | Expert-only training (100M params) |
| #7.2 | SmolVLA BC (scratch, 60k steps) | 56% | −14–20% | No VLM pretrain, plateaus |
| **#8** | **IQL weighted BC (VIP dense)** | **86–88%** | **+12–16%** | Best offline method |
| #8b | IQL weighted BC (sim sparse, VIP enc) | 76% | ~0% | Sparse reward insufficient |
| #8c | IQL weighted BC (sim sparse, ImageNet) | 74% | ~0% | Encoder irrelevant at low reward quality |
| #9b | BC + Learned Sampler (offline) | 80% | +4–10% | State-conditional x_0, 32 dims |
| #9b→8 | IQL + Sampler (sampler from BC) | 82% | +6–12% | Worse than IQL alone |
| #10a | IQL + PPO (200 updates) | 74% | −12–14% | PPO degraded IQL (86→64→74) |
| **#10b** | **BC + sampler + PPO (best@50)** | **90%** | **+14–20%** | **Best result** |

---

## Ranking

```
90%  BC + sampler + PPO @update_50     (#10b)  ← best
88%  IQL weighted BC                   (#8)
86%  BC + sampler + PPO @update_200    (#10b final)
82%  IQL + Sampler                     (#9b→8)
80%  BC + Learned Sampler              (#9b)
76%  SmolVLA BC baseline               (#7.1)
74%  IQL + PPO @update_200             (#10a)
70%  SmolVLA BC frozen VLM             (#7.1f)
56%  ACT                               (#5.1)
56%  SmolVLA BC scratch                (#7.2)
 0%  Diffusion Policy                  (#6)
 0%  SmolVLA BC robotics pretrain      (#7)
```

---

## Key findings

### Architecture

- SmolVLA >> ACT >> Diffusion for this task
- VLM pretraining is critical: +14–20% vs scratch; robotics pretraining hurts (0%)
- `--n-action-steps 15` is critical for SmolVLA (default 50 gives significantly worse results)
- Frozen vs unfrozen VLM: difference (70% vs 70–76%) is within the ±7% CI — not conclusive

### Offline RL (IQL)

- **Dense VIP rewards work** (+12–16% over baseline)
- **Sparse sim rewards do not improve** (76% ≈ baseline) — critic cannot distinguish "almost success" from "far away"
- **Encoder does not matter** at low reward quality: VIP 76% ≈ ImageNet 74%
- Key factor: reward signal quality, not encoder choice

### Learned Noise Sampler

- +4–10% over BC baseline (80% vs 70–76%)
- IQL + sampler (82%) is worse than IQL alone (86–88%) — sampler was trained on the BC velocity field, which changes after IQL fine-tuning
- Most useful as a precursor to PPO, not as a standalone method

### Online RL (PPO)

- PPO improves a weaker policy (80% → 90%) but degrades a stronger one (86–88% → 64%)
- Root cause: value head (random init) does not converge under raw VIP rewards (~−13/step, v_loss ~25k)
- Best checkpoint = update_50 — indicates instability; policy has not yet been degraded
- Potential fixes (`--normalize-rewards`, `--warmup-value`) added but not tested on this task

### Optimal pipeline

```
SmolVLA BC (30k)  →  Offline sampler  →  PPO @update_50
     70–76%              80%                  90%
```

Not via IQL: IQL reaches 86–88%, but PPO on top degrades it. PPO works better on top of a weaker starting point.

---

## Training cost (RTX 5070 Ti)

| Stage | Time | Result |
|-------|------|--------|
| Teleoperation (200 episodes) | ~2 h | dataset |
| SmolVLA BC (30k steps) | ~3 h | 70–76% |
| Offline sampler (500 steps) | ~15 min | 80% |
| PPO (50 updates) | ~1.2 h | 90% |
| **Total** | **~6.5 h** | **90%** |

For comparison: IQL pipeline (VIP labeling + critics + weighted BC) = ~3.5 h → 86–88%, but PPO on top does not work.

---

## Open questions

1. **PPO with `--normalize-rewards --warmup-value`** — would a converged value head allow training past update_50?
2. **PPO on top of IQL with normalize/warmup** — if v_loss converges, can IQL + PPO exceed 90%?
3. **Harder task** — tested on medium task (4× spawn area, 300 episodes). Results: BC 45%, IQL 48% (faster execution, SR difference not significant), PPO 30% (trained with DR, eval without). See [medium_task.md](medium_task.md).
4. **#8→9b** (sampler re-trained on IQL checkpoint, not BC) — not tested.
