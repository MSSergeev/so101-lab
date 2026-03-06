# Medium Task Results

Task: `figure_shape_placement` — place a cube into a slot on a platform.
Spawn zone: 8×8 cm (4× easy task area), platform yaw fixed at 90°. Dataset: 300 teleoperated episodes.

**Eval protocol:** 100 episodes, seed 1988042740, `--n-action-steps 50`, `--no-domain-rand`, relaxed thresholds (9cm XY, 30° yaw).

Statistical note: ±6% confidence interval for 100 episodes, ±4% for 200+ episodes.

---

## Full results table

| # | Method | SR | Δ vs BC baseline | Notes |
|---|--------|----|-----------------|-------|
| #6 | Diffusion Policy | 0–5% | — | Baseline config not suitable |
| #5.1 | ACT (chunk=15) | 23% | — | Different architecture, temporal ensemble |
| **#7.1** | **SmolVLA BC (VLM pretrained, 040000)** | **22%** | **baseline** | 600 ep, seed 7382910 |
| #9b | BC + Learned Sampler (offline, 32d) | 20% | −2% | No improvement on medium task |
| **#8** | **IQL weighted BC (006000)** | **32%** | **+10%** | Best offline method |
| #10b | BC + sampler + PPO v2 (000320, 600 updates) | 24% | +2% | Consistent direction, z-test p=0.60 |

---

## Ranking

```
32%  IQL weighted BC              (#8)      ← best
24%  BC + sampler + PPO v2        (#10b)
23%  ACT                          (#5.1)
22%  SmolVLA BC baseline          (#7.1)
20%  BC + Learned Sampler         (#9b)
 5%  Diffusion Policy             (#6)
```

---

## Key findings

### vs easy task

All methods degrade significantly with 4× spawn area and 300 episodes (vs 200):

| Method | Easy task | Medium task | Drop |
|--------|-----------|-------------|------|
| SmolVLA BC | 70–76% | 22% | −50 p.p. |
| IQL | 86–88% | 32% | −55 p.p. |
| BC + sampler + PPO | 90% | 24% | −66 p.p. |

IQL remains the best offline method (+10 p.p. over BC), consistent with easy task trend (+12–16 p.p.).

### Learned Noise Sampler

No effect on medium task (20% = BC baseline). On easy task gave +4–10%.
Likely reason: small val_loss improvement (~7.7%) → sampler learns a minor shift that doesn't generalize to the harder spawn distribution.

### Online RL (PPO)

- **Consistent direction:** PPO > BC in every single run across all seeds, but effect is small (+1–8 p.p.)
- **Not statistically significant:** z-test p=0.60 on 600 ep; ~2500 ep per checkpoint needed to confirm
- **Why PPO underperforms vs easy task:**
  - BC baseline too low (22% vs 70%) — less signal for reward improvement
  - Episodes too long (~400 steps vs ~150) — harder credit assignment over VIP reward
  - Single-env rollout (256 steps < 1 episode) — poor GAE estimates
- **IQL warm start mismatch:** IQL V-net (V≈−634) conflicts with normalized PPO VIP rewards → v_loss=32k+. Fix: omit `--iql-checkpoint`, use `--normalize-rewards --warmup-value 30`
- Value head converges with fix (v_loss 211→9), but reward signal flat throughout training

### Optimal pipeline

```
SmolVLA BC (45k)  →  IQL weighted BC
     22%                  32%
```

PPO adds a likely small positive effect but not confirmed statistically. IQL is the clear winner.

---

## Training cost (RTX 5070 Ti)

| Stage | Time | Result |
|-------|------|--------|
| Teleoperation (300 episodes) | ~3 h | dataset |
| SmolVLA BC (45k steps) | ~4.5 h | 22% |
| VIP labeling + IQL critics | ~5 min | critics |
| IQL weighted BC (10k steps) | ~1 h | 32% |
| **Total (BC → IQL)** | **~9 h** | **32%** |
| Offline sampler (5000 steps) | ~2.5 h | no gain |
| PPO (600 updates) | ~16 h | 24% |

---

## Open questions

1. **More data:** 300 episodes may be insufficient for 4× spawn area. 500+ episodes may close the gap to easy task.
2. **PPO with multiple envs:** single-env rollout (256 steps < 1 episode) gives poor GAE estimates. 8–16 parallel envs may unlock PPO on harder tasks.
3. **PPO statistical confirmation:** consistent direction observed but ~2500 ep needed for p<0.05. Small effect likely real.
4. **IQL from scratch:** current pipeline fine-tunes BC for 10k steps with advantage weighting. Training from scratch with IQL for full 45k steps not tested.
5. **Noise sampler with noise-dims=1600:** 32-dim sampler had no effect. Full x_0 control may help on harder spawn distribution.
