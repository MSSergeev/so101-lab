# Medium Task Results

Task: `figure_shape_placement` — place a cube into a slot on a platform.
Spawn zone: 8×8 cm (4× easy task area), platform yaw fixed at 90°. Dataset: 300 teleoperated episodes.

**Eval protocol:** 600 episodes, seed 7382910, `--n-action-steps 15`, `--no-domain-rand`, `--max-steps 600`, relaxed thresholds (9cm XY, 30° yaw).

Statistical note: ±4% confidence interval for 600 episodes.

> **2026-03-22:** Results corrected. Previous results (BC 22%, IQL 32%, PPO 24%) were collected with a bug in `smolvla_server.py` — the task language instruction was not passed to SmolVLA (empty string instead of the actual task description). Fixed in [`7cdae2d`](https://github.com/MSSergeev/so101-lab/commit/7cdae2d). SmolVLA-based results (#7.1, #8, #9b, #10b) re-evaluated below. ACT and Diffusion results are unaffected (they use separate eval scripts without gRPC).

---

## Full results table

| # | Method | SR | Δ vs BC baseline | Notes |
|---|--------|----|-----------------|-------|
| #6 | Diffusion Policy | 0–5% | — | Baseline config not suitable |
| #5.1 | ACT (chunk=15) | 23% | — | Different architecture, temporal ensemble |
| **#7.1** | **SmolVLA BC (VLM pretrained, 040000)** | **45%** | **baseline** | 600 ep, seed 7382910 |
| #9b | BC + Learned Sampler (offline, 32d) | ~~20%~~ | — | Invalidated (gRPC task bug), not re-evaluated |
| **#8** | **IQL weighted BC (006000)** | **48%** | **+3%** | SR not significant (z=1.04), but 22 steps faster (p<0.001) |
| #10b | BC + sampler + PPO v2 (000320, 600 updates) | 30% | −15% | Trained with DR, evaluated without — see note |

---

## Ranking

```
48%  IQL weighted BC              (#8)      ← best SR, significantly faster
45%  SmolVLA BC baseline          (#7.1)
30%  BC + sampler + PPO v2        (#10b)    trained with DR, eval without DR
23%  ACT                          (#5.1)
 5%  Diffusion Policy             (#6)
```

---

## Key findings

### vs easy task

All methods degrade with 4× spawn area and 300 episodes (vs 200):

| Method | Easy task | Medium task | Drop |
|--------|-----------|-------------|------|
| SmolVLA BC | 70–76% | 45% | −27 p.p. |
| IQL | 86–88% | 48% | −39 p.p. |
| BC + sampler + PPO | 90% | 30% | −60 p.p. |

### IQL vs BC

IQL SR advantage (+3 p.p.) is not statistically significant (z=1.04, p≈0.30 on 600 episodes). However, IQL was consistently slightly ahead of BC across all prior evaluations. More importantly, IQL is **significantly faster** when successful: avg 278 steps vs 300 steps (Welch t=3.48, p<0.001) — the policy is more decisive in its movements.

### Online RL (PPO)

PPO (30%) is below BC (45%). This does **not** necessarily mean PPO degraded the policy — the PPO checkpoint was trained with domain randomization enabled, while eval was run without DR (`--no-domain-rand`). The policy may perform better under DR conditions it was trained for. This hypothesis needs testing.


### Learned Noise Sampler (#9b)

Previous result (20%) was invalidated by the gRPC task bug. Not re-evaluated — the sampler checkpoint exists but re-eval is low priority given it showed no improvement even with the bug.

### Optimal pipeline

```
SmolVLA BC (45k)  →  IQL weighted BC
     45%                  48%
```

IQL improves execution speed significantly. SR improvement is directionally positive but not statistically confirmed.

---

## Training cost (RTX 5070 Ti)

| Stage | Time | Result |
|-------|------|--------|
| Teleoperation (300 episodes) | ~3 h | dataset |
| SmolVLA BC (45k steps) | ~4.5 h | 45% |
| VIP labeling + IQL critics | ~5 min | critics |
| IQL weighted BC (10k steps) | ~1 h | 48% |
| **Total (BC → IQL)** | **~9 h** | **48%** |
| PPO (600 updates) | ~16 h | 30% (with DR) |

---

## Open questions

1. **More data:** 300 episodes may be insufficient for 4× spawn area. 500+ episodes may close the gap to easy task.
2. **PPO with DR eval:** PPO trained with DR — eval with DR may show improvement over BC. Need to test.
3. **PPO with multiple envs:** single-env rollout (256 steps < 1 episode) gives poor GAE estimates. 8–16 parallel envs may unlock PPO on harder tasks.
4. **IQL from scratch:** current pipeline fine-tunes BC for 10k steps with advantage weighting. Training from scratch with IQL for full 45k steps not tested.
5. **Noise sampler re-eval:** #9b invalidated, re-evaluation needed if sampler approach is revisited.
