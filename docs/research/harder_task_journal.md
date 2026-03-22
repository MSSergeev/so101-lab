# Medium Task — Research Journal

**Task:** `figure_shape_placement` (medium variant, 4x easy task spawn area)
**Goal:** Run the full pipeline — create env → record → train → eval — with LeRobot 0.5.1.
**Reference:** `docs/workflow/` series (00–05b), `docs/results/easy_task.md` for baseline numbers.
**Agent instructions:** `docs/research/harder_task_agent.md`

## What we're tracking

- LeRobot 0.5.1 migration: what broke, what still works, what workarounds are still needed
- Issues found in docs, scripts, imports — and fixes applied
- Eval results for each method, compared to easy task baseline
- Open questions and unexpected findings

---

## Previous attempt (polar spawn zone) — concluded

Ran full pipeline on `figure_shape_placement` with large polar spawn zone. Results:

| Method | SR | Notes |
|--------|----|-------|
| ACT | ~16% | Single eval, 20 ep, step 35000, `--temporal-ensemble-coeff 0.01` |
| ACT sweep | 0% | All checkpoints, seed 1988042740, cause unknown |
| Diffusion sweep | 0% | All checkpoints, same seed |
| SmolVLA BC | 0% | All 7 checkpoints, 50 ep each |

**Conclusion:** spawn zone too large, 500 episodes insufficient. Redesigned task with rectangular spawn zone ~4x easy task area, fixed platform (same as easy).

---

## Phase 0: Setup and migration

### LeRobot version

- [x] Installed LeRobot 0.5.1 from source (`uv venv --python 3.12`, `uv pip install -e .`)
- [x] Installed so101-lab in lerobot-env (`uv pip install -e .`, `.[hardware]`)
- [x] Installed trackio, nvidia-ml-py
- [x] Blackwell GPU: torch 2.10.0+cu128 — CUDA works, no reinstall needed
- [x] Additional packages installed in isaaclab-env (diffusers, pygame, transformers, etc.)
- [x] Installed venvs/rerun and venvs/viewer
- [x] Verified setup: `test_env_spawn.py --gui --env figure_shape_placement_easy` — 10 resets OK
- [x] SmolVLA deps: `[smolvla]` + `[async]` installed — no torch downgrade in 0.5.1

### Isaac Lab install

- [x] Isaac Lab installed at `~/Robotics/IsaacLab2`, venv at `~/Robotics/IsaacLab2/env_isaaclab` (Python 3.11, uv)
- [x] so101-lab installed in isaaclab-env (`uv pip install -e .`)
- **Note:** Isaac Lab venv is created with uv — use `uv pip` not `pip`
- **Issue:** `isaaclab.sh --install` run inside lerobot-env → installed packages into lerobot-env, downgraded setuptools to 81.0.0. Fixed: create and activate a dedicated venv first (`uv venv --python 3.11`), restore setuptools with `uv pip install "setuptools>=82"`
- **Issue:** `_isaac_sim` symlink missing → `./isaaclab.sh` can't find Isaac Sim Python. Fixed: `ln -s /path/to/isaacsim /path/to/IsaacLab/_isaac_sim`
- **Issue:** `isaaclab.sh --install` does not patch `activate` script → `isaacsim` module not found. Fixed: manually append `ISAACLAB_PATH`, `setup_conda_env.sh` sourcing to `env_isaaclab/bin/activate`
- **Issue:** snap cmake (`/snap/bin/cmake`) fails with GLIBC errors when building `egl_probe`. Fixed: `sudo apt install cmake` → `/usr/bin/cmake`

### Breaking changes found

<!-- Log each issue: what broke, where (script/module), how fixed -->

---

## Phase 1: Environment

### Task

- [x] Redesigned `figure_shape_placement` spawn zone:
  - Cube: rectangular zone X∈[0.294, 0.374], Y∈[0.038, 0.118] (8×8 cm, 4x easy task area)
  - Platform: fixed (0.154, 0.048), yaw=90° (same as easy task)
  - Cube yaw: discrete 0–80°, 10° step (same as easy task)
- [x] Verified spawn: `test_env_spawn.py --gui --env figure_shape_placement --resets 20`

### Issues found

- **Issue:** `pyserial`, `deepdiff`, `feetech-servo-sdk` missing in isaaclab-env when using SO-101 leader arm. Fixed: `uv pip install pyserial deepdiff feetech-servo-sdk`. Added to `00_setup.md`.
- **Issue:** `CUBE_PLATFORM_GAP_REDUCTION` and `MAX_SPAWN_ATTEMPTS` removed from `figure_shape_placement/env_cfg.py` during spawn refactor — easy task imports them. Fixed: re-added with original values.

---

## Phase 2: Recording

- [x] Test recording: 50 episodes — 50/50 success, mean 399 frames (~13.3s), min/max 331/510
- [ ] Main dataset: target 300 episodes, recording in batches of 50 with `--diversity-keys cube_x,cube_y --diversity-ratio 1.2`
  - [x] Batch 1: 50 ep (total 50)
  - [x] Batch 2: +50 (total 100) — 41031 frames, diversity: 20 cells, mean=5.0, min=1, max=11, rerolls=72
  - [x] Batch 3: +50 (total 150) — 61275 frames, diversity: 29 cells, mean=5.2, min=1, max=8, rerolls=50
  - [x] Batch 4: +50 (total 200) — 79999 frames, diversity: 40 cells, mean=5.0, min=1, max=8, rerolls=40
  - [x] Batch 5: +50 (total 250) — 99341 frames, diversity: 51 cells, mean=4.9, min=1, max=8, rerolls=50
  - [x] Batch 6: +50 (total 300) — 118032 frames, diversity: 66 cells, mean=4.5, min=1, max=10, rerolls=49
- [x] Trim: `trim_after_success.py --keep-after 10 --reencode --crf 18` → 105145 frames (was 118032)
- [x] Verified: LeRobot API OK — 300 ep, 105145 frames, all features present

### Issues found

---

## Phase 3: BC Training

### ACT

- Config: `configs/policy/act/baseline.yaml` — steps=60000, chunk_size=15, n_action_steps=15 (was 45000/20/20)
- [x] Checkpoint: `outputs/act_figure_shape_placement_v1` — best loss 0.0252 at 60000 steps, 299 min
- [x] Sweep eval (single, 100 ep, seed 1988042740, `--temporal-ensemble-coeff 0.01`, max_steps=600):

| Checkpoint | SR | Avg steps (ok) | Avg steps (fail) |
|---|---|---|---|
| **best** | **23%** | 300 | 598 |
| latest (60k) | 17% | 314 | 598 |
| 50000 | 15% | 276 | 598 |
| 45000 | 15% | 296 | 598 |
| 40000 | 13% | 313 | 598 |
| 55000 | 10% | 381 | 598 |
| 30000 | 9% | 321 | 598 |
| 35000 | 6% | 288 | 598 |

- **Note:** parallel eval had a stale-camera bug (stale obs after auto-reset corrupted temporal ensembler buffer). Fixed — skip stale envs from blending, deferred re-init.

### Diffusion Policy

- [x] Checkpoint: `outputs/diffusion_figure_shape_placement_v1` — best loss 0.0012 at 50000 steps, 247 min
- [x] Sweep eval (single, 20 ep, seed 1988042740, max_steps=600): **0% all checkpoints** (latest: 5% = 1/20)
- Same pattern as easy task (0%). Diffusion baseline config not suitable for this task.

### SmolVLA BC

- Variant B: VLM pretrained (SmolVLM2), random action expert, unfrozen vision encoder
- [x] Checkpoint: `outputs/smolvla_figure_shape_placement_v1` — loss 0.013 at 45000 steps, 260 min
- [ ] Eval SR (SUCCESS_THRESHOLD=5cm): 0% — robot places cube near slot but not close enough
- Relaxed thresholds for research: SUCCESS_THRESHOLD 5cm → 9cm, ORIENTATION_THRESHOLD 15° → 30°
- [x] Sweep eval (100 ep, seed 1988042740, n-action-steps=50, relaxed thresholds):

| Checkpoint | SR | Avg steps (ok) | Avg steps (fail) |
|---|---|---|---|
| **040000** | **21%** | 345 | 600 |
| 015000 | 21% | 323 | 600 |
| 045000 | 19% | 373 | 600 |
| 025000 | 19% | 376 | 595 |
| 020000 | 18% | 386 | 600 |
| 035000 | 16% | 372 | 600 |

- Plateau 16–21% from 15k to 45k, differences within statistical noise (~±6% CI).

### Issues found

- **Issue:** `eval_vla_policy.py` used `sys.path` hack — incompatible with Python 3.11 (isaaclab-env) + lerobot 0.5.1 (Python 3.12). Fixed: gRPC policy server (`smolvla_server.py`) in lerobot-env, eval script connects as client (`GrpcPolicyClient`). Sweep auto-starts server; single eval uses `--auto-server`.
- **Issue:** Diffusion OOM with batch=32 on RTX 5070 Ti 16GB. Fixed: batch=16 in `configs/policy/diffusion/baseline.yaml`.
- **Issue:** RL env_cfg still used polar spawn (`CUBE_R_MAX_OFFSET`, `reset_cube_polar`) after main env was switched to rectangular. Fixed: added `reset_cube_rect` event, updated imports.
- **Issue:** VLA eval success detection used `MilestonePlaced` reward with hardcoded thresholds (4cm XY, 20° yaw), not `SUCCESS_THRESHOLD`/`ORIENTATION_THRESHOLD` from env_cfg. Fixed: wired MilestonePlaced params to env_cfg constants.
- **Issue:** gRPC policy client did not reset `_timestep` between episodes. Fixed.

---

## Phase 4: Offline RL (IQL)

IQL pipeline: label dataset with VIP dense rewards → train V/Q critics → compute advantage weights `exp(β·A)` → fine-tune SmolVLA with per-sample loss weighting (good demos get stronger gradient). On easy task this gave +12-16% over BC baseline.

- [x] VIP reward labels: `data/recordings/figure_shape_placement_v1_vip` — range [-16.38, -0.34], mean=-9.90
- [x] IQL critics: `outputs/iql_critics_figure_shape_placement_v1` — V≈-634, Q≈-635, 50k steps, 2.2 min
- [x] Advantage weights: β=1.0, mean=6.02, 36.7% >1, 1703 clipped
- [x] Weighted BC fine-tune: from SmolVLA 040000, 10k steps, loss 0.012, 58 min
- [x] Weighted BC eval (100 ep, seed 1988042740, n-action-steps=50, relaxed thresholds):

| Checkpoint | SR | Avg steps (ok) | Avg steps (fail) |
|---|---|---|---|
| **006000** | **32%** | 414 | 600 |
| 008000 | 28% | 375 | 600 |

- ~~IQL best 32% vs BC best 21% — +11 p.p.~~ **Corrected 2026-03-22:** these results were collected with gRPC eval bug (empty task string, [`7cdae2d`](https://github.com/MSSergeev/so101-lab/commit/7cdae2d)). Re-eval (600 ep, seed 7382910, n-action-steps=15): IQL 48% vs BC 45%. SR difference not significant (z=1.04), but IQL significantly faster (278 vs 300 steps, p<0.001).

### Issues found

---

## Phase 5: Online RL (PPO)

### Plan

Two pipelines, in order:

1. **BC → noise prior → PPO** (primary). Train offline noise sampler on BC checkpoint (040000), then PPO on top. This was the best pipeline on easy task (90% SR). Sampler is consistent with BC velocity field, PPO has room to improve from 21%.
2. **IQL → noise prior → PPO** (secondary). Train noise sampler on IQL checkpoint (006000), then PPO. On easy task IQL+PPO degraded (86%→74%), but with a sampler trained on the correct velocity field it may work better. Test only if pipeline 1 is feasible.

**Current step:** offline noise sampler training on BC (lerobot-env, no Isaac Sim needed).
**Blocker for PPO:** Python 3.11/3.12 incompatibility (see issues below).

### Phase 5a: Offline noise sampler

- [x] Train sampler on BC 040000 (`--noise-dims 32 --n-action-steps 50 --offline-steps 5000`)
  - Baseline val_loss: 0.04111 (N(0,I)), best val_loss: 0.03793 (step 800) — ~7.7% improvement
  - Val loss plateaus after step 800, oscillates 0.038–0.047. Sampler finds small shift quickly, doesn't learn further.
  - Output: `outputs/learned_sampler_figure_shape_placement_v1/noise_prior.pt`
  - Training time: 2h18m, batch=4, RTX 5070 Ti
- [x] Eval BC + sampler vs BC baseline (21%): **20%** (100 ep, seed 1988042740) — no improvement. Sampler has negligible effect, consistent with small val_loss gain.
- [ ] (optional) Train sampler on IQL 006000, eval

**Batch size benchmarks** (RTX 5070 Ti 16GB, 5000 steps, noise-dims=32, n-action-steps=50):

| Batch size | VRAM | Time | Notes |
|-----------|------|------|-------|
| 4 | 4 GB | ~2h | **used** |
| 8 | 8 GB | ~3h50m | |
| 16 | 12 GB | ~7h | fits but slow due to ODE backprop |

**Note:** total data seen = batch_size × steps. Batch 16 × 1250 steps ≈ same wall time as batch 4 × 5000 (~2h) but with more stable gradients. Worth trying if current run shows noisy val_loss.

### Phase 5b: PPO (gRPC split — two-process architecture)

- [x] Implemented gRPC PPO server/client (see workflow `05_online_rl.md` for commands)
- [x] Smoke test: 2 updates × 16 steps — passed (ratio=0.996, reward=-32, v_loss=10208)
- [x] Training run 1 (2026-03-14): BC 040000 + IQL V-net + learned sampler, labeled VIP goals, 100 updates × 256 steps, warmup-value 5, checkpoints every 10, `--tracker none`, **with DR**. Completed in 2.7h.
  - Output: `outputs/flow_noise_ppo_figure_shape_placement_v1`
  - v_loss unstable: oscillates 486–50k (same pattern as easy task). Value head not converging.
  - reward: -33 → -15...-29, no clear improvement trend
  - ratio stable ~1.0 throughout (policy updates conservative)
- [x] Sweep eval (100 ep, seed 1988042740, n-action-steps=50, relaxed thresholds, **with DR**):

| Checkpoint | SR | Avg steps (ok) | Avg steps (fail) |
|---|---|---|---|
| update_10 | 15% | 368 | 600 |
| update_20 | 18% | 379 | 600 |
| update_30 | 13% | 380 | 600 |
| update_40 | 18% | 362 | 600 |
| **update_50** | **23%** | 382 | 600 |
| update_60 | 17% | 389 | 600 |
| update_70 | 17% | 371 | 600 |
| update_80 | 16% | 385 | 600 |
| update_90 | 19% | 358 | 600 |
| final | 15% | 363 | 600 |

- Best: update_50 (23%) — matches BC baseline within CI (±8%). PPO did not improve over BC on medium task.
- Same pattern as easy task: best at update_50, degradation after. Value head instability (v_loss 486–50k) prevents stable policy improvement.
- **Note:** eval was run with DR (no `--no-domain-rand`). v2 sweep uses `--no-domain-rand` — results not directly comparable.

- [x] Training run 2 (2026-03-15): BC 040000 + learned sampler, labeled VIP goals, **no IQL warm start**, 200 updates × 256 steps, normalize-rewards, value-lr=3e-5, warmup-value=30, save-freq=20, **with DR**. Completed in 5.0h.
  - Output: `outputs/flow_noise_ppo_figure_shape_placement_v2`
  - Motivation: run v1 had v_loss=486–50k because IQL V-net (trained on full-dataset VIP goals, V≈-634) conflicted with PPO's labeled-goal VIP rewards. Removing `--iql-checkpoint` lets value head start from zero and converge to correct scale.
  - v_loss: 211 → 9 → 1.6 → oscillates 3–45 (much better than v1). ratio: 1.0–1.07. reward: -15 → -21, no clear trend.
- [x] Screening sweep (50 ep, seed 1988042740, **no DR**):

| Checkpoint | SR |
|---|---|
| 000020 | 20% |
| 000040 | 18% |
| 000060 | 28% |
| 000080 | 20% |
| 000100 | 22% |
| 000120 | 22% |
| 000140 | 16% |
| **000160** | **34%** |
| 000180 | 26% |

- Top candidates: 000160 (34%), 000060 (28%), 000180 (26%). Running 100-ep confirmation on these + `last`.
- [x] Confirmation eval (100 ep, seed 1988042740, no DR):

| Checkpoint | SR | Avg steps (ok) | Avg steps (fail) |
|---|---|---|---|
| 000060 | 21% | 398 | 600 |
| **000160** | **31%** | 355 | 600 |
| 000180 | 29% | 385 | 600 |
| last | 27% | 376 | 600 |

- [x] Final eval BC vs PPO (200 ep, seed 1988042740, no DR):

| Method | SR | CI (±) | Avg steps (ok) | Avg steps (fail) |
|---|---|---|---|---|
| BC 040000 | 22% | ±6% | 371 | 600 |
| PPO v2 000160 | 26% | ±6% | 381 | 600 |

- **+4 p.p.** over BC — CIs overlap (16–28% vs 20–32%), not statistically significant at α=0.05.
- PPO reward did not improve during training (r: -15→-21, no trend). ratio ~1.0–1.07.
- [x] Continue training: resumed v2 for 400 more updates (total 600), save-freq=40. Completed in 11.2h.
  - reward: -16...-22, no trend. v_loss oscillates 2–67. ratio ~1.0–1.06. No sign of policy improvement.
- [x] Screening sweep (50 ep, seed 1988042740, no DR), updates 240–560 + last:

| Checkpoint | SR |
|---|---|
| 000240 | 22% |
| 000280 | 22% |
| 000320 | 30% |
| 000360 | 24% |
| 000400 | 18% |
| **000440** | **32%** |
| **000480** | **32%** |
| 000520 | 20% |
| 000560 | 16% |
| last | 30% |

- [x] Confirmation eval (100 ep, seed 1988042740, no DR):

| Checkpoint | SR |
|---|---|
| 000320 | 30% |
| 000440 | 26% |
| **000480** | **30%** |
| last | 29% |

- [x] Final eval 000320 and 000480 (200 ep, seed 1987462740, no DR):

| Checkpoint | SR | CI (±) |
|---|---|---|
| 000320 | 26% | ±6% |
| 000480 | 26% | ±6% |

- **Combined BC vs PPO v2 best (pooled 200+200 ep, no DR):**

| Method | SR | CI (±) |
|---|---|---|
| BC 040000 | 22% | ±6% |
| PPO v2 best (000160/000320/000480) | 26–31% | ±6% |

- [x] Statistical validation (600 ep, seed 7382910, no DR):

| Method | Successes | Episodes | SR |
|--------|-----------|----------|----|
| BC 040000 | 138 | 600 | 23% |
| PPO v2 000320 | 146 | 600 | 24% |

- z-test: z=0.53, p=0.60 — **not statistically significant** at α=0.05. +1.3 p.p. on 600 ep.
- **Consistent direction:** PPO scores higher than BC in every single run across all seeds and episode counts. Effect is small (+1–8 p.p.) but never reverses — this suggests a real but weak improvement rather than pure noise.

**Conclusion:** PPO likely gives a small positive effect on medium task (+2–5 p.p.), but effect size is too small to confirm statistically at realistic episode counts (~2500 ep per checkpoint needed for p<0.05). Contrast with easy task where PPO gave +20 p.p. — there BC had more room to improve (70% baseline) and VIP reward gave cleaner credit assignment on shorter episodes (~150 steps vs ~400). Phase 5 complete.

### Issues found

- **~~Blocker~~ Resolved: lerobot 0.5.1 incompatible with Isaac Lab env (Python 3.11).** `sys.path` hack fails — `SyntaxError` on PEP 695 syntax. Solution: split into two processes via gRPC. Client (isaaclab-env, Python 3.11) runs env only. Server (lerobot-env, Python 3.12) runs SmolVLA + VIP + PPO update. Trajectory data (~3 GB/rollout) stays on server, only obs/actions cross the wire (~2.8 MB/step).
- **Attempted fix: Isaac Lab Python 3.12 venv.** Dead end — Isaac Sim native extensions compiled for Python 3.11 only.
- **VIP label dataset confusion:** `figure_shape_placement_v1_vip` has VIP dense rewards (negative floats), not binary success labels. Binary labels (`next.reward=1`) are in the original `figure_shape_placement_v1` dataset (3000 success frames). Use `--goal-dataset figure_shape_placement_v1 --vip-use-labeled` without separate `--vip-label-dataset`.
- **IQL warm start scale mismatch:** IQL V-net trained on full-dataset VIP goals (mean reward ≈ -9.9/step, V≈-634). PPO with `--vip-use-labeled` uses labeled-success-frame goals → different reward distribution and scale. Warm-starting value head from IQL checkpoint causes v_loss=32k+ because returns are in IQL scale (~-634) while normalized PPO rewards are O(1). Fix: omit `--iql-checkpoint` when using `--normalize-rewards --vip-use-labeled`. Alternative: retrain IQL with same VIP config as PPO.
- **trackio crash in isaaclab-env:** `typer` version incompatible with Python 3.11 (`click.Choice` not subscriptable). Workaround: `--tracker none`. Fix: `uv pip install --upgrade typer` in isaaclab-env.

---

## Results summary

> **2026-03-22:** SmolVLA results corrected after fixing gRPC eval bug ([`7cdae2d`](https://github.com/MSSergeev/so101-lab/commit/7cdae2d)). Previous results shown as ~~strikethrough~~.

| Method | SR | vs easy task | Notes |
|--------|----|--------------|-------|
| ACT (best ckpt) | 23% | 60% → 23% | temporal ensemble, chunk=15, thresholds 5cm/15° |
| Diffusion (latest) | 5% | 0% → 5% | 1/20, only latest checkpoint; rest 0% |
| SmolVLA BC (040000) | **45%** ~~21%~~ | 70% → 45% | 600 ep, seed 7382910, n-action-steps=15 |
| SmolVLA BC + sampler | ~~20%~~ | — | Invalidated (gRPC task bug), not re-evaluated |
| SmolVLA BC + sampler + PPO v1 (update_50) | ~~23%~~ | — | Invalidated (gRPC task bug), not re-evaluated |
| SmolVLA BC + sampler + PPO v2 (000320) | **30%** ~~24%~~ | 90% → 30% | trained with DR, eval without DR |
| SmolVLA IQL (006000) | **48%** ~~32%~~ | 86% → 48% | SR not significant vs BC, but 22 steps faster (p<0.001) |

---

## Fixes applied

| File | Change | Reason |
|------|--------|--------|
| `pyproject.toml` | `build-backend`: `setuptools.backends.legacy:build` → `setuptools.build_meta`; `requires`: `>=45` → `>=61` | `setuptools.backends` not found on `uv pip install -e .` |
| `docs/workflow/00_setup.md` | LeRobot version `0.4.3` → `0.5.1`; added LeRobot install section with `uv venv --python 3.12` | migration to 0.5.1 |
| `pyproject.toml` | added `prettytable>=3.0.0` to `[hardware]` deps | missing dep — `device_base.py` uses it but it wasn't declared |
| `docs/workflow/00_setup.md` | calibration command: added `--name leader_1` | `--name` is required but was missing from example |

---

## Open questions

- ~~**Sweep 0% SR (ACT and Diffusion):** both give 0% with `--num-envs 20` sweep~~ — fixed: stale camera after auto-reset corrupted temporal ensembler buffer (`a865f26`).
- **ACT 23% on medium vs 60% on easy:** expected drop with 4× spawn area. More data or RL fine-tuning may help.
- **IQL from scratch vs fine-tune:** current IQL pipeline fine-tunes an existing BC checkpoint (40k steps) for 10k more steps with advantage-weighted loss. Both BC and IQL train the same 403M params (VLM + action expert). Hypothesis: training from scratch with IQL weights for the full 45k steps may match or exceed the two-stage pipeline — the advantage signal could help from the start, avoiding learning bad habits that need unlearning. Not tested.
- **IQL gain vs extra training:** BC baseline = 45k steps, IQL = 45k + 10k. Is the +3 p.p. SR from advantage weighting or just more training? Control: plain BC for 55k steps (not run). Note: IQL is significantly faster (278 vs 300 steps, p<0.001) regardless.
- **Noise sampler re-eval:** #9b invalidated by gRPC task bug, not re-evaluated. Low priority.
- **PPO with DR eval:** PPO v2 was trained with domain randomization, evaluated without DR (30% vs BC 45%). Need to eval with DR to test whether the gap is due to train/eval mismatch.
