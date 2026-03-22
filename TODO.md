# TODO

---

## 1. Unify checkpoint directory structure across all training scripts

Current state: each training script saves checkpoints in its own format.

| Script | Format | Example |
|--------|--------|---------|
| SmolVLA BC (`lerobot-train`) | `checkpoints/NNNNNN/pretrained_model/` | `checkpoints/040000/pretrained_model/` |
| ACT (`train_act.py`) | `checkpoint_NNNNN/` | `checkpoint_35000/` |
| Diffusion (`train_diffusion.py`) | `checkpoint_NNNNN/` | `checkpoint_50000/` |
| PPO gRPC (`train_flow_noise_ppo_grpc.py`) | `checkpoints/NNNNNN/` + `last/` | `checkpoints/000050/pretrained_model/` |
| PPO original (`train_flow_noise_ppo.py`) | `update_N/` | `update_50/pretrained_model/` |
| SAC (`train_sac_composite_grpc.py`) | `checkpoints/NNNNNN/pretrained_model/` + `final/` | `checkpoints/000050/pretrained_model/` |
| Noise sampler | flat `noise_prior.pt` | `noise_prior.pt` |
| IQL critics | flat `critics.pt` | `final/critics.pt` |

Target: `checkpoints/NNNNNN/` everywhere (where applicable). Sweep scripts (`sweep_*_eval.py`) already expect specific formats — unifying would simplify eval workflows. Low priority — only matters when adding new sweep scripts or changing eval patterns.

---

## 2. Real robot deployment

HIL-SERL and real robot policy deployment are not yet implemented.
Current status: `so101_lab/rl/hil_input.py` and `hil_device.py` are stubs.

---

## 3. Reduce visual sim-to-real gap

Two approaches to explore (simplest first):

**3a. Digital twin via background replacement** — replace sim background with a real photo of the workspace using semantic segmentation (foreground = robot/objects, background = real photo). Inspired by leisaac's `ManagerBasedRLDigitalTwinEnv`. Effective for static top camera; limited for wrist camera due to changing perspective.

**3b. 3DGS scenes** — replace procedural sim scenes with 3D Gaussian Splatting reconstructions of the real workbench. Requires capturing multi-view images, reconstructing 3DGS, and integrating as Isaac Lab scene background. Better quality but significantly more effort.

---

## ~~4. Re-evaluate medium task results~~ (done 2026-03-22)

~~Medium task results in `docs/results/medium_task.md` (BC 22%, IQL 32%, PPO 24%) were collected with a bug in `smolvla_server.py`.~~ Fixed in [`7cdae2d`](https://github.com/MSSergeev/so101-lab/commit/7cdae2d). Re-evaluated: BC 45%, IQL 48%, PPO 30%. Results and docs updated.

---

## 5. Eval gRPC: support per-episode task strings

`smolvla_server.py` receives `task` once at `connect()` and uses it for all episodes. For multi-task evaluation (one model, different task instructions per episode), the server needs to accept `task` per `SendObservations` call.

---

## 6. PPO gRPC: support per-episode task strings

`ppo_server.py` receives the `task` string once at `Init` and bakes it into `FlowNoiseSmolVLA` for all episodes. For multi-task training, the server needs to accept `task` per `SampleAction` call, re-tokenize the language input, and invalidate the KV cache prefix. See [docs/reference/train_flow_noise_ppo_grpc.md](docs/reference/train_flow_noise_ppo_grpc.md#known-limitation-single-task-per-session).

