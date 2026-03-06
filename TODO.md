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

## 3. 3DGS scenes for sim-to-real

Replace procedural sim scenes with 3D Gaussian Splatting reconstructions of real workbench/environment. Goal: close the visual sim-to-real gap by training on photorealistic renders of the actual workspace. Requires capturing real scene with multi-view images, reconstructing 3DGS, and integrating as Isaac Lab scene background/environment.

