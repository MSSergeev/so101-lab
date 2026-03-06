# BC Training

Training and evaluating imitation learning policies: ACT, Diffusion Policy, SmolVLA.

All training runs in **lerobot-env**. All eval runs in **Isaac Lab env**.

Results for easy task: [results/easy_task.md](../results/easy_task.md).

---

## ACT

### Training

```bash
act-lerobot  # activate lerobot-env

python scripts/train/train_act.py \
    --dataset data/recordings/my_task_v1 \
    --config configs/policy/act/baseline.yaml \
    --name act_my_task_v1 \
    --tracker trackio --system-stats
```

`configs/policy/act/baseline.yaml`: 45k steps, batch 24, chunk 20, lr 1e-5, kl_weight 10.
Checkpoints every 5k steps + best by loss.

Variants: `high_kl.yaml`, `low_kl.yaml` — for tuning the KL weight.

### Eval

```bash
act-isaac  # activate Isaac Lab env

# Single checkpoint
python scripts/eval/eval_act_policy.py \
    --checkpoint outputs/act_my_task_v1 \
    --env figure_shape_placement_easy \
    --episodes 20 --max-steps 500 --headless

# Sweep multiple checkpoints (shared seed across runs)
python scripts/eval/sweep_act_eval_single.py \
    --checkpoint outputs/act_my_task_v1 \
    --checkpoints 25000 30000 35000 40000 --latest --best \
    --env figure_shape_placement_easy \
    --episodes 100 --max-steps 500
```

> **Note:** `sweep_act_eval_single.py` runs single-env eval (slower but supports `--save-episodes`). `sweep_act_eval.py` uses parallel envs for faster sweeps.

> **Temporal ensembling** (`--temporal-ensemble-coeff 0.01`) significantly hurts results at chunk=20 (observed: 15% vs 60%). Works better with small chunk sizes (1–5).

> **Loss vs success rate:** best checkpoint by loss does not always correspond to best success rate. Always sweep eval; do not rely on loss alone.

---

## Diffusion Policy

### Training

```bash
act-lerobot  # activate lerobot-env

python scripts/train/train_diffusion.py \
    --dataset data/recordings/my_task_v1 \
    --config configs/policy/diffusion/baseline.yaml \
    --name diffusion_my_task_v1 \
    --tracker trackio --system-stats
```

`configs/policy/diffusion/baseline.yaml`: 50k steps, batch 32, horizon 16, n_obs_steps 2, n_action_steps 8, lr 1e-4, 100 diffusion timesteps.

### Eval

```bash
python scripts/eval/sweep_diffusion_eval_single.py \
    --checkpoint outputs/diffusion_my_task_v1 \
    --checkpoints 5000 10000 20000 30000 40000 50000 --latest --best \
    --env figure_shape_placement_easy \
    --episodes 50 --max-steps 500 \
    --num-inference-steps 10
```

`--num-inference-steps 10` uses DDIM with 10 steps instead of 100 — ~10× faster eval.

> **Easy task result: 0% across all checkpoints.** Cause not investigated. Baseline config may not be suitable for this task.

---

## SmolVLA

SmolVLA is trained via `lerobot-train` CLI (LeRobot entry point). Four variants tested on easy task.

> **First time:** install `[smolvla]` extras before training — see [00_setup.md](00_setup.md#smolvla-fine-tuning-dependencies-lerobot-env-first-time-only).

### Key flags

| Flag | Effect |
|------|--------|
| `--policy.path=lerobot/smolvla_base` | Load pretrained SmolVLA (robotics weights) |
| `--policy.type=smolvla` | Create from scratch or load VLM-only weights |
| `--policy.load_vlm_weights=true` | Load SmolVLM2 vision+language weights, random action expert |
| `--policy.train_expert_only=false` | Train full model (required to unfreeze vision encoder) |
| `--policy.freeze_vision_encoder=false` | Unfreeze vision encoder (needs `train_expert_only=false`) |
| `--policy.empty_cameras=1` | Number of camera slots to leave empty in SmolVLA input |
| `--dataset.image_transforms.enable=true` | Augmentations: crop, color jitter |
| `--n-action-steps` | Steps to execute per inference call (eval only) |

> **`--rename_map`** is required only with `--policy.path` (pretrained SmolVLA has camera1/camera2 hardcoded). With `--policy.type=smolvla` the keys are taken from the dataset automatically — do not use rename_map, it corrupts `config.json` in the checkpoint.

### Variant A: pretrained SmolVLA (robotics weights)

```bash
act-lerobot  # activate lerobot-env

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=local/my_task_v1 \
    --dataset.root=data/recordings/my_task_v1 \
    --batch_size=8 --steps=30000 --save_freq=5000 \
    --output_dir=outputs/smolvla_pretrained_v1 \
    --policy.freeze_vision_encoder=false \
    --policy.push_to_hub=false \
    --policy.train_expert_only=false \
    --dataset.image_transforms.enable=true \
    --policy.empty_cameras=1 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

403M learnable / 450M total. ~3h on RTX 5070 Ti.

> **Easy task result: 0%.** Possible cause: Open X-Embodiment action distribution conflicts with sim. Supported indirectly by variant B (VLM-only weights) reaching 70–76%.

### Variant B: VLM pretrained, random action expert (recommended)

Vision/language weights from SmolVLM2 (web data), action expert initialized randomly.

```bash
lerobot-train \
    --policy.type=smolvla \
    --policy.load_vlm_weights=true \
    --dataset.repo_id=local/my_task_v1 \
    --dataset.root=data/recordings/my_task_v1 \
    --batch_size=8 --steps=30000 --save_freq=5000 \
    --output_dir=outputs/smolvla_vlm_v1 \
    --policy.freeze_vision_encoder=false \
    --policy.push_to_hub=false \
    --policy.train_expert_only=false \
    --dataset.image_transforms.enable=true \
    --policy.empty_cameras=1
```

**Easy task result: 70–76%** at checkpoint 030000.

### Variant C: frozen VLM, random action expert

`train_expert_only=true` (default) — only the action expert is trained, VLM frozen.

```bash
lerobot-train \
    --policy.type=smolvla \
    --policy.load_vlm_weights=true \
    --dataset.repo_id=local/my_task_v1 \
    --dataset.root=data/recordings/my_task_v1 \
    --batch_size=8 --steps=30000 --save_freq=5000 \
    --output_dir=outputs/smolvla_frozen_v1 \
    --policy.push_to_hub=false \
    --dataset.image_transforms.enable=true \
    --policy.empty_cameras=1
```

100M learnable / 450M total. ~1.7h.

**Easy task result: 70%** at checkpoint 020000. Difference vs variant B (70–76%) is within the ±7% confidence interval for 50 episodes — not conclusive.

### Variant D: from scratch

All weights random. Requires batch=4 (OOM at 8 — random init uses more VRAM).

```bash
lerobot-train \
    --policy.type=smolvla \
    --dataset.repo_id=local/my_task_v1 \
    --dataset.root=data/recordings/my_task_v1 \
    --batch_size=4 --steps=60000 --save_freq=5000 \
    --output_dir=outputs/smolvla_scratch_v1 \
    --policy.freeze_vision_encoder=false \
    --policy.push_to_hub=false \
    --policy.train_expert_only=false \
    --dataset.image_transforms.enable=true \
    --policy.empty_cameras=1
```

Use `--steps=60000` (3.2 epochs on 200 episodes) for a fair comparison with variants B/C.

> **Cosine scheduler and resume:** resuming from a checkpoint restores scheduler state. If the original run used `--steps=30000`, the lr is already at minimum (2.5e-6) at step 30k. The resume trains at minimum lr — progress is slow. To extend training properly, start a new run with the full step count from the beginning.

**Easy task result: 56%** at 60k steps.

### SmolVLA eval

```bash
act-isaac  # activate Isaac Lab env

python scripts/eval/sweep_vla_eval.py \
    --checkpoint outputs/smolvla_vlm_v1/checkpoints \
    --all \
    --env figure_shape_placement_easy \
    --episodes 50 --max-steps 500 --no-domain-rand \
    --n-action-steps 15
```

`--n-action-steps 15` is important — using default 50 gives significantly worse results on easy task.

### SmolVLA ablation summary (easy task)

| Variant | VLM weights | VLM frozen | Learnable | Batch | Epochs | Best SR | Best ckpt |
|---------|------------|------------|-----------|-------|--------|---------|-----------|
| A — pretrained SmolVLA | robotics | no | 403M | 8 | 3.2 | **0%** | — |
| B — VLM pretrained | web (SmolVLM2) | no | 403M | 8 | 3.2 | **70–76%** | 030000 |
| C — frozen VLM | web (SmolVLM2) | yes | 100M | 8 | 3.2 | **70%** | 020000 |
| D — scratch (60k) | random | no | 403M | 4 | 3.2 | **56%** | 030000 |

Key observations:
- **VLM pretraining matters**: B/C (70–76%) vs D (56%) — pretrained vision encoder gives +14–20%
- **Robotics pretraining hurts**: A (0%) — Open X-Embodiment weights conflict with sim distribution (possible cause, not verified)
- **Unfreezing VLM**: B vs C difference (70–76% vs 70%) is within statistical noise for 50 episodes
- **Scratch plateaus**: going from 1.6 to 3.2 epochs gives only +6%
