# Offline RL

IQL pipeline: reward labeling → critic training → advantage-weighted BC fine-tuning.

All steps run in **lerobot-env** except eval, which runs in **Isaac Lab env**.

Results for easy task: [results/easy_task.md](../results/easy_task.md).

---

## Overview

The IQL pipeline improves a BC policy by re-weighting demonstrations by their advantage:

1. **Label rewards** — assign a dense reward to each frame using VIP or sim rewards
2. **Train IQL critics** — learn V(s) and Q(s,a) from the labeled dataset
3. **Compute advantage weights** — `exp(β * A(s,a))` per frame, where `A = Q - V`
4. **Weighted BC fine-tune** — re-train the policy with per-sample loss scaling

Good demonstrations (high advantage) receive a higher loss weight; poor ones are suppressed.

---

## Step 1: Label rewards

### VIP dense rewards (recommended)

VIP computes a dense reward as the negative distance between the current frame and a goal embedding in a ResNet50 feature space trained on Ego4D video.

```bash
act-lerobot  # activate lerobot-env

python scripts/train/prepare_reward_dataset.py \
    --dataset data/recordings/my_task_v1 \
    --output data/recordings/my_task_v1_vip \
    --reward-source vip \
    --vip-goal-dataset data/recordings/my_task_v1 \
    --vip-use-labeled --vip-goal-mode mean
```

`--vip-use-labeled` uses only frames with `next.reward=1` as goal frames.
`--vip-goal-mode mean` averages them into a single goal embedding.

VIP weights (~98 MB) are downloaded automatically to `~/.vip/resnet50/` on first run.

The output dataset is a copy of the original with an additional `next.reward` column containing dense float rewards.

> **Easy task stats:** range [-16.18, -0.32], mean=-9.63. 1000 goal frames (subsampled from 2000 success frames).

### Sim sparse rewards (baseline)

If the dataset was recorded with `--reward-mode success`, the `next.reward` column already contains binary 0/1 rewards — no labeling needed. Pass the original dataset directly to the critic training step.

---

## Step 2: Train IQL critics

```bash
python scripts/train/train_iql_critics.py \
    --dataset data/recordings/my_task_v1_vip \
    --output outputs/my_task_iql_critics_v1
```

Default: 50k steps, batch 256, τ=0.7, γ=0.99, VIP encoder.

> **Easy task (VIP rewards, 50k steps):** V≈-618, Q≈-619. ~2.3 min on RTX 5070 Ti.

For sparse rewards, use more steps (V/Q take longer to converge on binary signal):

```bash
python scripts/train/train_iql_critics.py \
    --dataset data/recordings/my_task_v1 \
    --output outputs/my_task_iql_critics_sim_v1 \
    --num-steps 200000
```

Encoder options: `--encoder vip` (default) or `--encoder imagenet`.

---

## Step 3: Compute advantage weights

```bash
python scripts/tools/compute_iql_weights.py \
    --dataset data/recordings/my_task_v1_vip \
    --critics outputs/my_task_iql_critics_v1/final/critics.pt \
    --beta 1.0
```

Writes `iql_weights.pt` alongside the dataset. `β=1.0` is a reasonable default.

> **Easy task (VIP rewards):** advantage mean=-1.58, range [-31.8, +22.1]. Weights: mean=4.70, 28.5% >1 (better than average), 1.5% clipped.

---

## Step 4: Weighted BC fine-tuning

`train_vla_weighted_bc.py` wraps `lerobot-train` with a monkey-patch: `SmolVLAPolicy.forward()` multiplies per-sample loss by the corresponding `iql_weight`. This is a fine-tune, not a resume — new optimizer and scheduler are created from scratch.

```bash
python scripts/train/train_vla_weighted_bc.py \
    --policy.path=outputs/smolvla_vlm_v1/checkpoints/030000/pretrained_model \
    --dataset.repo_id=local/my_task_v1_vip \
    --dataset.root=data/recordings/my_task_v1_vip \
    --output_dir=outputs/smolvla_iql_v1 \
    --batch_size=8 --steps=10000 --save_freq=2000
```

The VLM is fully unfrozen (403M params) — inherited from the source checkpoint config (`train_expert_only=false`, `freeze_vision_encoder=false`).

> **Easy task:** loss 0.017→0.012 over 10k steps. ~30 min.

---

## Step 5: Eval

```bash
act-isaac  # activate Isaac Lab env

python scripts/eval/sweep_vla_eval.py \
    --checkpoint outputs/smolvla_iql_v1/checkpoints \
    --all \
    --env figure_shape_placement_easy \
    --episodes 50 --max-steps 500 --no-domain-rand \
    --n-action-steps 15 --seed 38531708
```

Use the same seed and `--n-action-steps 15` as BC eval for a fair comparison.

---

## Easy task results

| Variant | Encoder | Reward | Best SR | vs baseline |
|---------|---------|--------|---------|-------------|
| #8 — VIP dense | VIP | VIP dense float | **86–88%** | +12–16% |
| #8b — sim sparse | VIP | sim 0/1 | **76%** | ≈ baseline |
| #8c — ImageNet encoder | ImageNet | sim 0/1 | **74%** | ≈ baseline |

Baseline (SmolVLA BC, variant B): 70–76%.

Key findings:
- **VIP dense rewards** provide a useful advantage signal (+12–16% over baseline)
- **Sim sparse rewards** (0/1) do not improve results with either encoder
- **Encoder choice does not matter** when reward quality is low: VIP 76% ≈ ImageNet 74%
- The reward signal quality is the key factor, not the encoder

> **Note:** results are non-deterministic — flow matching ODE starts from random x_0, so each eval run gives different actions even on identical scenes. The seed controls only the initial object positions.

---

## Ablation: checkpoint sweep (easy task, #8)

| Checkpoint | Success% |
|------------|----------|
| 002000 | 40% |
| 004000 | 20% |
| 006000 | 52% |
| 008000 | 66% |
| **010000** | **86%** |

Loss decreases monotonically (0.017→0.012). Success rate is non-monotone early — use sweep eval, not loss.

---

## Open questions

- **Is the gain from IQL weights or extra training steps?** Baseline = 30k steps, IQL = 30k + 10k fine-tune. A control experiment (plain BC for 40k steps) would clarify whether the advantage weighting is necessary.
- **More IQL steps?** The trend is increasing (40%→86% over 2k→10k steps). 20k steps weighted BC not tested.
