# VIPReward

Zero-shot visual reward via pretrained ResNet50 (Ego4D). lerobot-env venv.

Reward = `-‖φ(current_image) − φ(goal_image)‖₂` — negative L2 distance in embedding space. Closer to goal → closer to 0.

Adapted from [facebookresearch/vip](https://github.com/facebookresearch/vip) (CC BY-NC 4.0). Standalone loader — no `vip-utils` package needed.

---

Runs in **lerobot-env** (`act-lerobot`). See [setup](../workflow/00_setup.md).

## Overview

VIP is useful when ground-truth sim rewards are not available or not transferable to real. It requires only a set of goal images from successful demonstrations — no reward labeling.

**When to use:**
- `--reward-mode vip_only` — pure visual reward (sim2real-friendly)
- `--reward-mode composite` + `--vip-goal-dataset` — add VIP on top of sim shaping
- `prepare_reward_dataset.py --reward-source vip` — bake VIP rewards into offline dataset

---

## Model

```
Input: observation.images.top (H×W×3 uint8)
  → Resize(224, 224)              # no CenterCrop — object may be at frame edge
  → /255, ImageNet normalize
  → ResNet50 (Ego4D pretrained, frozen)
  → fc(2048 → 1024)
  → embedding (1024,)
  → reward = -‖φ(current) − φ(goal)‖₂
```

Weights: `~/.vip/resnet50/model.pt` (~98 MB), auto-downloaded from PyTorch S3 on first use.

---

## API

```python
from so101_lab.rewards.vip_reward import VIPReward

# Mean goal embedding (default — fast, 1 comparison per step)
vip = VIPReward("data/recordings/figure_shape_placement_v4")

# Min distance to closest of N goal embeddings
vip = VIPReward("data/recordings/figure_shape_placement_v4", goal_mode="min")

# Use labeled success frames as goals
vip = VIPReward(
    "data/recordings/figure_shape_placement_v4",
    use_labeled=True,
    label_dataset_path="data/recordings/figure_shape_placement_v4_labeled",
)

# With normalization to [-1, 0]
vip = VIPReward(
    "data/recordings/figure_shape_placement_v4",
    normalize=True,
    scale_dataset_path="data/recordings/figure_shape_placement_v4_vip_128",  # optional
)

obs = {"observation.images.top": img_hwc_uint8}  # (H, W, 3) numpy uint8
reward = vip.compute_reward(obs)  # float, ≤ 0 (closer to 0 = closer to goal)
```

### Constructor arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `goal_dataset_path` | required | LeRobot dataset — source of goal images |
| `device` | `"cuda"` | Torch device |
| `image_key` | `"observation.images.top"` | Which camera to embed |
| `n_goal_frames` | 5 | Final frames per episode used as goals (when `use_labeled=False`) |
| `goal_mode` | `"mean"` | `"mean"` — averaged goal embedding; `"min"` — closest of all goals |
| `use_labeled` | `False` | Use frames with `next.reward > 0.5` as goals |
| `label_dataset_path` | `None` | Dataset with `next.reward` labels (images from `goal_dataset_path`) |
| `normalize` | `False` | Normalize reward to `[-1, 0]` |
| `scale_dataset_path` | `None` | Dataset with precomputed VIP rewards for normalization scale |
| `reward_clip` | `(-2.0, 0.0)` | Clip range after normalization |

---

## Goal Embedding

Goal frames are encoded once at init and cached to disk (`<goal_dataset_path>/vip_goal_cache_<hash>.pt`). Cache is keyed on `image_key`, `use_labeled`, `label_path`, `n_goal_frames`.

**Two goal sources:**

| Source | When | Description |
|--------|------|-------------|
| Final frames | `use_labeled=False` (default) | Last `n_goal_frames` of each episode |
| Labeled success | `use_labeled=True` | Frames with `next.reward > 0.5` from `label_dataset_path` |

Labeled source requires a gripper-labeled dataset (from `prepare_classifier_dataset.py`). Max 1000 frames are subsampled if more are available.

**Two aggregation modes:**

| Mode | Description | Speed |
|------|-------------|-------|
| `"mean"` | Single averaged embedding — one L2 comparison per step | Fast |
| `"min"` | All goal embeddings — reward = min distance to closest goal | Slower (N comparisons) |

`"min"` is more conservative: reward is high only when the current frame resembles at least one specific success frame. `"mean"` can be pulled by outlier goal frames.

---

## Reward Normalization

Raw VIP reward range is ~[−20, −3] (task-dependent). Without normalization, episode returns are on the order of −6000, which causes critic loss spikes in SAC.

`normalize=True` maps reward to [−1, 0] and clips to `reward_clip`:

```
normalized = raw / scale
clipped = clip(normalized, reward_clip)
```

Scale computation:
1. If `scale_dataset_path` has `next.reward` (precomputed VIP rewards) → `scale = |5th percentile|`
2. Otherwise → sample 100 random frames from `goal_dataset_path`, measure distances → `scale = 95th percentile`

**Always use `--vip-normalize` in SAC training** when VIP reward range is unknown.

---

## Usage in Training

### Online SAC

```bash
# Sim + VIP (normalized)
python scripts/train/train_sac_composite_grpc.py \
    --reward-mode sim_only \
    --vip-goal-dataset data/recordings/figure_shape_placement_v4 \
    --vip-label-dataset data/recordings/figure_shape_placement_v4_labeled \
    --vip-use-labeled --vip-goal-mode min \
    --w-vip 1.0 --vip-normalize \
    --num-steps 50000 --output outputs/sac_vip_v1 \
    --auto-server --image-size 128 --headless

# VIP only (no sim rewards — for sim2real)
python scripts/train/train_sac_composite_grpc.py \
    --reward-mode vip_only \
    --vip-goal-dataset data/recordings/figure_shape_placement_v4 \
    --vip-use-labeled \
    --vip-normalize --w-vip 1.0 \
    --num-steps 50000 --output outputs/sac_vip_only_v1 \
    --auto-server --image-size 128 --headless
```

### Bake into offline dataset

```bash
python scripts/train/prepare_reward_dataset.py \
    --reward-source vip \
    --dataset     data/recordings/figure_shape_placement_v4 \
    --output      data/recordings/figure_shape_placement_v4_vip_128 \
    --vip-goal-dataset  data/recordings/figure_shape_placement_v4 \
    --vip-label-dataset data/recordings/figure_shape_placement_v4_labeled \
    --vip-use-labeled --vip-goal-mode min \
    --image-size 128
```

Then use `--vip-normalize` + `--demo-dataset` in SAC to compute scale from the precomputed rewards.

### Diagnostic tool

```bash
act-lerobot  # activate lerobot-env

python scripts/tools/test_vip_reward.py

python scripts/tools/test_vip_reward.py \
    --goal-mode min --use-labeled \
    --goal-dataset data/recordings/figure_shape_placement_v4_labeled
```

---

## Notes

- **Resize not CenterCrop**: CenterCrop can cut the cube when it's near frame edge. `Resize(224, 224)` preserves the full image.
- **No `vip-utils` package**: `vip-utils` has incompatible deps (`Pillow==9.0.1`). Weights are loaded standalone by stripping `module.convnet.` prefix from the state dict.
- **Goal cache**: recomputed if any constructor arg changes. Delete `vip_goal_cache_*.pt` to force recompute.
- **Image source vs label source**: `label_dataset_path` supplies `next.reward` labels; images are always taken from `goal_dataset_path`. This allows using a 128×128 labeled dataset while keeping goal images at full resolution.
- VIP reward is not dt-scaled in the training loop (discrete-style signal, like classifier).

---

## Files

```
so101_lab/rewards/
├── vip_reward.py              # VIPReward class
└── __init__.py

scripts/tools/
└── test_vip_reward.py         # Diagnostic: reward range on demo episodes
```
