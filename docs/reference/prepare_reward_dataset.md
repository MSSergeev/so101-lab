# prepare_reward_dataset.py

Label a LeRobot dataset with `next.reward` for offline RL training (IQL critics, weighted BC, SAC replay buffer). lerobot-env venv.

---

## Overview

Takes a recorded demo dataset and writes `next.reward` into the parquet files. Three reward sources are supported:

| `--reward-source` | What `next.reward` contains | Use case |
|-------------------|-----------------------------|----------|
| `classifier` (default) | Binary 0/1 from image classifier | Cleanest signal; requires trained classifier |
| `vip` | Continuous distance in VIP embedding space | Dense reward; no labeled data needed |
| `composite` | `classifier + vip * w_vip` | Combines binary terminal + dense shaping |

This script is used to prepare data for:
- SAC replay buffer (offline seed transitions)
- IQL critics training (`train_iql_critics.py`)
- Weighted BC (`train_vla_weighted_bc.py`)

See also:
- Classifier training: [reference/reward_classifier.md](reward_classifier.md)
- VIP reward model: [reference/vip_reward.md](vip_reward.md)

---

## CLI

### Classifier only (default)

Binary reward from `RewardClassifier`. Requires a trained classifier checkpoint.

```bash
python scripts/train/prepare_reward_dataset.py \
    --dataset     data/recordings/figure_shape_placement_v4 \
    --output      data/recordings/figure_shape_placement_v4_clf \
    --reward-model outputs/reward_classifier_v2/best \
    --threshold   0.5
```

- `--threshold` — classifier confidence threshold for binary 0/1 (default: 0.5; train script uses 0.9 at inference — use 0.5 here for a softer boundary in offline data)
- Images are read from the output dataset (already resized if `--image-size` was applied)

### VIP only

Continuous reward from VIP embedding distance to goal. No classifier needed; goal frames are sampled from a reference dataset.

```bash
python scripts/train/prepare_reward_dataset.py \
    --reward-source vip \
    --dataset       data/recordings/figure_shape_placement_v4 \
    --output        data/recordings/figure_shape_placement_v4_vip \
    --vip-goal-dataset   data/recordings/figure_shape_placement_v4 \
    --vip-label-dataset  data/recordings/figure_shape_placement_v4_labeled \
    --vip-use-labeled \
    --vip-goal-mode min
```

- `--vip-goal-dataset` — dataset to sample goal frames from (can be same as `--dataset`)
- `--vip-use-labeled` + `--vip-label-dataset` — filter goal candidates to frames with `next.reward > 0.5` (i.e. actual success frames from gripper labeling)
- `--vip-goal-mode mean` — encode one mean goal embedding; `min` — take closest of N embeddings per step (more conservative)
- `--n-goal-frames` — how many final frames per episode to use as goal candidates (default: 5)
- `--vip-camera` — which image key to use (default: `observation.images.top`)
- VIP uses full-resolution images; its own `Resize(224, 224)` handles preprocessing internally

### Composite (classifier + VIP)

Combines binary termination signal (classifier) with dense distance-to-goal shaping (VIP):

```
next.reward = clf_reward + vip_reward * w_vip
```

```bash
python scripts/train/prepare_reward_dataset.py \
    --reward-source composite \
    --dataset       data/recordings/figure_shape_placement_v4 \
    --output        data/recordings/figure_shape_placement_v4_composite \
    --reward-model  outputs/reward_classifier_v2/best \
    --vip-goal-dataset data/recordings/figure_shape_placement_v4 \
    --w-vip         1.0 \
    --image-size    128
```

- `--w-vip` — weight for VIP component (default: 1.0)

---

## Common flags

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Source LeRobot dataset path |
| `--output` | required | Output labeled dataset path |
| `--reward-source` | `classifier` | `classifier` / `vip` / `composite` |
| `--image-size` | `0` | Resize videos to N×N before reward computation (0 = no resize; use 128 for classifier) |

`--image-size 128` triggers ffmpeg re-encoding of all videos and updates `info.json` metadata. The classifier was trained on 128×128 images; if the dataset is at full resolution (480×640), pass `--image-size 128` so the classifier reads the resized frames.

---

## How it works

1. **Copy dataset** to output path, adding a dummy `next.reward` column of zeros (via `lerobot.datasets.dataset_tools.add_features`) — or copy as-is if `next.reward` already exists
2. **Resize videos** (optional, via ffmpeg)
3. **Compute rewards** — iterate all frames, call classifier / VIP / both
4. **Write rewards** — patch `next.reward` column in every parquet file

If the source dataset already has `next.reward` (e.g. sim rewards written during recording), the dataset is copied without modification — rewards are not overwritten.

---

## Notes

- **Classifier reads resized images.** Always match `--image-size` to what the classifier was trained on (128 for v1/v2).
- **VIP reads original images.** Do not resize before VIP if using `vip` mode — pass `--image-size 0` or omit it.
- For **composite** mode, the script resizes first, then runs classifier on resized images and VIP on the original dataset (separate `LeRobotDataset` instances).
- The output dataset is a full copy of the input; no frames are dropped.
