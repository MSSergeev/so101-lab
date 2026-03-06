# Reward Classifier

Binary success/fail classifier for use as a reward signal in SAC/PPO training. lerobot-env venv.

Unlike sim ground-truth rewards, the classifier operates on images and can transfer sim→real.

---

## Overview

Pipeline: demo recordings → gripper-based labeling → classifier training → inference in training loop.

All scripts run in **lerobot-env** (no Isaac Sim needed).

---

## Step 1 — Label Dataset (`prepare_classifier_dataset.py`)

Adds a `next.reward` column to a LeRobot dataset based on gripper state, then re-encodes videos to 128×128 (required by SpatialLearnedEmbeddings — ResNet-18 stride=32, 128/32=4).

**Labeling logic:** In pick-and-place, the robot opens its gripper to release the object. The frame where gripper value crosses the threshold upward = moment of release. All frames after → reward 1.0; before → 0.0. Episodes without a crossing → all frames = 0.0 (warning printed).

```
observation.state[:, 5]  (gripper normalized)

100 ─┐
     │                    ╱──── success (reward=1.0)
29.5 ├──────threshold────/──────────────────────────
     │                  ╱
   0 ├──────────────────
     └──────────────────┬──────── time
                     release
```

```bash
python scripts/train/prepare_classifier_dataset.py \
    --dataset data/recordings/figure_shape_placement_v4 \
    --output  data/recordings/figure_shape_placement_v4_labeled \
    --gripper-threshold 29.5 \
    --gripper-index 5
```

---

## Step 2 — Train (`train_reward_classifier_v2.py`)

**v2 (recommended):** fine-tunes the ResNet-18 backbone end-to-end.

**v1 (`train_reward_classifier.py`) is deprecated:** LeRobot's `Classifier._get_encoder_output()` wraps the encoder in `torch.no_grad()`, so only the final `classifier_head` receives gradients. In practice this yields ~50% episode-level success recall (median probability on last frames ≈ 0.5).

### v2 differences from v1

- Backbone unfrozen + `torch.no_grad()` patched out of encoder forward
- Differential LR: backbone 1e-5, head 1e-4
- Balanced sampling: `WeightedRandomSampler` equalizes success/fail per batch
- Gradient clipping (norm=1.0) for stable fine-tuning

### Model architecture

```
Input: two 128×128 RGB images (top + wrist cameras)
  └─ ResNet-18 backbone (ImageNet pretrained) → [B, 512, 4, 4]
  └─ SpatialLearnedEmbeddings                → [B, 512×8] per camera
  └─ Linear → LayerNorm → Tanh              → [B, 256] per camera
  └─ Concat cameras                         → [B, 512]
  └─ Linear(512, 256) → Dropout → LayerNorm → ReLU → Linear(256, 1)
Training loss: BCEWithLogitsLoss
Inference:     sigmoid(logit) > threshold
```

### Dataset split

Split is by **episode** (not by frame — frames within an episode are correlated):
- Train 70% — backprop
- Val 15% — best checkpoint selection (early stopping on accuracy)
- Test 15% — final evaluation only, not used for checkpoint selection

```bash
python scripts/train/train_reward_classifier_v2.py \
    --dataset data/recordings/figure_shape_placement_v4_labeled \
    --output  outputs/reward_classifier_v2 \
    --epochs 15 --batch-size 32 \
    --backbone-lr 1e-5 --head-lr 1e-4 \
    --tracker trackio
```

---

## Step 3 — Evaluate (`eval_reward_classifier.py`)

Evaluates a checkpoint against the test set. Reads episode splits from `train_config.json`.

```bash
# Best checkpoint on test set:
python scripts/eval/eval_reward_classifier.py \
    --checkpoint outputs/reward_classifier_v2/best \
    --dataset     data/recordings/figure_shape_placement_v4_labeled \
    --train-config outputs/reward_classifier_v2/train_config.json

# All splits (train / val / test):
python scripts/eval/eval_reward_classifier.py \
    --checkpoint outputs/reward_classifier_v1/best \
    --dataset     data/recordings/figure_shape_placement_v4_labeled \
    --train-config outputs/reward_classifier_v1/train_config.json \
    --all-splits
```

### Analyze probability distribution (`analyze_classifier_distribution.py`)

Shows probability histogram, class statistics, grey zone, and accuracy at various thresholds. Useful for threshold selection and calibration checks.

```bash
python scripts/eval/analyze_classifier_distribution.py \
    --checkpoint   outputs/reward_classifier_v2/best \
    --dataset      data/recordings/figure_shape_placement_v4_labeled \
    --train-config outputs/reward_classifier_v2/train_config.json
```

---

## Inference API

```python
from so101_lab.rewards.classifier import RewardClassifier

clf = RewardClassifier(
    "outputs/reward_classifier_v2/best",
    device="cuda",
    threshold=0.9,   # default
)

obs = {
    "observation.images.top":   img_top,    # (H, W, 3) uint8 numpy
    "observation.images.wrist": img_wrist,  # (H, W, 3) uint8 numpy
}

reward = clf.predict_reward(obs)      # 0.0 or 1.0
prob   = clf.predict_probability(obs) # float [0.0, 1.0]
# Images are resized to 128×128 internally
```

`RewardClassifier` auto-detects image keys from `classifier.config.input_features` and handles the `observation.image.*` ↔ `observation.images.*` key name discrepancy.

---

## Threshold Selection

Default threshold: **0.9** (minimizes false positives in SAC training — false positives teach incorrect Q-values).

v1 results on test set (30 episodes):
- Class 0 (fail): 96% of predictions in [0.0–0.1], mean=0.018
- Class 1 (success): 90% in [0.9–1.0], mean=0.952
- Grey zone [0.3–0.7]: 1.5% of samples

| Threshold | Accuracy | False Positives | False Negatives |
|-----------|----------|-----------------|-----------------|
| 0.5 | 98.62% | ~1.1% | ~3.4% |
| 0.7 | 98.88% | ~0.6% | ~4.9% |
| 0.9 | ~99.0% | ~0.24% | ~9.7% |

Note: these numbers are from v1 (frozen backbone). The high test accuracy is misleading — v1 fails to recognize success in ~50% of episodes at inference time. Use v2.

---

## Domain Randomization and Classifier

The classifier must be trained on data matching the lighting conditions used during SAC training.

| Demo recording | SAC training | Result |
|----------------|--------------|--------|
| Fixed light | Fixed light | Works |
| Fixed light | Random light | Domain gap — spurious rewards |
| Random light | Random light | Works (retrain classifier on DR data) |

If the classifier was trained on fixed lighting, pass `--no-randomize-light` to SAC/PPO training scripts, or use `IsaacLabGymEnv(env_cfg, randomize_light=False)`.

---

## Output Artifacts

```
outputs/reward_classifier_v2/
├── config.json             # RewardClassifierConfig
├── model.safetensors       # Final model weights
├── policy_preprocessor.*   # Normalization stats
├── train_config.json       # Hyperparams + episode splits (train/val/test indices)
└── best/
    ├── config.json
    ├── model.safetensors
    ├── policy_preprocessor.*
    └── eval_results.json   # Populated after running eval_reward_classifier.py
```

`train_config.json` stores `train_episodes`, `val_episodes`, `test_episodes` lists for reproducibility.

---

## Files

```
scripts/train/
├── prepare_classifier_dataset.py     # Step 1: label + re-encode to 128×128
├── train_reward_classifier_v2.py     # Step 2: fine-tune (recommended)
└── train_reward_classifier.py        # Step 2: frozen backbone (deprecated)

scripts/eval/
├── eval_reward_classifier.py         # Step 3: evaluate on test set
└── analyze_classifier_distribution.py  # Probability histogram diagnostic

so101_lab/rewards/
├── classifier.py                     # RewardClassifier inference wrapper
└── __init__.py
```

See also: `prepare_reward_dataset.py` for labeling datasets with classifier/VIP/composite rewards for offline RL.
