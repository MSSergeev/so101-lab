#!/usr/bin/env python3
"""Analyze reward classifier probability distribution.

Shows histogram and statistics for classifier predictions on test set.
Helps understand classifier confidence and calibration.

Run in lerobot-env:
    eval "$(./activate_lerobot.sh)"

    python scripts/eval/analyze_classifier_distribution.py \
        --checkpoint outputs/reward_classifier_v1/best \
        --dataset data/recordings/figure_shape_placement_v4_labeled \
        --train-config outputs/reward_classifier_v1/train_config.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.policies.factory import make_pre_post_processors


@torch.no_grad()
def collect_predictions(policy, dataloader, preprocessor) -> tuple[np.ndarray, np.ndarray]:
    """Collect all predictions and labels."""
    policy.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Collecting predictions"):
        batch = preprocessor(batch)
        images = [batch[key] for key in policy.config.input_features if key.startswith("observation.image")]
        output = policy.predict(images)

        all_probs.extend(output.probabilities.cpu().numpy().tolist())
        all_labels.extend(batch["next.reward"].cpu().numpy().tolist())

    return np.array(all_probs), np.array(all_labels)


def print_histogram(probs: np.ndarray, labels: np.ndarray):
    """Print ASCII histogram of probability distribution."""
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Split by true label
    probs_0 = probs[labels == 0]
    probs_1 = probs[labels == 1]

    print("\n" + "=" * 60)
    print("PROBABILITY DISTRIBUTION")
    print("=" * 60)

    print(f"\nTrue class=0 (fail): {len(probs_0)} samples")
    print(f"True class=1 (success): {len(probs_1)} samples")

    # Histogram for class 0
    print("\n--- True class=0 (should predict LOW probability) ---")
    counts_0, _ = np.histogram(probs_0, bins=bins)
    max_count = max(counts_0) if len(counts_0) > 0 else 1
    for i in range(len(bins) - 1):
        bar_len = int(40 * counts_0[i] / max_count) if max_count > 0 else 0
        pct = 100 * counts_0[i] / len(probs_0) if len(probs_0) > 0 else 0
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}] {'█' * bar_len:<40} {counts_0[i]:>5} ({pct:>5.1f}%)")

    # Histogram for class 1
    print("\n--- True class=1 (should predict HIGH probability) ---")
    counts_1, _ = np.histogram(probs_1, bins=bins)
    max_count = max(counts_1) if len(counts_1) > 0 else 1
    for i in range(len(bins) - 1):
        bar_len = int(40 * counts_1[i] / max_count) if max_count > 0 else 0
        pct = 100 * counts_1[i] / len(probs_1) if len(probs_1) > 0 else 0
        print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}] {'█' * bar_len:<40} {counts_1[i]:>5} ({pct:>5.1f}%)")


def print_statistics(probs: np.ndarray, labels: np.ndarray):
    """Print detailed statistics."""
    probs_0 = probs[labels == 0]
    probs_1 = probs[labels == 1]

    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    # Basic stats
    print(f"\nClass 0 (fail) predictions:")
    print(f"  Mean prob: {probs_0.mean():.4f}")
    print(f"  Std:       {probs_0.std():.4f}")
    print(f"  Min/Max:   {probs_0.min():.4f} / {probs_0.max():.4f}")

    print(f"\nClass 1 (success) predictions:")
    print(f"  Mean prob: {probs_1.mean():.4f}")
    print(f"  Std:       {probs_1.std():.4f}")
    print(f"  Min/Max:   {probs_1.min():.4f} / {probs_1.max():.4f}")

    # Gray zone analysis
    gray_low, gray_high = 0.3, 0.7
    in_gray_0 = np.sum((probs_0 >= gray_low) & (probs_0 <= gray_high))
    in_gray_1 = np.sum((probs_1 >= gray_low) & (probs_1 <= gray_high))

    print(f"\nGray zone [{gray_low}-{gray_high}]:")
    print(f"  Class 0: {in_gray_0}/{len(probs_0)} ({100*in_gray_0/len(probs_0):.1f}%)")
    print(f"  Class 1: {in_gray_1}/{len(probs_1)} ({100*in_gray_1/len(probs_1):.1f}%)")
    print(f"  Total:   {in_gray_0 + in_gray_1}/{len(probs)} ({100*(in_gray_0+in_gray_1)/len(probs):.1f}%)")

    # Confidence analysis
    predictions = (probs > 0.5).astype(int)
    correct = predictions == labels
    confidence = np.abs(probs - 0.5) * 2  # 0 = uncertain, 1 = confident

    print(f"\nConfidence (distance from 0.5):")
    print(f"  Correct predictions:   mean={confidence[correct].mean():.4f}")
    print(f"  Incorrect predictions: mean={confidence[~correct].mean():.4f}")

    # Accuracy at different thresholds
    print(f"\nAccuracy at different thresholds:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (probs > thresh).astype(int)
        acc = 100 * np.mean(preds == labels)
        print(f"  threshold={thresh}: {acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze classifier distribution")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train-config", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = Classifier.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    # Load dataset
    dataset_path = Path(args.dataset).absolute()
    full_dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, dataset_stats=full_dataset.meta.stats,
    )

    # Get episodes for split
    episodes = None
    if args.train_config:
        with open(args.train_config) as f:
            train_config = json.load(f)
        if args.split != "all":
            episodes = train_config.get(f"{args.split}_episodes")
            print(f"Using {args.split} split: {len(episodes)} episodes")

    dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path, episodes=episodes)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type != "cpu",
    )

    print(f"Dataset: {len(dataset)} frames")

    # Collect predictions
    probs, labels = collect_predictions(policy, loader, preprocessor)

    # Print analysis
    print_histogram(probs, labels)
    print_statistics(probs, labels)


if __name__ == "__main__":
    main()
