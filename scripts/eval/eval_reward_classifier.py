#!/usr/bin/env python3
"""Evaluate reward classifier on test set.

Loads a trained checkpoint and evaluates on test episodes saved in train_config.json.
Can also evaluate on arbitrary episode lists or the full dataset.

Run in lerobot-env:
    eval "$(./activate_lerobot.sh)"

    # Evaluate best checkpoint on test set:
    python scripts/eval/eval_reward_classifier.py \
        --checkpoint outputs/reward_classifier_v1/best \
        --dataset data/recordings/figure_shape_placement_v4_labeled \
        --train-config outputs/reward_classifier_v1/train_config.json

    # Evaluate any checkpoint on test set:
    python scripts/eval/eval_reward_classifier.py \
        --checkpoint outputs/reward_classifier_v1/checkpoint_5 \
        --dataset data/recordings/figure_shape_placement_v4_labeled \
        --train-config outputs/reward_classifier_v1/train_config.json

    # Evaluate on all splits (train/val/test):
    python scripts/eval/eval_reward_classifier.py \
        --checkpoint outputs/reward_classifier_v1/best \
        --dataset data/recordings/figure_shape_placement_v4_labeled \
        --train-config outputs/reward_classifier_v1/train_config.json \
        --all-splits

    # Evaluate on full dataset (no split):
    python scripts/eval/eval_reward_classifier.py \
        --checkpoint outputs/reward_classifier_v1/best \
        --dataset data/recordings/figure_shape_placement_v4_labeled
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.reward_model.modeling_classifier import Classifier
from lerobot.policies.factory import make_pre_post_processors


@torch.no_grad()
def evaluate(policy, dataloader, preprocessor) -> dict:
    """Run evaluation and return metrics."""
    policy.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in dataloader:
        batch = preprocessor(batch)
        loss, output_dict = policy.forward(batch)

        total_loss += loss.item()
        total_correct += output_dict["correct"]
        total_samples += output_dict["total"]
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "accuracy": 100 * total_correct / total_samples,
        "correct": total_correct,
        "total": total_samples,
    }


def eval_split(
    policy, preprocessor, dataset_path: Path, episodes: list[int] | None,
    split_name: str, batch_size: int, num_workers: int, device: torch.device,
) -> dict:
    """Evaluate on a specific split."""
    dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path, episodes=episodes)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    metrics = evaluate(policy, loader, preprocessor)
    n_episodes = len(episodes) if episodes else dataset.meta.total_episodes
    print(f"  {split_name:>5s}: accuracy={metrics['accuracy']:.2f}% | "
          f"loss={metrics['loss']:.4f} | "
          f"{metrics['correct']}/{metrics['total']} frames | "
          f"{n_episodes} episodes")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate reward classifier")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to labeled LeRobot dataset")
    parser.add_argument("--train-config", type=str, default=None,
                        help="Path to train_config.json (for split info)")
    parser.add_argument("--all-splits", action="store_true",
                        help="Evaluate on all splits (train/val/test)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint).absolute()
    print(f"Loading checkpoint: {checkpoint_path}")
    policy = Classifier.from_pretrained(str(checkpoint_path))
    policy.to(device)
    policy.eval()

    # Load dataset for stats
    dataset_path = Path(args.dataset).absolute()
    full_dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy.config, dataset_stats=full_dataset.meta.stats,
    )

    # Load split info
    train_config = None
    if args.train_config:
        with open(args.train_config) as f:
            train_config = json.load(f)

    print(f"\nEvaluating on: {dataset_path}")
    results = {}

    if train_config and "test_episodes" in train_config:
        if args.all_splits:
            # Evaluate all three splits
            for split_name, key in [("train", "train_episodes"), ("val", "val_episodes"), ("test", "test_episodes")]:
                episodes = train_config[key]
                results[split_name] = eval_split(
                    policy, preprocessor, dataset_path, episodes,
                    split_name, args.batch_size, args.num_workers, device,
                )
        else:
            # Test only
            results["test"] = eval_split(
                policy, preprocessor, dataset_path, train_config["test_episodes"],
                "test", args.batch_size, args.num_workers, device,
            )
    else:
        # No split info — evaluate on full dataset
        print("  No train_config with splits provided, evaluating on full dataset")
        results["full"] = eval_split(
            policy, preprocessor, dataset_path, None,
            "full", args.batch_size, args.num_workers, device,
        )

    # Save results
    results_path = checkpoint_path / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
