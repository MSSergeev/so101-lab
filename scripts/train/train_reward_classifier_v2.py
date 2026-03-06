#!/usr/bin/env python3
"""Train reward classifier v2 with unfrozen backbone.

Key differences from v1 (train_reward_classifier.py):
- Unfreezes ResNet backbone for fine-tuning (v1 froze it)
- Removes torch.no_grad() from encoder forward pass
- Differential learning rates: backbone LR << head LR
- Balanced sampling: equal success/fail frames per batch

Run in lerobot-env:
    eval "$(./activate_lerobot.sh)"

    # Basic training
    python scripts/train/train_reward_classifier_v2.py \
        --dataset data/recordings/figure_shape_placement_v4_labeled \
        --output outputs/reward_classifier_v2

    # Custom LRs
    python scripts/train/train_reward_classifier_v2.py \
        --dataset data/recordings/figure_shape_placement_v4_labeled \
        --output outputs/reward_classifier_v2 \
        --backbone-lr 1e-5 --head-lr 1e-4 --epochs 15
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train reward classifier v2 (fine-tune backbone)")
    parser.add_argument("--dataset", type=str, required=True, help="Path to labeled LeRobot dataset")
    parser.add_argument("--output", type=str, default="outputs/reward_classifier_v2",
                        help="Output directory")
    parser.add_argument("--model-name", type=str, default="microsoft/resnet-18",
                        help="HuggingFace vision model (default: microsoft/resnet-18)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--backbone-lr", type=float, default=1e-5,
                        help="Learning rate for backbone (default: 1e-5)")
    parser.add_argument("--head-lr", type=float, default=1e-4,
                        help="Learning rate for head (default: 1e-4)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of episodes for validation (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.15,
                        help="Fraction of episodes for test (default: 0.15)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    from so101_lab.utils.tracker import add_tracker_args
    add_tracker_args(parser, default_project="so101-reward-classifier")
    return parser.parse_args()


def split_episodes(
    total_episodes: int, val_fraction: float, test_fraction: float, seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split episode indices into train/val/test by episode (no data leakage)."""
    all_episodes = list(range(total_episodes))
    rng = random.Random(seed)
    rng.shuffle(all_episodes)

    n_test = max(1, int(total_episodes * test_fraction))
    n_val = max(1, int(total_episodes * val_fraction))

    test_episodes = sorted(all_episodes[:n_test])
    val_episodes = sorted(all_episodes[n_test:n_test + n_val])
    train_episodes = sorted(all_episodes[n_test + n_val:])

    return train_episodes, val_episodes, test_episodes


def unfreeze_and_patch_classifier(policy):
    """Unfreeze backbone and patch _get_encoder_output to allow gradients."""
    classifier = policy

    # Unfreeze all encoder parameters
    for param in classifier.encoder.parameters():
        param.requires_grad = True

    # Also unfreeze per-camera encoders (contains SpatialLearnedEmbeddings etc.)
    for encoder in classifier.encoders.values():
        for param in encoder.parameters():
            param.requires_grad = True

    # Monkey-patch _get_encoder_output to remove torch.no_grad()
    original_method = classifier._get_encoder_output

    def _get_encoder_output_with_grad(self, x, image_key):
        if self.is_cnn:
            return self.encoders[image_key](x)
        else:
            outputs = self.encoder(x)
            return outputs.last_hidden_state[:, 0, :]

    import types
    classifier._get_encoder_output = types.MethodType(_get_encoder_output_with_grad, classifier)

    return classifier


def get_param_groups(policy, backbone_lr: float, head_lr: float):
    """Create parameter groups with differential learning rates."""
    backbone_params = []
    head_params = []

    # ResNet backbone parameters (shared encoder)
    backbone_param_ids = set()
    for param in policy.encoder.parameters():
        backbone_param_ids.add(id(param))
        if param.requires_grad:
            backbone_params.append(param)

    # Everything else (SpatialLearnedEmbeddings, per-camera layers, classifier head)
    for param in policy.parameters():
        if id(param) not in backbone_param_ids and param.requires_grad:
            head_params.append(param)

    print(f"  Backbone params: {sum(p.numel() for p in backbone_params):,} (lr={backbone_lr})")
    print(f"  Head params: {sum(p.numel() for p in head_params):,} (lr={head_lr})")

    return [
        {"params": backbone_params, "lr": backbone_lr},
        {"params": head_params, "lr": head_lr},
    ]


def build_balanced_sampler(dataset: LeRobotDataset) -> WeightedRandomSampler:
    """Create a sampler that balances success/fail frames."""
    rewards = []
    for i in range(len(dataset)):
        r = dataset.hf_dataset[i]["next.reward"]
        rewards.append(float(r))

    rewards = np.array(rewards)
    n_pos = (rewards > 0.5).sum()
    n_neg = (rewards <= 0.5).sum()

    weight_pos = 1.0 / max(n_pos, 1)
    weight_neg = 1.0 / max(n_neg, 1)

    weights = np.where(rewards > 0.5, weight_pos, weight_neg)

    print(f"  Balanced sampling: {n_pos} positive, {n_neg} negative")

    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).double(),
        num_samples=len(dataset),
        replacement=True,
    )


@torch.no_grad()
def evaluate(policy, dataloader, preprocessor) -> tuple[float, float]:
    """Run evaluation and return (avg_loss, accuracy%)."""
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

    avg_loss = total_loss / num_batches
    accuracy = 100 * total_correct / total_samples
    return avg_loss, accuracy


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset_path = Path(args.dataset).absolute()
    print(f"Loading dataset: {dataset_path}")

    full_dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path)
    camera_keys = full_dataset.meta.camera_keys
    print(f"Camera keys: {camera_keys}")

    if "next.reward" not in full_dataset.meta.features:
        raise ValueError("Dataset missing 'next.reward'. Run prepare_classifier_dataset.py first.")

    # Split by episodes
    total_episodes = full_dataset.meta.total_episodes
    train_episodes, val_episodes, test_episodes = split_episodes(
        total_episodes, args.val_split, args.test_split, args.seed,
    )
    print(f"Episodes: {total_episodes} total, {len(train_episodes)} train, "
          f"{len(val_episodes)} val, {len(test_episodes)} test")

    train_dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path, episodes=train_episodes)
    val_dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path, episodes=val_episodes)
    print(f"Frames: {len(train_dataset)} train, {len(val_dataset)} val")

    # Create classifier
    config = RewardClassifierConfig(
        num_cameras=len(camera_keys),
        device=str(device),
        model_name=args.model_name,
        learning_rate=args.head_lr,
    )

    policy = make_policy(config, ds_meta=full_dataset.meta)
    preprocessor, _ = make_pre_post_processors(policy_cfg=config, dataset_stats=full_dataset.meta.stats)

    # Unfreeze backbone and patch forward pass
    print("Unfreezing backbone and patching encoder forward...")
    unfreeze_and_patch_classifier(policy)

    policy.train()
    policy.to(device)

    num_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Differential LR optimizer
    param_groups = get_param_groups(policy, args.backbone_lr, args.head_lr)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Balanced sampler for training
    print("Building balanced sampler...")
    sampler = build_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"
    best_dir.mkdir(exist_ok=True)

    # Tracker
    from so101_lab.utils.tracker import setup_tracker, cleanup_tracker
    run_name = output_dir.name
    tracker, sys_monitor = setup_tracker(args, run_name)

    # Save training config
    train_config = {
        "dataset": str(args.dataset),
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "val_split": args.val_split,
        "test_split": args.test_split,
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "test_episodes": test_episodes,
        "best_val_accuracy": 0.0,
        "camera_keys": list(camera_keys),
        "version": "v2_finetune",
    }
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_accuracy = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        policy.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            batch = preprocessor(batch)

            loss, output_dict = policy.forward(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_correct += output_dict["correct"]
            total_samples += output_dict["total"]
            num_batches += 1

            pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}",
                             acc=f"{100 * total_correct / total_samples:.1f}%")

        train_loss = total_loss / num_batches
        train_accuracy = 100 * total_correct / total_samples

        # Validate
        val_loss, val_accuracy = evaluate(policy, val_loader, preprocessor)

        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"

        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"train_loss: {train_loss:.4f} | train_acc: {train_accuracy:.2f}% | "
              f"val_loss: {val_loss:.4f} | val_acc: {val_accuracy:.2f}% | "
              f"elapsed: {elapsed_str}")

        if tracker is not None:
            tracker.log({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "epoch": epoch + 1,
            })

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            policy.save_pretrained(best_dir)
            preprocessor.save_pretrained(best_dir)
            print(f"  -> New best val accuracy: {best_val_accuracy:.2f}%")

    # Save final model
    total_time = time.time() - start_time
    print(f"\nTraining complete in {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"Best val accuracy: {best_val_accuracy:.2f}%")

    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)

    train_config["best_val_accuracy"] = best_val_accuracy
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"Saved model to {output_dir}")

    cleanup_tracker(tracker, sys_monitor)


if __name__ == "__main__":
    main()
