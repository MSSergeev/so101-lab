#!/usr/bin/env python3
# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Example: examples/tutorial/diffusion/diffusion_training_example.py
# Original license: Apache 2.0
# Changes: Standalone training for SO-101 datasets with CLI args
"""Train Diffusion Policy on LeRobot dataset.

Run in lerobot-env:
    eval "$(./activate_lerobot.sh)"
    python scripts/train/train_diffusion.py --dataset data/recordings/figure_shape_placement_v1

With config file:
    python scripts/train/train_diffusion.py --dataset ... --config configs/policy/diffusion/baseline.yaml
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on LeRobot dataset")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (CLI args override config values)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to local LeRobot dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/",
        help="Output directory (default: outputs/)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name for output folder (default: diffusion_<dataset_name>)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50000,
        help="Training steps (default: 50000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Diffusion horizon / action prediction size (default: 16)",
    )
    parser.add_argument(
        "--n-obs-steps",
        type=int,
        default=2,
        help="Number of observation steps (default: 2)",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=8,
        help="Actions to execute per chunk (default: 8)",
    )
    parser.add_argument(
        "--num-train-timesteps",
        type=int,
        default=100,
        help="Number of diffusion training timesteps (default: 100)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Number of diffusion inference steps (default: same as train timesteps)",
    )
    from so101_lab.utils.tracker import add_tracker_args
    add_tracker_args(parser, default_project="so101-diffusion")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint directory",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (default: 5000)",
    )
    parser.add_argument(
        "--best-check-interval",
        type=int,
        default=100,
        help="Check for best model every N steps (default: 100)",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=None,
        help="Override tolerance_s for delta_timestamps (default: auto from fps)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers (default: 4, try 8 if GPU is underutilized)",
    )

    # First parse to get config file
    args, _ = parser.parse_known_args()

    # Load config and set as defaults (CLI will override)
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        key_map = {
            "batch_size": "batch_size",
            "n_obs_steps": "n_obs_steps",
            "n_action_steps": "n_action_steps",
            "num_train_timesteps": "num_train_timesteps",
            "num_inference_steps": "num_inference_steps",
            "checkpoint_interval": "checkpoint_interval",
            "best_check_interval": "best_check_interval",
        }
        defaults = {}
        for yaml_key, arg_dest in key_map.items():
            if yaml_key in config:
                defaults[arg_dest] = config[yaml_key]
        if "num_workers" in config:
            defaults["num_workers"] = config["num_workers"]
        for key in ["steps", "lr", "horizon", "seed", "output", "name", "wandb", "resume"]:
            if key in config:
                defaults[key] = config[key]
        parser.set_defaults(**defaults)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.config:
        print(f"Loaded config: {args.config}")
    print(f"Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, "
          f"horizon={args.horizon}, n_obs_steps={args.n_obs_steps}, "
          f"n_action_steps={args.n_action_steps}")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    if args.name:
        run_name = args.name
    else:
        dataset_name = Path(args.dataset).name
        run_name = f"diffusion_{dataset_name}"
    output_dir = Path(args.output) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load dataset metadata
    dataset_path = Path(args.dataset).absolute()
    print(f"Loading dataset: {dataset_path}")
    dataset_metadata = LeRobotDatasetMetadata(repo_id="local_dataset", root=dataset_path)

    # Extract features
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # Create Diffusion config
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        optimizer_lr=args.lr,
    )

    # Create policy and processors
    policy = DiffusionPolicy(cfg)
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    policy.train()
    policy.to(device)

    num_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Setup delta timestamps for observation history + action horizon
    # observation.state needs n_obs_steps timestamps
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # Image features also need n_obs_steps timestamps
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    print(f"Delta timestamps:")
    for k, v in delta_timestamps.items():
        print(f"  {k}: {v}")

    # Compute tolerance_s
    # Diffusion needs multi-timestep lookups, so tolerance needs to cover the gap
    if args.tolerance_s is not None:
        tolerance_s = args.tolerance_s
    else:
        tolerance_s = 1.0 / dataset_metadata.fps  # 1 frame tolerance
    print(f"tolerance_s: {tolerance_s}")

    # Load dataset with delta timestamps
    dataset = LeRobotDataset(
        repo_id="local_dataset",
        root=dataset_path,
        delta_timestamps=delta_timestamps,
        tolerance_s=tolerance_s,
    )
    print(f"Dataset size: {len(dataset)} frames")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Create optimizer (diffusion uses different defaults than ACT)
    optimizer = cfg.get_optimizer_preset().build(policy.get_optim_params())

    # Create LR scheduler
    scheduler_cfg = cfg.get_scheduler_preset()
    lr_scheduler = scheduler_cfg.build(optimizer, num_training_steps=args.steps)

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        resume_path = Path(args.resume)
        print(f"Resuming from checkpoint: {resume_path}")

        policy.load_state_dict(DiffusionPolicy.from_pretrained(resume_path).state_dict())

        train_state_path = resume_path / "train_state.pt"
        if train_state_path.exists():
            train_state = torch.load(train_state_path, map_location=device, weights_only=True)
            optimizer.load_state_dict(train_state["optimizer"])
            start_step = train_state["step"]
            if "lr_scheduler" in train_state:
                lr_scheduler.load_state_dict(train_state["lr_scheduler"])
            print(f"Resumed optimizer state, starting from step {start_step}")
        else:
            print("Warning: train_state.pt not found, optimizer state not restored")

    # Experiment tracker
    from so101_lab.utils.tracker import setup_tracker, cleanup_tracker
    tracker, sys_monitor = setup_tracker(args, run_name, config={
        "dataset": args.dataset, "steps": args.steps, "batch_size": args.batch_size,
        "lr": args.lr, "horizon": args.horizon, "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps, "num_train_timesteps": args.num_train_timesteps,
    })

    # Training loop
    best_loss = float("inf")
    start_time = time.time()

    pbar = tqdm(total=args.steps, initial=start_step, desc="Training", unit="step")
    step = start_step
    done = False

    while not done:
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch = preprocessor(batch)

            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            current_lr = lr_scheduler.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
            pbar.update(1)

            if step % 100 == 0 and tracker:
                tracker.log({"loss": loss.item(), "lr": current_lr}, step=step)

            # Save checkpoint
            if step > 0 and step % args.checkpoint_interval == 0:
                checkpoint_dir = output_dir / f"checkpoint_{step}"
                checkpoint_dir.mkdir(exist_ok=True)
                policy.save_pretrained(checkpoint_dir)
                preprocessor.save_pretrained(checkpoint_dir)
                postprocessor.save_pretrained(checkpoint_dir)
                torch.save(
                    {"optimizer": optimizer.state_dict(), "step": step,
                     "lr_scheduler": lr_scheduler.state_dict()},
                    checkpoint_dir / "train_state.pt",
                )
                tqdm.write(f"Saved checkpoint to {checkpoint_dir}")

            # Save best model
            if step % args.best_check_interval == 0 and loss.item() < best_loss:
                best_loss = loss.item()
                best_dir = output_dir / "best"
                best_dir.mkdir(exist_ok=True)
                policy.save_pretrained(best_dir)
                preprocessor.save_pretrained(best_dir)
                postprocessor.save_pretrained(best_dir)
                torch.save(
                    {"optimizer": optimizer.state_dict(), "step": step, "loss": best_loss,
                     "lr_scheduler": lr_scheduler.state_dict()},
                    best_dir / "train_state.pt",
                )

            step += 1
            if step >= args.steps:
                done = True
                break

    pbar.close()

    # Save final model
    total_time = time.time() - start_time
    print(f"Training complete in {int(total_time // 60)}m {int(total_time % 60)}s. Best loss: {best_loss:.4f}")
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    torch.save(
        {"optimizer": optimizer.state_dict(), "step": step,
         "lr_scheduler": lr_scheduler.state_dict()},
        output_dir / "train_state.pt",
    )

    # Save training config
    train_config = {
        "config": args.config,
        "name": run_name,
        "dataset": str(args.dataset),
        "fps": dataset_metadata.fps,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "horizon": args.horizon,
        "n_obs_steps": args.n_obs_steps,
        "n_action_steps": args.n_action_steps,
        "num_train_timesteps": args.num_train_timesteps,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "best_loss": best_loss,
    }
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(train_config, f, indent=2)

    print(f"Saved final model to {output_dir}")

    cleanup_tracker(tracker, sys_monitor)


if __name__ == "__main__":
    main()
