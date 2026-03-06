#!/usr/bin/env python3
"""Prepare LeRobot dataset with rewards.

Computes `next.reward` using classifier, VIP, or both. Optionally resizes videos.
Used for SAC training, IQL critics, weighted BC, etc.

Run in lerobot-env:
    eval "$(./activate_lerobot.sh)"

    # Classifier only (default)
    python scripts/train/prepare_reward_dataset.py \
        --dataset data/recordings/figure_shape_placement_v4 \
        --output data/recordings/figure_shape_placement_v4_clf \
        --reward-model outputs/reward_classifier_v1/best

    # VIP only
    python scripts/train/prepare_reward_dataset.py \
        --reward-source vip \
        --dataset data/recordings/figure_shape_placement_v4 \
        --output data/recordings/figure_shape_placement_v4_vip \
        --vip-goal-dataset data/recordings/figure_shape_placement_v4 \
        --vip-label-dataset data/recordings/figure_shape_placement_v4_labeled \
        --vip-use-labeled --vip-goal-mode min

    # Composite (classifier + VIP)
    python scripts/train/prepare_reward_dataset.py \
        --reward-source composite \
        --dataset data/recordings/figure_shape_placement_v4 \
        --output data/recordings/figure_shape_placement_v4_composite \
        --reward-model outputs/reward_classifier_v1/best \
        --vip-goal-dataset data/recordings/figure_shape_placement_v4 \
        --w-vip 1.0 --image-size 128
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Add so101_lab to path when running from lerobot-env
SO101_LAB_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SO101_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(SO101_LAB_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_lab.rewards.classifier import RewardClassifier


def _sample_to_hwc_uint8(sample, key: str) -> np.ndarray:
    """Convert LeRobot sample image to HWC uint8."""
    img = sample.get(key)
    if img is None:
        alt_key = key.replace("observation.image.", "observation.images.")
        if alt_key == key:
            alt_key = key.replace("observation.images.", "observation.image.")
        img = sample.get(alt_key)
    if img is None:
        raise KeyError(f"Missing image key: {key}")

    if isinstance(img, torch.Tensor):
        img = img.numpy()
    if img.shape[0] == 3:  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)
    return img


def compute_classifier_rewards(
    dataset: LeRobotDataset,
    reward_model: RewardClassifier,
) -> np.ndarray:
    """Compute per-frame rewards using classifier."""
    total_frames = len(dataset)
    rewards = np.zeros(total_frames, dtype=np.float32)
    image_keys = reward_model.image_keys

    print(f"Computing classifier rewards for {total_frames} frames...")
    print(f"  Image keys: {image_keys}")

    for i in tqdm(range(total_frames), desc="Classifier rewards"):
        sample = dataset[i]
        obs = {key: _sample_to_hwc_uint8(sample, key) for key in image_keys}
        rewards[i] = reward_model.predict_reward(obs)

    success_frames = (rewards > 0.5).sum()
    print(f"  Success frames: {success_frames} ({100 * success_frames / total_frames:.1f}%)")
    return rewards


def compute_vip_rewards(
    dataset: LeRobotDataset,
    vip_model,
    image_key: str,
) -> np.ndarray:
    """Compute per-frame VIP rewards."""
    total_frames = len(dataset)
    rewards = np.zeros(total_frames, dtype=np.float32)

    print(f"Computing VIP rewards for {total_frames} frames...")
    print(f"  Image key: {image_key}")

    for i in tqdm(range(total_frames), desc="VIP rewards"):
        sample = dataset[i]
        img = _sample_to_hwc_uint8(sample, image_key)
        rewards[i] = vip_model.compute_reward({image_key: img})

    print(f"  Range: [{rewards.min():.2f}, {rewards.max():.2f}], mean={rewards.mean():.2f}")
    return rewards


def resize_videos(output_path: Path, target_size: int) -> list[str]:
    """Re-encode videos to target size using ffmpeg."""
    video_dir = output_path / "videos"
    mp4_files = sorted(video_dir.glob("**/*.mp4"))

    if not mp4_files:
        return []

    patched = []
    print(f"\nRe-encoding {len(mp4_files)} videos to {target_size}x{target_size}...")

    for mp4 in tqdm(mp4_files, desc="Resizing videos"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=mp4.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp4),
                "-vf", f"scale={target_size}:{target_size}",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-an", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed for {mp4}: {result.stderr}")

        tmp_path.rename(mp4)
        patched.append(str(mp4.relative_to(output_path)))

    return patched


def patch_info_json(output_path: Path, target_size: int | None) -> list[str]:
    """Update info.json with video dimensions if resized."""
    info_path = output_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    patched_cameras = []
    if target_size:
        for key, feat in info["features"].items():
            if feat.get("dtype") == "video":
                feat["shape"] = [3, target_size, target_size]
                vi = feat.get("video_info", {})
                vi["video.height"] = target_size
                vi["video.width"] = target_size
                feat["video_info"] = vi
                patched_cameras.append(key)

        if patched_cameras:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)
            print(f"Patched camera metadata to {target_size}x{target_size}: {patched_cameras}")

    return patched_cameras


def fix_parquet_column_order(output_path: Path):
    """Fix parquet column order to match info.json features order."""
    info_path = output_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    expected_order = [k for k, v in info["features"].items() if v["dtype"] != "video"]

    for pq_file in sorted((output_path / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq_file)
        if list(df.columns) != expected_order:
            df[expected_order].to_parquet(pq_file, index=False)


def patch_total_tasks(dataset_path: Path):
    """Patch missing total_tasks for older datasets."""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if "total_tasks" not in info:
        tasks_path = dataset_path / "meta" / "tasks.parquet"
        if tasks_path.exists():
            info["total_tasks"] = len(pd.read_parquet(tasks_path))
        else:
            info["total_tasks"] = 1

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Patched info.json: added total_tasks={info['total_tasks']}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset with rewards for SAC training"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to LeRobot dataset"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for labeled dataset"
    )
    parser.add_argument(
        "--reward-source",
        type=str,
        default="classifier",
        choices=["classifier", "vip", "composite"],
        help="Reward source: classifier, vip, or composite (clf + vip)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="outputs/reward_classifier_v2/best",
        help="Path to trained reward classifier",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classifier threshold for binary reward",
    )
    # VIP args
    parser.add_argument(
        "--vip-goal-dataset", type=str, default=None,
        help="LeRobot dataset for VIP goal images",
    )
    parser.add_argument(
        "--vip-label-dataset", type=str, default=None,
        help="Dataset with next.reward labels for VIP goal filtering",
    )
    parser.add_argument("--vip-use-labeled", action="store_true",
                        help="Use next.reward > 0.5 frames as VIP goals")
    parser.add_argument("--vip-goal-mode", type=str, default="mean",
                        choices=["mean", "min"])
    parser.add_argument("--vip-camera", type=str,
                        default="observation.images.top")
    parser.add_argument("--n-goal-frames", type=int, default=5,
                        help="Final frames per episode for VIP goal")
    parser.add_argument("--w-vip", type=float, default=1.0,
                        help="VIP reward weight (for composite mode)")
    # Common
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        choices=[0, 128, 256],
        help="Resize videos to NxN (0 = no resize)",
    )
    args = parser.parse_args()

    # Validate args
    use_clf = args.reward_source in ("classifier", "composite")
    use_vip = args.reward_source in ("vip", "composite")

    if use_vip and not args.vip_goal_dataset:
        parser.error("--vip-goal-dataset required for vip/composite reward source")

    dataset_path = Path(args.dataset).absolute()
    output_path = Path(args.output).absolute()

    print(f"Loading dataset: {dataset_path}")
    print(f"Reward source: {args.reward_source}")

    # Patch missing total_tasks for older datasets
    patch_total_tasks(dataset_path)

    # Step 1: Copy dataset to output with dummy rewards (zeros)
    print(f"\nCopying dataset to {output_path}...")
    dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path)

    if "next.reward" in dataset.meta.features:
        # Dataset already has next.reward (e.g. sim rewards from recording)
        print("  next.reward already exists, copying dataset as-is...")
        import shutil
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(dataset_path, output_path)
    else:
        dummy_rewards = np.zeros(len(dataset), dtype=np.float32)
        add_features(
            dataset=dataset,
            features={
                "next.reward": (
                    dummy_rewards,
                    {"dtype": "float32", "shape": (1,), "names": None},
                ),
            },
            output_dir=output_path,
            repo_id="local_dataset_labeled",
        )
    fix_parquet_column_order(output_path)

    # Step 2: Resize videos (before reward computation for faster decoding)
    if args.image_size > 0:
        resize_videos(output_path, args.image_size)
        patch_info_json(output_path, args.image_size)

    # Step 3: Compute rewards
    total_frames = len(dataset)
    rewards = np.zeros(total_frames, dtype=np.float32)

    if use_clf:
        # Classifier uses resized images (trained on 128x128)
        output_dataset = LeRobotDataset(repo_id="local_dataset_labeled", root=output_path)
        print(f"\nLoading reward classifier: {args.reward_model}")
        reward_model = RewardClassifier(
            args.reward_model, device="cuda", threshold=args.threshold
        )
        rewards += compute_classifier_rewards(output_dataset, reward_model)

    if use_vip:
        # VIP uses original full-res images (its own Resize(224,224) handles preprocessing)
        from so101_lab.rewards.vip_reward import VIPReward
        print(f"\nLoading VIP reward (goal dataset: {args.vip_goal_dataset})...")
        vip = VIPReward(
            goal_dataset_path=args.vip_goal_dataset,
            device="cuda",
            image_key=args.vip_camera,
            n_goal_frames=args.n_goal_frames,
            goal_mode=args.vip_goal_mode,
            use_labeled=args.vip_use_labeled,
            label_dataset_path=args.vip_label_dataset,
        )
        vip_rewards = compute_vip_rewards(dataset, vip, args.vip_camera)
        rewards += vip_rewards * args.w_vip

    # Step 4: Write rewards to parquet
    print(f"\nFinal reward range: [{rewards.min():.4f}, {rewards.max():.4f}], "
          f"mean={rewards.mean():.4f}")
    print("Writing rewards to parquet...")
    for pq_file in sorted((output_path / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq_file)
        indices = df["index"].tolist()
        df["next.reward"] = [float(rewards[i]) for i in indices]
        df.to_parquet(pq_file, index=False)

    print("Done!")


if __name__ == "__main__":
    main()
