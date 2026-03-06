# Adapted from: facebookresearch/vip (https://github.com/facebookresearch/vip)
# Original license: CC BY-NC 4.0
# Changes: Standalone loader without vip-utils package; Resize(224,224) instead of CenterCrop

"""VIP (Value-Implicit Pre-training) reward model.

Zero-shot visual reward via pretrained ResNet50 (Ego4D).
Reward = -||phi(current_image) - phi(goal_image)||_2

Goal images: either final frames of demo episodes, or labeled success frames
(next.reward > 0.5) from a prepared dataset.

Usage:
    from so101_lab.rewards.vip_reward import VIPReward

    # Mean goal embedding (default)
    vip = VIPReward("data/recordings/figure_shape_placement_v4")

    # Min distance to closest goal
    vip = VIPReward("data/recordings/figure_shape_placement_v4", goal_mode="min")

    # Use labeled success frames instead of final frames
    vip = VIPReward("data/recordings/figure_shape_placement_v4_labeled", use_labeled=True)

    reward = vip.compute_reward(obs)  # scalar, negative (closer to 0 = better)
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

# Pretrained weights URL (PyTorch S3)
VIP_WEIGHTS_URL = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
VIP_CACHE_DIR = os.path.expanduser("~/.vip/resnet50")


def _load_vip_encoder(device: str = "cuda") -> nn.Module:
    """Load pretrained VIP ResNet50 encoder."""
    convnet = torchvision.models.resnet50(weights=None)
    convnet.fc = nn.Linear(2048, 1024)

    # Download weights if needed
    weights_path = os.path.join(VIP_CACHE_DIR, "model.pt")
    if not os.path.exists(weights_path):
        os.makedirs(VIP_CACHE_DIR, exist_ok=True)
        torch.hub.load_state_dict_from_url(VIP_WEIGHTS_URL, VIP_CACHE_DIR)

    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    # Strip 'module.convnet.' prefix from DataParallel + VIP wrapper
    state_dict = {
        k.replace("module.convnet.", ""): v
        for k, v in ckpt["vip"].items()
    }
    convnet.load_state_dict(state_dict)
    convnet.to(device)
    convnet.eval()
    return convnet


class VIPReward:
    """VIP reward: embedding distance to goal image.

    Accepts numpy images (HWC uint8) from IsaacLabGymEnv.
    Returns scalar reward (negative, closer to 0 = closer to goal).
    """

    def __init__(
        self,
        goal_dataset_path: str | Path,
        device: str = "cuda",
        image_key: str = "observation.images.top",
        n_goal_frames: int = 5,
        goal_mode: str = "mean",
        use_labeled: bool = False,
        label_dataset_path: str | Path | None = None,
        normalize: bool = False,
        scale_dataset_path: str | Path | None = None,
        reward_clip: tuple[float, float] = (-2.0, 0.0),
    ):
        """
        Args:
            goal_dataset_path: Path to LeRobot dataset (source of images).
            device: Torch device.
            image_key: Which camera to use for embeddings.
            n_goal_frames: Final frames per episode for goal (ignored if use_labeled=True).
            goal_mode: "mean" — single averaged goal embedding,
                       "min" — keep all goal embeddings, reward = min distance.
            use_labeled: If True, use frames with next.reward > 0.5 as goals.
            label_dataset_path: Dataset with next.reward labels (default: same as goal_dataset_path).
                                Images are always taken from goal_dataset_path.
            normalize: If True, normalize rewards to [-1, 0] using scale from data.
            scale_dataset_path: Dataset with precomputed VIP rewards for normalization
                                scale (typically --demo-dataset). Falls back to sampling
                                from goal_dataset if None.
            reward_clip: Clip range after normalization (default: (-2.0, 0.0)).
        """
        self.device = device
        self.image_key = image_key
        self.goal_mode = goal_mode
        self.normalize = normalize
        self.reward_clip = reward_clip
        self.reward_scale = 1.0  # updated below if normalize=True

        # Preprocessing: resize to 224x224 + ImageNet normalize
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load encoder
        self.encoder = _load_vip_encoder(device)

        # Precompute goal embeddings from dataset (with disk cache)
        goal_embeddings = self._load_or_compute_goal_embeddings(
            goal_dataset_path, use_labeled, label_dataset_path, n_goal_frames
        )

        if goal_mode == "mean":
            self.goal_embedding = goal_embeddings.mean(dim=0)  # (1024,)
            self.goal_embeddings = None
            norm = torch.norm(self.goal_embedding).item()
            print(f"  Goal mode: mean (single embedding, norm={norm:.2f})")
        else:
            self.goal_embedding = None
            self.goal_embeddings = goal_embeddings  # (N, 1024)
            print(f"  Goal mode: min (closest of {len(goal_embeddings)} embeddings)")

        # Compute normalization scale
        if normalize:
            self.reward_scale = self._compute_reward_scale(
                scale_dataset_path, goal_dataset_path
            )
            print(f"  Normalize: scale={self.reward_scale:.2f}, clip={reward_clip}")

        print(f"VIP reward initialized (camera={image_key})")

    def _load_or_compute_goal_embeddings(
        self,
        goal_dataset_path: str | Path,
        use_labeled: bool,
        label_dataset_path: str | Path | None,
        n_goal_frames: int,
    ) -> torch.Tensor:
        """Load cached goal embeddings or compute and cache them.

        Cache file: <goal_dataset_path>/vip_goal_cache_<key>.pt
        Key encodes: image_key, use_labeled, label_path, n_goal_frames.
        """
        import hashlib

        lbl_path = str(label_dataset_path or goal_dataset_path)
        cache_key_str = f"{self.image_key}|labeled={use_labeled}|lbl={lbl_path}|n={n_goal_frames}"
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:8]
        cache_path = Path(goal_dataset_path) / f"vip_goal_cache_{cache_hash}.pt"

        if cache_path.exists():
            goal_embeddings = torch.load(cache_path, map_location=self.device, weights_only=True)
            print(f"  Loaded cached goal embeddings: {cache_path} ({goal_embeddings.shape[0]} embeddings)")
            return goal_embeddings

        # Compute from scratch
        if use_labeled:
            goal_embeddings = self._compute_goal_from_labels(
                goal_dataset_path, label_dataset_path or goal_dataset_path
            )
        else:
            goal_embeddings = self._compute_goal_from_finals(
                goal_dataset_path, n_goal_frames
            )

        # Save cache
        torch.save(goal_embeddings.cpu(), cache_path)
        print(f"  Cached goal embeddings: {cache_path}")

        return goal_embeddings.to(self.device)

    def _encode_frame(self, dataset, frame_idx: int) -> torch.Tensor:
        """Encode a single dataset frame → (1, 1024) embedding."""
        frame = dataset[frame_idx]
        img = frame[self.image_key]  # (C, H, W) float [0, 1] from LeRobot
        img_uint8 = (img * 255).to(torch.uint8)
        img_preprocessed = self.preprocess(img_uint8).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encoder(img_preprocessed)  # (1, 1024)

    def _compute_goal_from_labels(
        self, image_dataset_path: str | Path, label_dataset_path: str | Path
    ) -> torch.Tensor:
        """Extract goal embeddings using labels from one dataset, images from another.

        Args:
            image_dataset_path: Dataset with full-res images for encoding.
            label_dataset_path: Dataset with next.reward column for filtering.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        label_ds = LeRobotDataset(repo_id="local", root=str(label_dataset_path))
        if str(image_dataset_path) == str(label_dataset_path):
            image_ds = label_ds
        else:
            image_ds = LeRobotDataset(repo_id="local", root=str(image_dataset_path))

        # Find success frame indices from label dataset
        success_indices = []
        for i in range(len(label_ds)):
            r = float(label_ds.hf_dataset[i]["next.reward"])
            if r > 0.5:
                success_indices.append(i)

        if not success_indices:
            raise ValueError(f"No success frames (next.reward > 0.5) in {label_dataset_path}")

        print(f"  Labels from: {label_dataset_path}")
        print(f"  Images from: {image_dataset_path}")
        print(f"  Found {len(success_indices)} success frames")

        # Subsample if too many (encode max 1000)
        if len(success_indices) > 1000:
            step = len(success_indices) // 1000
            success_indices = success_indices[::step][:1000]
            print(f"  Subsampled to {len(success_indices)} frames")

        # Encode images from image_ds (not label_ds)
        embeddings = []
        for idx in success_indices:
            emb = self._encode_frame(image_ds, idx)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)  # (N, 1024)

    def _compute_goal_from_finals(
        self, dataset_path: str | Path, n_goal_frames: int
    ) -> torch.Tensor:
        """Extract goal embeddings from final frames of each episode."""
        import pandas as pd
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id="local", root=str(dataset_path))

        ep_parquet = Path(dataset_path) / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ep_df = pd.read_parquet(ep_parquet)
        n_episodes = len(ep_df)
        print(f"  Extracting goal from {n_episodes} episodes "
              f"({n_goal_frames} final frames each)...")

        embeddings = []
        for _, row in ep_df.iterrows():
            ep_start = int(row["dataset_from_index"])
            ep_end = int(row["dataset_to_index"])
            ep_len = ep_end - ep_start

            n_take = min(n_goal_frames, ep_len)
            for offset in range(n_take):
                frame_idx = ep_end - 1 - offset
                emb = self._encode_frame(dataset, frame_idx)
                embeddings.append(emb)

        return torch.cat(embeddings, dim=0)  # (N, 1024)

    def _compute_reward_scale(
        self,
        scale_dataset_path: str | Path | None,
        goal_dataset_path: str | Path,
    ) -> float:
        """Compute reward normalization scale.

        If scale_dataset has next.reward (precomputed VIP rewards),
        use 5th percentile. Otherwise sample random frames from goal
        dataset and measure distances to goal embedding.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Try scale dataset with precomputed VIP rewards (typically --demo-dataset)
        if scale_dataset_path is not None:
            scale_ds = LeRobotDataset(repo_id="local", root=str(scale_dataset_path))
            if "next.reward" in scale_ds.meta.features:
                rewards = [
                    float(scale_ds.hf_dataset[i]["next.reward"])
                    for i in range(len(scale_ds))
                ]
                scale = -float(np.percentile(rewards, 5))
                print(f"  Scale from {scale_dataset_path}: p5={-scale:.2f}")
                return max(scale, 1.0)  # safety floor

        # Fallback: sample random frames from goal dataset
        print("  No precomputed rewards, sampling 100 random frames...")
        image_ds = LeRobotDataset(repo_id="local", root=str(goal_dataset_path))
        rng = np.random.default_rng(42)
        n_samples = min(100, len(image_ds))
        indices = rng.choice(len(image_ds), size=n_samples, replace=False)

        dists = []
        for idx in indices:
            emb = self._encode_frame(image_ds, int(idx)).squeeze(0)
            if self.goal_mode == "mean":
                d = torch.norm(emb - self.goal_embedding).item()
            else:
                d = torch.norm(self.goal_embeddings - emb.unsqueeze(0), dim=1).min().item()
            dists.append(d)

        scale = float(np.percentile(dists, 95))
        print(f"  Scale from sampled frames: p95={scale:.2f}")
        return max(scale, 1.0)

    def _prepare_image(self, obs: dict) -> torch.Tensor:
        """Convert raw observation image to preprocessed tensor."""
        img = obs[self.image_key]  # (H, W, 3) uint8 numpy

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        # (H, W, C) -> (C, H, W)
        if img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        # Preprocess: resize 224x224, /255, ImageNet normalize
        img = self.preprocess(img.unsqueeze(0).to(self.device))
        return img  # (1, 3, 224, 224)

    @torch.no_grad()
    def compute_reward(self, obs: dict) -> float:
        """Compute VIP reward: -||phi(current) - phi(goal)||_2.

        Args:
            obs: Dict with image arrays, e.g.:
                {"observation.images.top": (H, W, 3) uint8 numpy}

        Returns:
            Scalar reward (negative, closer to 0 = better).
            If normalize=True, scaled to [-1, 0] and clipped.
        """
        img = self._prepare_image(obs)
        emb = self.encoder(img).squeeze(0)  # (1024,)

        if self.goal_mode == "mean":
            raw = -torch.norm(emb - self.goal_embedding).item()
        else:
            dists = torch.norm(self.goal_embeddings - emb.unsqueeze(0), dim=1)  # (N,)
            raw = -dists.min().item()

        if self.normalize:
            normalized = raw / self.reward_scale
            return max(self.reward_clip[0], min(normalized, self.reward_clip[1]))
        return raw
