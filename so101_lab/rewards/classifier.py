"""Reward classifier inference wrapper.

Usage:
    from so101_lab.rewards.classifier import RewardClassifier

    clf = RewardClassifier("outputs/reward_classifier_v1/best")
    reward = clf.predict_reward(obs)      # 0.0 or 1.0
    prob = clf.predict_probability(obs)   # float [0, 1]
"""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F

from lerobot.policies.sac.reward_model.modeling_classifier import Classifier


class RewardClassifier:
    """Inference wrapper for trained reward classifier.

    Accepts numpy images (HWC uint8) from IsaacLabGymEnv and returns reward signals.
    """

    def __init__(
        self,
        pretrained_path: str | Path,
        device: str = "cuda",
        threshold: float = 0.9,
    ):
        self.device = device
        self.threshold = threshold

        self.classifier = Classifier.from_pretrained(
            str(pretrained_path),
        )
        self.classifier.to(device)
        self.classifier.eval()

        self.image_keys = [
            key for key in self.classifier.config.input_features
            if key.startswith("observation.image")
        ]

        print(f"Loaded reward classifier from {pretrained_path}")
        print(f"  Image keys: {self.image_keys}")
        print(f"  Threshold: {self.threshold}")

    def _prepare_batch(self, obs: dict) -> dict:
        """Convert numpy observations to torch batch.

        Args:
            obs: Dict with keys like "observation.images.top" containing (H, W, 3) uint8 arrays.
        """
        batch = {}
        for key in self.image_keys:
            # Try exact key first, then try with/without 's' in 'images'
            img = obs.get(key)
            if img is None:
                # observation.image.top <-> observation.images.top
                alt_key = key.replace("observation.image.", "observation.images.")
                if alt_key == key:
                    alt_key = key.replace("observation.images.", "observation.image.")
                img = obs.get(alt_key)

            if img is None:
                raise KeyError(f"Missing image key: {key} (or alternative)")

            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)

            # (H, W, C) -> (C, H, W)
            if img.dim() == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1)

            # Resize to 128x128 for SpatialLearnedEmbeddings compatibility
            h, w = img.shape[-2:]
            if h != 128 or w != 128:
                img = F.resize(img, [128, 128])

            # Add batch dim
            if img.dim() == 3:
                img = img.unsqueeze(0)

            img = img.to(dtype=torch.float32, device=self.device)

            # Normalize to [0, 1] then apply ImageNet normalization (MEAN_STD)
            img = img / 255.0
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            img = (img - mean) / std

            batch[key] = img

        return batch

    @torch.no_grad()
    def predict_reward(self, obs: dict) -> float:
        """Predict binary reward (0.0 or 1.0).

        Args:
            obs: Dict with image arrays, e.g.:
                {"observation.images.top": (H, W, 3) uint8, ...}
        """
        batch = self._prepare_batch(obs)
        images = [batch[key] for key in self.image_keys]
        output = self.classifier.predict(images)
        return float((output.probabilities > self.threshold).float().item())

    @torch.no_grad()
    def predict_probability(self, obs: dict) -> float:
        """Predict success probability as float [0.0, 1.0].

        Args:
            obs: Dict with image arrays.
        """
        batch = self._prepare_batch(obs)
        images = [batch[key] for key in self.image_keys]
        output = self.classifier.predict(images)
        return float(output.probabilities.item())
