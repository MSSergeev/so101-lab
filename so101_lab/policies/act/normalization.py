# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: Simplified normalizer for inference-only, loads from checkpoint safetensors
"""Normalization utilities for ACT inference."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor


class Normalizer:
    """Normalizer that loads statistics from checkpoint safetensors.

    Uses mean/std normalization (MEAN_STD mode from lerobot).
    Loads from policy_preprocessor_step_3_normalizer_processor.safetensors.
    """

    def __init__(self, checkpoint_path: Path | str, device: str = "cuda"):
        """Initialize normalizer from checkpoint directory.

        Args:
            checkpoint_path: Path to checkpoint directory containing *_processor.safetensors
            device: Device for tensors
        """
        checkpoint_path = Path(checkpoint_path)

        # Find normalizer safetensors file
        normalizer_file = checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        if not normalizer_file.exists():
            # Try parent directory (for best/ or checkpoint_N/ subdirs)
            normalizer_file = checkpoint_path.parent / "policy_preprocessor_step_3_normalizer_processor.safetensors"

        if not normalizer_file.exists():
            raise FileNotFoundError(f"Normalizer file not found in {checkpoint_path} or parent")

        # Load safetensors
        self._raw_stats = load_file(normalizer_file, device=device)
        self.device = device
        self._cache = {}

    def _get_stats_tensors(self, key: str) -> tuple[Tensor, Tensor]:
        """Get mean and std tensors for a feature key (cached)."""
        if key not in self._cache:
            mean_key = f"{key}.mean"
            std_key = f"{key}.std"

            if mean_key not in self._raw_stats:
                available = set(k.rsplit(".", 1)[0] for k in self._raw_stats.keys())
                raise KeyError(f"Feature '{key}' not found in stats. Available: {available}")

            mean = self._raw_stats[mean_key].to(dtype=torch.float32)
            std = self._raw_stats[std_key].to(dtype=torch.float32)

            # Ensure std is never zero
            std = torch.clamp(std, min=1e-8)

            self._cache[key] = (mean, std)

        return self._cache[key]

    def normalize(self, key: str, x: Tensor) -> Tensor:
        """Normalize tensor using mean/std.

        Args:
            key: Feature key (e.g., "observation.state", "action")
            x: Input tensor

        Returns:
            Normalized tensor: (x - mean) / std
        """
        mean, std = self._get_stats_tensors(key)
        return (x - mean) / std

    def denormalize(self, key: str, x: Tensor) -> Tensor:
        """Denormalize tensor back to original scale.

        Args:
            key: Feature key (e.g., "observation.state", "action")
            x: Normalized tensor

        Returns:
            Denormalized tensor: x * std + mean
        """
        mean, std = self._get_stats_tensors(key)
        return x * std + mean

    def normalize_images(self, images: Tensor, key: str | None = None) -> Tensor:
        """Normalize images using dataset statistics if available.

        Args:
            images: (B, C, H, W) uint8 or float [0, 255]
            key: Feature key (e.g. "observation.images.top") for stats lookup

        Returns:
            Normalized images (mean/std normalized if stats exist, else [0, 1])
        """
        if images.dtype == torch.uint8:
            images = images.float()
        images = images / 255.0

        if key and f"{key}.mean" in self._raw_stats:
            mean = self._raw_stats[f"{key}.mean"].to(dtype=torch.float32).reshape(1, -1, 1, 1)
            std = self._raw_stats[f"{key}.std"].to(dtype=torch.float32).reshape(1, -1, 1, 1)
            std = torch.clamp(std, min=1e-8)
            images = (images - mean) / std

        return images

    def to(self, device: str) -> "Normalizer":
        """Move cached tensors to device."""
        self.device = device
        # Move raw stats to new device
        self._raw_stats = {k: v.to(device) for k, v in self._raw_stats.items()}
        # Clear cache to force re-creation on new device
        self._cache = {}
        return self
