# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: MIN_MAX normalizer for diffusion inference, loads from checkpoint safetensors
"""MIN_MAX normalization utilities for Diffusion Policy inference."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor


class DiffusionNormalizer:
    """Normalizer using MIN_MAX mode for diffusion policy.

    Maps state/action values to [-1, 1] range using min/max statistics.
    Loads from policy_preprocessor_step_3_normalizer_processor.safetensors.
    """

    def __init__(self, checkpoint_path: Path | str, device: str = "cuda"):
        checkpoint_path = Path(checkpoint_path)

        normalizer_file = checkpoint_path / "policy_preprocessor_step_3_normalizer_processor.safetensors"
        if not normalizer_file.exists():
            normalizer_file = checkpoint_path.parent / "policy_preprocessor_step_3_normalizer_processor.safetensors"

        if not normalizer_file.exists():
            raise FileNotFoundError(f"Normalizer file not found in {checkpoint_path} or parent")

        self._raw_stats = load_file(normalizer_file, device=device)
        self.device = device
        self._cache = {}

        # Detect whether stats use min/max or mean/std keys
        sample_keys = list(self._raw_stats.keys())
        self._has_min_max = any(k.endswith(".min") for k in sample_keys)
        self._has_mean_std = any(k.endswith(".mean") for k in sample_keys)

    def _get_minmax_tensors(self, key: str) -> tuple[Tensor, Tensor]:
        """Get min and max tensors for a feature key (cached)."""
        if key not in self._cache:
            min_key = f"{key}.min"
            max_key = f"{key}.max"

            if min_key not in self._raw_stats:
                available = set(k.rsplit(".", 1)[0] for k in self._raw_stats.keys())
                raise KeyError(f"Feature '{key}' not found in stats. Available: {available}")

            vmin = self._raw_stats[min_key].to(dtype=torch.float32)
            vmax = self._raw_stats[max_key].to(dtype=torch.float32)

            # Avoid division by zero
            range_val = torch.clamp(vmax - vmin, min=1e-8)

            self._cache[key] = (vmin, range_val)

        return self._cache[key]

    def _get_meanstd_tensors(self, key: str) -> tuple[Tensor, Tensor]:
        """Get mean and std tensors (fallback for mean_std mode)."""
        cache_key = f"{key}_meanstd"
        if cache_key not in self._cache:
            mean_key = f"{key}.mean"
            std_key = f"{key}.std"

            if mean_key not in self._raw_stats:
                available = set(k.rsplit(".", 1)[0] for k in self._raw_stats.keys())
                raise KeyError(f"Feature '{key}' not found in stats. Available: {available}")

            mean = self._raw_stats[mean_key].to(dtype=torch.float32)
            std = self._raw_stats[std_key].to(dtype=torch.float32)
            std = torch.clamp(std, min=1e-8)

            self._cache[cache_key] = (mean, std)

        return self._cache[cache_key]

    def normalize(self, key: str, x: Tensor) -> Tensor:
        """Normalize tensor to [-1, 1] using MIN_MAX.

        normalize(x) = (x - min) / (max - min) * 2 - 1
        """
        if self._has_min_max:
            vmin, range_val = self._get_minmax_tensors(key)
            return (x - vmin) / range_val * 2.0 - 1.0
        else:
            # Fallback to mean/std
            mean, std = self._get_meanstd_tensors(key)
            return (x - mean) / std

    def denormalize(self, key: str, x: Tensor) -> Tensor:
        """Denormalize tensor from [-1, 1] back to original scale.

        denormalize(x) = (x + 1) / 2 * (max - min) + min
        """
        if self._has_min_max:
            vmin, range_val = self._get_minmax_tensors(key)
            return (x + 1.0) / 2.0 * range_val + vmin
        else:
            mean, std = self._get_meanstd_tensors(key)
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

    def to(self, device: str) -> "DiffusionNormalizer":
        """Move cached tensors to device."""
        self.device = device
        self._raw_stats = {k: v.to(device) for k, v in self._raw_stats.items()}
        self._cache = {}
        return self
