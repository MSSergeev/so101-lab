# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: Standalone inference wrapper for SO-101, handles rad<->normalized conversion,
#          multi-step observation history for diffusion policy
"""Diffusion Policy inference for Isaac Lab.

Usage:
    from so101_lab.policies.diffusion import DiffusionInference

    policy = DiffusionInference("outputs/diffusion_v1")
    policy.reset()
    action = policy.select_action(obs)
"""

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from so101_lab.data.converters import joint_rad_to_motor_normalized, motor_normalized_to_joint_rad

from .configuration_diffusion import DiffusionConfig
from .modeling_diffusion import DiffusionPolicy
from .normalization import DiffusionNormalizer


class DiffusionInference:
    """Diffusion Policy wrapper for Isaac Lab inference.

    Handles:
    - Loading checkpoint from lerobot training
    - Conversion between radians (Isaac Lab) and normalized motor (LeRobot)
    - MIN_MAX normalization using checkpoint statistics
    - Multi-step observation history (n_obs_steps=2)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        # Load config
        config_path = self.checkpoint_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        self.config = DiffusionConfig.from_dict(config_dict)

        # Create model
        self.policy = DiffusionPolicy(self.config)
        self.policy.to(device)
        self.policy.eval()

        # Load weights
        self._load_weights()

        # Setup normalizer from checkpoint (MIN_MAX mode)
        self.normalizer = DiffusionNormalizer(self.checkpoint_path, device=device)

        print(f"Loaded Diffusion policy from {checkpoint_path}")
        print(f"  n_obs_steps: {self.config.n_obs_steps}")
        print(f"  horizon: {self.config.horizon}")
        print(f"  n_action_steps: {self.config.n_action_steps}")
        print(f"  image_features: {self.config.image_features}")

    def _load_weights(self):
        """Load model weights from checkpoint."""
        safetensors_path = self.checkpoint_path / "model.safetensors"
        pytorch_path = self.checkpoint_path / "model.pt"

        if safetensors_path.exists():
            state_dict = load_file(safetensors_path, device=self.device)
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location=self.device)
        else:
            raise FileNotFoundError(f"No model weights found in {self.checkpoint_path}")

        # Handle potential prefix differences
        if any(k.startswith("diffusion.") for k in state_dict.keys()):
            state_dict = {k.replace("diffusion.", ""): v for k, v in state_dict.items()}

        # Load into policy.diffusion
        self.policy.diffusion.load_state_dict(state_dict, strict=False)

    def reset(self):
        """Call on episode start."""
        self.policy.reset()

    def _prepare_obs_batch(self, obs: dict) -> dict:
        """Convert raw observations to normalized batch dict."""
        # Convert joint_pos from radians to normalized motor format
        joint_pos_rad = obs["joint_pos"]
        if isinstance(joint_pos_rad, torch.Tensor):
            joint_pos_rad = joint_pos_rad.cpu().numpy()
        joint_pos_normalized = joint_rad_to_motor_normalized(joint_pos_rad)

        # Prepare state tensor
        state = torch.tensor(joint_pos_normalized, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dim

        # Normalize state
        state = self.normalizer.normalize("observation.state", state)

        batch = {"observation.state": state}

        # Process images
        if "images" in obs and self.config.image_features:
            for feat_key in self.config.image_features:
                cam_name = feat_key.split(".")[-1]
                if cam_name in obs["images"]:
                    img = obs["images"][cam_name]

                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)

                    if img.dim() == 3:
                        img = img.unsqueeze(0)
                    if img.shape[-1] == 3:
                        img = img.permute(0, 3, 1, 2)

                    img = img.to(dtype=torch.float32, device=self.device)
                    img = self.normalizer.normalize_images(img, key=feat_key)

                    batch[feat_key] = img

        return batch

    @torch.no_grad()
    def select_action(self, obs: dict) -> np.ndarray:
        """Select action given observation.

        Args:
            obs: Dict with:
                - "joint_pos": (6,) array in radians
                - "images": dict of {"top": (H, W, 3), "wrist": (H, W, 3)} uint8 arrays

        Returns:
            action: (6,) array in radians
        """
        batch = self._prepare_obs_batch(obs)

        # select_action handles observation queue and action queue internally
        action_normalized = self.policy.select_action(batch)

        # Remove batch dimension if needed
        if action_normalized.dim() > 1:
            action_normalized = action_normalized.squeeze(0)

        # Denormalize action
        action_normalized = self.normalizer.denormalize("action", action_normalized)

        # Convert back to numpy
        action_motor = action_normalized.cpu().numpy()

        # Convert from normalized motor to radians
        action_rad = motor_normalized_to_joint_rad(action_motor)

        return action_rad

    def select_action_batch(self, obs: dict) -> np.ndarray:
        """Select actions for batched observations (one action per env).

        For diffusion, this runs the full denoising loop for each env independently
        since each env has its own observation history queue. Falls back to per-env
        select_action calls.

        Args:
            obs: Dict with batched arrays:
                - "joint_pos": (B, 6) array in radians
                - "images": dict of {"top": (B, H, W, 3), ...} uint8 arrays

        Returns:
            actions: (B, 6) array in radians
        """
        batch_size = obs["joint_pos"].shape[0]
        actions = np.zeros((batch_size, 6), dtype=np.float32)

        for i in range(batch_size):
            single_obs = {
                "joint_pos": obs["joint_pos"][i],
                "images": {cam: img[i] for cam, img in obs.get("images", {}).items()},
            }
            actions[i] = self.select_action(single_obs)

        return actions

    @torch.no_grad()
    def select_action_chunk_batch(self, obs: dict, n_steps: int | None = None) -> np.ndarray:
        """Select action chunks for batched observations.

        Runs full denoising for each env and returns multiple actions per env
        for use with action queues in parallel eval.

        Args:
            obs: Dict with batched arrays:
                - "joint_pos": (B, 6) array in radians
                - "images": dict of {"top": (B, H, W, 3), ...} uint8 arrays
            n_steps: Number of steps to return (default: n_action_steps from config)

        Returns:
            actions: (B, n_steps, 6) array in radians
        """
        if n_steps is None:
            n_steps = self.config.n_action_steps

        batch_size = obs["joint_pos"].shape[0]

        # Convert all joint_pos
        joint_pos_rad = obs["joint_pos"]
        if isinstance(joint_pos_rad, torch.Tensor):
            joint_pos_rad = joint_pos_rad.cpu().numpy()
        joint_pos_normalized = joint_rad_to_motor_normalized(joint_pos_rad)

        # Build batched state tensor
        state = torch.tensor(joint_pos_normalized, dtype=torch.float32, device=self.device)
        state = self.normalizer.normalize("observation.state", state)

        batch = {"observation.state": state}

        # Process images
        if "images" in obs and self.config.image_features:
            for feat_key in self.config.image_features:
                cam_name = feat_key.split(".")[-1]
                if cam_name in obs["images"]:
                    img = obs["images"][cam_name]
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)
                    if img.shape[-1] == 3:
                        img = img.permute(0, 3, 1, 2)
                    img = img.to(dtype=torch.float32, device=self.device)
                    img = self.normalizer.normalize_images(img, key=feat_key)
                    batch[feat_key] = img

        # Stack images for model
        if self.config.image_features:
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        # Add obs_steps dimension (duplicate current obs for history)
        obs_state = batch["observation.state"].unsqueeze(1).expand(-1, self.config.n_obs_steps, -1)
        model_batch = {"observation.state": obs_state}

        if "observation.images" in batch:
            obs_images = batch["observation.images"].unsqueeze(1).expand(
                -1, self.config.n_obs_steps, *batch["observation.images"].shape[1:]
            )
            model_batch["observation.images"] = obs_images

        # Run denoising
        actions_chunk = self.policy.diffusion.generate_actions(model_batch)  # (B, n_action_steps, action_dim)
        actions_chunk = actions_chunk[:, :n_steps]

        # Denormalize each timestep and convert to radians
        actions_list = []
        for t in range(n_steps):
            action_t = self.normalizer.denormalize("action", actions_chunk[:, t])
            action_t_motor = action_t.cpu().numpy()
            action_t_rad = motor_normalized_to_joint_rad(action_t_motor)
            actions_list.append(action_t_rad)

        return np.stack(actions_list, axis=1)


__all__ = ["DiffusionInference", "DiffusionConfig", "DiffusionPolicy"]
