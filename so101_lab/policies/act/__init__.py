# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: Standalone inference wrapper for SO-101, handles rad<->normalized conversion
"""ACT policy inference for Isaac Lab.

Usage:
    from so101_lab.policies.act import ACTInference

    policy = ACTInference("outputs/act_v1")
    policy.reset()
    action = policy.select_action(obs)
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from so101_lab.data.converters import joint_rad_to_motor_normalized, motor_normalized_to_joint_rad

from .configuration_act import ACTConfig
from .modeling_act import ACTPolicy
from .normalization import Normalizer


class ACTInference:
    """ACT policy wrapper for Isaac Lab inference.

    Handles:
    - Loading checkpoint from lerobot training
    - Conversion between radians (Isaac Lab) and normalized motor (LeRobot)
    - Input/output normalization using checkpoint statistics
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
    ):
        """Initialize ACT inference.

        Args:
            checkpoint_path: Path to trained model directory (contains config.json, model.safetensors)
            device: Device to run on
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device

        # Load config
        config_path = self.checkpoint_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path) as f:
            config_dict = json.load(f)

        self.config = ACTConfig.from_dict(config_dict)

        # Create model
        self.policy = ACTPolicy(self.config)
        self.policy.to(device)
        self.policy.eval()

        # Load weights
        self._load_weights()

        # Setup normalizer from checkpoint
        self.normalizer = Normalizer(self.checkpoint_path, device=device)

        print(f"Loaded ACT policy from {checkpoint_path}")
        print(f"  chunk_size: {self.config.chunk_size}")
        print(f"  n_action_steps: {self.config.n_action_steps}")
        print(f"  image_features: {self.config.image_features}")

    def _load_weights(self):
        """Load model weights from checkpoint."""
        # Try safetensors first, then pytorch
        safetensors_path = self.checkpoint_path / "model.safetensors"
        pytorch_path = self.checkpoint_path / "model.pt"

        if safetensors_path.exists():
            state_dict = load_file(safetensors_path, device=self.device)
        elif pytorch_path.exists():
            state_dict = torch.load(pytorch_path, map_location=self.device)
        else:
            raise FileNotFoundError(f"No model weights found in {self.checkpoint_path}")

        # Handle potential prefix differences
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        # Load into policy.model
        self.policy.model.load_state_dict(state_dict, strict=False)

    def reset(self):
        """Call on episode start."""
        self.policy.reset()

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

        # Prepare batch
        batch = {"observation.state": state}

        # Process images
        if "images" in obs and self.config.image_features:
            for feat_key in self.config.image_features:
                # Extract camera name from feature key (e.g., "observation.images.top" -> "top")
                cam_name = feat_key.split(".")[-1]
                if cam_name in obs["images"]:
                    img = obs["images"][cam_name]

                    # Convert to tensor
                    if isinstance(img, np.ndarray):
                        img = torch.from_numpy(img)

                    # Ensure (B, C, H, W) format
                    if img.dim() == 3:
                        img = img.unsqueeze(0)  # Add batch
                    if img.shape[-1] == 3:
                        img = img.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

                    img = img.to(dtype=torch.float32, device=self.device)

                    # Normalize images
                    img = self.normalizer.normalize_images(img, key=feat_key)

                    batch[feat_key] = img

        # Get action from policy
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
        """Select actions for batched observations.

        Args:
            obs: Dict with batched arrays:
                - "joint_pos": (B, 6) array in radians
                - "images": dict of {"top": (B, H, W, 3), ...} uint8 arrays

        Returns:
            actions: (B, 6) array in radians
        """
        batch_size = obs["joint_pos"].shape[0]

        # Convert joint_pos from radians to normalized motor format
        joint_pos_rad = obs["joint_pos"]
        if isinstance(joint_pos_rad, torch.Tensor):
            joint_pos_rad = joint_pos_rad.cpu().numpy()
        joint_pos_normalized = joint_rad_to_motor_normalized(joint_pos_rad)

        # Prepare state tensor
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

        # Get actions
        with torch.no_grad():
            self.policy.eval()
            if self.config.image_features:
                batch_model = dict(batch)
                batch_model["observation.images"] = [batch[key] for key in self.config.image_features]
            else:
                batch_model = batch

            actions_chunk = self.policy.model(batch_model)[0]  # (B, chunk_size, action_dim)
            actions_normalized = actions_chunk[:, 0]  # Take first action

        # Denormalize
        actions_normalized = self.normalizer.denormalize("action", actions_normalized)

        # Convert to numpy and then to radians
        actions_motor = actions_normalized.cpu().numpy()
        actions_rad = motor_normalized_to_joint_rad(actions_motor)

        return actions_rad

    @torch.no_grad()
    def select_action_batch_ensemble(
        self, obs: dict, skip_env_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Select actions for batched observations using temporal ensembling.

        Runs forward pass for all envs and updates the shared temporal ensembler.
        Must be called every step (not compatible with action queue).

        Args:
            obs: Dict with batched arrays:
                - "joint_pos": (B, 6) array in radians
                - "images": dict of {"top": (B, H, W, 3), ...} uint8 arrays
            skip_env_ids: Env indices to exclude from ensembler update
                (e.g. envs with stale camera after auto-reset).
                Their ensembler state is untouched; call reset_envs() separately.

        Returns:
            actions: (B, 6) array in radians
        """
        # Convert joint_pos from radians to normalized motor format
        joint_pos_rad = obs["joint_pos"]
        if isinstance(joint_pos_rad, torch.Tensor):
            joint_pos_rad = joint_pos_rad.cpu().numpy()
        joint_pos_normalized = joint_rad_to_motor_normalized(joint_pos_rad)

        # Prepare state tensor
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

        # Get full action chunk via forward pass
        self.policy.eval()
        if self.config.image_features:
            batch_model = dict(batch)
            batch_model["observation.images"] = [batch[key] for key in self.config.image_features]
        else:
            batch_model = batch

        actions_chunk = self.policy.model(batch_model)[0]  # (B, chunk_size, action_dim)

        # Update temporal ensembler
        actions_normalized = self.policy.temporal_ensembler.update(
            actions_chunk, skip_env_ids=skip_env_ids,
        )  # (B, action_dim)

        # Denormalize and convert to radians
        actions_normalized = self.normalizer.denormalize("action", actions_normalized)
        actions_motor = actions_normalized.cpu().numpy()
        actions_rad = motor_normalized_to_joint_rad(actions_motor)

        return actions_rad

    @torch.no_grad()
    def pop_ensemble_actions(self) -> np.ndarray:
        """Pop next ensembled actions without running inference."""
        actions_normalized = self.policy.temporal_ensembler.pop()
        actions_normalized = self.normalizer.denormalize("action", actions_normalized)
        actions_motor = actions_normalized.cpu().numpy()
        return motor_normalized_to_joint_rad(actions_motor)

    def select_action_chunk_batch(self, obs: dict, n_steps: int | None = None) -> np.ndarray:
        """Select action chunks for batched observations.

        Returns multiple actions per env for use with action queues.

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

        # Convert joint_pos from radians to normalized motor format
        joint_pos_rad = obs["joint_pos"]
        if isinstance(joint_pos_rad, torch.Tensor):
            joint_pos_rad = joint_pos_rad.cpu().numpy()
        joint_pos_normalized = joint_rad_to_motor_normalized(joint_pos_rad)

        # Prepare state tensor
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

        # Get action chunk
        with torch.no_grad():
            self.policy.eval()
            if self.config.image_features:
                batch_model = dict(batch)
                batch_model["observation.images"] = [batch[key] for key in self.config.image_features]
            else:
                batch_model = batch

            actions_chunk = self.policy.model(batch_model)[0]  # (B, chunk_size, action_dim)
            actions_chunk = actions_chunk[:, :n_steps]  # (B, n_steps, action_dim)

        # Denormalize each timestep and convert to radians
        actions_list = []
        for t in range(n_steps):
            action_t = self.normalizer.denormalize("action", actions_chunk[:, t])
            action_t_motor = action_t.cpu().numpy()
            action_t_rad = motor_normalized_to_joint_rad(action_t_motor)
            actions_list.append(action_t_rad)

        # Stack: (B, n_steps, 6)
        return np.stack(actions_list, axis=1)


__all__ = ["ACTInference", "ACTConfig", "ACTPolicy"]
