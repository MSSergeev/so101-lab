"""Observation functions for figure shape placement RL environment."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.observations import image as _base_image
from isaaclab.managers import SceneEntityCfg


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw (Z rotation) from quaternion (w, x, y, z)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def cube_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube world position. Shape: (N, 3)"""
    cube = env.scene[asset_cfg.name]
    return cube.data.root_pos_w - env.scene.env_origins


def cube_quat_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    """Cube world quaternion. Shape: (N, 4)"""
    cube = env.scene[asset_cfg.name]
    return cube.data.root_quat_w


def slot_pos_w(
    env: ManagerBasedEnv,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    slot_offset: tuple[float, float] = (-0.0667, 0.0667),
) -> torch.Tensor:
    """Slot world position (computed from platform pos + rotated offset). Shape: (N, 3)"""
    platform = env.scene[platform_cfg.name]
    pos = platform.data.root_pos_w - env.scene.env_origins
    yaw = _quat_to_yaw(platform.data.root_quat_w)

    cos_a = torch.cos(yaw)
    sin_a = torch.sin(yaw)
    ox, oy = slot_offset
    slot_x = pos[:, 0] + ox * cos_a - oy * sin_a
    slot_y = pos[:, 1] + ox * sin_a + oy * cos_a
    slot_z = pos[:, 2]

    return torch.stack([slot_x, slot_y, slot_z], dim=-1)


def platform_yaw(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("platform")) -> torch.Tensor:
    """Platform Z rotation. Shape: (N, 1)"""
    platform = env.scene[asset_cfg.name]
    yaw = _quat_to_yaw(platform.data.root_quat_w)
    return yaw.unsqueeze(-1)


def image_with_noise(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    normalize: bool = False,
    brightness_range: tuple[float, float] = (-20.0, 20.0),
    contrast_range: tuple[float, float] = (0.8, 1.2),
    noise_std_range: tuple[float, float] = (3.0, 12.0),
) -> torch.Tensor:
    """Camera image with per-episode brightness/contrast/noise_std and per-step noise pattern.

    Per-episode (sampled at reset): brightness, contrast, noise_std.
    Per-step: Gaussian noise pattern (new random values each frame, fixed std for episode).
    """
    images = _base_image(env, sensor_cfg, data_type, normalize=normalize)
    # images: (N, H, W, 3) float (unnormalized = 0-255 range)

    n = images.shape[0]
    device = images.device
    cam_name = sensor_cfg.name

    # Per-episode brightness/contrast: resample after reset (episode_length_buf == 0)
    cache_attr = "_camera_noise_cache"
    if not hasattr(env, cache_attr):
        setattr(env, cache_attr, {})
    cache = getattr(env, cache_attr)

    needs_resample = cam_name not in cache or env.episode_length_buf[0].item() == 0
    if needs_resample:
        cache[cam_name] = {
            "brightness": torch.empty(n, 1, 1, 1, device=device).uniform_(*brightness_range),
            "contrast": torch.empty(n, 1, 1, 1, device=device).uniform_(*contrast_range),
            "noise_std": torch.empty(n, 1, 1, 1, device=device).uniform_(*noise_std_range),
        }

    # Apply per-episode brightness
    if brightness_range[0] != 0.0 or brightness_range[1] != 0.0:
        images = images + cache[cam_name]["brightness"]

    # Apply per-episode contrast
    if contrast_range[0] != 1.0 or contrast_range[1] != 1.0:
        mean = images.mean(dim=(1, 2, 3), keepdim=True)
        images = (images - mean) * cache[cam_name]["contrast"] + mean

    # Per-step Gaussian noise (pattern changes, std fixed per episode)
    if noise_std_range[1] > 0:
        images = images + torch.randn_like(images) * cache[cam_name]["noise_std"]

    return images.clamp(0, 255).clone()
