"""Termination functions for figure shape placement RL environment."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from so101_lab.tasks.figure_shape_placement.env_cfg import TABLE_HEIGHT


def cube_out_of_bounds(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    z_threshold: float = 0.05,
) -> torch.Tensor:
    """True if cube Z < TABLE_HEIGHT - z_threshold. Shape: (N,) bool"""
    cube = env.scene[asset_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return cube_z < (TABLE_HEIGHT - z_threshold)


def platform_out_of_bounds(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    z_threshold: float = 0.05,
) -> torch.Tensor:
    """True if platform Z < TABLE_HEIGHT - z_threshold. Shape: (N,) bool"""
    platform = env.scene[asset_cfg.name]
    platform_z = platform.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return platform_z < (TABLE_HEIGHT - z_threshold)
