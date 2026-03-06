"""Reward functions for figure shape placement RL environment.

GPU-native Isaac Lab reward terms using ManagerTermBase pattern.
"""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from so101_lab.tasks.figure_shape_placement.env_cfg import TABLE_HEIGHT
from .observations import slot_pos_w as _slot_pos_w, cube_pos_w as _cube_pos_w, _quat_to_yaw


def drop_penalty(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    z_threshold: float = 0.05,
) -> torch.Tensor:
    """1.0 if cube fell below table. Shape: (N,)"""
    cube = env.scene[cube_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return (cube_z < TABLE_HEIGHT - z_threshold).float()


def jerky_motion_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    vel_threshold: float = 5.0,
) -> torch.Tensor:
    """Excess joint velocity above threshold. Shape: (N,)"""
    robot = env.scene[asset_cfg.name]
    max_vel = robot.data.joint_vel.abs().max(dim=1).values
    return torch.clamp(max_vel - vel_threshold, min=0.0)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Squared L2 of action change: ||a_t - a_{t-1}||². Shape: (N,)"""
    action = env.action_manager.action
    prev = env.action_manager.prev_action
    return torch.sum((action - prev) ** 2, dim=-1)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant 1.0 per step. Shape: (N,)"""
    return torch.ones(env.num_envs, device=env.device)


def distance_cube_to_slot(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    slot_offset: tuple[float, float] = (-0.0667, 0.0667),
) -> torch.Tensor:
    """Negative XY distance cube→slot. Shape: (N,)"""
    cube_pos = _cube_pos_w(env, cube_cfg)
    slot_pos = _slot_pos_w(env, platform_cfg, slot_offset)
    return -torch.norm(cube_pos[:, :2] - slot_pos[:, :2], dim=1)


def distance_gripper_to_cube(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    """Negative 3D distance gripper→cube. Shape: (N,)"""
    ee = env.scene[ee_cfg.name]
    ee_pos = ee.data.target_pos_w[:, 0, :] - env.scene.env_origins
    cube_pos = _cube_pos_w(env, cube_cfg)
    return -torch.norm(ee_pos - cube_pos, dim=1)


def table_contact_penalty(
    env: ManagerBasedRLEnv,
    gripper_sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_contact"),
    jaw_sensor_cfg: SceneEntityCfg = SceneEntityCfg("jaw_contact"),
    table_filter_idx: int = 0,
    max_force: float = 10.0,
) -> torch.Tensor:
    """Force magnitude of gripper↔table contact, clamped to [0, max_force]. Shape: (N,)

    Uses force_matrix_w from ContactSensor with Table filter (index 0).
    Returns max force across both jaws — higher force = bigger penalty.
    Clamped so hard collisions don't dominate the reward signal.
    """
    g_sensor = env.scene[gripper_sensor_cfg.name]
    j_sensor = env.scene[jaw_sensor_cfg.name]
    # force_matrix_w: (N, B, M, 3) — B=1 body, M filters, 3D force
    g_force = torch.nan_to_num(g_sensor.data.force_matrix_w[:, 0, table_filter_idx], nan=0.0)
    j_force = torch.nan_to_num(j_sensor.data.force_matrix_w[:, 0, table_filter_idx], nan=0.0)
    g_mag = g_force.norm(dim=1)
    j_mag = j_force.norm(dim=1)
    return torch.clamp(torch.max(g_mag, j_mag), max=max_force)


class MilestonePicked(ManagerTermBase):
    """One-time bonus when cube is lifted above threshold."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._achieved = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._achieved[:] = False
        else:
            self._achieved[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        lift_height: float = TABLE_HEIGHT + 0.03,
    ) -> torch.Tensor:
        cube = env.scene[cube_cfg.name]
        cube_z = cube.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
        newly = (~self._achieved) & (cube_z > lift_height)
        self._achieved |= newly
        return newly.float()


class MilestonePlaced(ManagerTermBase):
    """One-time bonus when cube is placed in slot (XY + Z + orientation)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._achieved = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._achieved[:] = False
        else:
            self._achieved[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
        slot_offset: tuple[float, float] = (-0.0667, 0.0667),
        threshold: float = 0.04,
        z_threshold: float = 0.004,
        orientation_threshold: float = 20.0,
    ) -> torch.Tensor:
        cube_pos = _cube_pos_w(env, cube_cfg)
        slot_pos = _slot_pos_w(env, platform_cfg, slot_offset)
        dist_xy = torch.norm(cube_pos[:, :2] - slot_pos[:, :2], dim=1)
        # Z: expected cube height when placed in slot
        expected_z = TABLE_HEIGHT + 0.006
        dist_z = torch.abs(cube_pos[:, 2] - expected_z)
        # Orientation: cube vs platform yaw with 90° symmetry
        cube = env.scene[cube_cfg.name]
        platform = env.scene[platform_cfg.name]
        cube_yaw = _quat_to_yaw(cube.data.root_quat_w)
        platform_yaw = _quat_to_yaw(platform.data.root_quat_w)
        angle_diff = (cube_yaw - platform_yaw) % (math.pi / 2)
        threshold_rad = math.radians(orientation_threshold)
        rot_ok = (angle_diff < threshold_rad) | (angle_diff > (math.pi / 2 - threshold_rad))

        newly = (~self._achieved) & (dist_xy < threshold) & (dist_z < z_threshold) & rot_ok
        self._achieved |= newly
        return newly.float()
