"""Event functions for figure shape placement (easy) RL environment.

Overrides reset_cube_polar with reset_cube_rect (rectangular spawn zone).
All other events (reset_platform, randomize_light, reset_distractors) are
inherited from the original task via mdp/__init__.py re-export.
"""

from __future__ import annotations

import math

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

from so101_lab.tasks.figure_shape_placement.rl.mdp.events import _yaw_to_quat


def reset_cube_rect(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg,
    platform_cfg: SceneEntityCfg,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    cube_radius: float,
    platform_radius: float,
    gap_reduction: float,
    yaw_values_deg: tuple[float, ...],
    table_h: float,
    max_attempts: int,
):
    """Reset cube in rectangular zone with discrete yaw, avoiding platform."""
    cube = env.scene[cube_cfg.name]
    platform = env.scene[platform_cfg.name]
    n = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]

    min_dist = platform_radius + cube_radius - gap_reduction

    platform_pos = platform.data.root_pos_w[env_ids] - env_origins
    platform_x = platform_pos[:, 0]
    platform_y = platform_pos[:, 1]

    x = torch.empty(n, device=env.device)
    y = torch.empty(n, device=env.device)
    remaining = torch.ones(n, dtype=torch.bool, device=env.device)

    for _ in range(max_attempts):
        k = remaining.sum().item()
        if k == 0:
            break

        sx = torch.empty(k, device=env.device).uniform_(x_min, x_max)
        sy = torch.empty(k, device=env.device).uniform_(y_min, y_max)

        dist_to_platform = torch.sqrt(
            (sx - platform_x[remaining]) ** 2 + (sy - platform_y[remaining]) ** 2
        )
        valid = dist_to_platform >= min_dist

        remaining_idx = torch.where(remaining)[0]
        accepted_idx = remaining_idx[valid]
        x[accepted_idx] = sx[valid]
        y[accepted_idx] = sy[valid]
        remaining[accepted_idx] = False

    # Fallback
    if remaining.any():
        x[remaining] = platform_x[remaining] + min_dist + 0.02
        y[remaining] = platform_y[remaining]

    # Discrete yaw
    yaw_values_rad = torch.tensor(
        [math.radians(deg) for deg in yaw_values_deg], device=env.device
    )
    yaw_indices = torch.randint(0, len(yaw_values_rad), (n,), device=env.device)
    cube_yaw = yaw_values_rad[yaw_indices]

    root_state = cube.data.default_root_state[env_ids].clone()
    root_state[:, 0] = x + env_origins[:, 0]
    root_state[:, 1] = y + env_origins[:, 1]
    root_state[:, 2] = table_h + 0.009 + env_origins[:, 2]
    root_state[:, 3:7] = _yaw_to_quat(cube_yaw)

    cube.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    cube.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
