"""Event functions for figure shape placement RL environment.

Ported from FigureShapePlacementEnv._reset_idx() and _randomize_light().
"""

from __future__ import annotations

import math

import numpy as np
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw (Z rotation) to quaternion (w, x, y, z)."""
    half = yaw * 0.5
    w = torch.cos(half)
    z = torch.sin(half)
    zeros = torch.zeros_like(yaw)
    return torch.stack([w, zeros, zeros, z], dim=-1)


def _sample_in_ring(
    n: int,
    r_min: float,
    r_max: float,
    obj_radius: float,
    robot_x: float,
    robot_y: float,
    table_bounds: tuple[float, float, float, float],
    theta_min: float = 0.0,
    theta_max: float = 2 * math.pi,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample positions in polar coordinates within reach ring."""
    r = torch.empty(n, device=device).uniform_(r_min, r_max)
    theta = torch.empty(n, device=device).uniform_(theta_min, theta_max)
    x = robot_x + r * torch.cos(theta)
    y = robot_y + r * torch.sin(theta)

    x_min, x_max, y_min, y_max = table_bounds
    valid = (
        (x >= x_min + obj_radius) & (x <= x_max - obj_radius)
        & (y >= y_min + obj_radius) & (y <= y_max - obj_radius)
    )
    return x, y, valid


def reset_platform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    fixed_x: float,
    fixed_y: float,
    yaw_values_deg: tuple[float, ...],
    table_h: float,
):
    """Reset platform to fixed position with random discrete yaw."""
    platform = env.scene[asset_cfg.name]
    n = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]

    # Fixed XY, random discrete yaw
    px = torch.full((n,), fixed_x, device=env.device)
    py = torch.full((n,), fixed_y, device=env.device)

    yaw_values_rad = torch.tensor(
        [math.radians(deg) for deg in yaw_values_deg], device=env.device
    )
    yaw_indices = torch.randint(0, len(yaw_values_rad), (n,), device=env.device)
    platform_yaw = yaw_values_rad[yaw_indices]

    root_state = platform.data.default_root_state[env_ids].clone()
    root_state[:, 0] = px + env_origins[:, 0]
    root_state[:, 1] = py + env_origins[:, 1]
    root_state[:, 2] = table_h + env_origins[:, 2]
    root_state[:, 3:7] = _yaw_to_quat(platform_yaw)

    platform.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    platform.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_cube_polar(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg,
    platform_cfg: SceneEntityCfg,
    robot_x: float,
    robot_y: float,
    min_reach: float,
    max_reach: float,
    cube_radius: float,
    platform_radius: float,
    theta_min: float,
    theta_max: float,
    r_max_offset: float,
    gap_reduction: float,
    table_h: float,
    max_attempts: int,
    table_bounds: tuple[float, float, float, float],
):
    """Reset cube with rejection sampling in polar coordinates, avoiding platform."""
    cube = env.scene[cube_cfg.name]
    platform = env.scene[platform_cfg.name]
    n = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]

    r_min = min_reach + cube_radius
    r_max = max_reach - cube_radius - r_max_offset
    min_dist = platform_radius + cube_radius - gap_reduction

    # Get current platform XY (local, without env_origins)
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

        sx, sy, on_table = _sample_in_ring(
            k, r_min, r_max, cube_radius, robot_x, robot_y,
            table_bounds, theta_min, theta_max, env.device,
        )

        dist_to_platform = torch.sqrt(
            (sx - platform_x[remaining]) ** 2 + (sy - platform_y[remaining]) ** 2
        )
        valid = on_table & (dist_to_platform >= min_dist)

        remaining_idx = torch.where(remaining)[0]
        accepted_idx = remaining_idx[valid]
        x[accepted_idx] = sx[valid]
        y[accepted_idx] = sy[valid]
        remaining[accepted_idx] = False

    # Fallback
    if remaining.any():
        x[remaining] = platform_x[remaining] + min_dist + 0.02
        y[remaining] = platform_y[remaining]

    cube_yaw = torch.empty(n, device=env.device).uniform_(-math.pi, math.pi)

    root_state = cube.data.default_root_state[env_ids].clone()
    root_state[:, 0] = x + env_origins[:, 0]
    root_state[:, 1] = y + env_origins[:, 1]
    root_state[:, 2] = table_h + 0.009 + env_origins[:, 2]
    root_state[:, 3:7] = _yaw_to_quat(cube_yaw)

    cube.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
    cube.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)


def reset_cube_rect(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_cfg: SceneEntityCfg,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    yaw_values_deg: tuple[float, ...],
    table_h: float,
):
    """Reset cube to uniform random position in rectangular zone with discrete yaw."""
    cube = env.scene[cube_cfg.name]
    n = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]

    x = torch.empty(n, device=env.device).uniform_(x_min, x_max)
    y = torch.empty(n, device=env.device).uniform_(y_min, y_max)

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


_light_cache: dict[str, "pxr.Usd.Prim | None"] = {}


def _find_light_prim(stage, name: str) -> "pxr.Usd.Prim | None":
    """Find and cache a light prim by name."""
    if name in _light_cache:
        return _light_cache[name]
    for prim in stage.Traverse():
        if prim.GetName() == name and "Light" in prim.GetTypeName():
            _light_cache[name] = prim
            return prim
    _light_cache[name] = None
    return None


def randomize_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    intensity_step: float,
    color_temp_range: tuple[float, float],
    color_temp_step: float,
    window_pos_offset: float = 0.0,
    window_rot_offset_deg: float = 0.0,
    ceiling_intensity_range: tuple[float, float] | None = None,
    dome_intensity_range: tuple[float, float] | None = None,
):
    """Randomize lights via USD API.

    - Window (RectLight): intensity, color temp, position offset, rotation offset
    - CeilingLamp (SphereLight): intensity
    - EnvironmentLight (DomeLight): intensity

    Only runs for env_ids containing 0 (single shared light prims).
    """
    if 0 not in env_ids:
        return

    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()

    # --- Window light ---
    window = _find_light_prim(stage, "Window")
    if window is not None:
        intensity_values = np.arange(
            intensity_range[0], intensity_range[1] + 1, intensity_step
        )
        color_temp_values = np.arange(
            color_temp_range[0], color_temp_range[1] + 1, color_temp_step
        )
        window.GetAttribute("inputs:intensity").Set(
            float(np.random.choice(intensity_values))
        )
        window.GetAttribute("inputs:colorTemperature").Set(
            float(np.random.choice(color_temp_values))
        )

        # Position offset (XY)
        if window_pos_offset > 0:
            translate_attr = window.GetAttribute("xformOp:translate")
            if translate_attr and translate_attr.IsValid():
                # Cache original position
                if not hasattr(randomize_light, "_window_orig_pos"):
                    randomize_light._window_orig_pos = Gf.Vec3d(translate_attr.Get())
                orig = randomize_light._window_orig_pos
                dx = float(np.random.uniform(-window_pos_offset, window_pos_offset))
                dy = float(np.random.uniform(-window_pos_offset, window_pos_offset))
                translate_attr.Set(Gf.Vec3d(orig[0] + dx, orig[1] + dy, orig[2]))

        # Rotation offset (XY tilt)
        if window_rot_offset_deg > 0:
            rotate_attr = window.GetAttribute("xformOp:rotateXYZ")
            if rotate_attr and rotate_attr.IsValid():
                if not hasattr(randomize_light, "_window_orig_rot"):
                    randomize_light._window_orig_rot = Gf.Vec3f(rotate_attr.Get())
                orig = randomize_light._window_orig_rot
                rx = float(np.random.uniform(-window_rot_offset_deg, window_rot_offset_deg))
                ry = float(np.random.uniform(-window_rot_offset_deg, window_rot_offset_deg))
                rotate_attr.Set(Gf.Vec3f(orig[0] + rx, orig[1] + ry, orig[2]))

    # --- CeilingLamp ---
    if ceiling_intensity_range is not None:
        ceiling = _find_light_prim(stage, "CeilingLamp")
        if ceiling is not None:
            val = float(np.random.uniform(*ceiling_intensity_range))
            ceiling.GetAttribute("inputs:intensity").Set(val)

    # --- EnvironmentLight (DomeLight) ---
    if dome_intensity_range is not None:
        dome = _find_light_prim(stage, "EnvironmentLight")
        if dome is not None:
            val = float(np.random.uniform(*dome_intensity_range))
            dome.GetAttribute("inputs:intensity").Set(val)


def reset_distractors(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    distractor_cfgs: list[SceneEntityCfg],
    cube_cfg: SceneEntityCfg,
    platform_cfg: SceneEntityCfg,
    robot_x: float,
    robot_y: float,
    min_reach: float,
    max_reach: float,
    distractor_radius: float,
    cube_radius: float,
    platform_radius: float,
    table_h: float,
    table_bounds: tuple[float, float, float, float],
    max_active: int = 3,
):
    """Reset distractors: randomly place 0..max_active on the table, hide the rest."""
    n = len(env_ids)
    env_origins = env.scene.env_origins[env_ids]
    device = env.device

    # Get cube/platform positions (local)
    cube = env.scene[cube_cfg.name]
    platform = env.scene[platform_cfg.name]
    cube_pos = (cube.data.root_pos_w[env_ids] - env_origins)[:, :2]
    platform_pos = (platform.data.root_pos_w[env_ids] - env_origins)[:, :2]

    # Decide how many distractors active per env (0 to max_active)
    num_active = torch.randint(0, max_active + 1, (n,), device=device)

    # Positions of already-placed distractors (for inter-distractor rejection)
    placed_positions = []  # list of (n, 2) tensors

    for d_idx, d_cfg in enumerate(distractor_cfgs):
        dist_obj = env.scene[d_cfg.name]
        root_state = dist_obj.data.default_root_state[env_ids].clone()

        # Which envs have this distractor active?
        active_mask = num_active > d_idx  # (n,)

        # Sample positions for active ones
        n_active = active_mask.sum().item()
        if n_active > 0:
            active_idx = torch.where(active_mask)[0]

            # Try rejection sampling
            best_x = torch.zeros(n_active, device=device)
            best_y = torch.zeros(n_active, device=device)
            found = torch.zeros(n_active, dtype=torch.bool, device=device)

            for _ in range(50):
                remaining = (~found).sum().item()
                if remaining == 0:
                    break

                r_min = min_reach + distractor_radius
                r_max = max_reach - distractor_radius
                sx, sy, on_table = _sample_in_ring(
                    remaining, r_min, r_max, distractor_radius,
                    robot_x, robot_y, table_bounds, device=device,
                )

                # Check distance to cube
                cube_xy = cube_pos[active_idx[~found]]
                dist_cube = torch.sqrt((sx - cube_xy[:, 0]) ** 2 + (sy - cube_xy[:, 1]) ** 2)
                valid = on_table & (dist_cube >= cube_radius + distractor_radius)

                # Check distance to platform
                plat_xy = platform_pos[active_idx[~found]]
                dist_plat = torch.sqrt((sx - plat_xy[:, 0]) ** 2 + (sy - plat_xy[:, 1]) ** 2)
                valid = valid & (dist_plat >= platform_radius + distractor_radius)

                # Check distance to already-placed distractors
                for placed in placed_positions:
                    placed_xy = placed[active_idx[~found]]
                    dist_placed = torch.sqrt((sx - placed_xy[:, 0]) ** 2 + (sy - placed_xy[:, 1]) ** 2)
                    valid = valid & (dist_placed >= 2 * distractor_radius)

                remaining_idx = torch.where(~found)[0]
                accepted = remaining_idx[valid]
                best_x[accepted] = sx[valid]
                best_y[accepted] = sy[valid]
                found[accepted] = True

            # Fallback: place off-table
            if not found.all():
                best_x[~found] = 0.0
                best_y[~found] = 0.0

            # Set active positions
            root_state[active_mask, 0] = best_x + env_origins[active_mask, 0]
            root_state[active_mask, 1] = best_y + env_origins[active_mask, 1]
            root_state[active_mask, 2] = table_h + 0.01 + env_origins[active_mask, 2]

            # Random yaw
            yaw = torch.empty(n_active, device=device).uniform_(-math.pi, math.pi)
            root_state[active_mask, 3:7] = _yaw_to_quat(yaw)

            # Track placed positions
            placed_xy = torch.zeros(n, 2, device=device)
            placed_xy[active_mask, 0] = best_x
            placed_xy[active_mask, 1] = best_y
            placed_positions.append(placed_xy)

        # Inactive: hide below table
        inactive_mask = ~active_mask
        if inactive_mask.any():
            root_state[inactive_mask, 0] = env_origins[inactive_mask, 0]
            root_state[inactive_mask, 1] = env_origins[inactive_mask, 1]
            root_state[inactive_mask, 2] = -1.0 + env_origins[inactive_mask, 2]

        # Zero velocities
        root_state[:, 7:] = 0.0

        dist_obj.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        dist_obj.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)
