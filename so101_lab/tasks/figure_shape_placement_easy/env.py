"""Figure shape placement (easy) task environment.

Same as FigureShapePlacementEnv but with rectangular cube spawn zone
and discrete yaw values.
"""

from __future__ import annotations

import math

import torch

from so101_lab.tasks.figure_shape_placement.env import (
    FigureShapePlacementEnv,
    yaw_to_quat,
)
from .env_cfg import (
    CUBE_SPAWN_X_MIN,
    CUBE_SPAWN_X_MAX,
    CUBE_SPAWN_Y_MIN,
    CUBE_SPAWN_Y_MAX,
    CUBE_YAW_VALUES_DEG,
    CUBE_RADIUS,
    PLATFORM_RADIUS,
    CUBE_PLATFORM_GAP_REDUCTION,
    MAX_SPAWN_ATTEMPTS,
    PLATFORM_FIXED_X,
    PLATFORM_FIXED_Y,
    PLATFORM_YAW_VALUES_DEG,
    TABLE_HEIGHT,
    FigureShapePlacementEasyEnvCfg,
)


class FigureShapePlacementEasyEnv(FigureShapePlacementEnv):
    """Easy variant: small rectangular spawn zone, discrete yaw."""

    cfg: FigureShapePlacementEasyEnvCfg

    def _spawn_platform(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.full((n,), PLATFORM_FIXED_X, device=self.device),
            torch.full((n,), PLATFORM_FIXED_Y, device=self.device),
        )

    def _spawn_cube(
        self, n: int, platform_x: torch.Tensor, platform_y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spawn cube in rectangular zone with rejection sampling for platform collision."""
        min_dist = PLATFORM_RADIUS + CUBE_RADIUS - CUBE_PLATFORM_GAP_REDUCTION

        x = torch.empty(n, device=self.device)
        y = torch.empty(n, device=self.device)
        remaining = torch.ones(n, dtype=torch.bool, device=self.device)

        for _ in range(MAX_SPAWN_ATTEMPTS):
            k = remaining.sum().item()
            if k == 0:
                break

            sx = torch.empty(k, device=self.device).uniform_(CUBE_SPAWN_X_MIN, CUBE_SPAWN_X_MAX)
            sy = torch.empty(k, device=self.device).uniform_(CUBE_SPAWN_Y_MIN, CUBE_SPAWN_Y_MAX)

            dist_to_platform = torch.sqrt(
                (sx - platform_x[remaining]) ** 2 + (sy - platform_y[remaining]) ** 2
            )
            valid = dist_to_platform >= min_dist

            remaining_idx = torch.where(remaining)[0]
            accepted_idx = remaining_idx[valid]
            x[accepted_idx] = sx[valid]
            y[accepted_idx] = sy[valid]
            remaining[accepted_idx] = False

        if remaining.any():
            x[remaining] = platform_x[remaining] + min_dist + 0.02
            y[remaining] = platform_y[remaining]

        return x, y

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset with rectangular spawn and discrete cube yaw."""
        # Call grandparent (TemplateEnv._reset_idx), not parent
        super(FigureShapePlacementEnv, self)._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._platform._ALL_INDICES

        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]

        # Platform: easy position and yaw
        platform_x, platform_y = self._spawn_platform(n)
        yaw_values_rad = torch.tensor(
            [math.radians(deg) for deg in PLATFORM_YAW_VALUES_DEG],
            device=self.device,
        )
        yaw_indices = torch.randint(0, len(yaw_values_rad), (n,), device=self.device)
        platform_yaw = yaw_values_rad[yaw_indices]

        # Cube: rectangular spawn, discrete yaw
        cube_x, cube_y = self._spawn_cube(n, platform_x, platform_y)
        cube_yaw_values_rad = torch.tensor(
            [math.radians(deg) for deg in CUBE_YAW_VALUES_DEG],
            device=self.device,
        )
        cube_yaw_indices = torch.randint(0, len(cube_yaw_values_rad), (n,), device=self.device)
        cube_yaw = cube_yaw_values_rad[cube_yaw_indices]

        # Platform state
        platform_state = self._platform.data.default_root_state[env_ids].clone()
        platform_state[:, 0] = platform_x + env_origins[:, 0]
        platform_state[:, 1] = platform_y + env_origins[:, 1]
        platform_state[:, 2] = TABLE_HEIGHT + env_origins[:, 2]
        platform_state[:, 3:7] = yaw_to_quat(platform_yaw)

        # Cube state
        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, 0] = cube_x + env_origins[:, 0]
        cube_state[:, 1] = cube_y + env_origins[:, 1]
        cube_state[:, 2] = TABLE_HEIGHT + 0.009 + env_origins[:, 2]
        cube_state[:, 3:7] = yaw_to_quat(cube_yaw)

        self._platform.write_root_state_to_sim(platform_state, env_ids=env_ids)
        self._cube.write_root_state_to_sim(cube_state, env_ids=env_ids)

        initial_state = {
            "platform_x": float(platform_x[0]),
            "platform_y": float(platform_y[0]),
            "platform_yaw_deg": float(math.degrees(platform_yaw[0])),
            "cube_x": float(cube_x[0]),
            "cube_y": float(cube_y[0]),
            "cube_yaw_deg": float(math.degrees(cube_yaw[0])),
        }

        if self.cfg.randomize_light:
            light_state = self._randomize_light()
            if light_state:
                initial_state.update(light_state)

        self._last_initial_state = initial_state
