"""Figure shape placement task environment."""

from __future__ import annotations

import math
import torch
import numpy as np
from isaaclab.assets import RigidObject

from so101_lab.tasks.template.env import TemplateEnv
from .env_cfg import (
    FigureShapePlacementEnvCfg,
    TABLE_HEIGHT,
    SLOT_OFFSET,
    SUCCESS_THRESHOLD,
    ORIENTATION_THRESHOLD,
    CUBE_SPAWN_X_MIN,
    CUBE_SPAWN_X_MAX,
    CUBE_SPAWN_Y_MIN,
    CUBE_SPAWN_Y_MAX,
    CUBE_YAW_VALUES_DEG,
    PLATFORM_FIXED_X,
    PLATFORM_FIXED_Y,
    PLATFORM_YAW_VALUES_DEG,
)

# Light name to search for in scene
WINDOW_LIGHT_NAME = "Window"


def rotate_2d(offset: tuple[float, float], angle: torch.Tensor) -> torch.Tensor:
    """Rotate 2D offset by angle (radians)."""
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)
    x_rot = offset[0] * cos_a - offset[1] * sin_a
    y_rot = offset[0] * sin_a + offset[1] * cos_a
    return torch.stack([x_rot, y_rot], dim=-1)


def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """Extract yaw (Z rotation) from quaternion (w, x, y, z)."""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    """Convert yaw (Z rotation) to quaternion (w, x, y, z)."""
    half_yaw = yaw * 0.5
    w = torch.cos(half_yaw)
    z = torch.sin(half_yaw)
    zeros = torch.zeros_like(yaw)
    return torch.stack([w, zeros, zeros, z], dim=-1)


class FigureShapePlacementEnv(TemplateEnv):
    """Figure shape placement task.

    Goal: Place the cube into the slot on the platform.

    Success criteria:
    - Cube center within ±2mm of slot center (XY)
    - Cube orientation aligned with platform (±5°, with 90° symmetry)

    Termination:
    - Only timeout (human controls episode end for imitation learning)
    """

    cfg: FigureShapePlacementEnvCfg

    def __init__(self, cfg: FigureShapePlacementEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._window_light_prim = None  # Cached light prim for randomization
        self._last_initial_state: dict | None = None

    def _setup_scene(self):
        """Register platform and cube in the scene."""
        super()._setup_scene()

        self._platform = RigidObject(self.cfg.scene.platform)
        self.scene.rigid_objects["platform"] = self._platform

        self._cube = RigidObject(self.cfg.scene.cube)
        self.scene.rigid_objects["cube"] = self._cube

    def get_task_description(self) -> str:
        """Get task description for current episode."""
        return "Place the cube into the matching slot on the platform"

    def _get_rewards(self) -> torch.Tensor:
        """Placeholder reward (not used for imitation learning)."""
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions.

        When terminate_on_success=True (eval mode): terminates on success.
        When terminate_on_success=False (teleop/recording): only timeout.
        """
        if self.cfg.terminate_on_success:
            terminated = self.is_success()
        else:
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def is_success(self) -> torch.Tensor:
        """Check if cube is correctly placed in slot.

        Checks three conditions (all must be true):
        1. XY position: cube center within ±2mm of slot center
        2. Z position: cube at expected height (in slot, not floating)
        3. Orientation: cube aligned with platform (±5°, 90° symmetry)
        """
        platform_pos = self._platform.data.root_pos_w
        platform_quat = self._platform.data.root_quat_w
        cube_pos = self._cube.data.root_pos_w
        cube_quat = self._cube.data.root_quat_w

        platform_yaw = quat_to_yaw(platform_quat)
        cube_yaw = quat_to_yaw(cube_quat)

        # Compute slot world position
        slot_offset_rotated = rotate_2d(SLOT_OFFSET, platform_yaw)
        slot_world = platform_pos[:, :2] + slot_offset_rotated

        # XY position check
        cube_to_slot_xy = torch.norm(cube_pos[:, :2] - slot_world, dim=-1)
        pos_xy_ok = cube_to_slot_xy < SUCCESS_THRESHOLD

        # Z position check (cube must be in slot, not floating)
        expected_cube_z = TABLE_HEIGHT + 0.006
        z_tolerance = 0.004
        pos_z_ok = torch.abs(cube_pos[:, 2] - expected_cube_z) < z_tolerance

        # Orientation check (90° symmetry)
        angle_diff = (cube_yaw - platform_yaw) % (math.pi / 2)
        threshold_rad = math.radians(ORIENTATION_THRESHOLD)
        rot_ok = (angle_diff < threshold_rad) | (angle_diff > (math.pi / 2 - threshold_rad))

        return pos_xy_ok & pos_z_ok & rot_ok

    def _spawn_platform(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.full((n,), PLATFORM_FIXED_X, device=self.device),
            torch.full((n,), PLATFORM_FIXED_Y, device=self.device),
        )

    def _spawn_cube(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.empty(n, device=self.device).uniform_(CUBE_SPAWN_X_MIN, CUBE_SPAWN_X_MAX)
        y = torch.empty(n, device=self.device).uniform_(CUBE_SPAWN_Y_MIN, CUBE_SPAWN_Y_MAX)
        return x, y

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._platform._ALL_INDICES

        n = len(env_ids)

        # Get env origins for multi-env support
        env_origins = self.scene.env_origins[env_ids]

        # Step 1: Spawn platform (fixed position, fixed yaw)
        platform_x, platform_y = self._spawn_platform(n)
        platform_yaw = torch.full(
            (n,), math.radians(PLATFORM_YAW_VALUES_DEG[0]), device=self.device
        )

        # Step 2: Spawn cube (rectangular zone, discrete yaw)
        cube_x, cube_y = self._spawn_cube(n)
        cube_yaw_values_rad = torch.tensor(
            [math.radians(d) for d in CUBE_YAW_VALUES_DEG], device=self.device
        )
        yaw_indices = torch.randint(0, len(cube_yaw_values_rad), (n,), device=self.device)
        cube_yaw = cube_yaw_values_rad[yaw_indices]

        # Build platform state (add env_origins for multi-env)
        platform_state = self._platform.data.default_root_state[env_ids].clone()
        platform_state[:, 0] = platform_x + env_origins[:, 0]
        platform_state[:, 1] = platform_y + env_origins[:, 1]
        platform_state[:, 2] = TABLE_HEIGHT + env_origins[:, 2]
        platform_state[:, 3:7] = yaw_to_quat(platform_yaw)

        # Build cube state (add env_origins for multi-env)
        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, 0] = cube_x + env_origins[:, 0]
        cube_state[:, 1] = cube_y + env_origins[:, 1]
        cube_state[:, 2] = TABLE_HEIGHT + 0.009 + env_origins[:, 2]
        cube_state[:, 3:7] = yaw_to_quat(cube_yaw)

        # Write states to simulation
        self._platform.write_root_state_to_sim(platform_state, env_ids=env_ids)
        self._cube.write_root_state_to_sim(cube_state, env_ids=env_ids)

        # Capture initial state (env 0 only, for single-env recording)
        initial_state = {
            "platform_x": float(platform_x[0]),
            "platform_y": float(platform_y[0]),
            "platform_yaw_deg": float(math.degrees(platform_yaw[0])),
            "cube_x": float(cube_x[0]),
            "cube_y": float(cube_y[0]),
            "cube_yaw_deg": float(math.degrees(cube_yaw[0])),
        }

        # Randomize light if enabled
        if self.cfg.randomize_light:
            light_state = self._randomize_light()
            if light_state:
                initial_state.update(light_state)

        self._last_initial_state = initial_state

    def reset_to_state(self, state: dict) -> tuple[dict, dict]:
        """Reset env to exact object positions from state dict.

        Used for replay recording — bypasses randomization.

        Args:
            state: Dict with platform_x, platform_y, platform_yaw_deg,
                   cube_x, cube_y, cube_yaw_deg, and optional light params.

        Returns:
            (obs_dict, info)
        """
        env_ids = self._platform._ALL_INDICES

        # Reset robot to default (parent handles joints + actuators)
        super()._reset_idx(env_ids)

        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]

        # Read exact positions from state
        platform_x = torch.full((n,), state["platform_x"], device=self.device)
        platform_y = torch.full((n,), state["platform_y"], device=self.device)
        platform_yaw = torch.full(
            (n,), math.radians(state["platform_yaw_deg"]), device=self.device
        )

        cube_x = torch.full((n,), state["cube_x"], device=self.device)
        cube_y = torch.full((n,), state["cube_y"], device=self.device)
        cube_yaw = torch.full(
            (n,), math.radians(state["cube_yaw_deg"]), device=self.device
        )

        # Build platform state
        platform_state = self._platform.data.default_root_state[env_ids].clone()
        platform_state[:, 0] = platform_x + env_origins[:, 0]
        platform_state[:, 1] = platform_y + env_origins[:, 1]
        platform_state[:, 2] = TABLE_HEIGHT + env_origins[:, 2]
        platform_state[:, 3:7] = yaw_to_quat(platform_yaw)

        # Build cube state
        cube_state = self._cube.data.default_root_state[env_ids].clone()
        cube_state[:, 0] = cube_x + env_origins[:, 0]
        cube_state[:, 1] = cube_y + env_origins[:, 1]
        cube_state[:, 2] = TABLE_HEIGHT + 0.009 + env_origins[:, 2]
        cube_state[:, 3:7] = yaw_to_quat(cube_yaw)

        # Write states to simulation
        self._platform.write_root_state_to_sim(platform_state, env_ids=env_ids)
        self._cube.write_root_state_to_sim(cube_state, env_ids=env_ids)

        # Apply light if present in state
        if "light_intensity" in state and "light_color_temp" in state:
            self._apply_light(state["light_intensity"], state["light_color_temp"])

        self._last_initial_state = state

        # Settle physics — match DirectRLEnv.reset() sequence
        self.scene.write_data_to_sim()
        self.sim.forward()

        if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
            for _ in range(self.cfg.num_rerenders_on_reset):
                self.sim.render()

        obs_dict = self._get_observations()
        return obs_dict, {}

    def _apply_light(self, intensity: float, color_temp: float) -> None:
        """Apply exact light parameters (for replay)."""
        import omni.usd

        if self._window_light_prim is None:
            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                if prim.GetName() == WINDOW_LIGHT_NAME and "Light" in prim.GetTypeName():
                    self._window_light_prim = prim
                    break

        if self._window_light_prim is not None:
            self._window_light_prim.GetAttribute("inputs:intensity").Set(intensity)
            self._window_light_prim.GetAttribute("inputs:colorTemperature").Set(color_temp)

    def get_initial_state(self) -> dict | None:
        """Return spawn state for metadata recording."""
        return self._last_initial_state

    def _randomize_light(self) -> dict | None:
        """Randomize window (sun) light intensity and color temperature.

        Returns:
            Dict with light values, or None if light not found.
        """
        import omni.usd

        # Find and cache light prim on first call
        if self._window_light_prim is None:
            stage = omni.usd.get_context().get_stage()
            for prim in stage.Traverse():
                if prim.GetName() == WINDOW_LIGHT_NAME and "Light" in prim.GetTypeName():
                    self._window_light_prim = prim
                    break

            if self._window_light_prim is None:
                print(f"[WARN] Light '{WINDOW_LIGHT_NAME}' not found in scene")
                return None

        # Build discrete value arrays
        intensity_values = np.arange(
            self.cfg.light_intensity_range[0],
            self.cfg.light_intensity_range[1] + 1,
            self.cfg.light_intensity_step,
        )
        color_temp_values = np.arange(
            self.cfg.light_color_temp_range[0],
            self.cfg.light_color_temp_range[1] + 1,
            self.cfg.light_color_temp_step,
        )

        # Sample random values
        intensity = float(np.random.choice(intensity_values))
        color_temp = float(np.random.choice(color_temp_values))

        # Apply to light prim
        self._window_light_prim.GetAttribute("inputs:intensity").Set(intensity)
        self._window_light_prim.GetAttribute("inputs:colorTemperature").Set(color_temp)

        print(f"[LIGHT] intensity={intensity:.0f}, colorTemp={color_temp:.0f}K")

        return {
            "light_intensity": intensity,
            "light_color_temp": color_temp,
        }
