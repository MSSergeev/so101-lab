"""Pick cube manipulation task environment."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject

from so101_lab.tasks.template.env import TemplateEnv
from .env_cfg import PickCubeEnvCfg


class PickCubeEnv(TemplateEnv):
    """Pick cube manipulation task.

    Goal: Move end-effector to cube and lift it above a threshold height.

    Rewards:
    - Distance from gripper to cube (negative, encourages reaching)
    - Cube height above table (positive when lifted)
    - Action penalty (small, encourages smooth motions)

    Termination:
    - Success: cube lifted above 0.15m
    - Failure: cube falls off table (|x| > 0.4 or |y| > 0.3)
    - Timeout: episode_length exceeds maximum
    """

    cfg: PickCubeEnvCfg

    def __init__(self, cfg: PickCubeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Cache initial cube height for reward calculation
        self._cube_init_height = 0.76

    def _setup_scene(self):
        """Register cube in the scene."""
        super()._setup_scene()

        # Add cube as RigidObject
        self._cube = RigidObject(self.cfg.scene.cube)
        self.scene.rigid_objects["cube"] = self._cube

    def _get_observations(self) -> dict:
        """Get observations (can optionally add cube position)."""
        obs_dict = super()._get_observations()

        # Optionally add cube position to observations
        # cube_pos = self.scene["cube"].data.root_pos_w  # (num_envs, 3)
        # obs_dict["policy"]["cube_pos"] = cube_pos

        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        """Compute task-specific reward.

        Reward components:
        1. Distance reward: negative distance from gripper to cube (encourages reaching)
        2. Lift reward: positive reward for lifting cube above table (encourages lifting)
        3. Action penalty: small penalty for large actions (encourages smooth motions)
        """
        # Get gripper position from end-effector frame
        gripper_pos = self.scene["ee_frame"].data.target_pos_w[:, 0, :]  # (num_envs, 3)

        # Get cube position
        cube_pos = self.scene["cube"].data.root_pos_w  # (num_envs, 3)

        # 1. Distance reward: negative distance to encourage reaching
        distance = torch.norm(gripper_pos - cube_pos, dim=-1)
        distance_reward = -distance * 2.0

        # 2. Lift reward: reward for lifting cube above initial height
        cube_height = cube_pos[:, 2] - self._cube_init_height
        lift_reward = torch.clamp(cube_height, 0.0, 0.2) * 10.0

        # 3. Action penalty: small penalty for large actions
        action_penalty = torch.sum(self._actions ** 2, dim=-1) * 0.001

        return distance_reward + lift_reward - action_penalty

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check success/failure conditions.

        Returns:
            terminated: Success (cube lifted) or failure (cube fell off table)
            time_out: Episode timeout
        """
        cube_pos = self.scene["cube"].data.root_pos_w

        # Success: cube lifted above 15cm from table
        success = (cube_pos[:, 2] - self._cube_init_height) > 0.15

        # Failure: cube fell off table (table is ~40×30cm)
        out_of_bounds = (torch.abs(cube_pos[:, 0]) > 0.4) | (torch.abs(cube_pos[:, 1]) > 0.3)

        # Terminated = success or failure
        terminated = torch.logical_or(success, out_of_bounds)

        # Timeout from parent class
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset robot and cube positions.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.
        """
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._cube._ALL_INDICES

        # Reset cube to default position (can add randomization later)
        # For now, use default state from config
        self._cube.write_root_state_to_sim(
            self._cube.data.default_root_state[env_ids].clone(),
            env_ids=env_ids
        )
