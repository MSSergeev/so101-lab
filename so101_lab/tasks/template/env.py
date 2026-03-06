"""Template environment for SO-101 manipulation tasks."""

from __future__ import annotations

import torch

from isaaclab.envs import DirectRLEnv

from .env_cfg import TemplateEnvCfg


class TemplateEnv(DirectRLEnv):
    """Template environment for SO-101 manipulation tasks.

    This is a base environment that provides:
    - Robot with 6 DOF arm control
    - Two cameras (top + wrist)
    - End-effector frame tracking
    - Workbench scene

    Concrete tasks should inherit and implement:
    - _get_rewards() - task-specific reward function
    - _get_dones() - task-specific termination conditions
    - Add objects to the scene via SceneCfg
    """

    cfg: TemplateEnvCfg

    def __init__(self, cfg: TemplateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Action buffers (initialized after parent __init__)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)

        # Action manager (only for teleop mode)
        if cfg.actions is not None:
            from isaaclab.managers import ActionManager

            self.action_manager = ActionManager(cfg.actions, self)
        else:
            self.action_manager = None

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Store previous actions and update current actions."""
        self._previous_actions[:] = self._actions[:]
        self._actions[:] = actions.clone()

    def _apply_action(self) -> None:
        """Apply joint position targets to the robot.

        In direct mode: Actions are absolute joint positions.
        In teleop mode: ActionManager processes IK and applies actions.
        """
        if self.action_manager is not None:
            # Teleop mode: use ActionManager (processes IK)
            self.action_manager.process_action(self._actions)
            self.action_manager.apply_action()
        else:
            # Direct mode: absolute joint positions
            self.scene["robot"].set_joint_position_target(self._actions)

    def _get_observations(self) -> dict:
        """Get observations from the environment.

        Returns dict with "policy" group containing:
        - joint_pos: current joint positions
        - actions: previous actions
        - joint_pos_target: target joint positions
        - top: RGB image from top camera (if available)
        - wrist: RGB image from wrist camera (if available)
        """
        obs = {
            "joint_pos": self.scene["robot"].data.joint_pos[:, :6],  # First 6 joints (arm only)
            "actions": self._actions,
            "joint_pos_target": self.scene["robot"].data.joint_pos_target[:, :6],
        }

        # Add camera observations if configured
        if "top" in self.scene.sensors:
            obs["top"] = self.scene["top"].data.output["rgb"]
        if "wrist" in self.scene.sensors:
            obs["wrist"] = self.scene["wrist"].data.output["rgb"]

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Template: no reward (override in subclasses).

        Concrete tasks should implement task-specific reward functions here.
        """
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Template: only timeout termination (override in subclasses).

        Returns:
            terminated: Task-specific success/failure conditions
            time_out: Episode timeout

        Concrete tasks should add success/failure conditions to 'terminated'.
        """
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset environments at given indices.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.

        Note:
            Alternative approach (not tested): use reset_scene_to_default from Isaac Lab
            which resets ALL scene objects (robots, rigid objects, deformables) automatically:

                from isaaclab.envs.mdp.events import reset_scene_to_default
                reset_scene_to_default(self, env_ids, reset_joint_targets=True)
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.scene["robot"]._ALL_INDICES

        # Call parent first to reset scene (including robot actuators)
        super()._reset_idx(env_ids)

        # Reset robot joint positions to default state (from init_state in ArticulationCfg)
        robot = self.scene["robot"]
        default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)

        # Also reset joint targets to default positions
        robot.set_joint_position_target(default_joint_pos, env_ids=env_ids)

        # Reset action buffers to zeros (actions are deltas in teleop mode)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

    def get_initial_state(self) -> dict | None:
        """Return initial prim values after reset for metadata recording.

        Override in subclasses to capture task-specific spawn state
        (object positions, light values, etc.).
        """
        return None
