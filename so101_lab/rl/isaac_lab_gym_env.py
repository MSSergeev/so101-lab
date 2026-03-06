"""Gymnasium wrapper around Isaac Lab ManagerBasedRLEnv for LeRobot SAC compatibility."""

from __future__ import annotations

import gymnasium
import numpy as np
import torch

from so101_lab.data.converters import joint_rad_to_motor_normalized, motor_normalized_to_joint_rad


class IsaacLabGymEnv(gymnasium.Env):
    """Wraps ManagerBasedRLEnv into a standard gymnasium.Env.

    Converts between Isaac Lab radians and LeRobot normalized motor space.
    Single-env only (num_envs=1).

    AppLauncher must be initialized before creating this env.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, env_cfg, randomize_light: bool = True):
        super().__init__()

        from isaaclab.envs import ManagerBasedRLEnv

        # Optionally disable light randomization (for consistency with classifier training data)
        if not randomize_light and hasattr(env_cfg.events, "randomize_light"):
            env_cfg.events.randomize_light = None

        self._env = ManagerBasedRLEnv(cfg=env_cfg)

        self.observation_space = gymnasium.spaces.Dict({
            "observation.state": gymnasium.spaces.Box(
                low=-100.0, high=100.0, shape=(6,), dtype=np.float32,
            ),
            "observation.images.top": gymnasium.spaces.Box(
                low=0, high=255, shape=(480, 640, 3), dtype=np.uint8,
            ),
            "observation.images.wrist": gymnasium.spaces.Box(
                low=0, high=255, shape=(480, 640, 3), dtype=np.uint8,
            ),
        })

        self.action_space = gymnasium.spaces.Box(
            low=-100.0, high=100.0, shape=(6,), dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        # normalized → radians
        action_rad = motor_normalized_to_joint_rad(action)
        # Add batch dim, convert to torch
        action_tensor = torch.from_numpy(action_rad).unsqueeze(0).float()

        obs_dict, reward, terminated, truncated, info = self._env.step(action_tensor)
        obs = self._convert_obs(obs_dict)

        reward = float(reward[0])
        terminated = bool(terminated)
        truncated = bool(truncated)

        info["ground_truth"] = self._extract_ground_truth()

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        obs_dict, info = self._env.reset(seed=seed, options=options)
        obs = self._convert_obs(obs_dict)
        info["ground_truth"] = self._extract_ground_truth()
        return obs, info

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()

    @property
    def step_dt(self) -> float:
        return self._env.step_dt

    def get_reward_details(self) -> dict[str, float]:
        """Per-term reward breakdown from RewardManager."""
        rm = self._env.reward_manager
        details = {}
        for i, name in enumerate(rm._term_names):
            details[f"reward/{name}_weighted"] = float(rm._step_reward[0, i])
        details["reward/sim_total"] = float(self._env.reward_buf[0])
        return details

    def _extract_ground_truth(self) -> dict:
        """Extract simulation ground truth for reward computation.

        Returns dict with cube_pos, cube_quat, slot_pos, gripper_pos, joint_vel.
        Returns {} for tasks without these scene entities.
        """
        try:
            scene = self._env.scene
            origins = scene.env_origins

            # Cube
            cube = scene["cube"]
            cube_pos = (cube.data.root_pos_w - origins)[0].cpu().numpy()
            cube_quat = cube.data.root_quat_w[0].cpu().numpy()

            # Slot from platform
            platform = scene["platform"]
            plat_pos = (platform.data.root_pos_w - origins)[0]
            plat_quat = platform.data.root_quat_w[0]
            # yaw from quaternion (w, x, y, z)
            w, x, y, z = plat_quat[0], plat_quat[1], plat_quat[2], plat_quat[3]
            yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            ox, oy = -0.0667, 0.0667  # SLOT_OFFSET
            slot_x = plat_pos[0] + ox * torch.cos(yaw) - oy * torch.sin(yaw)
            slot_y = plat_pos[1] + ox * torch.sin(yaw) + oy * torch.cos(yaw)
            slot_z = plat_pos[2]
            slot_pos = torch.stack([slot_x, slot_y, slot_z]).cpu().numpy()

            # Gripper (ee_frame)
            ee = scene["ee_frame"]
            gripper_pos = (ee.data.target_pos_w[:, 0, :] - origins)[0].cpu().numpy()

            # Joint velocities
            robot = scene["robot"]
            joint_vel = robot.data.joint_vel[0].cpu().numpy()

            return {
                "cube_pos": cube_pos,
                "cube_quat": cube_quat,
                "slot_pos": slot_pos,
                "gripper_pos": gripper_pos,
                "joint_vel": joint_vel,
            }
        except (KeyError, AttributeError):
            return {}

    def _convert_obs(self, obs_dict: dict) -> dict:
        """Convert Isaac Lab obs to LeRobot normalized format."""
        policy = obs_dict["policy"]

        # joint_pos: (1, 6) tensor → (6,) numpy normalized
        joint_pos = policy["joint_pos"]
        if isinstance(joint_pos, torch.Tensor):
            joint_pos = joint_pos.cpu().numpy()
        joint_pos = joint_pos.squeeze(0)  # remove batch dim
        state = joint_rad_to_motor_normalized(joint_pos)

        # images: (1, H, W, 3) tensor → (H, W, 3) numpy uint8
        img_top = policy["images_top"]
        if isinstance(img_top, torch.Tensor):
            img_top = img_top.cpu().numpy()
        img_top = img_top.squeeze(0).astype(np.uint8)

        img_wrist = policy["images_wrist"]
        if isinstance(img_wrist, torch.Tensor):
            img_wrist = img_wrist.cpu().numpy()
        img_wrist = img_wrist.squeeze(0).astype(np.uint8)

        return {
            "observation.state": state,
            "observation.images.top": img_top,
            "observation.images.wrist": img_wrist,
        }
