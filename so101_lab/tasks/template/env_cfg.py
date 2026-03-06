"""Configuration for template SO-101 environment."""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, OffsetCfg, TiledCameraCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from so101_lab.assets.robots.so101 import SO101_CFG
from so101_lab.assets.scenes.workbench import WORKBENCH_CLEAN_CFG, GROUND_PLANE_CFG


@configclass
class TemplateSceneCfg(InteractiveSceneCfg):
    """Template scene configuration for SO-101 tasks."""

    # Ground plane (cloned per-env)
    ground: AssetBaseCfg = GROUND_PLANE_CFG

    # Workbench scene (table, lighting) - cloned per-env
    scene: AssetBaseCfg = WORKBENCH_CLEAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # SO-101 robot
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # End-effector frame tracking (use gripper_frame_link for tool center point)
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_frame_link",
                name="gripper",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ),
        ],
    )

    # Top-down camera (fixed in world, looking at table)
    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",  # Scene-level, not attached to robot
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.309, -0.457, 1.217),
            rot=(0.84493, 0.44012, -0.17551, -0.24817),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=23.5,  # H-FOV ~78° (matching real Logitech Brio 90)
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 5.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # Wrist camera (attached to camera_wrist link)
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_wrist/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.00112, 0, 0.022),
            rot=(0.00000, -0.00000, -0.99999, 0.00436),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=23.5,  # H-FOV ~78° (matching real OV2710 + 6205 lens)
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 5.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    # Lighting is included in workbench_v1.usd scene:
    # - CeilingLamp: SphereLight (4000K, 500000 intensity, diffuse=0.9, specular=1.2)
    # - Window: RectLight with ShapingAPI (6500K, 150000 intensity, 2x1.6m)
    # - EnvironmentLight: DomeLight (500 intensity, slightly blue)
    #
    # All lights can be randomized via USD API in event functions
    # See workbench_v1_backup.usd for reference parameters


@configclass
class TemplateActionsCfg:
    """Action configuration for teleop mode."""

    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class TemplateEnvCfg(DirectRLEnvCfg):
    """Template environment configuration."""

    # Physics configuration
    sim: SimulationCfg = SimulationCfg(
        physx=PhysxCfg(
            # Solve articulation contacts last for better gripper behavior
            # See: https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html
            solve_articulation_contact_last=True,
        )
    )

    scene: TemplateSceneCfg = TemplateSceneCfg(num_envs=1, env_spacing=10.0)

    # Action space: 6 DOF joint positions (arm only, no gripper)
    action_space = 6

    # Optional ActionsCfg for teleop mode
    actions: TemplateActionsCfg | None = None

    # State space: all available information
    state_space = {
        "joint_pos": 6,
        "joint_vel": 6,
        "actions": 6,
        "ee_frame_state": 7,  # pos (3) + quat (4)
        "joint_pos_target": 6,
        "top": [480, 640, 3],
        "wrist": [480, 640, 3],
    }

    # Observation space: subset used by policy
    observation_space = {
        "joint_pos": 6,
        "actions": 6,
        "joint_pos_target": 6,
        "top": [480, 640, 3],
        "wrist": [480, 640, 3],
    }

    action_scale = 1.0
    decimation = 1
    episode_length_s = 25.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.5)

    def use_teleop_device(self, device_type: str) -> None:
        """Configure environment for teleoperation.

        WARNING: Uses DifferentialIK which is SIMULATION-ONLY.

        Args:
            device_type: Device type ("keyboard", "gamepad")

        Modifies:
            - action_space: Set to 8 DOF for keyboard/gamepad
            - actions: Initialize ActionsCfg with DifferentialIK
            - scene.robot: Disable gravity for IK stability
        """
        from so101_lab.devices.action_process import init_action_cfg

        if device_type in ["keyboard", "gamepad"]:
            self.action_space = 8
            self.actions = TemplateActionsCfg()
            init_action_cfg(self.actions, device_type)
            self.scene.robot.spawn.rigid_props.disable_gravity = True
        elif device_type == "so101leader":
            self.action_space = 6
            self.actions = TemplateActionsCfg()
            init_action_cfg(self.actions, device_type)
            self.scene.robot.spawn.rigid_props.disable_gravity = True
        else:
            raise ValueError(f"Unsupported device type: {device_type}")

    def preprocess_device_action(self, action_dict: dict, teleop_device) -> torch.Tensor:
        """Convert device action dict to environment action tensor.

        Args:
            action_dict: Dict from device.advance() containing "joint_state"
            teleop_device: Device instance

        Returns:
            Action tensor [num_envs, action_dim]
        """
        from so101_lab.devices.action_process import preprocess_device_action

        return preprocess_device_action(action_dict, teleop_device)
