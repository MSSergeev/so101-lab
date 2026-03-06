# Adapted from: leisaac (https://github.com/huggingface/LeIsaac)
# Original license: Apache-2.0
# Changes: ManagerBasedRLEnvCfg for SO-101 figure shape placement task

"""RL environment configuration for figure shape placement task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from so101_lab.tasks.figure_shape_placement.env_cfg import (
    CUBE_RADIUS,
    CUBE_SPAWN_X_MAX,
    CUBE_SPAWN_X_MIN,
    CUBE_SPAWN_Y_MAX,
    CUBE_SPAWN_Y_MIN,
    CUBE_YAW_VALUES_DEG,
    DISTRACTOR_RADIUS,
    DISTRACTOR_USD_PATHS,
    MAX_REACH,
    MIN_REACH,
    ORIENTATION_THRESHOLD,
    PLATFORM_FIXED_X,
    PLATFORM_FIXED_Y,
    PLATFORM_RADIUS,
    PLATFORM_YAW_VALUES_DEG,
    ROBOT_X,
    ROBOT_Y,
    SUCCESS_THRESHOLD,
    TABLE_HEIGHT,
    TABLE_X_MAX,
    TABLE_X_MIN,
    TABLE_Y_MAX,
    TABLE_Y_MIN,
    FigureShapePlacementSceneCfg,
)

from . import mdp
from .mdp import events, terminations


@configclass
class FigureShapePlacementRLSceneCfg(FigureShapePlacementSceneCfg):
    """Scene for RL — inherits all objects from IL scene + contact sensors."""

    gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Scene/Table"],
        update_period=0.0,
    )
    jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Scene/Table"],
        update_period=0.0,
    )

    distractor_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DISTRACTOR_USD_PATHS[0]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -1)),
    )
    distractor_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DISTRACTOR_USD_PATHS[1]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -1)),
    )
    distractor_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Distractor2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DISTRACTOR_USD_PATHS[2]),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, -1)),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        images_top = ObsTerm(
            func=mdp.image_with_noise,
            params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False},
        )
        images_wrist = ObsTerm(
            func=mdp.image_with_noise,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action configuration — joint position targets for all 6 DOF."""

    joint_pos: mdp.JointPositionActionCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                     "wrist_flex", "wrist_roll", "gripper"],
        scale=1.0,
        use_default_offset=False,
    )


@configclass
class EventsCfg:
    """Event configuration — reset and domain randomization."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_platform = EventTerm(
        func=events.reset_platform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "fixed_x": PLATFORM_FIXED_X,
            "fixed_y": PLATFORM_FIXED_Y,
            "yaw_values_deg": PLATFORM_YAW_VALUES_DEG,
            "table_h": TABLE_HEIGHT,
        },
    )

    reset_cube = EventTerm(
        func=events.reset_cube_rect,
        mode="reset",
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "x_min": CUBE_SPAWN_X_MIN,
            "x_max": CUBE_SPAWN_X_MAX,
            "y_min": CUBE_SPAWN_Y_MIN,
            "y_max": CUBE_SPAWN_Y_MAX,
            "yaw_values_deg": CUBE_YAW_VALUES_DEG,
            "table_h": TABLE_HEIGHT,
        },
    )

    randomize_light = EventTerm(
        func=events.randomize_light,
        mode="reset",
        params={
            "intensity_range": (0.0, 200000.0),
            "intensity_step": 10000.0,
            "color_temp_range": (3500.0, 6500.0),
            "color_temp_step": 500.0,
            "window_pos_offset": 0.5,
            "window_rot_offset_deg": 15.0,
            "ceiling_intensity_range": (100000.0, 800000.0),
            "dome_intensity_range": (200.0, 1000.0),
        },
    )

    randomize_cube_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 16,
        },
    )
    randomize_platform_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.3),
            "num_buckets": 16,
        },
    )
    randomize_cube_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cube"),
            "mass_distribution_params": (0.8, 1.5),
            "operation": "scale",
        },
    )

    reset_distractors = EventTerm(
        func=events.reset_distractors,
        mode="reset",
        params={
            "distractor_cfgs": [
                SceneEntityCfg("distractor_0"),
                SceneEntityCfg("distractor_1"),
                SceneEntityCfg("distractor_2"),
            ],
            "cube_cfg": SceneEntityCfg("cube"),
            "platform_cfg": SceneEntityCfg("platform"),
            "robot_x": ROBOT_X,
            "robot_y": ROBOT_Y,
            "min_reach": MIN_REACH,
            "max_reach": MAX_REACH,
            "distractor_radius": DISTRACTOR_RADIUS,
            "cube_radius": CUBE_RADIUS,
            "platform_radius": PLATFORM_RADIUS,
            "table_h": TABLE_HEIGHT,
            "table_bounds": (TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX),
        },
    )


@configclass
class RewardsCfg:
    """Reward configuration — sim-based dense rewards."""

    time_out_penalty = RewTerm(func=mdp.is_terminated, weight=-1.0)
    drop_penalty = RewTerm(
        func=mdp.drop_penalty, weight=-10.0,
        params={"cube_cfg": SceneEntityCfg("cube"), "z_threshold": 0.05},
    )
    jerky_motion_penalty = RewTerm(
        func=mdp.jerky_motion_penalty, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "vel_threshold": 5.0},
    )
    action_smoothness_penalty = RewTerm(
        func=mdp.action_smoothness_penalty, weight=-0.1,
    )
    time_penalty = RewTerm(
        func=mdp.time_penalty, weight=-0.05,
    )
    distance_cube_to_slot = RewTerm(
        func=mdp.distance_cube_to_slot, weight=0.5,
        params={"cube_cfg": SceneEntityCfg("cube"), "platform_cfg": SceneEntityCfg("platform")},
    )
    distance_gripper_to_cube = RewTerm(
        func=mdp.distance_gripper_to_cube, weight=0.5,
        params={"ee_cfg": SceneEntityCfg("ee_frame"), "cube_cfg": SceneEntityCfg("cube")},
    )
    milestone_picked = RewTerm(
        func=mdp.MilestonePicked, weight=5.0,
        params={"cube_cfg": SceneEntityCfg("cube"), "lift_height": TABLE_HEIGHT + 0.03},
    )
    milestone_placed = RewTerm(
        func=mdp.MilestonePlaced, weight=5.0,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "platform_cfg": SceneEntityCfg("platform"),
            "threshold": SUCCESS_THRESHOLD,
            "z_threshold": 0.004,
            "orientation_threshold": ORIENTATION_THRESHOLD,
        },
    )
    table_contact_penalty = RewTerm(
        func=mdp.table_contact_penalty, weight=-0.1,
        params={
            "gripper_sensor_cfg": SceneEntityCfg("gripper_contact"),
            "jaw_sensor_cfg": SceneEntityCfg("jaw_contact"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination configuration."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_out_of_bounds = DoneTerm(
        func=terminations.cube_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("cube"), "z_threshold": 0.05},
    )
    platform_out_of_bounds = DoneTerm(
        func=terminations.platform_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("platform"), "z_threshold": 0.05},
    )


@configclass
class FigureShapePlacementRLEnvCfg(ManagerBasedRLEnvCfg):
    """ManagerBasedRLEnvCfg for figure shape placement task."""

    scene: FigureShapePlacementRLSceneCfg = FigureShapePlacementRLSceneCfg(
        num_envs=1, env_spacing=10.0
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    sim: SimulationCfg = SimulationCfg(
        physx=PhysxCfg(solve_articulation_contact_last=True)
    )

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (1.4, -0.9, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.5)
