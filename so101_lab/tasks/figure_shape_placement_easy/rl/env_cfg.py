"""RL environment configuration for figure shape placement (easy) task.

Inherits everything from original FigureShapePlacementRLEnvCfg,
overrides only EventsCfg to use rectangular cube spawn.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from so101_lab.tasks.figure_shape_placement.env_cfg import (
    CUBE_PLATFORM_GAP_REDUCTION,
    CUBE_RADIUS,
    DISTRACTOR_RADIUS,
    MAX_REACH,
    MAX_SPAWN_ATTEMPTS,
    MIN_REACH,
    PLATFORM_RADIUS,
    ROBOT_X,
    ROBOT_Y,
    TABLE_HEIGHT,
    TABLE_X_MAX,
    TABLE_X_MIN,
    TABLE_Y_MAX,
    TABLE_Y_MIN,
)
from so101_lab.tasks.figure_shape_placement.rl.env_cfg import (
    FigureShapePlacementRLEnvCfg,
    FigureShapePlacementRLSceneCfg,
)
from so101_lab.tasks.figure_shape_placement.rl.mdp import events as orig_events

from so101_lab.tasks.figure_shape_placement_easy.env_cfg import (
    CUBE_SPAWN_X_MAX,
    CUBE_SPAWN_X_MIN,
    CUBE_SPAWN_Y_MAX,
    CUBE_SPAWN_Y_MIN,
    CUBE_YAW_VALUES_DEG,
    PLATFORM_FIXED_X,
    PLATFORM_FIXED_Y,
    PLATFORM_YAW_VALUES_DEG,
)

from . import mdp
from .mdp import events


@configclass
class EventsCfg:
    """Events — same as original but with rectangular cube spawn."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_platform = EventTerm(
        func=orig_events.reset_platform,
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
            "platform_cfg": SceneEntityCfg("platform"),
            "x_min": CUBE_SPAWN_X_MIN,
            "x_max": CUBE_SPAWN_X_MAX,
            "y_min": CUBE_SPAWN_Y_MIN,
            "y_max": CUBE_SPAWN_Y_MAX,
            "cube_radius": CUBE_RADIUS,
            "platform_radius": PLATFORM_RADIUS,
            "gap_reduction": CUBE_PLATFORM_GAP_REDUCTION,
            "yaw_values_deg": CUBE_YAW_VALUES_DEG,
            "table_h": TABLE_HEIGHT,
            "max_attempts": MAX_SPAWN_ATTEMPTS,
        },
    )

    randomize_light = EventTerm(
        func=orig_events.randomize_light,
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
        func=orig_events.reset_distractors,
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
class FigureShapePlacementEasyRLEnvCfg(FigureShapePlacementRLEnvCfg):
    """RL env cfg — inherits scene/obs/actions/rewards/terminations, overrides events."""

    events: EventsCfg = EventsCfg()
