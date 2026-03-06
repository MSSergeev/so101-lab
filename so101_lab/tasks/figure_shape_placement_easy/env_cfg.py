"""Configuration for figure shape placement (easy) task.

Same scene and objects as figure_shape_placement, but with simplified spawn:
- Cube spawns in a small rectangular zone (±2cm around center)
- Discrete cube yaw: 0-80° with 10° step
- Platform closer to robot, fixed yaw 90°

Reuses TABLE_HEIGHT, SUCCESS_THRESHOLD, ORIENTATION_THRESHOLD, SLOT_OFFSET,
and all object geometry from the original task.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from so101_lab.tasks.figure_shape_placement.env_cfg import (
    CUBE_PLATFORM_GAP_REDUCTION,
    CUBE_RADIUS,
    CUBE_SIZE,
    DISTRACTOR_RADIUS,
    DISTRACTOR_USD_PATHS,
    MAX_REACH,
    MAX_SPAWN_ATTEMPTS,
    MIN_REACH,
    ORIENTATION_THRESHOLD,
    PLATFORM_RADIUS,
    PLATFORM_SIZE,
    ROBOT_X,
    ROBOT_Y,
    SLOT_OFFSET,
    SUCCESS_THRESHOLD,
    TABLE_HEIGHT,
    TABLE_X_MAX,
    TABLE_X_MIN,
    TABLE_Y_MAX,
    TABLE_Y_MIN,
    FigureShapePlacementEnvCfg,
    FigureShapePlacementSceneCfg,
)

# --- Easy spawn constants ---

# Cube rectangular spawn zone (world coordinates)
# Center: (0.334, 0.098) — 8cm left (+X) and 18cm forward (-Y) from robot pivot
CUBE_SPAWN_X_MIN = 0.314
CUBE_SPAWN_X_MAX = 0.354
CUBE_SPAWN_Y_MIN = 0.058
CUBE_SPAWN_Y_MAX = 0.098

# Discrete cube yaw values (degrees) — 0° to 80° with 10° step
CUBE_YAW_VALUES_DEG = tuple(float(d) for d in range(0, 81, 10))

# Platform: 1cm right (-X) from previous (0.164, 0.048)
PLATFORM_FIXED_X = 0.154
PLATFORM_FIXED_Y = 0.048

# Single platform yaw
PLATFORM_YAW_VALUES_DEG = (90.0,)


# Re-export constants used by other modules (rewards, terminations, observations)
__all__ = [
    "CUBE_SPAWN_X_MIN", "CUBE_SPAWN_X_MAX",
    "CUBE_SPAWN_Y_MIN", "CUBE_SPAWN_Y_MAX",
    "CUBE_YAW_VALUES_DEG",
    "PLATFORM_FIXED_X", "PLATFORM_FIXED_Y", "PLATFORM_YAW_VALUES_DEG",
    # From original (re-exported)
    "TABLE_HEIGHT", "TABLE_X_MIN", "TABLE_X_MAX", "TABLE_Y_MIN", "TABLE_Y_MAX",
    "ROBOT_X", "ROBOT_Y", "MIN_REACH", "MAX_REACH",
    "PLATFORM_RADIUS", "PLATFORM_SIZE", "CUBE_RADIUS", "CUBE_SIZE",
    "MAX_SPAWN_ATTEMPTS", "CUBE_PLATFORM_GAP_REDUCTION",
    "SLOT_OFFSET", "SUCCESS_THRESHOLD", "ORIENTATION_THRESHOLD",
    "DISTRACTOR_RADIUS", "DISTRACTOR_USD_PATHS",
]


@configclass
class FigureShapePlacementEasySceneCfg(FigureShapePlacementSceneCfg):
    """Scene is identical to original — same objects, cameras, etc."""
    pass


@configclass
class FigureShapePlacementEasyEnvCfg(FigureShapePlacementEnvCfg):
    """Env config inherits from original — only scene class differs."""

    scene: FigureShapePlacementEasySceneCfg = FigureShapePlacementEasySceneCfg(
        num_envs=1, env_spacing=10.0
    )
