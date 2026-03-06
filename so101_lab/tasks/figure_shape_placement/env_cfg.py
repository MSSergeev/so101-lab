"""Configuration for figure shape placement task.

Spawn Randomization
===================

Cube spawns uniformly in a rectangular zone (world coordinates).
Platform is fixed position and orientation (same as easy task).
Cube yaw is discrete: 0-80° in 10° steps.

Spawn zone is ~2x easy task in each dimension (4x area):
- Easy:   X∈[0.314, 0.354], Y∈[0.058, 0.098]  (4×4 cm)
- Medium: X∈[0.294, 0.374], Y∈[0.038, 0.118]  (8×8 cm)
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from so101_lab.tasks.template.env_cfg import TemplateEnvCfg, TemplateSceneCfg

# Asset paths
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "props"
PLATFORM_USD_PATH = ASSETS_DIR / "platfrom_full.usd"
CUBE_USD_PATH = ASSETS_DIR / "light_blue_cube.usd"

# Table geometry
TABLE_HEIGHT = 0.75
TABLE_X_MIN, TABLE_X_MAX = -0.45, 0.45
TABLE_Y_MIN, TABLE_Y_MAX = -0.30, 0.30

# Robot position
ROBOT_X, ROBOT_Y = 0.254, 0.278

# Reachability from robot pivot to grasp point
MIN_REACH = 0.105  # 10.5cm - closer hits robot base (+2cm from original 8.5cm)
MAX_REACH = 0.40   # 40cm - further is unreachable

# Object radii (half diagonal for worst-case rotation)
PLATFORM_SIZE = 0.20  # 20cm side
CUBE_SIZE = 0.04      # 4cm side
PLATFORM_RADIUS = PLATFORM_SIZE * math.sqrt(2) / 2  # ~0.142m
CUBE_RADIUS = CUBE_SIZE * math.sqrt(2) / 2          # ~0.028m

# Cube rectangular spawn zone (world coordinates)
CUBE_SPAWN_X_MIN = 0.294
CUBE_SPAWN_X_MAX = 0.374
CUBE_SPAWN_Y_MIN = 0.038
CUBE_SPAWN_Y_MAX = 0.118

# Discrete cube yaw values (degrees) — 0° to 80° with 10° step
CUBE_YAW_VALUES_DEG = tuple(float(d) for d in range(0, 81, 10))

# Platform spawn — fixed position and yaw (same as easy task)
PLATFORM_FIXED_X = 0.154
PLATFORM_FIXED_Y = 0.048
PLATFORM_YAW_VALUES_DEG = (90.0,)

# Task geometry
SLOT_OFFSET = (-0.0667, 0.0667)  # slot position relative to platform center
SUCCESS_THRESHOLD = 0.090  # 12cm XY distance to slot
ORIENTATION_THRESHOLD = 30.0  # ±30° (with 90° symmetry)

# Spawn attempt limit (used by easy task polar spawn)
MAX_SPAWN_ATTEMPTS = 100

# Gap reduction for cube-platform proximity check
CUBE_PLATFORM_GAP_REDUCTION = 0.01  # 1cm reduction in combined radii

# Distractor objects
DISTRACTOR_USD_PATHS = [
    ASSETS_DIR / "yellow_cylinder.usd",
    ASSETS_DIR / "deep_blue_hexagon.usd",
    ASSETS_DIR / "pink_diamond.usd",
]
DISTRACTOR_RADIUS = 0.03  # approximate bounding radius


@configclass
class FigureShapePlacementSceneCfg(TemplateSceneCfg):
    """Scene configuration for figure shape placement task."""

    # Override top camera position/orientation for this task
    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, -0.075, 1.725),
            rot=(0.99966, 0.02618, 0.0, 0.0),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=41.6,  # H-FOV (like in real)
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 5.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,
    )

    platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PLATFORM_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.05, 0.12, TABLE_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CUBE_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.10, 0.05, TABLE_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class FigureShapePlacementEnvCfg(TemplateEnvCfg):
    """Environment configuration for figure shape placement task."""

    scene: FigureShapePlacementSceneCfg = FigureShapePlacementSceneCfg(
        num_envs=1, env_spacing=10.0
    )

    # Terminate episode on success (for eval; disable for teleop/recording)
    terminate_on_success: bool = False

    # Light randomization (Window/sun light)
    randomize_light: bool = False
    light_intensity_range: tuple[float, float] = (0.0, 200000.0)
    light_intensity_step: float = 10000.0
    light_color_temp_range: tuple[float, float] = (3500.0, 6500.0)
    light_color_temp_step: float = 500.0
