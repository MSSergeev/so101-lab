"""Configuration for full game demo scene.

All 9 shape sorting figures spawned on the platform simultaneously.
Intended for teleop recording demos — no task goal or success criteria.

Figure layout from full_game.usd (3x3 grid, 6.67cm spacing):

    Y+  light_blue_cube    pink_diamond       yellow_cylinder
     |  red_clover         deep_green_star    deep_blue_hexagon
     |  lg_triangle_prism  yellow_parallel.   orange_x
     +---------> X+

Platform center at world (0, 0) in USD, figures offset by ±0.0667m.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from so101_lab.tasks.template.env_cfg import TemplateEnvCfg, TemplateSceneCfg

# Asset paths
ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "props"
PLATFORM_USD_PATH = ASSETS_DIR / "platfrom_full.usd"

# Table geometry (shared with other tasks)
TABLE_HEIGHT = 0.75

# Platform position (same as figure_shape_placement)
PLATFORM_X = 0.154
PLATFORM_Y = 0.048

# Figure positions relative to platform center (from full_game.usd)
# Each figure spawns at platform_pos + offset, Z = TABLE_HEIGHT + 0.006
FIGURE_DEFS: list[tuple[str, str, float, float]] = [
    # (prim_name, usd_filename, offset_x, offset_y)
    ("DeepGreenStar", "deep_green_star.usd", 0.0, 0.0),
    ("DeepBlueHexagon", "deep_blue_hexagon.usd", 0.0667, 0.0),
    ("RedClover", "red_clover.usd", -0.0667, 0.0),
    ("OrangeX", "orange_x.usd", 0.0667, -0.0667),
    ("YellowParallelepiped", "yellow_parallelepiped.usd", 0.0, -0.0667),
    ("LightGreenTrianglePrism", "light_green_triangle_prism.usd", -0.0609, -0.0667),
    ("YellowCylinder", "yellow_cylinder.usd", 0.0667, 0.0667),
    ("PinkDiamond", "pink_diamond.usd", 0.0, 0.0667),
    ("LightBlueCube", "light_blue_cube.usd", -0.0667, 0.0667),
]

# Shared rigid body properties for figures (matching full_game.usd: 32 pos / 8 vel)
_FIGURE_RIGID_PROPS = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=1.0,
    solver_position_iteration_count=32,
    solver_velocity_iteration_count=8,
)


def _make_figure_cfg(prim_name: str, usd_filename: str, offset_x: float, offset_y: float) -> RigidObjectCfg:
    """Create RigidObjectCfg for a single figure at its platform slot."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS_DIR / usd_filename),
            rigid_props=_FIGURE_RIGID_PROPS,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PLATFORM_X + offset_x, PLATFORM_Y + offset_y, TABLE_HEIGHT + 0.006),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


@configclass
class FullGameDemoSceneCfg(TemplateSceneCfg):
    """Scene with platform and all 9 shape sorting figures."""

    platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(PLATFORM_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(PLATFORM_X, PLATFORM_Y, TABLE_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Each figure as a separate RigidObjectCfg
    deep_green_star: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[0])
    deep_blue_hexagon: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[1])
    red_clover: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[2])
    orange_x: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[3])
    yellow_parallelepiped: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[4])
    light_green_triangle_prism: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[5])
    yellow_cylinder: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[6])
    pink_diamond: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[7])
    light_blue_cube: RigidObjectCfg = _make_figure_cfg(*FIGURE_DEFS[8])


@configclass
class FullGameDemoEnvCfg(TemplateEnvCfg):
    """Environment configuration for full game demo."""

    # Match full_game.usd physics: 120 Hz, higher solver iterations
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        physx=PhysxCfg(
            solve_articulation_contact_last=True,
            enable_external_forces_every_iteration=True,
        )
    )
    decimation = 2  # 120 Hz physics / 2 = 60 Hz control (same as other tasks)

    scene: FullGameDemoSceneCfg = FullGameDemoSceneCfg(num_envs=1, env_spacing=10.0)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.render_interval = self.decimation
