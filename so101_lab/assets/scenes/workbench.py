"""Workbench scene configuration (table + lighting).

Uses workbench_table_only.usd which excludes PhysicsScene and GroundPlane
(both are added by Isaac Lab globally) — required for multi-env support.
"""

from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

SCENES_ROOT = Path(__file__).resolve().parents[3] / "assets" / "scenes"

WORKBENCH_TABLE_USD_PATH = str(SCENES_ROOT / "workbench_table_only.usd")

WORKBENCH_CLEAN_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=WORKBENCH_TABLE_USD_PATH)
)

# Ground plane spawned per-env
GROUND_PLANE_CFG = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/GroundPlane",
    spawn=sim_utils.CuboidCfg(
        size=(3.0, 3.0, 0.01),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.20, 0.12, 0.07),
            roughness=0.95,
            metallic=0.0,
        ),
    ),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.005)),  # top surface at z=0
)
