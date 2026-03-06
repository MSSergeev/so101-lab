"""Configuration for pick cube task."""

from __future__ import annotations

from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from so101_lab.tasks.template.env_cfg import TemplateEnvCfg, TemplateSceneCfg

# Path to cube USD file
CUBE_USD_PATH = Path(__file__).resolve().parents[3] / "assets" / "props" / "light_blue_cube.usd"


@configclass
class PickCubeSceneCfg(TemplateSceneCfg):
    """Scene configuration for pick cube task."""

    # Add cube programmatically via RigidObjectCfg
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CUBE_USD_PATH),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.1, 0.76),  # On table in front of robot (z = 0.75 table + 0.01)
            rot=(1.0, 0.0, 0.0, 0.0),  # No rotation (quaternion: w, x, y, z)
        ),
    )


@configclass
class PickCubeEnvCfg(TemplateEnvCfg):
    """Environment configuration for pick cube task."""

    scene: PickCubeSceneCfg = PickCubeSceneCfg(num_envs=1, env_spacing=2.5)
