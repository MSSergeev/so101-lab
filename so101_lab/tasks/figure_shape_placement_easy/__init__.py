"""Figure shape placement (easy) task.

Same task as figure_shape_placement but with simplified spawn:
- Cube in small rectangular zone (±2cm) instead of polar quarter-circle
- Discrete cube yaw (0-80° step 10°) instead of full rotation
- Platform closer to robot, single yaw
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "FigureShapePlacementEasyEnv":
        from .env import FigureShapePlacementEasyEnv
        return FigureShapePlacementEasyEnv
    if name == "FigureShapePlacementEasyEnvCfg":
        from .env_cfg import FigureShapePlacementEasyEnvCfg
        return FigureShapePlacementEasyEnvCfg
    if name == "FigureShapePlacementEasySceneCfg":
        from .env_cfg import FigureShapePlacementEasySceneCfg
        return FigureShapePlacementEasySceneCfg
    if name == "rl":
        from . import rl
        return rl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FigureShapePlacementEasyEnv",
    "FigureShapePlacementEasyEnvCfg",
    "FigureShapePlacementEasySceneCfg",
]
