"""Figure shape placement task.

Lazy imports: env.py requires omni.physics (Isaac Sim runtime),
so we defer imports until actual attribute access.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "FigureShapePlacementEnv":
        from .env import FigureShapePlacementEnv
        return FigureShapePlacementEnv
    if name == "FigureShapePlacementEnvCfg":
        from .env_cfg import FigureShapePlacementEnvCfg
        return FigureShapePlacementEnvCfg
    if name == "FigureShapePlacementSceneCfg":
        from .env_cfg import FigureShapePlacementSceneCfg
        return FigureShapePlacementSceneCfg
    if name == "rl":
        from . import rl
        return rl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FigureShapePlacementEnv", "FigureShapePlacementEnvCfg", "FigureShapePlacementSceneCfg"]
