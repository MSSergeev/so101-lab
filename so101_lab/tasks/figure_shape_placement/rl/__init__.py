"""RL environment for figure shape placement task."""

import gymnasium as gym

gym.register(
    id="SO101-FigureShapePlacement-RL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FigureShapePlacementRLEnvCfg",
    },
)
