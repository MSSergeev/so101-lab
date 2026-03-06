"""Full game demo environment — all figures on the platform for teleop."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject

from so101_lab.tasks.template.env import TemplateEnv
from .env_cfg import FullGameDemoEnvCfg, TABLE_HEIGHT, FIGURE_DEFS, PLATFORM_X, PLATFORM_Y


class FullGameDemoEnv(TemplateEnv):
    """Demo scene with all 9 shape sorting figures on the platform.

    No task goal — purely for teleop recording demos.
    All figures spawn at their slots on the platform (fixed positions).
    """

    cfg: FullGameDemoEnvCfg

    def __init__(self, cfg: FullGameDemoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene(self):
        """Register platform and all figures in the scene."""
        super()._setup_scene()

        self._platform = RigidObject(self.cfg.scene.platform)
        self.scene.rigid_objects["platform"] = self._platform

        # Register each figure as a separate rigid object
        self._figures: dict[str, RigidObject] = {}
        figure_attr_names = [
            "deep_green_star", "deep_blue_hexagon", "red_clover",
            "orange_x", "yellow_parallelepiped", "light_green_triangle_prism",
            "yellow_cylinder", "pink_diamond", "light_blue_cube",
        ]
        for attr_name in figure_attr_names:
            cfg = getattr(self.cfg.scene, attr_name)
            obj = RigidObject(cfg)
            self._figures[attr_name] = obj
            self.scene.rigid_objects[attr_name] = obj

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._platform._ALL_INDICES

        n = len(env_ids)
        env_origins = self.scene.env_origins[env_ids]

        # Reset platform to fixed position
        platform_state = self._platform.data.default_root_state[env_ids].clone()
        platform_state[:, 0] = PLATFORM_X + env_origins[:, 0]
        platform_state[:, 1] = PLATFORM_Y + env_origins[:, 1]
        platform_state[:, 2] = TABLE_HEIGHT + env_origins[:, 2]
        self._platform.write_root_state_to_sim(platform_state, env_ids=env_ids)

        # Reset each figure to its slot on the platform
        figure_attr_names = [
            "deep_green_star", "deep_blue_hexagon", "red_clover",
            "orange_x", "yellow_parallelepiped", "light_green_triangle_prism",
            "yellow_cylinder", "pink_diamond", "light_blue_cube",
        ]
        for attr_name, (_, _, offset_x, offset_y) in zip(figure_attr_names, FIGURE_DEFS):
            obj = self._figures[attr_name]
            state = obj.data.default_root_state[env_ids].clone()
            state[:, 0] = PLATFORM_X + offset_x + env_origins[:, 0]
            state[:, 1] = PLATFORM_Y + offset_y + env_origins[:, 1]
            state[:, 2] = TABLE_HEIGHT + 0.006 + env_origins[:, 2]
            obj.write_root_state_to_sim(state, env_ids=env_ids)
