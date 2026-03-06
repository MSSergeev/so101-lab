"""Task environments for SO-101 robot."""


def get_task_registry():
    """Lazy load task registry to avoid circular imports."""
    from so101_lab.tasks.template.env import TemplateEnv
    from so101_lab.tasks.template.env_cfg import TemplateEnvCfg
    from so101_lab.tasks.pick_cube.env import PickCubeEnv
    from so101_lab.tasks.pick_cube.env_cfg import PickCubeEnvCfg
    from so101_lab.tasks.figure_shape_placement.env import FigureShapePlacementEnv
    from so101_lab.tasks.figure_shape_placement.env_cfg import FigureShapePlacementEnvCfg

    from so101_lab.tasks.figure_shape_placement_easy.env import FigureShapePlacementEasyEnv
    from so101_lab.tasks.figure_shape_placement_easy.env_cfg import FigureShapePlacementEasyEnvCfg
    from so101_lab.tasks.full_game_demo.env import FullGameDemoEnv
    from so101_lab.tasks.full_game_demo.env_cfg import FullGameDemoEnvCfg
    import so101_lab.tasks.figure_shape_placement.rl  # noqa: F401 — gym registration
    import so101_lab.tasks.figure_shape_placement_easy.rl  # noqa: F401 — gym registration

    return {
        "template": (TemplateEnv, TemplateEnvCfg),
        "pick_cube": (PickCubeEnv, PickCubeEnvCfg),
        "figure_shape_placement": (FigureShapePlacementEnv, FigureShapePlacementEnvCfg),
        "figure_shape_placement_easy": (FigureShapePlacementEasyEnv, FigureShapePlacementEasyEnvCfg),
        "full_game_demo": (FullGameDemoEnv, FullGameDemoEnvCfg),
    }


def get_task(task_name: str) -> tuple:
    """Get environment class and config class by task name.

    Args:
        task_name: One of: template, pick_cube, figure_shape_placement, figure_shape_placement_easy

    Returns:
        (EnvClass, EnvCfgClass)
    """
    registry = get_task_registry()
    if task_name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"Unknown task: {task_name}. Available: {available}")
    return registry[task_name]


def list_tasks() -> list[str]:
    """Return list of available task names."""
    return list(get_task_registry().keys())


def _get_rl_task_registry() -> dict[str, type]:
    """Lazy load RL task registry to avoid circular imports."""
    import so101_lab.tasks.figure_shape_placement.rl  # noqa: F401 — gym registration
    import so101_lab.tasks.figure_shape_placement_easy.rl  # noqa: F401 — gym registration

    from so101_lab.tasks.figure_shape_placement.rl.env_cfg import FigureShapePlacementRLEnvCfg
    from so101_lab.tasks.figure_shape_placement_easy.rl.env_cfg import FigureShapePlacementEasyRLEnvCfg

    return {
        "figure_shape_placement": FigureShapePlacementRLEnvCfg,
        "figure_shape_placement_easy": FigureShapePlacementEasyRLEnvCfg,
    }


def get_rl_task(task_name: str) -> type:
    """Get RL env config class by task name.

    Args:
        task_name: One of: figure_shape_placement

    Returns:
        ManagerBasedRLEnvCfg subclass
    """
    registry = _get_rl_task_registry()
    if task_name not in registry:
        available = ", ".join(registry.keys())
        raise ValueError(f"Unknown RL task: {task_name}. Available: {available}")
    return registry[task_name]
