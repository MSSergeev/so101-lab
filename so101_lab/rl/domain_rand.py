"""Shared domain randomization helpers for RL scripts."""

from __future__ import annotations


def apply_domain_rand_flags(env_cfg, args) -> None:
    """Disable domain randomization components based on CLI flags.

    Expects args with: no_domain_rand, no_randomize_light,
    no_randomize_physics, no_camera_noise, no_distractors.
    """
    from isaaclab.envs.mdp import image
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg

    if args.no_domain_rand or args.no_randomize_light:
        env_cfg.events.randomize_light = None
    if args.no_domain_rand or args.no_randomize_physics:
        env_cfg.events.randomize_cube_material = None
        env_cfg.events.randomize_platform_material = None
        env_cfg.events.randomize_cube_mass = None
    if args.no_domain_rand or args.no_camera_noise:
        env_cfg.observations.policy.images_top = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False},
        )
        env_cfg.observations.policy.images_wrist = ObsTerm(
            func=image,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )
    if args.no_domain_rand or args.no_distractors:
        env_cfg.events.reset_distractors = None
