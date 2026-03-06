# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: Standalone inference-only config for SO-101, removed PreTrainedConfig base
"""Diffusion Policy configuration for inference."""

from dataclasses import dataclass, field


@dataclass
class DiffusionConfig:
    """Configuration for Diffusion Policy inference.

    Loaded from config.json saved by lerobot training.
    """

    # Inputs / output structure
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    # Vision backbone
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # UNet
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # Noise scheduler
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Feature shapes (set at load time from config.json)
    state_dim: int = 6
    action_dim: int = 6
    image_features: list[str] = field(default_factory=list)
    image_shapes: dict[str, list[int]] = field(default_factory=dict)

    # Environment state (usually None for SO-101)
    env_state_dim: int | None = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "DiffusionConfig":
        """Create config from dictionary (loaded from config.json)."""
        kwargs = {}

        direct_fields = [
            "n_obs_steps", "horizon", "n_action_steps",
            "vision_backbone", "crop_shape", "crop_is_random",
            "pretrained_backbone_weights", "use_group_norm",
            "spatial_softmax_num_keypoints", "use_separate_rgb_encoder_per_camera",
            "down_dims", "kernel_size", "n_groups", "diffusion_step_embed_dim",
            "use_film_scale_modulation", "noise_scheduler_type",
            "num_train_timesteps", "beta_schedule", "beta_start", "beta_end",
            "prediction_type", "clip_sample", "clip_sample_range",
            "num_inference_steps",
        ]

        for field_name in direct_fields:
            if field_name in config_dict:
                val = config_dict[field_name]
                # Convert lists to tuples for tuple fields
                if field_name in ("crop_shape", "down_dims") and isinstance(val, list):
                    val = tuple(val)
                kwargs[field_name] = val

        # Extract feature dimensions from input_features/output_features
        if "input_features" in config_dict:
            input_features = config_dict["input_features"]
            for key, feat in input_features.items():
                if "state" in key.lower() and "image" not in key.lower() and "environment" not in key.lower():
                    shape = feat.get("shape", [6])
                    kwargs["state_dim"] = shape[0] if isinstance(shape, list) else shape

            image_features = []
            image_shapes = {}
            for key, feat in input_features.items():
                if "image" in key.lower():
                    image_features.append(key)
                    shape = feat.get("shape", [3, 480, 640])
                    image_shapes[key] = shape if isinstance(shape, list) else list(shape)
            kwargs["image_features"] = image_features
            kwargs["image_shapes"] = image_shapes

            # Environment state
            for key, feat in input_features.items():
                if "environment" in key.lower():
                    shape = feat.get("shape", [0])
                    kwargs["env_state_dim"] = shape[0] if isinstance(shape, list) else shape

        if "output_features" in config_dict:
            output_features = config_dict["output_features"]
            for key, feat in output_features.items():
                if "action" in key.lower():
                    shape = feat.get("shape", [6])
                    kwargs["action_dim"] = shape[0] if isinstance(shape, list) else shape

        return cls(**kwargs)
