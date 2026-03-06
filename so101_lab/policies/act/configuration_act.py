# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: Standalone inference-only config for SO-101, removed PreTrainedConfig base
"""ACT configuration for inference."""

from dataclasses import dataclass, field


@dataclass
class ACTConfig:
    """Configuration for ACT inference.

    Loaded from config.json saved by lerobot training.
    """

    # Model architecture
    chunk_size: int = 100
    n_action_steps: int = 100
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    pre_norm: bool = False
    dropout: float = 0.1

    # VAE
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Vision
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # Feature shapes (set at load time from config.json)
    state_dim: int = 6
    action_dim: int = 6
    image_features: list[str] = field(default_factory=list)
    image_shapes: dict[str, list[int]] = field(default_factory=dict)

    # Inference mode
    temporal_ensemble_coeff: float | None = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ACTConfig":
        """Create config from dictionary (loaded from config.json)."""
        # Extract relevant fields, handling nested structure
        kwargs = {}

        # Direct mappings
        direct_fields = [
            "chunk_size", "n_action_steps", "dim_model", "n_heads",
            "dim_feedforward", "feedforward_activation", "n_encoder_layers",
            "n_decoder_layers", "pre_norm", "dropout", "use_vae", "latent_dim",
            "n_vae_encoder_layers", "vision_backbone", "pretrained_backbone_weights",
            "replace_final_stride_with_dilation", "temporal_ensemble_coeff",
        ]

        for field_name in direct_fields:
            if field_name in config_dict:
                kwargs[field_name] = config_dict[field_name]

        # Extract feature dimensions from input_features/output_features
        if "input_features" in config_dict:
            input_features = config_dict["input_features"]
            # State dimension
            for key, feat in input_features.items():
                if "state" in key.lower() and "image" not in key.lower():
                    shape = feat.get("shape", [6])
                    kwargs["state_dim"] = shape[0] if isinstance(shape, list) else shape

            # Image features
            image_features = []
            image_shapes = {}
            for key, feat in input_features.items():
                if "image" in key.lower():
                    image_features.append(key)
                    shape = feat.get("shape", [3, 480, 640])
                    image_shapes[key] = shape if isinstance(shape, list) else list(shape)
            kwargs["image_features"] = image_features
            kwargs["image_shapes"] = image_shapes

        if "output_features" in config_dict:
            output_features = config_dict["output_features"]
            for key, feat in output_features.items():
                if "action" in key.lower():
                    shape = feat.get("shape", [6])
                    kwargs["action_dim"] = shape[0] if isinstance(shape, list) else shape

        return cls(**kwargs)
