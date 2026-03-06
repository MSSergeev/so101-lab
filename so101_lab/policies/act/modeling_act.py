# Adapted from: lerobot (https://github.com/huggingface/lerobot)
# Original license: Apache 2.0
# Changes: Standalone inference-only ACT for SO-101, removed HF Hub/training deps
"""Action Chunking Transformer model for inference.

Based on "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
https://huggingface.co/papers/2304.13705
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from .configuration_act import ACTConfig


class ACTPolicy(nn.Module):
    """Action Chunking Transformer Policy for inference."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config
        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(
                config.temporal_ensemble_coeff, config.chunk_size
            )

        self.reset()

    def reset(self):
        """Reset on episode start."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select single action given observations.

        Args:
            batch: Dict with "observation.state" and image keys

        Returns:
            (action_dim,) action tensor
        """
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict chunk of actions."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions


class ACTTemporalEnsembler:
    """Temporal ensembling for smoother actions.

    Supports batched operation (B envs) with per-env reset via reset_envs().
    Count is per-env (B, T, 1) so staggered resets blend correctly.
    Based on Algorithm 2 of https://huggingface.co/papers/2304.13705.
    """

    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Reset all envs."""
        self.ensembled_actions = None
        self.ensembled_actions_count = None
        self._pending_reset_ids = None

    def reset_envs(self, env_ids: list[int]):
        """Mark specific envs for re-initialization on next update()."""
        if self.ensembled_actions is None:
            return
        self._pending_reset_ids = env_ids

    def pop(self) -> Tensor:
        """Pop next action without new predictions (for ensemble_interval > 1)."""
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[:, 1:],
        )
        return action

    def update(self, actions: Tensor, skip_env_ids: list[int] | None = None) -> Tensor:
        """Update ensemble and return next action.

        Args:
            actions: (B, chunk_size, action_dim) predicted action chunks
            skip_env_ids: Env indices to exclude from blending.
                Their buffer is still popped (shared T dimension) but not
                updated with new predictions. Use reset_envs() after to
                re-initialize them on the next update() call.

        Returns:
            action: (B, action_dim) next ensembled action
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)

        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            B = actions.shape[0]
            self.ensembled_actions_count = torch.ones(
                (B, self.chunk_size, 1), dtype=torch.long, device=actions.device
            )
        else:
            M = self.ensembled_actions.shape[1]

            # Re-initialize reset envs before blending
            if self._pending_reset_ids is not None:
                self.ensembled_actions[self._pending_reset_ids] = actions[self._pending_reset_ids, :M]
                self.ensembled_actions_count[self._pending_reset_ids] = 1
                self._pending_reset_ids = None

            # Determine which envs to update
            if skip_env_ids is not None and len(skip_env_ids) > 0:
                active = torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device)
                active[skip_env_ids] = False
                idx = torch.where(active)[0]
            else:
                idx = None  # all

            if idx is not None:
                # Update only active envs
                ea = self.ensembled_actions[idx]
                ec = self.ensembled_actions_count[idx]
                ea *= self.ensemble_weights_cumsum[ec - 1]
                ea += actions[idx, :M] * self.ensemble_weights[ec]
                ea /= self.ensemble_weights_cumsum[ec]
                self.ensembled_actions[idx] = ea
                self.ensembled_actions_count[idx] = torch.clamp(ec + 1, max=self.chunk_size)
            else:
                self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
                self.ensembled_actions += actions[:, :M] * self.ensemble_weights[self.ensembled_actions_count]
                self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
                self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)

            # Append remaining new actions with count=1
            tail = actions[:, M:]
            if tail.shape[1] > 0:
                self.ensembled_actions = torch.cat([self.ensembled_actions, tail], dim=1)
                new_count = torch.ones(
                    (actions.shape[0], tail.shape[1], 1), dtype=torch.long, device=actions.device
                )
                self.ensembled_actions_count = torch.cat(
                    [self.ensembled_actions_count, new_count], dim=1
                )

        return self.pop()


class ACT(nn.Module):
    """Action Chunking Transformer network."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        # VAE encoder (only used during training, kept for weight loading)
        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            self.vae_encoder_robot_state_input_proj = nn.Linear(config.state_dim, config.dim_model)
            self.vae_encoder_action_input_proj = nn.Linear(config.action_dim, config.dim_model)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)

            num_input_token_encoder = 1 + config.chunk_size + 1  # cls + actions + state
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # Vision backbone
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})
            self._backbone_out_channels = backbone_model.fc.in_features

        # Transformer encoder/decoder
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Input projections
        self.encoder_robot_state_input_proj = nn.Linear(config.state_dim, config.dim_model)
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)

        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                self._backbone_out_channels, config.dim_model, kernel_size=1
            )

        # Positional embeddings
        n_1d_tokens = 2  # latent + state
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)

        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Action head
        self.action_head = nn.Linear(config.dim_model, config.action_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor | None, Tensor | None]]:
        """Forward pass.

        Args:
            batch: Dict with:
                - "observation.state": (B, state_dim)
                - "observation.images": list of (B, C, H, W) image tensors

        Returns:
            actions: (B, chunk_size, action_dim)
            (mu, log_sigma_x2): VAE params (None during inference)
        """
        # Determine batch size
        if "observation.images" in batch:
            batch_size = batch["observation.images"][0].shape[0]
        else:
            batch_size = batch["observation.state"].shape[0]

        device = batch["observation.state"].device

        # During inference, use zero latent
        mu = log_sigma_x2 = None
        latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32, device=device)

        # Build encoder input tokens
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))

        # Robot state token
        encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch["observation.state"]))

        # Image tokens
        if self.config.image_features and "observation.images" in batch:
            for img in batch["observation.images"]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack tokens
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # Forward through transformer
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=device,
        )

        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    """Single encoder layer."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    """Transformer decoder."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    """Single decoder layer."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x

        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x

        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)

        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """Create 1D sinusoidal positional embeddings."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings for images."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        not_mask = torch.ones_like(x[0, :1])
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        x_range = x_range.unsqueeze(-1) / inverse_frequency
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Get activation function by name."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
