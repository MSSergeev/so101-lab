# Adapted from: IQL (https://arxiv.org/abs/2110.06169)
# Changes: VIP ResNet50 frozen backbone for image encoding, twin Q-networks

"""Q and V critic networks for IQL with VIP ResNet50 backbone.

Architecture:
  VIP encoder (frozen ResNet50, Ego4D pretrained) extracts 1024-dim
  embeddings from top and wrist camera images. These are concatenated
  with state (6-dim) and optionally action (6-dim) for Q-networks.

  V(s): [vip(top), vip(wrist), state] → MLP → scalar
  Q(s,a): [vip(top), vip(wrist), state, action] → MLP → scalar

Usage:
    from so101_lab.policies.rl.critics import VNetwork, QNetwork, TwinQ

    v_net = VNetwork(device="cuda")
    q_net = TwinQ(device="cuda")

    # Forward pass
    v = v_net(images_top, images_wrist, state)        # (B, 1)
    q1, q2 = q_net(images_top, images_wrist, state, action)  # (B, 1), (B, 1)
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from so101_lab.rewards.vip_reward import _load_vip_encoder


class VIPBackbone(nn.Module):
    """Frozen VIP ResNet50 encoder for two camera images.

    Takes raw uint8 images, preprocesses and encodes them.
    Output: concatenated embeddings (B, 2048) for top + wrist.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.encoder = _load_vip_encoder(device)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def forward(
        self, images_top: torch.Tensor, images_wrist: torch.Tensor
    ) -> torch.Tensor:
        """Encode two camera images.

        Args:
            images_top: (B, H, W, 3) uint8 or (B, 3, H, W) float
            images_wrist: same format

        Returns:
            (B, 2048) concatenated embeddings.
        """
        emb_top = self._encode(images_top)
        emb_wrist = self._encode(images_wrist)
        return torch.cat([emb_top, emb_wrist], dim=-1)  # (B, 2048)

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images → (B, 1024)."""
        # Handle HWC uint8 format
        if images.dim() == 4 and images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)  # (B, 3, H, W)

        images = images.to(self.device)

        # Preprocess each image in batch
        preprocessed = torch.stack([self.preprocess(img) for img in images])
        return self.encoder(preprocessed)  # (B, 1024)


class VNetwork(nn.Module):
    """Value network V(s) for IQL.

    Input: images (top + wrist) + state → scalar value.
    """

    def __init__(
        self,
        state_dim: int = 6,
        hidden_dims: list[int] = [256, 256],
        device: str = "cuda",
        backbone: VIPBackbone | None = None,
    ):
        super().__init__()
        self.backbone = backbone or VIPBackbone(device)
        # 2048 (VIP top+wrist) + state_dim
        input_dim = 2048 + state_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self.to(device)

    def forward(
        self,
        images_top: torch.Tensor,
        images_wrist: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images_top: (B, H, W, 3) uint8
            images_wrist: (B, H, W, 3) uint8
            state: (B, 6) float

        Returns:
            (B, 1) value estimate.
        """
        visual = self.backbone(images_top, images_wrist)  # (B, 2048)
        x = torch.cat([visual, state.to(visual.device)], dim=-1)
        return self.mlp(x)


class QNetwork(nn.Module):
    """Single Q-network Q(s, a) for IQL.

    Input: images (top + wrist) + state + action → scalar Q-value.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 6,
        hidden_dims: list[int] = [256, 256],
        device: str = "cuda",
        backbone: VIPBackbone | None = None,
    ):
        super().__init__()
        self.backbone = backbone or VIPBackbone(device)
        # 2048 (VIP top+wrist) + state_dim + action_dim
        input_dim = 2048 + state_dim + action_dim

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.SiLU(),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)
        self.to(device)

    def forward(
        self,
        images_top: torch.Tensor,
        images_wrist: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images_top: (B, H, W, 3) uint8
            images_wrist: (B, H, W, 3) uint8
            state: (B, 6) float
            action: (B, 6) float

        Returns:
            (B, 1) Q-value estimate.
        """
        visual = self.backbone(images_top, images_wrist)  # (B, 2048)
        x = torch.cat(
            [visual, state.to(visual.device), action.to(visual.device)],
            dim=-1,
        )
        return self.mlp(x)


class TwinQ(nn.Module):
    """Twin Q-networks for IQL (min of two Q-values).

    Shares a single VIP backbone between both Q-networks.
    """

    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int = 6,
        hidden_dims: list[int] = [256, 256],
        device: str = "cuda",
    ):
        super().__init__()
        self.backbone = VIPBackbone(device)
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims, device, self.backbone)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims, device, self.backbone)

    def forward(
        self,
        images_top: torch.Tensor,
        images_wrist: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (q1, q2), each (B, 1)."""
        return (
            self.q1(images_top, images_wrist, state, action),
            self.q2(images_top, images_wrist, state, action),
        )

    def min_q(
        self,
        images_top: torch.Tensor,
        images_wrist: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Returns min(Q1, Q2), shape (B, 1)."""
        q1, q2 = self.forward(images_top, images_wrist, state, action)
        return torch.min(q1, q2)
