"""Neural IK inference wrapper for SO-101 arm (5 DOF).

Input: desired EE pose in robot base frame (wxyz quaternion convention, matching IsaacLab).
Output: (N, 5) arm joint angles in radians.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralIKNet(nn.Module):
    def __init__(self, hidden_size: int = 256, num_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(12, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class NeuralIK:
    """Neural IK for SO-101 arm.

    Args:
        checkpoint_dir: directory with neural_ik.pt and neural_ik_meta.json
        device: torch device
    """

    def __init__(self, checkpoint_dir: str | Path, device: str | torch.device = "cpu", iters: int = 1):
        checkpoint_dir = Path(checkpoint_dir)
        with open(checkpoint_dir / "neural_ik_meta.json") as f:
            meta = json.load(f)

        self.device = torch.device(device)
        self.lo = torch.FloatTensor(meta["joint_limits_low"]).to(self.device)
        self.hi = torch.FloatTensor(meta["joint_limits_high"]).to(self.device)
        self.pos_mean = torch.FloatTensor(meta["pos_mean"]).to(self.device)
        self.pos_std = torch.FloatTensor(meta["pos_std"]).to(self.device)

        self.model = torch.load(
            checkpoint_dir / "neural_ik.pt", map_location=self.device, weights_only=False
        )
        self.model.eval()
        self.iters = iters
        self._mid = (self.hi + self.lo) / 2
        self._rng = (self.hi - self.lo) / 2

    @torch.no_grad()
    def compute(
        self,
        ee_pos_b: torch.Tensor,        # (N, 3) desired EE position in base frame
        ee_quat_b: torch.Tensor,       # (N, 4) desired EE quaternion wxyz in base frame
        current_joints: torch.Tensor,  # (N, 5) current arm joint angles
    ) -> torch.Tensor:
        """Returns (N, 5) predicted joint angles, clamped to limits.

        With iters>1 uses iterative refinement: feeds each prediction back as
        current_joints for the next pass, converging from noisy starting points.
        """
        ee_p = (ee_pos_b - self.pos_mean) / self.pos_std
        ee_q = F.normalize(ee_quat_b, dim=-1)
        sign = ee_q[:, :1].sign()
        sign[sign == 0] = 1.0
        ee_q = ee_q * sign

        pred = current_joints
        for _ in range(self.iters):
            curr = (pred - self._mid) / self._rng
            pred_norm = self.model(torch.cat([ee_p, ee_q, curr], dim=-1))
            pred = (pred_norm * self._rng + self._mid).clamp(self.lo, self.hi)
        return pred
