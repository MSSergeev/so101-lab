"""Learned noise sampler for SmolVLA flow matching.

State-conditional MLP that replaces N(0,I) with N(mu(s), sigma(s))
for x_0 dims in flow matching ODE. Inspired by Green-VLA.

noise_dims controls how many dims of x_0 the sampler generates:
  - 6: first 6 dims of token 0 (original joint actions)
  - 32: all 32 dims of token 0
  - 1600: all 50 tokens × 32 dims (full x_0)

Two training modes:
  - Offline: backprop through ODE, loss = MSE(predicted, expert_action)
  - Online: REINFORCE with episode-level reward
"""

import torch
import torch.nn as nn

from so101_lab.policies.rl.critics import VIPBackbone

# Default hidden dims per noise_dims setting
_DEFAULT_HIDDEN = {6: 256, 32: 512, 1600: 1024}


class LearnedNoiseSampler(nn.Module):
    """State-conditioned noise sampler for flow matching x_0.

    Architecture: VIP(2048) + state(6) -> MLP -> mu(noise_dims) + log_sigma(noise_dims)
    """

    def __init__(
        self,
        device: str = "cuda",
        noise_dims: int = 6,
        hidden_dim: int | None = None,
        log_sigma_min: float = -2.0,
        log_sigma_max: float = 2.0,
    ):
        super().__init__()
        self.device = device
        self.noise_dims = noise_dims
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

        if hidden_dim is None:
            hidden_dim = _DEFAULT_HIDDEN.get(noise_dims, 512)

        self.vip_backbone = VIPBackbone(device)

        # MLP: (2048 + 6) -> hidden -> (noise_dims * 2)
        self.mlp = nn.Sequential(
            nn.Linear(2048 + 6, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, noise_dims * 2),
        ).to(device)

        # Zero init last layer → starts as N(0, 1)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, vip_emb: torch.Tensor, state: torch.Tensor):
        """Predict noise distribution params.

        Args:
            vip_emb: (B, 2048) pre-computed VIP embedding (no grad).
            state: (B, 6) observation state.

        Returns:
            mu: (B, noise_dims), log_sigma: (B, noise_dims).
        """
        x = torch.cat([vip_emb, state], dim=-1)
        out = self.mlp(x)
        mu = out[:, :self.noise_dims]
        log_sigma = torch.clamp(
            out[:, self.noise_dims:], self.log_sigma_min, self.log_sigma_max
        )
        return mu, log_sigma

    def forward_from_obs(self, obs: dict):
        """Convenience: extract VIP + state from raw obs dict.

        Args:
            obs: env observation with images (numpy HWC uint8) and state.

        Returns:
            mu: (1, noise_dims), log_sigma: (1, noise_dims)
        """
        import numpy as np

        img_top = torch.from_numpy(
            np.ascontiguousarray(obs["observation.images.top"])
        ).unsqueeze(0)
        img_wrist = torch.from_numpy(
            np.ascontiguousarray(obs["observation.images.wrist"])
        ).unsqueeze(0)
        state = (
            torch.from_numpy(obs["observation.state"])
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        with torch.no_grad():
            vip_emb = self.vip_backbone(img_top, img_wrist)

        return self.forward(vip_emb, state)

    def _fill_x0(self, x_0: torch.Tensor, learned: torch.Tensor) -> torch.Tensor:
        """Fill x_0 with learned noise values based on noise_dims."""
        B = learned.shape[0]
        if self.noise_dims <= 32:
            x_0[:, 0, :self.noise_dims] = learned
        else:  # 1600 = 50 * 32
            x_0[:, :, :] = learned.reshape(B, 50, 32)
        return x_0

    def sample_x0(self, shape: tuple[int, ...], obs: dict) -> torch.Tensor:
        """Sample x_0 with learned conditional noise.

        Args:
            shape: full noise shape (1, chunk_size, max_action_dim).
            obs: current env observation.

        Returns:
            x_0 with learned dims from N(mu(obs), sigma(obs)).
        """
        mu, log_sigma = self.forward_from_obs(obs)
        sigma = torch.exp(log_sigma)

        x_0 = torch.randn(shape, device=self.device)
        eps = torch.randn(self.noise_dims, device=self.device)
        learned = mu.squeeze(0) + sigma.squeeze(0) * eps
        x_0 = self._fill_x0(x_0, learned.unsqueeze(0))
        return x_0

    def sample_x0_differentiable(
        self, shape: tuple[int, ...], vip_emb: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Sample x_0 with gradient flow through mu (for offline training).

        Uses reparameterization trick: x_action = mu + sigma * eps.
        Gradient flows: loss → ODE → x_0 → mu → sampler MLP.

        Args:
            shape: full noise shape (B, chunk_size, max_action_dim).
            vip_emb: (B, 2048) VIP features (detached).
            state: (B, 6) observation state.

        Returns:
            x_0 tensor with grad through learned dims.
        """
        mu, log_sigma = self.forward(vip_emb, state)
        sigma = torch.exp(log_sigma)

        B = shape[0]
        x_0 = torch.randn(shape, device=self.device)
        eps = torch.randn(B, self.noise_dims, device=self.device)
        learned = mu + sigma * eps
        x_0 = x_0.clone()
        x_0 = self._fill_x0(x_0, learned)
        return x_0

    def log_prob(
        self, x_action: torch.Tensor, vip_emb: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """Log probability of sampled noise under current params (for REINFORCE).

        Args:
            x_action: (B, noise_dims) actual noise values used.
            vip_emb: (B, 2048) VIP features.
            state: (B, 6) observation state.

        Returns:
            (B,) log probabilities with gradient.
        """
        mu, log_sigma = self.forward(vip_emb, state)
        sigma = torch.exp(log_sigma)
        lp = -0.5 * ((x_action - mu) / sigma).pow(2) - log_sigma
        return lp.sum(dim=-1)

    def patch_model(self, model, obs_ref: list):
        """Monkey-patch model.sample_noise with learned sampler.

        Caller must update obs_ref[0] = obs before each policy forward.
        """
        original_sample_noise = model.sample_noise
        sampler = self

        def patched_sample_noise(shape, device):
            if obs_ref[0] is not None:
                return sampler.sample_x0(shape, obs_ref[0])
            return original_sample_noise(shape, device)

        model.sample_noise = patched_sample_noise

    def save(self, path: str, extra: dict | None = None):
        """Save checkpoint compatible with eval --noise-prior."""
        data = {
            "noise_prior": {
                "type": "learned",
                "noise_dims": self.noise_dims,
                "mlp": self.mlp.state_dict(),
                "log_sigma_min": self.log_sigma_min,
                "log_sigma_max": self.log_sigma_max,
            },
        }
        if extra:
            data.update(extra)
        torch.save(data, path)

    def load(self, path: str):
        """Load from checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        state = ckpt["noise_prior"]
        saved_dims = state.get("noise_dims", 6)
        if saved_dims != self.noise_dims:
            raise ValueError(
                f"Checkpoint noise_dims={saved_dims} != sampler noise_dims={self.noise_dims}"
            )
        self.mlp.load_state_dict(state["mlp"])
        self.log_sigma_min = state.get("log_sigma_min", self.log_sigma_min)
        self.log_sigma_max = state.get("log_sigma_max", self.log_sigma_max)
