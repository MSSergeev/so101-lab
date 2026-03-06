"""Noise prior for SmolVLA flow matching.

A fixed mu (6,) shift applied to x_0[:, 0, :6] before the ODE.
Updated via human feedback: "better" moves mu toward current noise,
"worse" moves it away.
"""

import torch


class NoisePrior:
    """Fixed mu shift for initial noise x_0[:, 0, :6].

    Each step of the episode samples x_0 with action dims = mu + randn(6).
    After human feedback, mu is updated via exponential moving average.
    """

    def __init__(self, lr: float = 0.1, device: str = "cuda"):
        self.mu = torch.zeros(6, device=device)
        self.lr = lr
        self.device = device

    def sample_x0(self, shape: tuple[int, ...]) -> torch.Tensor:
        """Sample x_0 with mu-shifted action dims.

        Args:
            shape: full noise shape, e.g. (1, 50, 32).

        Returns:
            x_0 tensor with x_0[0, 0, :6] = mu + randn(6).
        """
        x_0 = torch.randn(shape, device=self.device)
        x_0[0, 0, :6] = self.mu + torch.randn(6, device=self.device)
        return x_0

    def update(self, feedback: str, x_0_history: list[torch.Tensor]):
        """Update mu based on human feedback.

        Args:
            feedback: "better" or "worse".
            x_0_history: list of x_0 tensors used during the episode.
        """
        if not x_0_history:
            return

        # Average of action-dim noise used in this episode
        ep_mean = torch.stack([x[0, 0, :6] for x in x_0_history]).mean(dim=0)
        direction = ep_mean - self.mu

        if feedback == "better":
            self.mu = self.mu + self.lr * direction
        elif feedback == "worse":
            self.mu = self.mu - self.lr * direction

    def patch_model(self, model):
        """Monkey-patch model.sample_noise to add mu shift.

        After patching, standard SmolVLA inference (select_action, etc.)
        will automatically use the mu-shifted noise. Works with eval scripts.

        Args:
            model: SmolVLA model (policy.model).
        """
        original_sample_noise = model.sample_noise
        mu = self.mu

        def patched_sample_noise(shape, device):
            noise = original_sample_noise(shape, device)
            noise[0, 0, :6] = mu.to(device) + noise[0, 0, :6]
            return noise

        model.sample_noise = patched_sample_noise

    def state_dict(self) -> dict:
        return {"mu": self.mu.cpu(), "lr": self.lr}

    def load_state_dict(self, state: dict):
        self.mu = state["mu"].to(self.device)
        self.lr = state.get("lr", self.lr)
