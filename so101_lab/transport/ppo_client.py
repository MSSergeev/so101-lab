# Runs in: isaaclab-env (Python 3.11)
"""gRPC client for PPO training — sends obs from Isaac Sim, receives actions."""

import pickle  # nosec B403
from typing import Any

import grpc
import numpy as np

from so101_lab.transport import ppo_pb2, ppo_pb2_grpc
from so101_lab.transport.utils import (
    grpc_channel_options,
    send_bytes_in_chunks,
    receive_bytes_in_chunks,
    serialize,
)


class PPOTrainingClient:
    """Client-side PPO training interface for isaaclab-env."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8081) -> None:
        self.host = host
        self.port = port
        self._channel = None
        self._stub = None

    def connect(self) -> None:
        options = grpc_channel_options()
        self._channel = grpc.insecure_channel(f"{self.host}:{self.port}", options=options)
        self._stub = ppo_pb2_grpc.PPOTrainingStub(self._channel)
        self._stub.Ready(ppo_pb2.Empty())

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def init(self, config: dict) -> dict:
        """Send config, server loads model+VIP. Returns resume metadata."""
        data = serialize(config)
        resp = self._stub.Init(send_bytes_in_chunks(data, ppo_pb2.DataChunk))
        return {
            "start_update": resp.start_update,
            "best_sr": resp.best_sr,
            "ep_count": resp.ep_count,
        }

    def sample_action(self, obs: dict) -> tuple[np.ndarray, float, float]:
        """Send obs, receive (action, log_prob, value)."""
        data = serialize(obs)
        resp = self._stub.SampleAction(send_bytes_in_chunks(data, ppo_pb2.DataChunk))
        action = np.frombuffer(resp.action, dtype=np.float32).copy()
        return action, resp.log_prob, resp.value

    def send_step_result(self, obs_next: dict, sim_rewards: dict, done: bool) -> None:
        """Send obs_next (for VIP reward) + sim rewards + done flag."""
        payload = {
            "obs_next": obs_next,
            "sim_rewards": sim_rewards,
            "done": done,
        }
        data = serialize(payload)
        self._stub.SendStepResult(send_bytes_in_chunks(data, ppo_pb2.DataChunk))

    def run_ppo_update(self, bootstrap_obs: dict, config: dict) -> dict:
        """Run PPO update on server. Returns metrics dict."""
        payload = {"bootstrap_obs": bootstrap_obs, "config": config}
        data = serialize(payload)
        resp = self._stub.RunPPOUpdate(send_bytes_in_chunks(data, ppo_pb2.DataChunk))
        return {
            "actor_loss": resp.actor_loss,
            "value_loss": resp.value_loss,
            "ratio_mean": resp.ratio_mean,
            "ratio_std": resp.ratio_std,
            "ratio_max": resp.ratio_max,
            "ratio_min": resp.ratio_min,
            "reward_mean": resp.reward_mean,
            "reward_std": resp.reward_std,
            "advantage_mean": resp.advantage_mean,
            "log_prob_mean": resp.log_prob_mean,
            "value_mean": resp.value_mean,
        }

    def set_mode(self, mode: str) -> None:
        """Set server to 'train' or 'eval' mode."""
        self._stub.SetMode(ppo_pb2.ModeRequest(mode=mode))

    def sample_action_deterministic(self, obs: dict) -> np.ndarray:
        """Deterministic action for evaluation."""
        data = serialize(obs)
        resp = self._stub.SampleActionDeterministic(
            send_bytes_in_chunks(data, ppo_pb2.DataChunk)
        )
        return np.frombuffer(resp.action, dtype=np.float32).copy()

    def reset_policy(self) -> None:
        """Reset policy state (action chunking, KV cache, etc.)."""
        self._stub.ResetPolicy(ppo_pb2.Empty())

    def save_checkpoint(self, path: str, metadata: dict | None = None) -> None:
        """Tell server to save checkpoint."""
        payload = {"path": path, "metadata": metadata or {}}
        data = serialize(payload)
        self._stub.SaveCheckpoint(send_bytes_in_chunks(data, ppo_pb2.DataChunk))
