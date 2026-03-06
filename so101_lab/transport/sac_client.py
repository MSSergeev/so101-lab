# Runs in: isaaclab-env (Python 3.11)
"""gRPC client for SAC training — sends obs from Isaac Sim, receives actions + rewards."""

import pickle  # nosec B403
from typing import Any

import grpc
import numpy as np

from so101_lab.transport import sac_pb2, sac_pb2_grpc
from so101_lab.transport.utils import (
    grpc_channel_options,
    send_bytes_in_chunks,
    receive_bytes_in_chunks,
    serialize,
)


class SACTrainingClient:
    """Client-side SAC training interface for isaaclab-env."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8082) -> None:
        self.host = host
        self.port = port
        self._channel = None
        self._stub = None

    def connect(self) -> None:
        options = grpc_channel_options()
        self._channel = grpc.insecure_channel(f"{self.host}:{self.port}", options=options)
        self._stub = sac_pb2_grpc.SACTrainingStub(self._channel)
        self._stub.Ready(sac_pb2.Empty())

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def init(self, config: dict) -> dict:
        """Send config, server loads policy + buffers + reward models. Returns resume metadata."""
        data = serialize(config)
        resp = self._stub.Init(send_bytes_in_chunks(data, sac_pb2.DataChunk))
        return {
            "start_step": resp.start_step,
            "start_episode": resp.start_episode,
            "best_sr": resp.best_sr,
        }

    def sample_action(self, obs: dict) -> np.ndarray:
        """Send obs, receive action (env-space)."""
        data = serialize(obs)
        resp_bytes = receive_bytes_in_chunks(
            self._stub.SampleAction(send_bytes_in_chunks(data, sac_pb2.DataChunk))
        )
        return np.frombuffer(resp_bytes, dtype=np.float32).copy()

    def send_step_result(
        self,
        obs: dict,
        obs_next: dict,
        action: np.ndarray,
        sim_reward: float,
        done: bool,
        truncated: bool,
        is_intervention: bool = False,
        image_size: int = 0,
    ) -> dict:
        """Send transition data. Server computes reward, stores in buffer. Returns reward breakdown."""
        payload = {
            "obs": obs,
            "obs_next": obs_next,
            "action": action,
            "sim_reward": sim_reward,
            "done": done,
            "truncated": truncated,
            "is_intervention": is_intervention,
            "image_size": image_size,
        }
        data = serialize(payload)
        resp = self._stub.SendStepResult(send_bytes_in_chunks(data, sac_pb2.DataChunk))
        return {
            "reward": resp.reward,
            "classifier_pred": resp.classifier_pred,
            "vip_reward": resp.vip_reward,
            "bonus_given": resp.bonus_given,
        }

    def run_sac_update(self, step: int, config: dict) -> dict:
        """Run SAC update on server. Returns metrics dict."""
        payload = {"step": step, "config": config}
        data = serialize(payload)
        resp = self._stub.RunSACUpdate(send_bytes_in_chunks(data, sac_pb2.DataChunk))
        return {
            "loss_critic": resp.loss_critic,
            "loss_actor": resp.loss_actor,
            "loss_temperature": resp.loss_temperature,
            "temperature": resp.temperature,
            "q_actor_mean": resp.q_actor_mean,
            "entropy": resp.entropy,
            "batch_reward_mean": resp.batch_reward_mean,
            "batch_reward_min": resp.batch_reward_min,
            "batch_reward_max": resp.batch_reward_max,
        }

    def set_mode(self, mode: str) -> None:
        """Set server to 'train' or 'eval' mode."""
        self._stub.SetMode(sac_pb2.ModeRequest(mode=mode))

    def sample_action_deterministic(self, obs: dict) -> np.ndarray:
        """Deterministic action for evaluation."""
        data = serialize(obs)
        resp_bytes = receive_bytes_in_chunks(
            self._stub.SampleActionDeterministic(
                send_bytes_in_chunks(data, sac_pb2.DataChunk)
            )
        )
        return np.frombuffer(resp_bytes, dtype=np.float32).copy()

    def save_checkpoint(self, path: str, metadata: dict | None = None) -> None:
        """Tell server to save checkpoint."""
        payload = {"path": path, "metadata": metadata or {}}
        data = serialize(payload)
        self._stub.SaveCheckpoint(send_bytes_in_chunks(data, sac_pb2.DataChunk))
