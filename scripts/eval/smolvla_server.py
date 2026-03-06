# Runs in: lerobot-env (Python 3.12)
"""gRPC policy server for sim eval — receives obs in lerobot format, returns action chunks.

Usage:
    python scripts/eval/smolvla_server.py --port 8080 --host 127.0.0.1
"""

import io
import logging
import pickle  # nosec
import threading
from concurrent import futures
from queue import Empty, Queue

import grpc
import numpy as np
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import receive_bytes_in_chunks, send_bytes_in_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smolvla_server")


class SimPolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self):
        self.shutdown_event = threading.Event()
        self.observation_queue: Queue = Queue(maxsize=1)
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        self.actions_per_chunk: int = 15
        self.device: str = "cuda"
        self.policy_type: str = "smolvla"

    def _reset(self) -> None:
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

    def Ready(self, request, context):  # noqa: N802
        logger.info(f"Client connected: {context.peer()}")
        self._reset()
        self.shutdown_event.clear()
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        specs = pickle.loads(request.data)  # nosec
        self.checkpoint = specs["checkpoint"]
        self.policy_type = specs.get("policy_type", "smolvla")
        self.actions_per_chunk = specs.get("actions_per_chunk", 15)
        self.device = specs.get("device", "cuda")
        task = specs.get("task", "")
        logger.info(f"Loading {self.policy_type} from {self.checkpoint} on {self.device}")

        # Register processor module
        if self.policy_type == "smolvla":
            import lerobot.policies.smolvla.processor_smolvla  # noqa: F401
        elif self.policy_type in ("pi0", "pi0fast"):
            import lerobot.policies.pi0.processor_pi0  # noqa: F401

        policy_cls = get_policy_class(self.policy_type)
        self.policy = policy_cls.from_pretrained(self.checkpoint)
        self.policy.to(self.device)
        self.policy.eval()

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.checkpoint,
        )

        noise_prior_path = specs.get("noise_prior")
        if noise_prior_path:
            import torch
            ckpt = torch.load(noise_prior_path, map_location=self.device, weights_only=False)
            np_data = ckpt.get("noise_prior", {})
            if isinstance(np_data, dict) and np_data.get("type") == "learned":
                from so101_lab.policies.rl.learned_noise_sampler import LearnedNoiseSampler
                noise_dims = np_data.get("noise_dims", 6)
                sampler = LearnedNoiseSampler(device=self.device, noise_dims=noise_dims)
                sampler.load(noise_prior_path)
                sampler.eval()
                sampler.patch_model(self.policy.model, [None])
                logger.info(f"Learned noise sampler loaded (noise_dims={noise_dims})")
            else:
                from so101_lab.policies.rl.noise_prior import NoisePrior
                noise_prior = NoisePrior(device=self.device)
                noise_prior.load_state_dict(ckpt["noise_prior"])
                noise_prior.patch_model(self.policy.model)
                mu_str = ", ".join(f"{v:.3f}" for v in noise_prior.mu.cpu().numpy())
                logger.info(f"Noise prior loaded: mu = [{mu_str}]")

        logger.info("Policy and processors loaded")
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, logger
        )
        if raw is None:
            return services_pb2.Empty()

        obs_dict = pickle.loads(raw)  # nosec

        # Build batch in lerobot format
        batch = {
            "observation.state": torch.from_numpy(obs_dict["observation.state"]).float(),
            "task": obs_dict.get("task", ""),
        }
        for key in ("observation.images.top", "observation.images.wrist"):
            if key in obs_dict:
                img = obs_dict[key]
                batch[key] = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        timestep = obs_dict.get("timestep", 0)

        # Preprocess and run inference
        try:
            batch = self.preprocessor(batch)
            with torch.no_grad():
                chunk = self.policy.predict_action_chunk(batch)
            # chunk shape: (B, chunk_size, action_dim) or (chunk_size, action_dim)
            if chunk.ndim == 2:
                chunk = chunk.unsqueeze(0)
            chunk = chunk[:, : self.actions_per_chunk, :]

            # Postprocess each action in chunk
            # postprocessor: PolicyAction (Tensor) -> PolicyAction (Tensor)
            processed = []
            for i in range(chunk.shape[1]):
                single = chunk[:, i, :]  # (B, action_dim)
                result = self.postprocessor(single)
                # result may be Tensor or dict depending on pipeline config
                if isinstance(result, dict):
                    result = result.get("action", single)
                processed.append(result)
            chunk_np = torch.stack(processed, dim=1).squeeze(0).detach().cpu().numpy()

            # Enqueue, dropping old if full
            if self.observation_queue.full():
                try:
                    self.observation_queue.get_nowait()
                except Empty:
                    pass
            self.observation_queue.put({"timestep": timestep, "chunk": chunk_np})
            logger.info(f"Inference done for timestep {timestep}, chunk {chunk_np.shape}")
        except Exception as e:
            logger.error(f"Inference error: {e}")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        try:
            item = self.observation_queue.get(timeout=10.0)
            chunk_np: np.ndarray = item["chunk"]
        except Empty:
            logger.warning("GetActions timed out — no inference result available")
            chunk_np = np.zeros((self.actions_per_chunk, 6), dtype=np.float32)
        buf = io.BytesIO()
        np.save(buf, chunk_np)
        return services_pb2.Actions(data=buf.getvalue())


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    policy_server = SimPolicyServer()
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{host}:{port}")
    logger.info(f"Server starting on {host}:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
