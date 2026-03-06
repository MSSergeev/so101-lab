# Runs in: lerobot-env (Python 3.12)
"""gRPC server for PPO training — holds policy, VIP reward, and rollout buffer.

All ML stays server-side: FlowNoiseSmolVLA + VIPReward + PPO update.
Trajectory data (~3 GB/rollout) never crosses the wire.

Usage:
    python scripts/train/ppo_server.py --port 8081
"""

import logging
import os
import pickle  # nosec
from concurrent import futures

import grpc
import numpy as np
import torch

from so101_lab.transport import ppo_pb2, ppo_pb2_grpc
from so101_lab.transport.utils import (
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    deserialize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ppo_server")


# --- GAE (same as original) ---

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# --- Reward normalization ---

class RunningMeanStd:
    """Welford's online algorithm for running mean/variance."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, x: torch.Tensor):
        batch_mean = x.mean().item()
        batch_var = x.var().item() if len(x) > 1 else 0.0
        batch_count = len(x)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / max(total, 1)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / max(total, 1)
        self.var = M2 / max(total, 1)
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var**0.5 + 1e-8)


class PPOTrainingServer(ppo_pb2_grpc.PPOTrainingServicer):
    def __init__(self):
        self.policy = None
        self.vip_reward = None
        self.actor_optimizer = None
        self.value_optimizer = None
        self.reward_rms = None
        self.config = None

        # Rollout buffer (populated per-step, consumed by PPO update)
        self._reset_rollout()

    def _reset_rollout(self):
        self.rollout_trajectories = []
        self.rollout_vip_embs = []
        self.rollout_states = []
        self.rollout_log_probs = []
        self.rollout_values = []
        self.rollout_rewards = []
        self.rollout_dones = []

    def Ready(self, request, context):  # noqa: N802
        logger.info(f"Client connected: {context.peer()}")
        return ppo_pb2.Empty()

    def Init(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        config = deserialize(raw)
        self.config = config
        logger.info(f"Init with config keys: {list(config.keys())}")

        # Load policy
        from so101_lab.policies.rl.flow_noise_smolvla import FlowNoiseSmolVLA

        self.policy = FlowNoiseSmolVLA(
            checkpoint_path=config["checkpoint"],
            device="cuda",
            iql_checkpoint=config.get("iql_checkpoint"),
            task_string=config.get("task", "Place the cube into the matching slot on the platform"),
            kv_cache_device=config.get("kv_cache_device", "cpu"),
        )

        # Apply noise prior if provided
        if config.get("noise_prior"):
            self._apply_noise_prior(config["noise_prior"])

        # VIP reward
        from so101_lab.rewards.vip_reward import VIPReward

        logger.info(f"Loading VIP reward from {config['goal_dataset']}")
        self.vip_reward = VIPReward(
            config["goal_dataset"],
            device="cuda",
            use_labeled=config.get("vip_use_labeled", False),
            label_dataset_path=config.get("vip_label_dataset"),
            goal_mode=config.get("vip_goal_mode", "mean"),
            n_goal_frames=config.get("n_goal_frames", 5),
        )

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.policy.trainable_actor_params(), lr=config.get("actor_lr", 3e-5)
        )
        self.value_optimizer = torch.optim.Adam(
            self.policy.value_mlp.parameters(), lr=config.get("value_lr", 1e-4)
        )

        # Reward normalization
        if config.get("normalize_rewards", False):
            self.reward_rms = RunningMeanStd()
        else:
            self.reward_rms = None

        # Resume
        start_update = 0
        best_sr = -1.0
        ep_count = 0
        if config.get("resume"):
            optimizers = {"actor": self.actor_optimizer, "value": self.value_optimizer}
            meta = self.policy.load_checkpoint(config["resume"], optimizers)
            start_update = meta.get("update", 0)
            best_sr = meta.get("best_sr", -1.0)
            ep_count = meta.get("ep_count", 0)
            logger.info(f"Resumed from update {start_update}, best_sr={best_sr:.0%}")

        logger.info("Init complete — policy, VIP, optimizers ready")
        return ppo_pb2.InitResponse(
            start_update=start_update,
            best_sr=best_sr,
            ep_count=ep_count,
        )

    def _apply_noise_prior(self, noise_prior_path: str):
        ckpt = torch.load(noise_prior_path, map_location="cuda", weights_only=False)
        np_data = ckpt.get("noise_prior", {})
        if isinstance(np_data, dict) and np_data.get("type") == "learned":
            from so101_lab.policies.rl.learned_noise_sampler import LearnedNoiseSampler
            noise_dims = np_data.get("noise_dims", 6)
            sampler = LearnedNoiseSampler(device="cuda", noise_dims=noise_dims)
            sampler.load(noise_prior_path)
            sampler.eval()
            sampler.patch_model(self.policy.policy.model, [None])
            logger.info(f"Learned noise sampler loaded (noise_dims={noise_dims})")
        else:
            from so101_lab.policies.rl.noise_prior import NoisePrior
            noise_prior = NoisePrior(device="cuda")
            noise_prior.load_state_dict(np_data)
            noise_prior.patch_model(self.policy.policy.model)
            mu_str = ", ".join(f"{v:.3f}" for v in noise_prior.mu.cpu().numpy())
            logger.info(f"Noise prior loaded: mu = [{mu_str}]")

    def SampleAction(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        obs = deserialize(raw)

        # Cache VIP embedding for value head
        vip_emb = self.policy.cache_vip_embedding(obs)
        state_tensor = torch.from_numpy(obs["observation.state"]).float().to("cuda")

        # Value (no grad)
        with torch.no_grad():
            value = self.policy.get_value_from_cached(
                vip_emb.unsqueeze(0), state_tensor.unsqueeze(0)
            )

        # Sample action with log_prob + trajectory data
        action, log_prob, trajectory = self.policy.sample_actions_with_log_prob(obs)

        # Store in rollout buffer (stays on server — ~3 GB never transferred)
        self.rollout_trajectories.append(trajectory)
        self.rollout_vip_embs.append(vip_emb)
        self.rollout_states.append(state_tensor)
        self.rollout_log_probs.append(log_prob)
        self.rollout_values.append(value.squeeze())

        return ppo_pb2.ActionResult(
            action=action.astype(np.float32).tobytes(),
            log_prob=float(log_prob),
            value=float(value),
        )

    def SendStepResult(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        payload = deserialize(raw)

        obs_next = payload["obs_next"]
        sim_rewards = payload["sim_rewards"]
        done = payload["done"]

        # Compute combined reward (VIP on obs_next + sim rewards)
        reward = 0.0
        cfg = self.config

        if cfg.get("vip_weight", 1.0) > 0:
            reward += cfg["vip_weight"] * self.vip_reward.compute_reward(obs_next)

        if cfg.get("sim_weight", 0.0) > 0:
            reward += cfg["sim_weight"] * sim_rewards.get(
                "reward/distance_cube_to_slot_weighted", 0.0
            )

        if cfg.get("success_bonus", 0.0) > 0:
            reward += cfg["success_bonus"] * sim_rewards.get(
                "reward/milestone_placed_weighted", 0.0
            )

        self.rollout_rewards.append(float(reward))
        self.rollout_dones.append(float(done))

        return ppo_pb2.Empty()

    def RunPPOUpdate(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        payload = deserialize(raw)

        bootstrap_obs = payload["bootstrap_obs"]
        update_config = payload["config"]
        cfg = self.config

        # Bootstrap last value
        with torch.no_grad():
            last_vip = self.policy.cache_vip_embedding(bootstrap_obs)
            last_state = torch.from_numpy(
                bootstrap_obs["observation.state"]
            ).float().to("cuda")
            last_value = self.policy.get_value_from_cached(
                last_vip.unsqueeze(0), last_state.unsqueeze(0)
            ).item()

        # Convert to tensors
        raw_rewards_t = torch.tensor(self.rollout_rewards, dtype=torch.float32)
        if self.reward_rms is not None:
            self.reward_rms.update(raw_rewards_t)
            rewards_t = self.reward_rms.normalize(raw_rewards_t)
        else:
            rewards_t = raw_rewards_t
        values_t = torch.stack(self.rollout_values).cpu()
        dones_t = torch.tensor(self.rollout_dones, dtype=torch.float32)
        old_log_probs_t = torch.stack(self.rollout_log_probs).cpu()

        vip_embs_t = torch.stack(self.rollout_vip_embs)  # (T, 2048)
        states_t = torch.stack(self.rollout_states)  # (T, 6)

        # GAE
        gamma = cfg.get("gamma", 0.99)
        gae_lambda = cfg.get("gae_lambda", 0.95)
        advantages, returns = compute_gae(
            rewards_t, values_t, dones_t, last_value, gamma, gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Move to GPU
        advantages = advantages.to("cuda")
        returns = returns.to("cuda")
        old_log_probs_t = old_log_probs_t.to("cuda")
        vip_embs_t = vip_embs_t.to("cuda")
        states_t = states_t.to("cuda")

        # PPO update
        update_epochs = cfg.get("update_epochs", 4)
        batch_size = cfg.get("batch_size", 64)
        clip_ratio = cfg.get("clip_ratio", 0.2)
        max_grad_norm = cfg.get("max_grad_norm", 1.0)
        reeval_batch_size = cfg.get("reeval_batch_size", 1)
        is_warmup = update_config.get("is_warmup", False)

        T = len(self.rollout_rewards)
        indices = np.arange(T)
        actor_losses = []
        value_losses = []
        ratios_all = []

        for epoch in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                batch_idx = indices[start:end]

                # Re-evaluate values using cached embeddings
                batch_vip = vip_embs_t[batch_idx]
                batch_states = states_t[batch_idx]
                new_values = self.policy.get_value_from_cached(batch_vip, batch_states)

                # Value loss
                value_loss = 0.5 * (new_values - returns[batch_idx]).pow(2).mean()
                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.value_mlp.parameters(), max_grad_norm
                )
                self.value_optimizer.step()
                value_losses.append(value_loss.item())

                if is_warmup:
                    actor_losses.append(0.0)
                    ratios_all.append(torch.ones(len(batch_idx)))
                    continue

                # Re-evaluate log_prob under current params
                if reeval_batch_size > 1:
                    batch_log_probs = []
                    for rb_start in range(0, len(batch_idx), reeval_batch_size):
                        rb_end = min(rb_start + reeval_batch_size, len(batch_idx))
                        rb_trajs = [
                            self.rollout_trajectories[batch_idx[j]]
                            for j in range(rb_start, rb_end)
                        ]
                        rb_lps = self.policy.reeval_log_prob_batched(rb_trajs)
                        batch_log_probs.append(rb_lps)
                    new_log_probs = torch.cat(batch_log_probs)
                else:
                    batch_log_probs = []
                    for i in batch_idx:
                        new_lp = self.policy.reeval_log_prob(
                            None, self.rollout_trajectories[i]
                        )
                        batch_log_probs.append(new_lp)
                    new_log_probs = torch.stack(batch_log_probs)

                # PPO ratio
                ratio = torch.exp(new_log_probs - old_log_probs_t[batch_idx])
                adv = advantages[batch_idx]

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.trainable_actor_params()), max_grad_norm
                )
                self.actor_optimizer.step()

                actor_losses.append(actor_loss.item())
                ratios_all.append(ratio.detach().cpu())

        all_ratios = torch.cat(ratios_all)

        # Clear rollout buffer
        self._reset_rollout()

        return ppo_pb2.UpdateMetrics(
            actor_loss=float(np.mean(actor_losses)),
            value_loss=float(np.mean(value_losses)),
            ratio_mean=float(all_ratios.mean()),
            ratio_std=float(all_ratios.std()),
            ratio_max=float(all_ratios.max()),
            ratio_min=float(all_ratios.min()),
            reward_mean=float(raw_rewards_t.mean()),
            reward_std=float(raw_rewards_t.std()),
            advantage_mean=float(advantages.mean()),
            log_prob_mean=float(old_log_probs_t.mean()),
            value_mean=float(values_t.mean()),
        )

    def SetMode(self, request, context):  # noqa: N802
        mode = request.mode
        if mode == "eval":
            self.policy.set_eval_noise_bounds()
        else:
            self.policy.set_train_noise_bounds()
        logger.info(f"Mode set to: {mode}")
        return ppo_pb2.Empty()

    def SampleActionDeterministic(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        obs = deserialize(raw)
        action = self.policy.sample_actions_deterministic(obs)
        return ppo_pb2.ActionResult(
            action=action.astype(np.float32).tobytes(),
            log_prob=0.0,
            value=0.0,
        )

    def ResetPolicy(self, request, context):  # noqa: N802
        self.policy.policy.reset()
        return ppo_pb2.Empty()

    def SaveCheckpoint(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        payload = deserialize(raw)
        path = payload["path"]
        metadata = payload.get("metadata", {})

        optimizers = {"actor": self.actor_optimizer, "value": self.value_optimizer}
        self.policy.save_checkpoint(path, optimizers, metadata)
        logger.info(f"Checkpoint saved: {path}")
        return ppo_pb2.Empty()


def serve(host: str = "127.0.0.1", port: int = 8081) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    ppo_server = PPOTrainingServer()
    ppo_pb2_grpc.add_PPOTrainingServicer_to_server(ppo_server, server)
    server.add_insecure_port(f"{host}:{port}")
    logger.info(f"PPO server starting on {host}:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PPO training gRPC server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
