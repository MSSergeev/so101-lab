# Runs in: lerobot-env (Python 3.12)
"""gRPC server for SAC training — holds SACPolicy, replay buffers, reward models, SAC update.

All ML stays server-side: SACPolicy + RewardClassifier + VIPReward + replay buffers.
Only raw observations (~1.8 MB) and actions cross the wire per step.

Usage:
    python scripts/train/sac_server.py --port 8082
"""

import json
import logging
import os
import pickle  # nosec
from concurrent import futures

from so101_lab.utils.compat import patch_hf_custom_models
patch_hf_custom_models()

import grpc
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.nn.utils import clip_grad_norm_

from so101_lab.transport import sac_pb2, sac_pb2_grpc
from so101_lab.transport.utils import (
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    deserialize,
    serialize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("sac_server")

# --- Constants ---

ACTION_MIN = np.array([-100, -100, -100, -100, -100, 0], dtype=np.float32)
ACTION_MAX = np.array([100, 100, 100, 100, 100, 100], dtype=np.float32)
STATE_MIN = ACTION_MIN
STATE_MAX = ACTION_MAX

STATE_KEYS = ["observation.state", "observation.images.top", "observation.images.wrist"]


# --- Helper functions ---

def normalize_action(action: np.ndarray) -> np.ndarray:
    """env [-100/0, 100] -> policy [-1, 1]"""
    return 2 * (action - ACTION_MIN) / (ACTION_MAX - ACTION_MIN) - 1


def unnormalize_action(action: np.ndarray) -> np.ndarray:
    """policy [-1, 1] -> env [-100/0, 100]"""
    return (action + 1) / 2 * (ACTION_MAX - ACTION_MIN) + ACTION_MIN


def normalize_state(state: np.ndarray) -> np.ndarray:
    return 2 * (state - STATE_MIN) / (STATE_MAX - STATE_MIN) - 1


def normalize_state_tensor(state: torch.Tensor) -> torch.Tensor:
    s_min = torch.from_numpy(STATE_MIN)
    s_max = torch.from_numpy(STATE_MAX)
    return 2 * (state - s_min) / (s_max - s_min) - 1


def normalize_action_tensor(action: torch.Tensor) -> torch.Tensor:
    a_min = torch.from_numpy(ACTION_MIN)
    a_max = torch.from_numpy(ACTION_MAX)
    return 2 * (action - a_min) / (a_max - a_min) - 1


def resize_image_if_needed(img: torch.Tensor, target_size: int) -> torch.Tensor:
    """Resize CHW image to target_size x target_size if needed."""
    if target_size == 0:
        return img
    _, h, w = img.shape
    if h == target_size and w == target_size:
        return img
    return TF.resize(img, [target_size, target_size])


def obs_to_policy_format(obs: dict, image_size: int = 0) -> dict[str, torch.Tensor]:
    """Env numpy obs -> policy tensors (no batch dim)."""
    state = torch.from_numpy(normalize_state(obs["observation.state"]))
    img_top = torch.from_numpy(obs["observation.images.top"]).permute(2, 0, 1).float() / 255.0
    img_wrist = torch.from_numpy(obs["observation.images.wrist"]).permute(2, 0, 1).float() / 255.0

    img_top = resize_image_if_needed(img_top, image_size)
    img_wrist = resize_image_if_needed(img_wrist, image_size)

    return {
        "observation.state": state,
        "observation.images.top": img_top,
        "observation.images.wrist": img_wrist,
    }


def get_observation_features(
    policy,
    observations: dict[str, torch.Tensor],
    next_observations: dict[str, torch.Tensor],
) -> tuple[dict | None, dict | None]:
    """Cache image features to avoid redundant encoder calls."""
    if policy.config.vision_encoder_name is None or not policy.config.freeze_vision_encoder:
        return None, None

    with torch.no_grad():
        obs_features = policy.actor.encoder.get_cached_image_features(observations)
        next_obs_features = policy.actor.encoder.get_cached_image_features(next_observations)

    return obs_features, next_obs_features


def build_sac_config(config: dict):
    """Build SACConfig from Init config dict."""
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.sac.configuration_sac import SACConfig

    image_size = config.get("image_size", 0)
    if image_size > 0:
        img_shape = (3, image_size, image_size)
    else:
        img_shape = (3, 480, 640)

    return SACConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=img_shape),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=img_shape),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
        dataset_stats={
            "observation.state": {
                "min": STATE_MIN.tolist(),
                "max": STATE_MAX.tolist(),
            },
            "observation.images.top": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "observation.images.wrist": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "action": {
                "min": ACTION_MIN.tolist(),
                "max": ACTION_MAX.tolist(),
            },
        },
        device="cuda",
        vision_encoder_name="helper2424/resnet10",
        freeze_vision_encoder=True,
        shared_encoder=True,
        num_critics=2,
        discount=config.get("discount", 0.99),
        actor_lr=config.get("actor_lr", 3e-4),
        critic_lr=config.get("critic_lr", 3e-4),
        temperature_lr=config.get("temperature_lr", 3e-4),
        utd_ratio=config.get("utd_ratio", 2),
        policy_update_freq=config.get("policy_update_freq", 1),
        online_step_before_learning=config.get("warmup_steps", 500),
        use_torch_compile=False,
    )


def make_optimizers(policy) -> dict[str, torch.optim.Adam]:
    """Create optimizers following learner.py pattern."""
    config = policy.config

    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not config.shared_encoder or not n.startswith("encoder")
        ],
        lr=config.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(
        params=policy.critic_ensemble.parameters(),
        lr=config.critic_lr,
    )
    optimizer_temperature = torch.optim.Adam(
        params=[policy.log_alpha],
        lr=config.temperature_lr,
    )
    return {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }


def save_policy_checkpoint(
    policy,
    output_dir: str,
    name: str,
    optimizers: dict[str, torch.optim.Adam] | None = None,
    training_state: dict | None = None,
):
    """Save policy checkpoint with optional training state for resume."""
    checkpoint_dir = os.path.join(output_dir, name, "pretrained_model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    policy.save_pretrained(checkpoint_dir)

    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        config_data = json.load(f)
    if "type" not in config_data:
        config_data = {"type": "sac", **config_data}
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=4)

    if training_state is not None:
        state_to_save = dict(training_state)
        if optimizers is not None:
            state_to_save["optimizers"] = {k: v.state_dict() for k, v in optimizers.items()}
        state_path = os.path.join(output_dir, name, "training_state.pt")
        torch.save(state_to_save, state_path)

    logger.info(f"Saved checkpoint: {checkpoint_dir}")


def load_training_state(
    resume_path: str,
    optimizers: dict[str, torch.optim.Adam],
) -> dict:
    """Load training state from checkpoint."""
    candidate = os.path.join(resume_path, "..", "training_state.pt")
    state_path = os.path.normpath(candidate)
    if not os.path.exists(state_path):
        candidate = os.path.join(resume_path, "..", "..", "training_state.pt")
        state_path = os.path.normpath(candidate)

    if not os.path.exists(state_path):
        logger.info("No training_state.pt found, starting counters from 0")
        return {}

    state = torch.load(state_path, weights_only=False)
    logger.info(f"Loaded training state from: {state_path}")

    if "optimizers" in state:
        for name, opt in optimizers.items():
            if name in state["optimizers"]:
                opt.load_state_dict(state["optimizers"][name])
                logger.info(f"  Restored {name} optimizer")
        del state["optimizers"]

    return state


def create_offline_buffer(
    dataset_path: str,
    image_size: int,
    device: str = "cuda",
    storage_device: str = "cpu",
    num_workers: int = 4,
    max_episodes: int | None = None,
    success_bonus: float = 0.0,
    reward_scale: float = 1.0,
    reward_clip: tuple | None = None,
):
    """Load offline dataset into ReplayBuffer using parallel DataLoader."""
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from tqdm import tqdm

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.rl.buffer import ReplayBuffer

    image_transforms = transforms.Resize([image_size, image_size])

    episodes = None
    if max_episodes is not None:
        episodes = list(range(max_episodes))

    dataset = LeRobotDataset(
        repo_id="local",
        root=dataset_path,
        image_transforms=image_transforms,
        episodes=episodes,
    )

    if "next.reward" not in dataset.meta.features:
        raise ValueError(
            f"Dataset {dataset_path} missing 'next.reward'. "
            "Run prepare_reward_dataset.py first."
        )

    logger.info(f"Loading offline dataset: {dataset_path} ({len(dataset)} frames, {num_workers} workers)")

    episode_indices = [dataset.hf_dataset[i]["episode_index"] for i in range(len(dataset))]

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    buffer = ReplayBuffer(
        capacity=len(dataset),
        device=device,
        state_keys=STATE_KEYS,
        storage_device=storage_device,
        optimize_memory=True,
        use_drq=False,
    )

    prev_state = None
    prev_action = None
    prev_reward = None
    prev_idx = None
    ep_success_given = False
    bonus_count = 0

    for i, sample in enumerate(tqdm(dataloader, desc="Loading offline data")):
        state = {
            "observation.state": normalize_state_tensor(sample["observation.state"].squeeze(0)).unsqueeze(0),
            "observation.images.top": sample["observation.images.top"],
            "observation.images.wrist": sample["observation.images.wrist"],
        }
        action = normalize_action_tensor(sample["action"].squeeze(0)).unsqueeze(0)
        reward = sample["next.reward"].item()

        if prev_state is not None:
            done = episode_indices[prev_idx] != episode_indices[i]

            adjusted_reward = prev_reward
            if prev_reward > 0.5 and not ep_success_given and success_bonus > 0:
                adjusted_reward += success_bonus
                ep_success_given = True
                bonus_count += 1

            if reward_scale != 1.0:
                adjusted_reward /= reward_scale
                if reward_clip:
                    adjusted_reward = max(reward_clip[0], min(adjusted_reward, reward_clip[1]))

            if done:
                next_state = {k: v.clone() for k, v in prev_state.items()}
                ep_success_given = False
            else:
                next_state = {k: v.to(storage_device) for k, v in state.items()}

            buffer.add(
                state={k: v.to(storage_device) for k, v in prev_state.items()},
                action=prev_action.to(storage_device),
                reward=adjusted_reward,
                next_state=next_state,
                done=done,
                truncated=done,
            )

        prev_state = state
        prev_action = action
        prev_reward = reward
        prev_idx = i

    if prev_state is not None:
        adjusted_reward = prev_reward
        if prev_reward > 0.5 and not ep_success_given and success_bonus > 0:
            adjusted_reward += success_bonus
            bonus_count += 1
        if reward_scale != 1.0:
            adjusted_reward /= reward_scale
            if reward_clip:
                adjusted_reward = max(reward_clip[0], min(adjusted_reward, reward_clip[1]))
        buffer.add(
            state={k: v.to(storage_device) for k, v in prev_state.items()},
            action=prev_action.to(storage_device),
            reward=adjusted_reward,
            next_state={k: v.clone().to(storage_device) for k, v in prev_state.items()},
            done=True,
            truncated=True,
        )

    scale_info = f", reward_scale={reward_scale:.2f}" if reward_scale != 1.0 else ""
    logger.info(f"Offline buffer loaded: {len(buffer)} transitions, {bonus_count} terminal bonuses{scale_info}")
    return buffer


class SACTrainingServer(sac_pb2_grpc.SACTrainingServicer):
    def __init__(self):
        self.policy = None
        self.classifier = None
        self.vip = None
        self.optimizers = None
        self.config = None
        self.clip_grad_norm_value = 10.0

        # Buffers
        self.online_buffer = None
        self.offline_buffer = None
        self.intervention_buffer = None

        # Per-episode state
        self.ep_success_given = False

    def Ready(self, request, context):  # noqa: N802
        logger.info(f"Client connected: {context.peer()}")
        return sac_pb2.Empty()

    def Init(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        config = deserialize(raw)
        self.config = config
        logger.info(f"Init with config keys: {list(config.keys())}")

        from lerobot.policies.sac.modeling_sac import SACPolicy
        from lerobot.rl.buffer import ReplayBuffer

        # Build or load policy
        resume_path = config.get("resume")
        if resume_path:
            logger.info(f"Resuming from checkpoint: {resume_path}")
            self.policy = SACPolicy.from_pretrained(resume_path)
            self.policy.to("cuda")
            self.policy.train()
        else:
            sac_config = build_sac_config(config)
            self.policy = SACPolicy(sac_config)
            self.policy.to("cuda")
            self.policy.train()

        # torch.compile
        if config.get("torch_compile", False):
            logger.info("Applying torch.compile to encoder, actor, critic_ensemble, critic_target...")
            self.policy.actor.encoder.image_encoder.forward = torch.compile(
                self.policy.actor.encoder.image_encoder.forward
            )
            self.policy.actor.forward = torch.compile(self.policy.actor.forward)
            self.policy.critic_ensemble.forward = torch.compile(self.policy.critic_ensemble.forward)
            self.policy.critic_target.forward = torch.compile(self.policy.critic_target.forward)

        self.optimizers = make_optimizers(self.policy)
        self.clip_grad_norm_value = self.policy.config.grad_clip_norm

        # Resume training state
        start_step = 0
        start_episode = 0
        best_sr = -1.0
        if resume_path:
            state = load_training_state(resume_path, self.optimizers)
            start_step = state.get("opt_step", 0)
            start_episode = state.get("ep_count", 0)
            best_sr = state.get("best_sr", -1.0)
            logger.info(f"Resumed from step {start_step}, episode {start_episode}, best_sr={best_sr:.0%}")

        # Reward models
        if config.get("reward_model") and config.get("reward_mode") in ("composite", "classifier_only"):
            from so101_lab.rewards.classifier import RewardClassifier
            self.classifier = RewardClassifier(config["reward_model"], device="cuda")
            logger.info(f"Classifier loaded from {config['reward_model']}")

        if config.get("vip_goal_dataset"):
            from so101_lab.rewards.vip_reward import VIPReward
            self.vip = VIPReward(
                config["vip_goal_dataset"],
                device="cuda",
                image_key=config.get("vip_camera", "observation.images.top"),
                goal_mode=config.get("vip_goal_mode", "mean"),
                use_labeled=config.get("vip_use_labeled", False),
                label_dataset_path=config.get("vip_label_dataset"),
                normalize=config.get("vip_normalize", False),
                scale_dataset_path=config.get("demo_dataset"),
            )
            logger.info(f"VIP reward loaded (mode={config.get('vip_goal_mode', 'mean')})")

        # Buffers
        buffer_image_size = config.get("buffer_image_size", 128)

        if not config.get("no_online", False):
            self.online_buffer = ReplayBuffer(
                capacity=config.get("online_capacity", 30000),
                device="cuda",
                state_keys=STATE_KEYS,
                storage_device="cpu",
                optimize_memory=True,
                use_drq=False,
            )
            logger.info(f"Online buffer: capacity={config.get('online_capacity', 30000)}")

        if not config.get("no_offline", False) and config.get("demo_dataset"):
            vip_scale = 1.0
            vip_clip = None
            if self.vip and config.get("vip_normalize", False):
                vip_scale = self.vip.reward_scale
                vip_clip = (-2.0, 0.0)
            self.offline_buffer = create_offline_buffer(
                config["demo_dataset"],
                image_size=buffer_image_size,
                device="cuda",
                storage_device="cpu",
                num_workers=config.get("num_workers", 4),
                max_episodes=config.get("max_episodes"),
                success_bonus=config.get("success_bonus", 0.0),
                reward_scale=vip_scale,
                reward_clip=vip_clip,
            )

        if config.get("hil", False):
            self.intervention_buffer = ReplayBuffer(
                capacity=config.get("intervention_capacity", 20000),
                device="cuda",
                state_keys=STATE_KEYS,
                storage_device="cpu",
                optimize_memory=True,
                use_drq=False,
            )
            logger.info(f"Intervention buffer: capacity={config.get('intervention_capacity', 20000)}")

        logger.info("Init complete — policy, reward models, buffers ready")
        return sac_pb2.InitResponse(
            start_step=start_step,
            start_episode=start_episode,
            best_sr=best_sr,
        )

    def SampleAction(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        obs_raw = deserialize(raw)

        obs_t = obs_to_policy_format(obs_raw, self.config.get("image_size", 0))
        batch = {k: v.unsqueeze(0).to("cuda") for k, v in obs_t.items()}
        with torch.no_grad():
            action_norm = self.policy.select_action(batch).squeeze(0).cpu().numpy()
        action_env = unnormalize_action(action_norm)

        resp_data = action_env.astype(np.float32).tobytes()
        return send_bytes_in_chunks(resp_data, sac_pb2.DataChunk)

    def SendStepResult(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        payload = deserialize(raw)

        obs_raw = payload["obs"]
        obs_next_raw = payload["obs_next"]
        action_env = payload["action"]
        sim_reward = payload["sim_reward"]
        done = payload["done"]
        truncated = payload["truncated"]
        is_intervention = payload.get("is_intervention", False)
        image_size = payload.get("image_size", 0)

        # Compute composite reward
        reward = sim_reward
        cls_pred = 0.0
        vip_reward_val = 0.0
        bonus_given = False

        if self.classifier is not None:
            cls_pred = self.classifier.predict_reward(obs_next_raw)
            w_classifier = self.config.get("w_classifier", 1.0)
            reward += cls_pred * w_classifier
            if cls_pred > 0.5 and not self.ep_success_given:
                reward += self.config.get("success_bonus", 10.0)
                self.ep_success_given = True
                bonus_given = True

        if self.vip is not None:
            vip_reward_val = self.vip.compute_reward(obs_next_raw)
            reward += vip_reward_val * self.config.get("w_vip", 1.0)

        # Add to buffer (normalized)
        buffer_image_size = self.config.get("buffer_image_size", 128)
        obs_t = obs_to_policy_format(obs_raw, buffer_image_size)
        next_obs_t = obs_to_policy_format(obs_next_raw, buffer_image_size)
        action_norm_np = normalize_action(
            action_env if isinstance(action_env, np.ndarray) else np.array(action_env, dtype=np.float32)
        )

        transition_kwargs = dict(
            state={k: v.unsqueeze(0) for k, v in obs_t.items()},
            action=torch.from_numpy(action_norm_np).unsqueeze(0),
            reward=float(reward),
            next_state={k: v.unsqueeze(0) for k, v in next_obs_t.items()},
            done=done,
            truncated=truncated,
        )

        if self.online_buffer is not None:
            self.online_buffer.add(**transition_kwargs)
        if is_intervention and self.intervention_buffer is not None:
            self.intervention_buffer.add(**transition_kwargs)

        # Reset per-episode state on done
        if done:
            self.ep_success_given = False

        return sac_pb2.StepResultResponse(
            reward=float(reward),
            classifier_pred=float(cls_pred),
            vip_reward=float(vip_reward_val),
            bonus_given=bonus_given,
        )

    def RunSACUpdate(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        payload = deserialize(raw)

        step = payload["step"]
        update_config = payload["config"]

        from lerobot.rl.buffer import concatenate_batch_transitions

        batch_size = self.config.get("batch_size", 256)
        utd_ratio = self.config.get("utd_ratio", 2)
        policy_update_freq = self.config.get("policy_update_freq", 1)

        training_infos = {}

        for utd_i in range(utd_ratio):
            active_buffers = []
            if self.online_buffer is not None and len(self.online_buffer) >= 2:
                active_buffers.append(self.online_buffer)
            if self.offline_buffer is not None and len(self.offline_buffer) >= 2:
                active_buffers.append(self.offline_buffer)
            if self.intervention_buffer is not None and len(self.intervention_buffer) >= 2:
                active_buffers.append(self.intervention_buffer)

            n_bufs = len(active_buffers)
            if n_bufs == 0:
                break
            elif n_bufs == 1:
                forward_batch = active_buffers[0].sample(batch_size)
            elif n_bufs == 2:
                half = batch_size // 2
                batch_a = active_buffers[0].sample(half)
                batch_b = active_buffers[1].sample(batch_size - half)
                forward_batch = concatenate_batch_transitions(batch_a, batch_b)
            else:  # 3 buffers
                third = batch_size // 3
                remainder = batch_size - 2 * third
                batch_a = active_buffers[0].sample(third)
                batch_b = active_buffers[1].sample(third)
                batch_c = active_buffers[2].sample(remainder)
                forward_batch = concatenate_batch_transitions(batch_a, batch_b)
                forward_batch = concatenate_batch_transitions(forward_batch, batch_c)

            # Cache image features
            obs_features, next_obs_features = get_observation_features(
                self.policy, forward_batch["state"], forward_batch["next_state"]
            )
            if obs_features is not None:
                forward_batch["observation_feature"] = obs_features
                forward_batch["next_observation_feature"] = next_obs_features

            # Critic update
            critic_output = self.policy.forward(forward_batch, model="critic")
            loss_critic = critic_output["loss_critic"]
            self.optimizers["critic"].zero_grad()
            loss_critic.backward()
            clip_grad_norm_(self.policy.critic_ensemble.parameters(), self.clip_grad_norm_value)
            self.optimizers["critic"].step()
            self.policy.update_target_networks()

            training_infos["loss_critic"] = loss_critic.item()

            batch_rewards = forward_batch["reward"]
            training_infos["batch_reward_mean"] = batch_rewards.mean().item()
            training_infos["batch_reward_min"] = batch_rewards.min().item()
            training_infos["batch_reward_max"] = batch_rewards.max().item()

        # Actor + temperature update
        opt_step = update_config.get("opt_step", step)
        if opt_step % policy_update_freq == 0:
            actor_output = self.policy.forward(forward_batch, model="actor")
            loss_actor = actor_output["loss_actor"]
            self.optimizers["actor"].zero_grad()
            loss_actor.backward()
            clip_grad_norm_(self.policy.actor.parameters(), self.clip_grad_norm_value)
            self.optimizers["actor"].step()

            temp_output = self.policy.forward(forward_batch, model="temperature")
            loss_temp = temp_output["loss_temperature"]
            self.optimizers["temperature"].zero_grad()
            loss_temp.backward()
            clip_grad_norm_([self.policy.log_alpha], self.clip_grad_norm_value)
            self.optimizers["temperature"].step()

            min_temperature = self.config.get("min_temperature", 0.0)
            if min_temperature > 0:
                with torch.no_grad():
                    self.policy.log_alpha.clamp_(min=np.log(min_temperature))
            training_infos["loss_actor"] = loss_actor.item()
            training_infos["loss_temperature"] = loss_temp.item()
            training_infos["temperature"] = self.policy.temperature

            # Q-values and entropy diagnostic
            with torch.no_grad():
                obs_b = forward_batch["state"]
                obs_feat_b = forward_batch.get("observation_feature")
                a_pi, lp, _ = self.policy.actor(obs_b, obs_feat_b)
                q_pi = self.policy.critic_forward(obs_b, a_pi, use_target=False, observation_features=obs_feat_b)
                training_infos["q_actor_mean"] = q_pi.min(dim=0)[0].mean().item()
                training_infos["entropy"] = -lp.mean().item()

        return sac_pb2.UpdateMetrics(
            loss_critic=training_infos.get("loss_critic", 0.0),
            loss_actor=training_infos.get("loss_actor", 0.0),
            loss_temperature=training_infos.get("loss_temperature", 0.0),
            temperature=training_infos.get("temperature", 0.0),
            q_actor_mean=training_infos.get("q_actor_mean", 0.0),
            entropy=training_infos.get("entropy", 0.0),
            batch_reward_mean=training_infos.get("batch_reward_mean", 0.0),
            batch_reward_min=training_infos.get("batch_reward_min", 0.0),
            batch_reward_max=training_infos.get("batch_reward_max", 0.0),
        )

    def SetMode(self, request, context):  # noqa: N802
        mode = request.mode
        if mode == "eval":
            self.policy.eval()
        else:
            self.policy.train()
        logger.info(f"Mode set to: {mode}")
        return sac_pb2.Empty()

    def SampleActionDeterministic(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        obs_raw = deserialize(raw)

        obs_t = obs_to_policy_format(obs_raw, self.config.get("image_size", 0))
        batch = {k: v.unsqueeze(0).to("cuda") for k, v in obs_t.items()}
        with torch.no_grad():
            action_norm = self.policy.select_action(batch).squeeze(0).cpu().numpy()
        action_env = unnormalize_action(action_norm)

        resp_data = action_env.astype(np.float32).tobytes()
        return send_bytes_in_chunks(resp_data, sac_pb2.DataChunk)

    def SaveCheckpoint(self, request_iterator, context):  # noqa: N802
        raw = receive_bytes_in_chunks(request_iterator)
        payload = deserialize(raw)
        path = payload["path"]
        metadata = payload.get("metadata", {})

        save_policy_checkpoint(
            self.policy,
            os.path.dirname(path),
            os.path.basename(path),
            self.optimizers,
            metadata,
        )
        return sac_pb2.Empty()


def serve(host: str = "127.0.0.1", port: int = 8082) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    sac_server = SACTrainingServer()
    sac_pb2_grpc.add_SACTrainingServicer_to_server(sac_server, server)
    server.add_insecure_port(f"{host}:{port}")
    logger.info(f"SAC server starting on {host}:{port}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAC training gRPC server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
