"""IQL offline pretraining for SAC policy.

Pretrains both actor and critic on offline data using Implicit Q-Learning
(Kostrikov et al. 2022). Produces a checkpoint compatible with SAC --resume.

Usage:
    source ~/Robotics/Lerobot/so-101_base_single/lerobot-env/bin/activate
    python scripts/train/train_iql_pretrain.py \
        --demo-dataset data/recordings/figure_shape_placement_v5_vip_128 \
        --num-steps 50000 --output outputs/iql_v1 --tracker trackio
"""

import argparse
import json
import os
import time

from so101_lab.utils.compat import patch_hf_custom_models
patch_hf_custom_models()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.buffer import ReplayBuffer

# Constants
ACTION_MIN = np.array([-100, -100, -100, -100, -100, 0], dtype=np.float32)
ACTION_MAX = np.array([100, 100, 100, 100, 100, 100], dtype=np.float32)
STATE_MIN = ACTION_MIN
STATE_MAX = ACTION_MAX
STATE_KEYS = ["observation.state", "observation.images.top", "observation.images.wrist"]


class ValueNetwork(nn.Module):
    """V(s) - state value network, takes pre-computed encoder features."""

    def __init__(self, input_dim, hidden_dims=(256, 256)):
        super().__init__()
        layers = []
        prev = input_dim
        for dim in hidden_dims:
            layers.extend([nn.Linear(prev, dim), nn.SiLU()])
            prev = dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features).squeeze(-1)


def normalize_state_tensor(state):
    s_min = torch.from_numpy(STATE_MIN).to(state.device)
    s_max = torch.from_numpy(STATE_MAX).to(state.device)
    return 2 * (state - s_min) / (s_max - s_min) - 1


def normalize_action_tensor(action):
    a_min = torch.from_numpy(ACTION_MIN).to(action.device)
    a_max = torch.from_numpy(ACTION_MAX).to(action.device)
    return 2 * (action - a_min) / (a_max - a_min) - 1


def build_sac_config(args):
    img_shape = (3, args.image_size, args.image_size)
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
            "observation.state": {"min": STATE_MIN.tolist(), "max": STATE_MAX.tolist()},
            "observation.images.top": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "observation.images.wrist": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "action": {"min": ACTION_MIN.tolist(), "max": ACTION_MAX.tolist()},
        },
        device="cuda",
        vision_encoder_name="helper2424/resnet10",
        freeze_vision_encoder=True,
        shared_encoder=True,
        num_critics=2,
        discount=args.discount,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        temperature_lr=3e-4,
        use_torch_compile=False,
    )


def create_offline_buffer(dataset_path, image_size, num_workers=4, max_episodes=None, success_bonus=0.0,
                          reward_scale=1.0, reward_clip=None):
    from torch.utils.data import DataLoader
    from torchvision import transforms

    image_transforms = transforms.Resize([image_size, image_size])
    episodes = list(range(max_episodes)) if max_episodes else None

    dataset = LeRobotDataset(
        repo_id="local",
        root=dataset_path,
        image_transforms=image_transforms,
        episodes=episodes,
    )

    if "next.reward" not in dataset.meta.features:
        raise ValueError(
            f"Dataset {dataset_path} missing 'next.reward'. Run prepare_reward_dataset.py first."
        )

    print(f"Loading offline dataset: {dataset_path} ({len(dataset)} frames, {num_workers} workers)")
    episode_indices = [dataset.hf_dataset[i]["episode_index"] for i in range(len(dataset))]

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        pin_memory=True, prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    buffer = ReplayBuffer(
        capacity=len(dataset), device="cuda", state_keys=STATE_KEYS,
        storage_device="cpu", optimize_memory=True, use_drq=False,
    )

    prev_state = prev_action = prev_reward = prev_idx = None
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
                next_state = {k: v.to("cpu") for k, v in state.items()}
            buffer.add(
                state={k: v.to("cpu") for k, v in prev_state.items()},
                action=prev_action.to("cpu"),
                reward=adjusted_reward,
                next_state=next_state,
                done=done, truncated=done,
            )

        prev_state, prev_action, prev_reward, prev_idx = state, action, reward, i

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
            state={k: v.to("cpu") for k, v in prev_state.items()},
            action=prev_action.to("cpu"),
            reward=adjusted_reward,
            next_state={k: v.clone().to("cpu") for k, v in prev_state.items()},
            done=True, truncated=True,
        )

    scale_info = f", reward_scale={reward_scale:.2f}" if reward_scale != 1.0 else ""
    print(f"Offline buffer loaded: {len(buffer)} transitions, {bonus_count} terminal bonuses{scale_info}")
    return buffer


def save_policy_checkpoint(policy, output_dir, name, v_net=None):
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

    if v_net is not None:
        torch.save(v_net.state_dict(), os.path.join(output_dir, name, "v_network.pt"))

    print(f"Saved checkpoint: {checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="IQL offline pretraining for SAC")
    parser.add_argument("--demo-dataset", type=str, required=True)
    parser.add_argument("--num-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--expectile", type=float, default=0.7, help="Expectile tau for V-loss")
    parser.add_argument("--beta", type=float, default=3.0, help="Advantage temperature for actor loss")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=3e-4)
    parser.add_argument("--output", type=str, default="outputs/iql_v1")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--success-bonus", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--vip-normalize", action="store_true",
                        help="Normalize VIP rewards to [-1,0] (match online SAC --vip-normalize)")
    from so101_lab.utils.tracker import add_tracker_args
    add_tracker_args(parser, default_project="so101-iql")
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--save-freq", type=int, default=10000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 1. Compute VIP reward scale if needed
    _reward_scale = 1.0
    _reward_clip = None
    if args.vip_normalize:
        from torchvision import transforms as _T
        _episodes = list(range(args.max_episodes)) if args.max_episodes else None
        _tmp_ds = LeRobotDataset(
            repo_id="local", root=args.demo_dataset,
            image_transforms=_T.Resize([args.image_size, args.image_size]),
            episodes=_episodes,
        )
        raw_rewards = [float(_tmp_ds.hf_dataset[i]["next.reward"]) for i in range(len(_tmp_ds))]
        _reward_scale = max(-float(np.percentile(raw_rewards, 5)), 1.0)
        _reward_clip = (-2.0, 0.0)
        print(f"VIP normalize: reward_scale={_reward_scale:.2f}, clip={_reward_clip}")
        del _tmp_ds

    # 2. Load offline buffer
    buffer = create_offline_buffer(
        args.demo_dataset, image_size=args.image_size,
        num_workers=args.num_workers, max_episodes=args.max_episodes,
        success_bonus=args.success_bonus,
        reward_scale=_reward_scale, reward_clip=_reward_clip,
    )

    # 3. Build policy
    config = build_sac_config(args)
    policy = SACPolicy(config)
    policy.to("cuda")
    policy.train()

    encoder = policy.critic_ensemble.encoder
    encoder_dim = encoder.output_dim

    # 4. V-network
    v_net = ValueNetwork(encoder_dim).to("cuda")

    # 5. Optimizers
    optimizer_actor = torch.optim.Adam(
        [p for n, p in policy.actor.named_parameters()
         if not config.shared_encoder or not n.startswith("encoder")],
        lr=args.actor_lr,
    )
    optimizer_critic = torch.optim.Adam(
        policy.critic_ensemble.parameters(), lr=args.critic_lr,
    )
    optimizer_value = torch.optim.Adam(v_net.parameters(), lr=args.value_lr)

    clip_grad_norm_value = policy.config.grad_clip_norm

    # 6. Tracker
    from so101_lab.utils.tracker import setup_tracker, cleanup_tracker
    run_name = os.path.basename(os.path.normpath(args.output))
    tracker, sys_monitor = setup_tracker(args, run_name)

    # 7. Training loop
    print(f"\nStarting IQL pretraining: {args.num_steps} steps")
    print(f"  Buffer: {len(buffer)} transitions")
    print(f"  Encoder dim: {encoder_dim}")
    print(f"  Expectile: {args.expectile}, Beta: {args.beta}")
    print(f"  Batch size: {args.batch_size}")
    print()

    pbar = tqdm(range(args.num_steps), desc="IQL", unit="step")
    t_start = time.time()

    for step in pbar:
        batch = buffer.sample(args.batch_size)
        obs = batch["state"]
        next_obs = batch["next_state"]
        actions = batch["action"]
        rewards = batch["reward"]
        dones = batch["done"]

        # Cache image features for efficiency (encoder is frozen)
        with torch.no_grad():
            obs_cache = encoder.get_cached_image_features(obs)
            next_obs_cache = encoder.get_cached_image_features(next_obs)

        # Full encoder features for V-network (images + state)
        with torch.no_grad():
            obs_feat = encoder(obs, cache=obs_cache)
            next_obs_feat = encoder(next_obs, cache=next_obs_cache)

        # --- V-loss: expectile regression ---
        with torch.no_grad():
            q_target = policy.critic_forward(obs, actions, use_target=True, observation_features=obs_cache)
            q_target_min = q_target.min(dim=0)[0]

        v_pred = v_net(obs_feat)
        diff = q_target_min - v_pred
        weight = torch.where(diff > 0, args.expectile, 1.0 - args.expectile)
        loss_v = (weight * diff.pow(2)).mean()

        optimizer_value.zero_grad()
        loss_v.backward()
        optimizer_value.step()

        # --- Q-loss: TD with V(s') bootstrap ---
        with torch.no_grad():
            v_next = v_net(next_obs_feat)
            td_target = rewards + (1 - dones) * args.discount * v_next

        q_pred = policy.critic_forward(obs, actions, use_target=False, observation_features=obs_cache)
        td_target_exp = td_target.unsqueeze(0).expand_as(q_pred)
        loss_q = F.mse_loss(q_pred, td_target_exp, reduction="none").mean(dim=1).sum()

        optimizer_critic.zero_grad()
        loss_q.backward()
        clip_grad_norm_(policy.critic_ensemble.parameters(), clip_grad_norm_value)
        optimizer_critic.step()

        # Target Q update (EMA)
        policy.update_target_networks()

        # --- Actor loss: advantage-weighted MSE ---
        with torch.no_grad():
            q_val = policy.critic_forward(obs, actions, use_target=True, observation_features=obs_cache)
            advantage = q_val.min(dim=0)[0] - v_net(obs_feat)
            weights = torch.exp(args.beta * advantage).clamp(max=100.0)

        _, _, means = policy.actor(obs, obs_cache)
        predicted = torch.tanh(means)
        loss_actor = (weights * (predicted - actions).pow(2).sum(dim=-1)).mean()

        optimizer_actor.zero_grad()
        loss_actor.backward()
        clip_grad_norm_(policy.actor.parameters(), clip_grad_norm_value)
        optimizer_actor.step()

        # Logging
        pbar.set_postfix({
            "Lv": f"{loss_v.item():.3f}",
            "Lq": f"{loss_q.item():.3f}",
            "La": f"{loss_actor.item():.3f}",
        })

        if step % args.log_freq == 0 and tracker:
            tracker.log({
                "train/loss_v": loss_v.item(),
                "train/loss_q": loss_q.item(),
                "train/loss_actor": loss_actor.item(),
                "train/v_mean": v_pred.mean().item(),
                "train/q_mean": q_pred.mean().item(),
                "train/advantage_mean": advantage.mean().item(),
                "train/weights_mean": weights.mean().item(),
                "train/weights_max": weights.max().item(),
            }, step=step)

        # Checkpoints
        if step > 0 and step % args.save_freq == 0:
            save_policy_checkpoint(policy, args.output, f"step_{step}", v_net)

    pbar.close()

    # Final save
    save_policy_checkpoint(policy, args.output, "final", v_net)
    elapsed = time.time() - t_start
    print(f"\nIQL pretraining complete in {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"Output: {args.output}")

    cleanup_tracker(tracker, sys_monitor)


if __name__ == "__main__":
    main()
