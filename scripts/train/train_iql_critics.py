"""Train standalone IQL critics (Q + V) with image encoder.

Trains Q(s,a) and V(s) networks on a LeRobot dataset.
Image encoder (VIP or ImageNet ResNet50) is used as state representation.
Rewards come from the dataset's next.reward column (sim or VIP).

Two phases:
  1. Cache: one-time image encoding of all frames → .pt file (~7 min)
  2. Train: fast MLP training on cached embeddings (~5 min for 50k steps)

Pipeline:
  1. prepare_reward_dataset.py  → copies dataset + writes VIP rewards
  2. train_iql_critics.py       → trains Q and V (this script)
  3. compute_iql_weights.py     → computes w = exp(A/β) per frame
  4. lerobot-train              → weighted BC with iql_weight column

Usage:
    eval "$(./activate_lerobot.sh)"
    python scripts/train/train_iql_critics.py \
        --dataset data/recordings/figure_shape_placement_v5_vip \
        --output outputs/iql_critics_v1 \
        --num-steps 50000 --batch-size 256 \
        --reward-normalize
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Action/state bounds (env space)
ACTION_MIN = np.array([-100, -100, -100, -100, -100, 0], dtype=np.float32)
ACTION_MAX = np.array([100, 100, 100, 100, 100, 100], dtype=np.float32)
STATE_MIN = ACTION_MIN
STATE_MAX = ACTION_MAX


def normalize_to_policy(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    """Normalize from env space to [-1, 1]."""
    return 2 * (x - x_min) / (x_max - x_min) - 1


def _load_imagenet_encoder(device: str = "cuda") -> torch.nn.Module:
    """Load ImageNet-pretrained ResNet50 with 2048→1024 projection (same dim as VIP)."""
    import torchvision
    convnet = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    convnet.fc = torch.nn.Linear(2048, 1024)
    torch.nn.init.xavier_uniform_(convnet.fc.weight)
    convnet.to(device).eval()
    return convnet


def cache_image_embeddings(dataset_path: str, cache_path: str,
                           encoder_type: str = "vip", device: str = "cuda",
                           batch_size: int = 64) -> torch.Tensor:
    """Encode all frames with image encoder and cache to disk.

    Returns (N, 2048) tensor: concat of top(1024) + wrist(1024) embeddings.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached image embeddings: {cache_path}")
        embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  Shape: {embeddings.shape}")
        return embeddings

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import torchvision.transforms as T

    if encoder_type == "vip":
        from so101_lab.rewards.vip_reward import _load_vip_encoder
        encoder = _load_vip_encoder(device)
    else:
        encoder = _load_imagenet_encoder(device)

    ds = LeRobotDataset(repo_id="local", root=dataset_path)
    total_frames = len(ds)
    print(f"Caching image embeddings ({encoder_type}) for {total_frames} frames...")

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    embeddings = torch.zeros(total_frames, 2048, dtype=torch.float32)

    for i in tqdm(range(total_frames), desc="Image encoding"):
        frame = ds[i]

        with torch.no_grad():
            # Top camera
            img_top = (frame["observation.images.top"] * 255).to(torch.uint8)
            img_top = preprocess(img_top).unsqueeze(0).to(device)
            emb_top = encoder(img_top).squeeze(0).cpu()  # (1024,)

            # Wrist camera
            img_wrist = (frame["observation.images.wrist"] * 255).to(torch.uint8)
            img_wrist = preprocess(img_wrist).unsqueeze(0).to(device)
            emb_wrist = encoder(img_wrist).squeeze(0).cpu()  # (1024,)

        embeddings[i] = torch.cat([emb_top, emb_wrist])  # (2048,)

        if (i + 1) % 10000 == 0:
            print(f"  [{i+1}/{total_frames}]")

    torch.save(embeddings, cache_path)
    size_mb = os.path.getsize(cache_path) / 1024 / 1024
    print(f"  Saved: {cache_path} ({size_mb:.1f} MB)")
    return embeddings


class IQLCachedDataset(Dataset):
    """Dataset of (emb, emb_next, state, action, reward, done) transitions.

    Uses pre-cached VIP embeddings instead of decoding video on the fly.
    """

    def __init__(
        self,
        dataset_path: str,
        embeddings: torch.Tensor,
        reward_normalize: bool = False,
        reward_clip: tuple[float, float] = (-2.0, 0.0),
    ):
        self.embeddings = embeddings  # (N, 2048)

        # Load parquet for state/action/reward
        data_parquet = os.path.join(dataset_path, "data", "chunk-000", "file-000.parquet")
        df = pd.read_parquet(data_parquet)

        self.episode_indices = df["episode_index"].values
        self.states = np.stack(df["observation.state"].values).astype(np.float32)
        self.actions = np.stack(df["action"].values).astype(np.float32)
        self.rewards = df["next.reward"].values.astype(np.float32)

        total_frames = len(df)
        print(f"IQLCachedDataset: {total_frames} frames")
        print(f"  Raw reward range: [{self.rewards.min():.4f}, {self.rewards.max():.4f}], "
              f"mean={self.rewards.mean():.4f}")

        # Normalize rewards
        self.reward_scale = 1.0
        if reward_normalize:
            p5 = float(np.percentile(self.rewards, 5))
            self.reward_scale = max(-p5, 1.0)
            self.rewards = self.rewards / self.reward_scale
            self.rewards = np.clip(self.rewards, reward_clip[0], reward_clip[1])
            print(f"  Normalized: scale={self.reward_scale:.2f}, clip={reward_clip}")
            print(f"  After norm: [{self.rewards.min():.4f}, {self.rewards.max():.4f}], "
                  f"mean={self.rewards.mean():.4f}")

        # Normalize states and actions to [-1, 1]
        self.states_norm = normalize_to_policy(self.states, STATE_MIN, STATE_MAX)
        self.actions_norm = normalize_to_policy(self.actions, ACTION_MIN, ACTION_MAX)

        self.n_transitions = total_frames - 1

    def __len__(self):
        return self.n_transitions

    def __getitem__(self, idx):
        done = self.episode_indices[idx] != self.episode_indices[idx + 1]
        return {
            "emb": self.embeddings[idx],                                 # (2048,)
            "next_emb": self.embeddings[idx + 1],                       # (2048,)
            "state": torch.from_numpy(self.states_norm[idx]),            # (6,)
            "next_state": torch.from_numpy(self.states_norm[idx + 1]),   # (6,)
            "action": torch.from_numpy(self.actions_norm[idx]),          # (6,)
            "reward": torch.tensor(self.rewards[idx]),                   # scalar
            "done": torch.tensor(float(done)),                           # scalar
        }


class VHead(torch.nn.Module):
    """V(s) MLP head on pre-computed VIP embeddings + state."""

    def __init__(self, emb_dim: int = 2048, state_dim: int = 6,
                 hidden_dims: list[int] = [256, 256]):
        super().__init__()
        input_dim = emb_dim + state_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev, h),
                torch.nn.LayerNorm(h),
                torch.nn.SiLU(),
            ])
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, state], dim=-1)
        return self.net(x)  # (B, 1)


class QHead(torch.nn.Module):
    """Q(s,a) MLP head on pre-computed VIP embeddings + state + action."""

    def __init__(self, emb_dim: int = 2048, state_dim: int = 6, action_dim: int = 6,
                 hidden_dims: list[int] = [256, 256]):
        super().__init__()
        input_dim = emb_dim + state_dim + action_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev, h),
                torch.nn.LayerNorm(h),
                torch.nn.SiLU(),
            ])
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, emb: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb, state, action], dim=-1)
        return self.net(x)  # (B, 1)


def parse_args():
    p = argparse.ArgumentParser(description="Train IQL critics with image encoder")
    p.add_argument("--dataset", type=str, required=True, help="LeRobot dataset path")
    p.add_argument("--output", type=str, default="outputs/iql_critics_v1")
    p.add_argument("--num-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--expectile", type=float, default=0.7, help="Expectile tau for V-loss")
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--target-tau", type=float, default=0.005, help="Target Q EMA rate")
    p.add_argument("--reward-normalize", action="store_true",
                   help="Normalize VIP rewards to ~[-1,0] using p5 percentile")
    p.add_argument("--reward-clip", type=float, nargs=2, default=[-2.0, 0.0],
                   help="Clip range after normalization (default: -2.0 0.0)")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--save-freq", type=int, default=10000)
    from so101_lab.utils.tracker import add_tracker_args
    add_tracker_args(p, default_project="so101-iql-critics")
    p.add_argument("--encoder", type=str, default="vip", choices=["vip", "imagenet"],
                   help="Image encoder: vip (VIP ResNet50) or imagenet (torchvision pretrained)")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # 1. Cache image embeddings (one-time, ~15-30 min for 194k frames)
    cache_path = os.path.join(args.dataset, f"{args.encoder}_embeddings.pt")
    embeddings = cache_image_embeddings(args.dataset, cache_path, encoder_type=args.encoder)

    # 2. Build dataset and dataloader
    dataset = IQLCachedDataset(
        args.dataset, embeddings,
        reward_normalize=args.reward_normalize,
        reward_clip=tuple(args.reward_clip),
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )
    data_iter = iter(dataloader)

    # 3. Build MLP heads (no VIP backbone needed during training)
    v_net = VHead(hidden_dims=args.hidden_dims).cuda()
    q1_net = QHead(hidden_dims=args.hidden_dims).cuda()
    q2_net = QHead(hidden_dims=args.hidden_dims).cuda()
    q1_target = QHead(hidden_dims=args.hidden_dims).cuda()
    q2_target = QHead(hidden_dims=args.hidden_dims).cuda()
    q1_target.load_state_dict(q1_net.state_dict())
    q2_target.load_state_dict(q2_net.state_dict())

    optimizer_v = torch.optim.Adam(v_net.parameters(), lr=args.value_lr)
    optimizer_q = torch.optim.Adam(
        list(q1_net.parameters()) + list(q2_net.parameters()),
        lr=args.critic_lr,
    )

    total_params_v = sum(p.numel() for p in v_net.parameters())
    total_params_q = sum(p.numel() for p in q1_net.parameters()) * 2
    print(f"\nV-network params: {total_params_v:,}")
    print(f"Q-networks params: {total_params_q:,} (2x twin)")

    # 4. Tracker
    from so101_lab.utils.tracker import setup_tracker, cleanup_tracker
    run_name = os.path.basename(os.path.normpath(args.output))
    tracker, sys_monitor = setup_tracker(args, run_name)

    # 5. Training loop
    reward_scale = dataset.reward_scale

    print(f"\nStarting IQL critic training: {args.num_steps} steps")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Expectile τ: {args.expectile}")
    print(f"  Discount γ: {args.discount}")
    print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Reward normalize: {args.reward_normalize} (scale={reward_scale:.2f})")
    print()

    pbar = tqdm(range(args.num_steps), desc="IQL Critics", unit="step")
    t_start = time.time()

    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        emb = batch["emb"].cuda()
        next_emb = batch["next_emb"].cuda()
        states = batch["state"].cuda()
        next_states = batch["next_state"].cuda()
        actions = batch["action"].cuda()
        rewards = batch["reward"].cuda()
        dones = batch["done"].cuda()

        # --- V-loss: expectile regression ---
        with torch.no_grad():
            q1_tgt = q1_target(emb, states, actions).squeeze(-1)
            q2_tgt = q2_target(emb, states, actions).squeeze(-1)
            q_target_min = torch.min(q1_tgt, q2_tgt)

        v_pred = v_net(emb, states).squeeze(-1)
        diff = q_target_min - v_pred
        weight = torch.where(diff > 0, args.expectile, 1.0 - args.expectile)
        loss_v = (weight * diff.pow(2)).mean()

        optimizer_v.zero_grad()
        loss_v.backward()
        optimizer_v.step()

        # --- Q-loss: TD with V(s') bootstrap ---
        with torch.no_grad():
            v_next = v_net(next_emb, next_states).squeeze(-1)
            td_target = rewards + (1 - dones) * args.discount * v_next

        q1_pred = q1_net(emb, states, actions).squeeze(-1)
        q2_pred = q2_net(emb, states, actions).squeeze(-1)
        loss_q1 = F.mse_loss(q1_pred, td_target)
        loss_q2 = F.mse_loss(q2_pred, td_target)
        loss_q = loss_q1 + loss_q2

        optimizer_q.zero_grad()
        loss_q.backward()
        optimizer_q.step()

        # --- Target Q update (EMA) ---
        with torch.no_grad():
            tau = args.target_tau
            for p, p_tgt in zip(q1_net.parameters(), q1_target.parameters()):
                p_tgt.data.mul_(1 - tau).add_(p.data, alpha=tau)
            for p, p_tgt in zip(q2_net.parameters(), q2_target.parameters()):
                p_tgt.data.mul_(1 - tau).add_(p.data, alpha=tau)

        # --- Logging ---
        pbar.set_postfix({
            "Lv": f"{loss_v.item():.4f}",
            "Lq": f"{loss_q.item():.4f}",
            "V": f"{v_pred.mean().item():.3f}",
            "Q": f"{q1_pred.mean().item():.3f}",
        })

        if step % args.log_freq == 0 and tracker:
            with torch.no_grad():
                advantage = q_target_min - v_pred
            tracker.log({
                "train/loss_v": loss_v.item(),
                "train/loss_q": loss_q.item(),
                "train/loss_q1": loss_q1.item(),
                "train/loss_q2": loss_q2.item(),
                "train/v_mean": v_pred.mean().item(),
                "train/v_std": v_pred.std().item(),
                "train/q1_mean": q1_pred.mean().item(),
                "train/q2_mean": q2_pred.mean().item(),
                "train/advantage_mean": advantage.mean().item(),
                "train/advantage_std": advantage.std().item(),
                "train/reward_mean": rewards.mean().item(),
                "train/td_target_mean": td_target.mean().item(),
            }, step=step)

        # --- Checkpoints ---
        if step > 0 and step % args.save_freq == 0:
            save_critics(args.output, f"step_{step}",
                         v_net, q1_net, q2_net, q1_target, q2_target,
                         args, reward_scale)

    pbar.close()

    save_critics(args.output, "final",
                 v_net, q1_net, q2_net, q1_target, q2_target,
                 args, reward_scale)

    elapsed = time.time() - t_start
    print(f"\nIQL critic training complete in {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"Output: {args.output}")

    cleanup_tracker(tracker, sys_monitor)


def save_critics(output_dir, name, v_net, q1_net, q2_net, q1_target, q2_target,
                  args, reward_scale):
    """Save critic checkpoints."""
    ckpt_dir = os.path.join(output_dir, name)
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.save({
        "v_net": v_net.state_dict(),
        "q1_net": q1_net.state_dict(),
        "q2_net": q2_net.state_dict(),
        "q1_target": q1_target.state_dict(),
        "q2_target": q2_target.state_dict(),
        "config": {
            "hidden_dims": args.hidden_dims,
            "state_dim": 6,
            "action_dim": 6,
            "emb_dim": 2048,
            "expectile": args.expectile,
            "discount": args.discount,
            "encoder_type": args.encoder,
            "image_keys": ["observation.images.top", "observation.images.wrist"],
            "reward_normalize": args.reward_normalize,
            "reward_scale": reward_scale,
            "reward_clip": list(args.reward_clip),
        },
    }, os.path.join(ckpt_dir, "critics.pt"))

    print(f"  Saved: {ckpt_dir}/critics.pt")


if __name__ == "__main__":
    main()
