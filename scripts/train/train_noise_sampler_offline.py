"""Offline training for learned noise sampler (backprop through SmolVLA ODE).

No Isaac Sim dependency — runs in lerobot-env.

Usage:
    python scripts/train/train_noise_sampler_offline.py \
        --checkpoint outputs/easy_smolvla_vlm_pretrained_v1/checkpoints/030000/pretrained_model \
        --dataset data/recordings/easy_task_v1 \
        --output outputs/easy_learned_sampler_offline_v1 \
        --noise-dims 32 --batch-size 4
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

# LeRobot src path
def _get_lerobot_src() -> str:
    import os
    from pathlib import Path
    if src := os.environ.get("LEROBOT_SRC"):
        return os.path.expanduser(src)
    env_file = Path(__file__).parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("LEROBOT_SRC="):
                return os.path.expanduser(line.split("=", 1)[1].strip())
    raise RuntimeError("LEROBOT_SRC not set. Add it to .env or set the environment variable.")

_lerobot_src = _get_lerobot_src()
if _lerobot_src not in sys.path:
    sys.path.insert(0, _lerobot_src)

from lerobot.policies.factory import get_policy_class
from lerobot.processor.pipeline import PolicyProcessorPipeline
import lerobot.policies.smolvla.processor_smolvla  # noqa: F401

from so101_lab.policies.rl.learned_noise_sampler import LearnedNoiseSampler

ENV_IMAGE_KEYS = ["observation.images.top", "observation.images.wrist"]


def parse_args():
    parser = argparse.ArgumentParser(description="Offline noise sampler training")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SmolVLA pretrained_model dir")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to LeRobot dataset")
    parser.add_argument("--task", type=str,
                        default="Place the cube into the matching slot on the platform")
    parser.add_argument("--n-action-steps", type=int, default=None)
    parser.add_argument("--noise-dims", type=int, default=6, choices=[6, 32, 1600])
    parser.add_argument("--offline-steps", type=int, default=5000)
    parser.add_argument("--offline-chunk-len", type=int, default=None,
                        help="Action steps to compare (default: n_action_steps)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-freq", type=int, default=100,
                        help="Log + save every N steps")
    return parser.parse_args()


def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load SmolVLA policy with processors."""
    policy_cls = get_policy_class("smolvla")
    policy = policy_cls.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()

    pre = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, "policy_preprocessor.json")
    post = PolicyProcessorPipeline.from_pretrained(
        checkpoint_path, "policy_postprocessor.json")

    policy_img_keys = sorted(policy.config.image_features.keys())
    env_keys = list(ENV_IMAGE_KEYS)
    if all(k in policy_img_keys for k in env_keys):
        img_map = {k: k for k in env_keys}
    else:
        img_map = {}
        for i, pk in enumerate(policy_img_keys):
            img_map[pk] = env_keys[i] if i < len(env_keys) else None

    return policy, pre, post, img_map


def frame_to_smolvla_batch(frame: dict, task_str: str, img_map: dict):
    """Convert LeRobotDataset frame to preprocessor-ready dict."""
    batch = {
        "observation.state": frame["observation.state"],
        "task": task_str,
    }
    for policy_key, env_key in img_map.items():
        if policy_key in frame:
            batch[policy_key] = frame[policy_key]
        elif env_key is not None and env_key in frame:
            batch[env_key] = frame[env_key]
    return batch


def load_offline_dataset(dataset_path: str, sampler, device: str = "cuda"):
    """Load LeRobot dataset and pre-compute VIP embeddings.

    Returns (ds, vip_embeddings, episode_indices).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset as LDS
    import pandas as pd

    ds = LDS(repo_id="local", root=dataset_path)
    n_frames = len(ds)
    print(f"Loaded dataset: {n_frames} frames, {ds.meta.total_episodes} episodes")

    # Episode indices from parquet
    parquet_files = sorted((ds.root / "data").rglob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    episode_indices = df["episode_index"].values

    # VIP embeddings cache
    cache_path = os.path.join(dataset_path, "vip_embeddings.pt")
    if os.path.exists(cache_path):
        print(f"Loading VIP cache from {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        vip_embeddings = data["vip_embeddings"] if isinstance(data, dict) else data
    else:
        print("Computing VIP embeddings (one-time)...")
        vip_embeddings = torch.zeros(n_frames, 2048, dtype=torch.float32)
        batch_sz = 32
        for i in range(0, n_frames, batch_sz):
            end = min(i + batch_sz, n_frames)
            tops, wrists = [], []
            for j in range(i, end):
                frame = ds[j]
                img_top = (frame["observation.images.top"] * 255).to(
                    torch.uint8).permute(1, 2, 0)
                img_wrist = (frame["observation.images.wrist"] * 255).to(
                    torch.uint8).permute(1, 2, 0)
                tops.append(img_top)
                wrists.append(img_wrist)
            tops = torch.stack(tops)
            wrists = torch.stack(wrists)
            with torch.no_grad():
                emb = sampler.vip_backbone(tops, wrists)
            vip_embeddings[i:end] = emb.cpu()
            if (i // batch_sz) % 20 == 0:
                print(f"  VIP: {i}/{n_frames}")
        torch.save(vip_embeddings, cache_path)
        print(f"Saved VIP cache: {cache_path}")

    return ds, vip_embeddings, episode_indices


def build_valid_indices(episode_indices: np.ndarray, chunk_len: int):
    """Frame indices with >= chunk_len frames remaining in same episode."""
    valid = []
    n = len(episode_indices)
    for i in range(n):
        end = i + chunk_len - 1
        if end < n and episode_indices[i] == episode_indices[end]:
            valid.append(i)
    return np.array(valid)


def load_action_chunks(ds, indices, chunk_len: int, action_dim: int,
                       device: str = "cuda"):
    """Load action chunks for a batch of indices.

    Returns (B, chunk_len, action_dim) tensor in env space.
    """
    chunks = []
    for idx in indices:
        actions = torch.stack(
            [ds[int(idx) + j]["action"][:action_dim] for j in range(chunk_len)])
        chunks.append(actions)
    return torch.stack(chunks).to(device)


def prepare_batch(ds, indices, task_str, img_map, pre, policy, device="cuda"):
    """Process B frames through preprocessor individually, stack for batched ODE.

    Returns (images, img_masks, lang_tokens, lang_masks, state_padded).
    """
    images_per_cam = None
    masks_per_cam = None
    states = []
    lang_tokens = None
    lang_masks = None

    for i, idx in enumerate(indices):
        frame = ds[int(idx)]
        batch = pre(frame_to_smolvla_batch(frame, task_str, img_map))
        imgs, masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)

        if images_per_cam is None:
            images_per_cam = [[] for _ in range(len(imgs))]
            masks_per_cam = [[] for _ in range(len(masks))]

        for cam_idx in range(len(imgs)):
            images_per_cam[cam_idx].append(imgs[cam_idx])
            masks_per_cam[cam_idx].append(masks[cam_idx])
        states.append(state)

        if lang_tokens is None:
            lang_tokens = batch["observation.language.tokens"]
            lang_masks = batch["observation.language.attention_mask"]

    B = len(indices)
    images = [torch.cat(cam_imgs, dim=0) for cam_imgs in images_per_cam]
    img_masks = [torch.cat(cam_masks, dim=0) for cam_masks in masks_per_cam]
    state_padded = torch.cat(states, dim=0)
    lang_tokens = lang_tokens.expand(B, -1)
    lang_masks = lang_masks.expand(B, -1)

    return images, img_masks, lang_tokens, lang_masks, state_padded


def train_offline(args):
    device = "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # Load policy
    print(f"\nLoading SmolVLA from {args.checkpoint}")
    policy, pre, post, img_map = load_policy(args.checkpoint)

    if args.n_action_steps is not None:
        old = policy.config.n_action_steps
        policy.config.n_action_steps = args.n_action_steps
        print(f"n_action_steps: {old} → {args.n_action_steps}")

    # Freeze VLA
    for p in policy.parameters():
        p.requires_grad = False

    # Sampler
    sampler = LearnedNoiseSampler(device=device, noise_dims=args.noise_dims)
    if args.resume:
        sampler.load(args.resume)
        print(f"Resumed sampler from {args.resume}")

    trainable = sum(p.numel() for p in sampler.mlp.parameters())
    print(f"Sampler: noise_dims={args.noise_dims}, trainable params={trainable:,}")

    # Dataset
    ds, vip_embeddings, episode_indices = load_offline_dataset(
        args.dataset, sampler, device)

    # Action normalization (ODE output is MEAN_STD normalized)
    stats_path = os.path.join(args.dataset, "meta", "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    action_dim = min(args.noise_dims, 6)  # compare first 6 action dims
    action_mean = torch.tensor(stats["action"]["mean"][:action_dim], device=device)
    action_std = torch.tensor(stats["action"]["std"][:action_dim], device=device)

    # Chunk and indices
    chunk_len = args.offline_chunk_len or policy.config.n_action_steps
    noise_shape = (args.batch_size, policy.config.chunk_size,
                   policy.config.max_action_dim)
    valid_indices = build_valid_indices(episode_indices, chunk_len)

    # Held-out validation set (100 random frames)
    rng = np.random.default_rng(args.seed)
    val_size = min(100, len(valid_indices) // 10)
    perm = rng.permutation(len(valid_indices))
    val_set = valid_indices[perm[:val_size]]
    train_set = valid_indices[perm[val_size:]]

    optimizer = torch.optim.Adam(sampler.mlp.parameters(), lr=args.lr)
    best_val_loss = float("inf")
    log_path = os.path.join(args.output, "train_log.jsonl")
    log_file = open(log_path, "a")

    B = args.batch_size

    # Baseline val loss (N(0,I) noise, before training)
    baseline_val = compute_val_loss(
        val_set, ds, vip_embeddings, sampler, policy, pre, img_map,
        args.task, noise_shape, chunk_len, action_dim,
        action_mean, action_std, device, B)

    print(f"\n{'='*60}")
    print(f"Offline Training (backprop through ODE)")
    print(f"{'='*60}")
    print(f"  Dataset: {len(ds)} frames, {len(train_set)} train, {val_size} val")
    print(f"  Chunk len: {chunk_len}, noise_dims: {args.noise_dims}")
    print(f"  Batch size: {B} (true GPU batch)")
    print(f"  Steps: {args.offline_steps}, LR: {args.lr}")
    print(f"  Baseline val_loss: {baseline_val:.5f} (N(0,I) noise)")
    print(f"{'='*60}\n")

    from tqdm import tqdm

    pbar = tqdm(range(args.offline_steps), desc="Offline", dynamic_ncols=True)
    running_loss = None

    for step in pbar:
        optimizer.zero_grad()

        # Sample batch of valid indices
        batch_indices = train_set[rng.integers(len(train_set), size=B)]

        # VIP + state for sampler
        vip_batch = vip_embeddings[batch_indices].to(device)
        state_batch = torch.stack(
            [ds[int(idx)]["observation.state"][:6] for idx in batch_indices]
        ).to(device).float()

        # Sampler → x_0 (differentiable)
        x_0 = sampler.sample_x0_differentiable(noise_shape, vip_batch, state_batch)

        # Prepare SmolVLA inputs (batched)
        images, img_masks, lang_tokens, lang_masks, state_padded = prepare_batch(
            ds, batch_indices, args.task, img_map, pre, policy, device)

        # Forward ODE with gradient
        predicted = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state_padded, noise=x_0)
        predicted_actions = predicted[:, :chunk_len, :action_dim]

        # Expert actions (env space → normalized)
        expert_chunk = load_action_chunks(
            ds, batch_indices, chunk_len, action_dim, device)
        expert_norm = (expert_chunk - action_mean) / action_std

        loss = F.mse_loss(predicted_actions, expert_norm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sampler.mlp.parameters(), 1.0)
        optimizer.step()

        cur_loss = loss.item()
        running_loss = cur_loss if running_loss is None else (
            0.95 * running_loss + 0.05 * cur_loss)
        grad_norm = sum(
            p.grad.norm().item() ** 2
            for p in sampler.mlp.parameters() if p.grad is not None
        ) ** 0.5
        pbar.set_postfix(loss=f"{running_loss:.5f}", grad=f"{grad_norm:.4f}")

        # Periodic eval + save
        if (step + 1) % args.eval_freq == 0:
            val_loss = compute_val_loss(
                val_set, ds, vip_embeddings, sampler, policy, pre, img_map,
                args.task, noise_shape, chunk_len, action_dim,
                action_mean, action_std, device, B)
            pbar.write(f"  Step {step+1}: train={cur_loss:.5f}, val={val_loss:.5f}")

            log_entry = {
                "step": step + 1, "train_loss": cur_loss, "val_loss": val_loss}
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                sampler.save(
                    os.path.join(args.output, "noise_prior.pt"),
                    extra={"val_loss": val_loss, "step": step + 1},
                )
                pbar.write(f"  Saved best (val_loss={val_loss:.5f})")
        elif (step + 1) % 100 == 0:
            log_entry = {"step": step + 1, "train_loss": cur_loss}
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

    log_file.close()
    sampler.save(
        os.path.join(args.output, "noise_prior_final.pt"),
        extra={"total_steps": args.offline_steps, "best_val_loss": best_val_loss},
    )
    print(f"\nOffline training complete. Best val loss: {best_val_loss:.5f}")


@torch.no_grad()
def compute_val_loss(val_set, ds, vip_embeddings, sampler, policy, pre, img_map,
                     task_str, noise_shape, chunk_len, action_dim,
                     action_mean, action_std, device, batch_size):
    """Compute MSE on held-out frames (deterministic, mu only)."""
    sampler.eval()
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(val_set), batch_size):
        batch_indices = val_set[start:start + batch_size]
        B = len(batch_indices)
        shape = (B, noise_shape[1], noise_shape[2])

        vip_batch = vip_embeddings[batch_indices].to(device)
        state_batch = torch.stack(
            [ds[int(idx)]["observation.state"][:6] for idx in batch_indices]
        ).to(device).float()

        # Deterministic: use mu only (no sigma noise)
        mu, _ = sampler.forward(vip_batch, state_batch)
        x_0 = torch.randn(shape, device=device)
        x_0 = x_0.clone()
        if sampler.noise_dims <= 32:
            x_0[:, 0, :sampler.noise_dims] = mu
        else:
            x_0[:, :, :] = mu.reshape(B, 50, 32)

        images, img_masks, lang_tokens, lang_masks, state_padded = prepare_batch(
            ds, batch_indices, task_str, img_map, pre, policy, device)

        predicted = policy.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state_padded, noise=x_0)
        predicted_actions = predicted[:, :chunk_len, :action_dim]

        expert_chunk = load_action_chunks(
            ds, batch_indices, chunk_len, action_dim, device)
        expert_norm = (expert_chunk - action_mean) / action_std

        total_loss += F.mse_loss(predicted_actions, expert_norm).item()
        n_batches += 1

    sampler.train()
    return total_loss / max(n_batches, 1)


if __name__ == "__main__":
    args = parse_args()
    train_offline(args)
