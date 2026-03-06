"""Compute IQL advantage weights for weighted BC.

Loads trained Q/V critics and cached VIP embeddings, computes
A(s,a) = Q(s,a) - V(s) for every frame, converts to weights
w = exp(A/β), and writes `iql_weight` column into the dataset parquet.

Uses cached VIP embeddings from train_iql_critics.py (vip_embeddings.pt).

Usage:
    eval "$(./activate_lerobot.sh)"
    python scripts/tools/compute_iql_weights.py \
        --dataset data/recordings/figure_shape_placement_v5_vip \
        --critics outputs/iql_critics_v1/final/critics.pt \
        --beta 1.0

    # Dry run (print stats, don't write)
    python scripts/tools/compute_iql_weights.py \
        --dataset data/recordings/figure_shape_placement_v5_vip \
        --critics outputs/iql_critics_v1/final/critics.pt \
        --beta 1.0 --dry-run
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Action/state bounds (env space)
ACTION_MIN = np.array([-100, -100, -100, -100, -100, 0], dtype=np.float32)
ACTION_MAX = np.array([100, 100, 100, 100, 100, 100], dtype=np.float32)
STATE_MIN = ACTION_MIN
STATE_MAX = ACTION_MAX


def normalize_to_policy(x: np.ndarray, x_min: np.ndarray, x_max: np.ndarray) -> np.ndarray:
    return 2 * (x - x_min) / (x_max - x_min) - 1


def parse_args():
    p = argparse.ArgumentParser(description="Compute IQL advantage weights")
    p.add_argument("--dataset", type=str, required=True, help="LeRobot dataset path")
    p.add_argument("--critics", type=str, required=True, help="Path to critics.pt checkpoint")
    p.add_argument("--beta", type=float, default=1.0,
                   help="Advantage temperature (higher → flatter weights)")
    p.add_argument("--max-weight", type=float, default=np.exp(5),
                   help="Clamp max weight (default: exp(5) ≈ 148)")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="Batch size for inference (embeddings are small)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true",
                   help="Print stats without writing to parquet")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_path = args.dataset

    # 1. Load critics checkpoint
    print(f"Loading critics: {args.critics}")
    ckpt = torch.load(args.critics, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    print(f"  Config: hidden_dims={config['hidden_dims']}, "
          f"reward_scale={config.get('reward_scale', 1.0):.2f}")

    # 2. Build MLP heads and load weights
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))
    from train_iql_critics import VHead, QHead

    v_net = VHead(hidden_dims=config["hidden_dims"]).to(args.device)
    q1_net = QHead(hidden_dims=config["hidden_dims"]).to(args.device)
    q2_net = QHead(hidden_dims=config["hidden_dims"]).to(args.device)

    v_net.load_state_dict(ckpt["v_net"])
    q1_net.load_state_dict(ckpt["q1_net"])
    q2_net.load_state_dict(ckpt["q2_net"])
    v_net.eval()
    q1_net.eval()
    q2_net.eval()
    print("  Networks loaded")

    # 3. Load cached image embeddings
    encoder_type = config.get("encoder_type", "vip")
    cache_path = os.path.join(dataset_path, f"{encoder_type}_embeddings.pt")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Image embeddings not found: {cache_path}\n"
            "Run train_iql_critics.py first to cache embeddings."
        )
    embeddings = torch.load(cache_path, map_location="cpu", weights_only=True)
    total_frames = embeddings.shape[0]
    print(f"  Embeddings: {cache_path} ({total_frames} frames, {embeddings.shape[1]}-dim)")

    # 4. Load states/actions from parquet
    data_parquet = os.path.join(dataset_path, "data", "chunk-000", "file-000.parquet")
    df = pd.read_parquet(data_parquet)

    states = np.stack(df["observation.state"].values).astype(np.float32)
    actions = np.stack(df["action"].values).astype(np.float32)
    states_norm = torch.from_numpy(normalize_to_policy(states, STATE_MIN, STATE_MAX))
    actions_norm = torch.from_numpy(normalize_to_policy(actions, ACTION_MIN, ACTION_MAX))

    print(f"Dataset: {dataset_path} ({total_frames} frames)")

    # 5. Compute advantages in batches (fast — no video decoding)
    print(f"\nComputing advantages (β={args.beta})...")
    advantages = torch.zeros(total_frames, dtype=torch.float32)

    for start in tqdm(range(0, total_frames, args.batch_size), desc="IQL weights"):
        end = min(start + args.batch_size, total_frames)
        emb = embeddings[start:end].to(args.device)
        state = states_norm[start:end].to(args.device)
        action = actions_norm[start:end].to(args.device)

        with torch.no_grad():
            v = v_net(emb, state).squeeze(-1)
            q1 = q1_net(emb, state, action).squeeze(-1)
            q2 = q2_net(emb, state, action).squeeze(-1)
            q_min = torch.min(q1, q2)
            advantages[start:end] = (q_min - v).cpu()

    advantages = advantages.numpy()

    # 6. Compute weights
    weights = np.exp(advantages / args.beta)
    weights = np.clip(weights, 0.0, args.max_weight)

    # 7. Stats
    print(f"\nAdvantage stats:")
    print(f"  mean={advantages.mean():.4f}, std={advantages.std():.4f}")
    print(f"  min={advantages.min():.4f}, max={advantages.max():.4f}")
    print(f"  p5={np.percentile(advantages, 5):.4f}, p95={np.percentile(advantages, 95):.4f}")

    print(f"\nWeight stats (β={args.beta}):")
    print(f"  mean={weights.mean():.4f}, std={weights.std():.4f}")
    print(f"  min={weights.min():.4f}, max={weights.max():.4f}")
    print(f"  p5={np.percentile(weights, 5):.4f}, p95={np.percentile(weights, 95):.4f}")
    print(f"  >1.0: {(weights > 1.0).sum()} ({(weights > 1.0).mean()*100:.1f}%)")
    print(f"  <1.0: {(weights < 1.0).sum()} ({(weights < 1.0).mean()*100:.1f}%)")
    print(f"  clipped to max: {(weights >= args.max_weight).sum()}")

    if args.dry_run:
        print("\n--dry-run: not writing to parquet")
        return

    # 8. Write to parquet
    print(f"\nWriting iql_weight to {data_parquet}...")
    df["iql_weight"] = weights.astype(np.float32)
    df.to_parquet(data_parquet, index=False)

    # Verify
    df_check = pd.read_parquet(data_parquet)
    assert "iql_weight" in df_check.columns
    assert np.allclose(df_check["iql_weight"].values, weights, atol=1e-6)
    print("  Written and verified")

    # 9. Update info.json
    info_path = os.path.join(dataset_path, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if "iql_weight" not in info.get("features", {}):
            info["features"]["iql_weight"] = {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            }
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            print(f"  Added iql_weight to {info_path}")

    # 10. Update stats.json
    stats_path = os.path.join(dataset_path, "meta", "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        stats["iql_weight"] = {
            "mean": [float(weights.mean())],
            "std": [float(weights.std())],
            "min": [float(weights.min())],
            "max": [float(weights.max())],
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Updated {stats_path}")

    print(f"\nDone! β={args.beta}, mean_weight={weights.mean():.3f}")


if __name__ == "__main__":
    main()
