"""Train Neural IK MLP on FK dataset.

Usage:
    python scripts/tools/train_neural_ik.py \
        --dataset data/neural_ik_data.npz \
        --output outputs/neural_ik
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from so101_lab.policies.rl.neural_ik import NeuralIKNet


def normalize_joints(q, lo, hi):
    mid = (hi + lo) / 2
    rng = (hi - lo) / 2
    return (q - mid) / rng


def denormalize_joints(q_norm, lo, hi):
    mid = (hi + lo) / 2
    rng = (hi - lo) / 2
    return q_norm * rng + mid


def build_input(ee_pos, ee_quat, curr_joints, pos_mean, pos_std, lo, hi):
    """Concatenate normalized [ee_pos(3), ee_quat(4), curr_joints(5)] → (N,12)."""
    ee_p = (ee_pos - pos_mean) / pos_std
    ee_q = F.normalize(ee_quat, dim=-1)
    # Canonical sign: w >= 0
    sign = ee_q[:, :1].sign()
    sign[sign == 0] = 1.0
    ee_q = ee_q * sign
    curr = normalize_joints(curr_joints, lo, hi)
    return torch.cat([ee_p, ee_q, curr], dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/neural_ik_data.npz")
    parser.add_argument("--output", default="outputs/neural_ik")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.3,
                        help="Std of noise added to joints to form current_joints input")
    parser.add_argument("--val-split", type=float, default=0.02)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    raw = np.load(args.dataset)
    joints = torch.FloatTensor(raw["joints"])   # (N, 5)
    ee_pos = torch.FloatTensor(raw["ee_pos"])   # (N, 3)
    ee_quat = torch.FloatTensor(raw["ee_quat"]) # (N, 4) wxyz
    lo = torch.FloatTensor(raw["joint_limits_low"]).to(device)
    hi = torch.FloatTensor(raw["joint_limits_high"]).to(device)

    pos_mean = ee_pos.mean(0).to(device)
    pos_std = ee_pos.std(0).clamp(min=1e-6).to(device)

    N = len(joints)
    n_val = int(N * args.val_split)
    perm = torch.randperm(N)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    j_tr = joints[train_idx].to(device)
    p_tr = ee_pos[train_idx].to(device)
    q_tr = ee_quat[train_idx].to(device)

    j_val = joints[val_idx].to(device)
    p_val = ee_pos[val_idx].to(device)
    q_val = ee_quat[val_idx].to(device)

    model = NeuralIKNet(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Architecture: {args.num_layers}×{args.hidden_size}  params={n_params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_tr = len(j_tr)
    steps = n_tr // args.batch_size
    best_val = float("inf")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train: {n_tr:,}  Val: {n_val:,}  Steps/epoch: {steps}")

    for epoch in range(args.epochs):
        model.train()
        idx_perm = torch.randperm(n_tr, device=device)
        tr_loss = 0.0
        for s in range(steps):
            b = idx_perm[s * args.batch_size:(s + 1) * args.batch_size]
            noise = torch.randn(len(b), 5, device=device) * args.noise_std
            curr = (j_tr[b] + noise).clamp(lo, hi)
            x = build_input(p_tr[b], q_tr[b], curr, pos_mean, pos_std, lo, hi)
            pred = model(x)
            target = normalize_joints(j_tr[b], lo, hi)
            loss = F.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        scheduler.step()
        tr_loss /= steps

        model.eval()
        with torch.no_grad():
            x_val = build_input(p_val, q_val, j_val, pos_mean, pos_std, lo, hi)
            pred_val = model(x_val)
            val_loss = F.mse_loss(pred_val, normalize_joints(j_val, lo, hi)).item()

            # Joint-space MAE in degrees (on first 2000 val samples)
            pred_rad = denormalize_joints(pred_val[:2000], lo, hi).clamp(lo, hi)
            mae_deg = (pred_rad - j_val[:2000]).abs().mean().item() * 57.3

        print(f"Epoch {epoch+1:3d}/{args.epochs}  train={tr_loss:.5f}  val={val_loss:.5f}  mae={mae_deg:.2f}°")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model, out_dir / "neural_ik.pt")

    meta = {
        "joint_names": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        "joint_limits_low": lo.cpu().tolist(),
        "joint_limits_high": hi.cpu().tolist(),
        "pos_mean": pos_mean.cpu().tolist(),
        "pos_std": pos_std.cpu().tolist(),
    }
    with open(out_dir / "neural_ik_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved best model (val={best_val:.5f}) to {out_dir}/")


if __name__ == "__main__":
    main()
