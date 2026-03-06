"""Verify Neural IK accuracy via pinocchio FK.

Loads trained model, predicts joint angles for val samples,
runs FK on predictions and measures EE position error (mm).

Usage:
    python scripts/tools/verify_neural_ik.py \
        --checkpoint outputs/neural_ik \
        --dataset data/neural_ik_data.npz \
        --n 5000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import torch
import torch.nn.functional as F

from so101_lab.policies.rl.neural_ik import NeuralIKNet  # noqa: F401 — needed for torch.load

URDF_PATH = Path(__file__).resolve().parents[2] / "assets/robots/so101/urdf/so101_new_calib.urdf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/neural_ik")
    parser.add_argument("--dataset", default="data/neural_ik_data.npz",
                        help="Dataset to take val samples from (uses last --n samples)")
    parser.add_argument("--n", type=int, default=5000, help="Number of test samples")
    parser.add_argument("--noise-std", type=float, default=0.0,
                        help="Noise on current_joints (0=exact, >0=simulate inference drift)")
    parser.add_argument("--iters", type=int, default=1,
                        help="Iterative refinement passes (1=single, 3=refine 3x)")
    args = parser.parse_args()

    # Load model + meta
    ckpt = Path(args.checkpoint)
    with open(ckpt / "neural_ik_meta.json") as f:
        meta = json.load(f)

    lo = torch.FloatTensor(meta["joint_limits_low"])
    hi = torch.FloatTensor(meta["joint_limits_high"])
    pos_mean = torch.FloatTensor(meta["pos_mean"])
    pos_std = torch.FloatTensor(meta["pos_std"])

    model = torch.load(ckpt / "neural_ik.pt", map_location="cpu", weights_only=False)
    model.eval()

    # Load test samples (last N from dataset = held-out val portion)
    raw = np.load(args.dataset)
    N = min(args.n, len(raw["joints"]))
    joints_gt = torch.FloatTensor(raw["joints"][-N:])   # ground truth joint config
    ee_pos_gt = torch.FloatTensor(raw["ee_pos"][-N:])   # FK of joints_gt
    ee_quat_gt = torch.FloatTensor(raw["ee_quat"][-N:])

    # Build input
    ee_p = (ee_pos_gt - pos_mean) / pos_std
    ee_q = F.normalize(ee_quat_gt, dim=-1)
    sign = ee_q[:, :1].sign()
    sign[sign == 0] = 1.0
    ee_q = ee_q * sign

    mid = (hi + lo) / 2
    rng = (hi - lo) / 2

    def predict(curr_joints):
        curr_norm = (curr_joints - mid) / rng
        with torch.no_grad():
            pred_norm = model(torch.cat([ee_p, ee_q, curr_norm], dim=-1))
        return (pred_norm * rng + mid).clamp(lo, hi)

    curr = joints_gt.clone()
    if args.noise_std > 0:
        curr = (curr + torch.randn_like(curr) * args.noise_std).clamp(lo, hi)

    # Pinocchio for FK error
    pin_model = pin.buildModelFromUrdf(str(URDF_PATH))
    pin_data = pin_model.createData()
    fid = pin_model.getFrameId("gripper_frame_link")
    q = pin.neutral(pin_model)
    target_pos = raw["ee_pos"][-N:]

    def fk_errors(pred_joints_t):
        pj = pred_joints_t.numpy()
        pred_pos = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            q[:5] = pj[i]
            pin.forwardKinematics(pin_model, pin_data, q)
            pin.updateFramePlacements(pin_model, pin_data)
            pred_pos[i] = pin_data.oMf[fid].translation
        return np.linalg.norm(pred_pos - target_pos, axis=1) * 1000  # mm

    def print_stats(errs, label):
        print(f"\n{label}")
        print(f"  mean={errs.mean():.1f}  median={np.median(errs):.1f}"
              f"  90th={np.percentile(errs,90):.1f}  95th={np.percentile(errs,95):.1f}  max={errs.max():.1f} mm")
        print(f"  <1mm={(errs<1).mean()*100:.1f}%  <2mm={(errs<2).mean()*100:.1f}%"
              f"  <3mm={(errs<3).mean()*100:.1f}%  <5mm={(errs<5).mean()*100:.1f}%"
              f"  <10mm={(errs<10).mean()*100:.1f}%  <35mm={(errs<35).mean()*100:.1f}%")

    print(f"n={N}, noise_std={args.noise_std:.2f} rad, iters={args.iters}")

    pred = predict(curr)
    errs = fk_errors(pred)
    print_stats(errs, "iter 1")

    for i in range(2, args.iters + 1):
        pred = predict(pred)
        errs = fk_errors(pred)
        print_stats(errs, f"iter {i}")

    joint_err_deg = (pred - joints_gt).abs().numpy() * (180 / np.pi)
    print(f"\nJoint MAE (final): {joint_err_deg.mean():.2f}°  "
          + "  ".join(f"{v:.2f}°" for v in joint_err_deg.mean(axis=0)))


if __name__ == "__main__":
    main()
