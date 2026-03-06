"""Generate Neural IK training dataset via pinocchio FK.

Usage:
    python scripts/tools/generate_ik_dataset.py \
        --output data/neural_ik_data.npz --samples 1000000
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pinocchio as pin

URDF_PATH = Path(__file__).resolve().parents[2] / "assets/robots/so101/urdf/so101_new_calib.urdf"

# Arm joint limits (rad) — from SO101_USD_JOINT_LIMITS in so101_lab/assets/robots/so101.py
JOINT_LIMITS_LOW = np.array([-1.91986, -1.74533, -1.68977, -1.65806, -2.74017], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([1.91986, 1.74533, 1.68977, 1.65806, 2.82743], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/neural_ik_data.npz")
    parser.add_argument("--samples", type=int, default=1_000_000)
    args = parser.parse_args()

    model = pin.buildModelFromUrdf(str(URDF_PATH))
    data = model.createData()
    frame_id = model.getFrameId("gripper_frame_link")

    q = pin.neutral(model)
    N = args.samples

    joints = np.random.uniform(JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH, size=(N, 5)).astype(np.float32)
    ee_pos = np.zeros((N, 3), dtype=np.float32)
    ee_quat = np.zeros((N, 4), dtype=np.float32)  # wxyz (IsaacLab convention)

    print(f"Generating {N:,} samples...")
    t0 = time.time()
    for i in range(N):
        if i % 100_000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {i:,}/{N:,}  ({i/elapsed:.0f} samples/sec)")

        q[:5] = joints[i]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T = data.oMf[frame_id]

        ee_pos[i] = T.translation

        quat = pin.Quaternion(T.rotation)
        # Store wxyz, canonical sign: w >= 0
        w, x, y, z = quat.w, quat.x, quat.y, quat.z
        if w < 0:
            w, x, y, z = -w, -x, -y, -z
        ee_quat[i] = [w, x, y, z]

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({N/elapsed:.0f} samples/sec)")
    print(f"EE pos range: {ee_pos.min(axis=0)} .. {ee_pos.max(axis=0)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        joints=joints,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        joint_limits_low=JOINT_LIMITS_LOW,
        joint_limits_high=JOINT_LIMITS_HIGH,
    )
    print(f"Saved: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
