#!/usr/bin/env python3
"""Save snapshots from USB cameras for sim-vs-real comparison.

Usage:
    python scripts/tools/camera_snapshot.py --cameras 0 2 --names top wrist
"""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 2])
    parser.add_argument("--names", nargs="+", type=str, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    names = args.names or [f"cam_{i}" for i in args.cameras]

    for idx, name in zip(args.cameras, names):
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not cap.isOpened():
            print(f"Cannot open camera {idx}")
            continue

        # Warm up (skip first frames)
        for _ in range(10):
            cap.read()

        ret, frame = cap.read()
        if ret:
            path = f"{args.output_dir}/snapshot_{name}.png"
            cv2.imwrite(path, frame)
            print(f"Saved: {path} ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"Failed to capture from camera {idx}")

        cap.release()


if __name__ == "__main__":
    main()
