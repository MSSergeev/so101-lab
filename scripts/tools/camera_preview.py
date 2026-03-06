#!/usr/bin/env python3
"""Live preview of USB cameras. Use to align real cameras with sim setup.

Usage:
    python scripts/tools/camera_preview.py                    # all cameras
    python scripts/tools/camera_preview.py --cameras 0 2      # specific indices
    python scripts/tools/camera_preview.py --cameras 0 --name top
"""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Camera preview")
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 2],
                        help="Camera indices (default: 0 2)")
    parser.add_argument("--names", nargs="+", type=str, default=None,
                        help="Camera names (default: cam_0, cam_1, ...)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    names = args.names or [f"cam_{i}" for i in args.cameras]
    if len(names) < len(args.cameras):
        names += [f"cam_{i}" for i in args.cameras[len(names):]]

    caps = []
    for idx in args.cameras:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, args.fps)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if not cap.isOpened():
            print(f"Cannot open camera {idx}")
            continue
        caps.append((cap, names[len(caps)], idx))
        print(f"Opened camera {idx} as '{names[len(caps)-1]}'")

    if not caps:
        print("No cameras available")
        return

    print("Press 'q' to quit")

    while True:
        for cap, name, idx in caps:
            ret, frame = cap.read()
            if ret:
                cv2.imshow(f"{name} (video{idx})", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap, _, _ in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
