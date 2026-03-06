#!/usr/bin/env python3
"""OpenCV viewer for HIL (human-in-the-loop) SAC training preview.

Reads camera images from /dev/shm written by train_sac.py.
Displays cameras side-by-side with HIL status overlay.

Setup:
    python3 -m venv venvs/viewer
    source venvs/viewer/bin/activate
    pip install opencv-python numpy

Usage:
    source venvs/viewer/bin/activate
    python scripts/tools/hil_viewer.py

Controls:
    Space  - Toggle HIL takeover ON/OFF
    X      - Reset episode
    Escape - Quit
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

SHM_DIR = "/dev/shm"
SHM_PREFIX = "so101_camera"
COMMAND_FILE = f"{SHM_DIR}/so101_command.json"


def write_command(cmd: str):
    """Write command to shared memory for train_sac.py to read."""
    tmp_path = f"{SHM_DIR}/.so101_command.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump({"command": cmd, "timestamp": time.time()}, f)
    os.rename(tmp_path, COMMAND_FILE)


def read_status() -> dict:
    """Read training status from shared memory."""
    try:
        with open(f"{SHM_DIR}/so101_status.json") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"state": "POLICY", "teleop": False,
                "episode": 0, "frame": 0, "status_text": ""}


def draw_overlay(img: np.ndarray, status: dict, label: str = "",
                 preview_fps: float = 0.0) -> np.ndarray:
    """Draw HIL training status overlay on image."""
    img = img.copy()
    h, w = img.shape[:2]

    state = status.get("state", "POLICY")
    teleop = status.get("teleop", False)
    episode = status.get("episode", 0)
    frame = status.get("frame", 0)
    status_text = status.get("status_text", "")

    # Camera label (top-right corner)
    if label:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(img, label, (w - label_size[0] - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # State indicator (top-left)
    if teleop:
        # HIL ON: green circle + HIL ON text
        cv2.circle(img, (30, 30), 12, (0, 255, 0), -1)
        cv2.putText(img, "HIL ON", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, f"Ep:{episode} Fr:{frame}", (155, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    else:
        # Policy running: blue circle + POLICY text
        cv2.circle(img, (30, 30), 12, (255, 165, 0), -1)
        cv2.putText(img, "POLICY", (50, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(img, f"Ep:{episode} Fr:{frame}", (155, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Status text (step, reward, etc.)
    if status_text:
        cv2.putText(img, status_text, (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Preview FPS (top-right, below camera label)
    if preview_fps > 0 and label:
        fps_text = f"preview: {preview_fps:.0f} fps"
        fps_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.putText(img, fps_text, (w - fps_size[0] - 10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Key hints at bottom
    cv2.putText(img, "Space:Toggle HIL  X:Reset episode  Esc:Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img


def main():
    parser = argparse.ArgumentParser(description="HIL training preview viewer")
    parser.add_argument("--scale", type=float, default=1.5,
                        help="Scale factor for preview window")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target display FPS")
    args = parser.parse_args()

    window_name = "SO101 HIL Training"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    base_w, base_h = 640, 480
    win_w = int(base_w * 2 * args.scale)
    win_h = int(base_h * args.scale)

    cv2.resizeWindow(window_name, win_w, win_h)
    cv2.moveWindow(window_name, 50, 50)

    print("=" * 50)
    print("HIL Training Viewer")
    print("=" * 50)
    print(f"Reading from: {SHM_DIR}/{SHM_PREFIX}_*.jpg")
    print(f"Scale: {args.scale}, FPS: {args.fps}")
    print("-" * 50)
    print("Controls:")
    print("  Space  - Toggle HIL takeover ON/OFF")
    print("  X      - Reset episode")
    print("  Escape - Quit")
    print("=" * 50)

    frame_dt = 1.0 / args.fps
    last_frame_time = time.time()
    waiting_for_data = True

    # Preview FPS calculation
    last_status_timestamp = 0.0
    fps_samples = []
    preview_fps = 0.0

    while True:
        top_path = f"{SHM_DIR}/{SHM_PREFIX}_top.jpg"
        wrist_path = f"{SHM_DIR}/{SHM_PREFIX}_wrist.jpg"

        top_img = cv2.imread(top_path) if os.path.exists(top_path) else None
        wrist_img = cv2.imread(wrist_path) if os.path.exists(wrist_path) else None

        status = read_status()

        # Calculate sim FPS from status timestamp updates
        current_status_ts = status.get("timestamp", 0.0)
        if current_status_ts > last_status_timestamp and last_status_timestamp > 0:
            dt = current_status_ts - last_status_timestamp
            if dt > 0:
                fps_samples.append(1.0 / dt)
                if len(fps_samples) > 10:
                    fps_samples.pop(0)
                preview_fps = sum(fps_samples) / len(fps_samples)
        last_status_timestamp = current_status_ts

        if top_img is None and wrist_img is None:
            if waiting_for_data:
                print("Waiting for camera data... Start train_sac.py with --hil")
                waiting_for_data = False
            placeholder = np.zeros((base_h, base_w * 2, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Waiting for data...", (base_w - 150, base_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)
            cv2.imshow(window_name, placeholder)
        else:
            waiting_for_data = True

            if top_img is None:
                top_img = np.zeros((base_h, base_w, 3), dtype=np.uint8)
                cv2.putText(top_img, "No top camera", (200, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            if wrist_img is None:
                wrist_img = np.zeros((base_h, base_w, 3), dtype=np.uint8)
                cv2.putText(wrist_img, "No wrist camera", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            top_display = draw_overlay(top_img, status, label="TOP", preview_fps=preview_fps)
            wrist_display = draw_overlay(wrist_img, status, label="WRIST")

            combined = np.hstack([top_display, wrist_display])
            cv2.imshow(window_name, combined)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Escape
            print("\n[CMD] Sending QUIT command...")
            write_command("quit")
            break
        elif key == ord(' '):  # Space - toggle HIL
            print("[CMD] Sending TELEOP (toggle HIL) command...")
            write_command("teleop")
        elif key == ord('x') or key == ord('X'):  # X - reset episode
            print("[CMD] Sending DISCARD (reset) command...")
            write_command("discard")

        # Rate limiting
        elapsed = time.time() - last_frame_time
        sleep_time = frame_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_frame_time = time.time()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
