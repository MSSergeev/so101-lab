#!/usr/bin/env python3
"""OpenCV viewer for policy evaluation preview.

Reads camera images from /dev/shm written by eval_act_policy.py.
Displays cameras side-by-side with evaluation status overlay.

Setup:
    python3 -m venv venvs/viewer
    source venvs/viewer/bin/activate
    pip install opencv-python numpy

Usage:
    source venvs/viewer/bin/activate
    python scripts/tools/eval_viewer.py

Controls:
    Space  - Pause/Resume
    N      - Next episode (skip current)
    R      - Restart current episode
    Escape - Quit evaluation
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
    """Write command to shared memory for eval_act_policy.py to read."""
    tmp_path = f"{SHM_DIR}/.so101_command.json.tmp"
    with open(tmp_path, "w") as f:
        json.dump({"command": cmd, "timestamp": time.time()}, f)
    os.rename(tmp_path, COMMAND_FILE)


def read_status() -> dict:
    """Read evaluation status from shared memory."""
    try:
        with open(f"{SHM_DIR}/so101_status.json") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "state": "EVAL",
            "episode": 0,
            "total_episodes": 0,
            "step": 0,
            "max_steps": 0,
            "seed": None,
            "success": None,
            "successes": 0,
            "status_text": "",
        }


def draw_overlay(img: np.ndarray, status: dict, label: str = "") -> np.ndarray:
    """Draw evaluation status overlay on image."""
    img = img.copy()
    h, w = img.shape[:2]

    episode = status.get("episode", 0)
    total_episodes = status.get("total_episodes", 0)
    step = status.get("step", 0)
    max_steps = status.get("max_steps", 0)
    seed = status.get("seed")
    success = status.get("success")
    successes = status.get("successes", 0)
    status_text = status.get("status_text", "")

    # Camera label (top-right corner)
    if label:
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.putText(img, label, (w - label_size[0] - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Episode and step (top-left)
    ep_text = f"Ep: {episode}/{total_episodes}"
    cv2.putText(img, ep_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    step_text = f"Step: {step}/{max_steps}"
    cv2.putText(img, step_text, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Seed (below step)
    if seed is not None:
        seed_text = f"Seed: {seed}"
        cv2.putText(img, seed_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # Success rate (top-right area, below camera label)
    if total_episodes > 0:
        completed = episode - 1 if success is None else episode
        if completed > 0:
            rate = successes / completed * 100
            rate_text = f"Success: {successes}/{completed} ({rate:.0f}%)"
            rate_color = (0, 255, 0) if rate >= 50 else (0, 165, 255) if rate >= 25 else (0, 0, 255)
            rate_size = cv2.getTextSize(rate_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(img, rate_text, (w - rate_size[0] - 10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, rate_color, 2)

    # Status text (e.g., "Running..." or custom message)
    if status_text:
        cv2.putText(img, status_text, (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # Key hints at bottom
    cv2.putText(img, "Space:Pause  N:Next  R:Restart  Esc:Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img


def main():
    parser = argparse.ArgumentParser(description="Evaluation preview viewer")
    parser.add_argument("--scale", type=float, default=1.5,
                        help="Scale factor for preview window")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target display FPS")
    args = parser.parse_args()

    window_name = "SO101 Eval"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    base_w, base_h = 640, 480
    win_w = int(base_w * 2 * args.scale)
    win_h = int(base_h * args.scale)

    cv2.resizeWindow(window_name, win_w, win_h)
    cv2.moveWindow(window_name, 50, 50)

    print("=" * 50)
    print("Evaluation Viewer")
    print("=" * 50)
    print(f"Reading from: {SHM_DIR}/{SHM_PREFIX}_*.jpg")
    print(f"Scale: {args.scale}, FPS: {args.fps}")
    print("-" * 50)
    print("Controls:")
    print("  Space  - Pause/Resume")
    print("  N      - Next episode (skip current)")
    print("  R      - Restart current episode")
    print("  Escape - Quit evaluation")
    print("=" * 50)

    frame_dt = 1.0 / args.fps
    last_frame_time = time.time()
    waiting_for_data = True

    while True:
        top_path = f"{SHM_DIR}/{SHM_PREFIX}_top.jpg"
        wrist_path = f"{SHM_DIR}/{SHM_PREFIX}_wrist.jpg"

        top_img = cv2.imread(top_path) if os.path.exists(top_path) else None
        wrist_img = cv2.imread(wrist_path) if os.path.exists(wrist_path) else None

        status = read_status()

        if top_img is None and wrist_img is None:
            if waiting_for_data:
                print("Waiting for camera data... Start eval_act_policy.py with --preview")
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

            top_display = draw_overlay(top_img, status, label="TOP")
            wrist_display = draw_overlay(wrist_img, status, label="WRIST")

            combined = np.hstack([top_display, wrist_display])
            cv2.imshow(window_name, combined)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # Escape
            print("\n[CMD] Sending QUIT command...")
            write_command("quit")
            break
        elif key == ord(' '):  # Space
            print("[CMD] Sending PAUSE command...")
            write_command("pause")
        elif key == ord('n') or key == ord('N'):
            print("[CMD] Sending NEXT command...")
            write_command("next")
        elif key == ord('r') or key == ord('R'):
            print("[CMD] Sending RESTART command...")
            write_command("restart")

        elapsed = time.time() - last_frame_time
        sleep_time = frame_dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        last_frame_time = time.time()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
