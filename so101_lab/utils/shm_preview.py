"""Shared memory camera preview: write images + status, read commands.

Used by record_episodes.py and train_sac.py to communicate with camera_viewer.py.
"""

import json
import os
import signal
import subprocess
import time

import numpy as np
from PIL import Image

SHM_DIR = "/dev/shm"
SHM_PREFIX = "so101_camera"
COMMAND_FILE = f"{SHM_DIR}/so101_command.json"

_last_command_time = 0.0


def write_camera_to_shm(name: str, img: np.ndarray):
    """Write image to shared memory with atomic rename using PIL."""
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)
    tmp_path = f"{SHM_DIR}/.{SHM_PREFIX}_{name}.jpg.tmp"
    final_path = f"{SHM_DIR}/{SHM_PREFIX}_{name}.jpg"
    pil_img.save(tmp_path, "JPEG", quality=90)
    os.rename(tmp_path, final_path)


def write_status_to_shm(status_dict: dict):
    """Write status dict to shared memory."""
    tmp_path = f"{SHM_DIR}/.so101_status.json.tmp"
    final_path = f"{SHM_DIR}/so101_status.json"
    with open(tmp_path, "w") as f:
        json.dump(status_dict, f)
    os.rename(tmp_path, final_path)


def read_command() -> str | None:
    """Read command from shared memory (written by camera_viewer.py).

    Returns: 'teleop', 'record', 'save', 'discard', 'quit', or None
    """
    global _last_command_time
    try:
        with open(COMMAND_FILE) as f:
            data = json.load(f)
        cmd_time = data.get("timestamp", 0)
        if cmd_time > _last_command_time:
            _last_command_time = cmd_time
            return data.get("command")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None


def cleanup_shm():
    """Remove shared memory files on exit."""
    for name in ["top", "wrist"]:
        path = f"{SHM_DIR}/{SHM_PREFIX}_{name}.jpg"
        tmp_path = f"{SHM_DIR}/.{SHM_PREFIX}_{name}.jpg.tmp"
        for p in [path, tmp_path]:
            if os.path.exists(p):
                os.remove(p)
    for p in [f"{SHM_DIR}/so101_status.json", f"{SHM_DIR}/.so101_status.json.tmp"]:
        if os.path.exists(p):
            os.remove(p)


def cleanup_command_file():
    """Remove command file on exit."""
    if os.path.exists(COMMAND_FILE):
        os.remove(COMMAND_FILE)
    tmp = f"{SHM_DIR}/.so101_command.json.tmp"
    if os.path.exists(tmp):
        os.remove(tmp)


def launch_viewer(script: str = "camera_viewer.py") -> subprocess.Popen | None:
    """Launch viewer subprocess.

    Args:
        script: Viewer script name in scripts/tools/ (e.g. "camera_viewer.py", "hil_viewer.py").
    """
    project_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    viewer_python = os.path.join(project_root, "venvs", "viewer", "bin", "python")
    viewer_script = os.path.join(project_root, "scripts", "tools", script)

    if not os.path.exists(viewer_python):
        print(f"\n[WARNING] Viewer venv not found: {viewer_python}")
        print("To enable preview, run:")
        print("  python3 -m venv venvs/viewer")
        print("  source venvs/viewer/bin/activate")
        print("  pip install opencv-python numpy")
        print("Continuing without preview...\n")
        return None

    if not os.path.exists(viewer_script):
        print(f"\n[WARNING] Viewer script not found: {viewer_script}")
        print("Continuing without preview...\n")
        return None

    try:
        clean_env = os.environ.copy()
        for key in list(clean_env.keys()):
            if key.startswith(("PYTHONPATH", "CARB_", "OMNI_", "ISAAC_")):
                del clean_env[key]
        if "DISPLAY" not in clean_env:
            clean_env["DISPLAY"] = ":0"

        proc = subprocess.Popen(
            [viewer_python, viewer_script],
            env=clean_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return proc
    except Exception as e:
        print(f"\n[WARNING] Failed to launch viewer: {e}")
        print("Continuing without preview...\n")
        return None


def stop_viewer(proc: subprocess.Popen | None):
    """Stop viewer subprocess."""
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
