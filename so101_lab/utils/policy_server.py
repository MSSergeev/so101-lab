# Runs in: isaaclab-env (Python 3.11)
"""Helpers for starting the policy server subprocess."""

import os
import socket
import subprocess
import time
from pathlib import Path


def get_lerobot_python() -> str:
    if lerobot_env := os.environ.get("LEROBOT_ENV"):
        return os.path.expanduser(os.path.join(lerobot_env, "bin", "python"))
    env_file = Path(__file__).parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("LEROBOT_ENV="):
                val = os.path.expanduser(line.split("=", 1)[1].strip())
                return os.path.join(val, "bin", "python")
    raise RuntimeError("LEROBOT_ENV not set. Add it to .env or set the environment variable.")


def start_policy_server(port: int = 8080, host: str = "127.0.0.1") -> subprocess.Popen:
    """Start smolvla_server.py in lerobot-env. Returns the Popen handle."""
    python = get_lerobot_python()
    server_script = str(Path(__file__).parents[2] / "scripts" / "eval" / "smolvla_server.py")
    cmd = [python, server_script, "--host", host, "--port", str(port)]
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    proc = subprocess.Popen(cmd, env=env)
    print(f"Policy server started (pid={proc.pid}, port={port})")

    for _ in range(30):
        time.sleep(1.0)
        try:
            with socket.create_connection((host, port), timeout=1):
                print("Policy server is ready")
                return proc
        except OSError:
            pass

    proc.terminate()
    raise RuntimeError("Policy server did not start in time")


def start_ppo_server(port: int = 8081, host: str = "127.0.0.1") -> subprocess.Popen:
    """Start ppo_server.py in lerobot-env. Returns the Popen handle."""
    python = get_lerobot_python()
    server_script = str(Path(__file__).parents[2] / "scripts" / "train" / "ppo_server.py")
    cmd = [python, server_script, "--host", host, "--port", str(port)]
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    proc = subprocess.Popen(cmd, env=env)
    print(f"PPO server started (pid={proc.pid}, port={port})")

    for _ in range(30):
        time.sleep(1.0)
        try:
            with socket.create_connection((host, port), timeout=1):
                print("PPO server is ready")
                return proc
        except OSError:
            pass

    proc.terminate()
    raise RuntimeError("PPO server did not start in time")


def start_sac_server(port: int = 8082, host: str = "127.0.0.1") -> subprocess.Popen:
    """Start sac_server.py in lerobot-env. Returns the Popen handle."""
    python = get_lerobot_python()
    server_script = str(Path(__file__).parents[2] / "scripts" / "train" / "sac_server.py")
    cmd = [python, server_script, "--host", host, "--port", str(port)]
    env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}
    proc = subprocess.Popen(cmd, env=env)
    print(f"SAC server started (pid={proc.pid}, port={port})")

    for _ in range(30):
        time.sleep(1.0)
        try:
            with socket.create_connection((host, port), timeout=1):
                print("SAC server is ready")
                return proc
        except OSError:
            pass

    proc.terminate()
    raise RuntimeError("SAC server did not start in time")
