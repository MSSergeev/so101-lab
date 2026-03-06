#!/usr/bin/env python3
"""Record teleoperation demonstrations with optional reward annotations.

This script provides a workflow for recording demonstrations directly
to LeRobot format (parquet + video).

States:
    IDLE      - Robot stationary, no teleop, no recording
    TELEOP    - Teleop active, no recording
    RECORDING - Teleop active, recording to dataset

Controls (same keys in Isaac Sim window and camera_viewer.py):
    Space  - Start teleop (IDLE -> TELEOP)
    N      - Start recording (IDLE/TELEOP -> RECORDING)
    F      - Finish: save episode + reset (RECORDING -> IDLE)
    X      - Discard: reset without saving (TELEOP/RECORDING -> IDLE)
    Escape - Quit (saves if recording)

Usage:
    # Basic recording with keyboard control
    python scripts/teleop/record_episodes.py --teleop-device=keyboard

    # Headless mode with camera preview (recommended for performance)
    python scripts/teleop/record_episodes.py --headless --teleop-device=keyboard

    # Recording with leader arm
    python scripts/teleop/record_episodes.py \\
        --teleop-device=so101leader \\
        --calibration-file=my_leader.json \\
        --output=data/recordings/pick_cube

    # Recording with reward annotations
    python scripts/teleop/record_episodes.py \\
        --reward-mode sim+success \\
        --env figure_shape_placement \\
        --output data/recordings/rewards_test

    # Headless + auto-restart (batch recording)
    python scripts/teleop/record_episodes.py \\
        --headless \\
        --teleop-device=so101leader \\
        --reset-mode=auto \\
        --reset-delay=3.0

    # Headless without preview (maximum performance)
    python scripts/teleop/record_episodes.py --headless --no-preview

Reward modes:
    none         - No reward recorded (default)
    success      - Binary: 0 before is_success(), 1 after
    sim          - Weighted sum of sim reward terms
    sim+success  - sim rewards + bonus on is_success()

Sim reward terms are auto-discovered from so101_lab/tasks/<env>/rl/mdp/rewards.py.
If the module doesn't exist, sim/sim+success modes are unavailable.
"""

import argparse
import sys
from enum import Enum

from isaaclab.app import AppLauncher

# Parse arguments before Isaac Sim launch
parser = argparse.ArgumentParser(description="Record demonstrations with shared memory preview")
parser.add_argument("--env", type=str, default="template",
                    help="Task environment (template, pick_cube, figure_shape_placement)")
parser.add_argument("--teleop-device", type=str, default="keyboard",
                    choices=["keyboard", "gamepad", "so101leader"],
                    help="Teleoperation device type")
parser.add_argument("--port", type=str, default="/dev/ttyACM0",
                    help="Serial port for SO101Leader device")
parser.add_argument("--calibration-file", type=str, default="so101_leader.json",
                    help="Calibration file for SO101Leader")
parser.add_argument("--recalibrate", action="store_true",
                    help="Force recalibration of SO101Leader")
parser.add_argument("--output", type=str, default="data/recordings/dataset",
                    help="Output directory for LeRobot dataset")
parser.add_argument("--task", type=str, default="default task",
                    help="Task description for the dataset")
parser.add_argument("--episode-length", type=float, default=120.0,
                    help="Maximum episode length in seconds")
parser.add_argument("--physics-hz", type=int, default=120,
                    help="Physics simulation frequency (PhysX timestep)")
parser.add_argument("--policy-hz", type=int, default=30,
                    help="Policy/control frequency (how often actions are updated)")
parser.add_argument("--recording-hz", type=int, default=30,
                    help="Dataset recording frequency (must equal policy-hz)")
parser.add_argument("--render-hz", type=int, default=30,
                    help="Rendering frequency (scene + cameras)")
parser.add_argument("--preview-hz", type=int, default=30,
                    help="OpenCV preview update frequency (0 = no throttling)")
parser.add_argument("--reset-mode", type=str, default="pause",
                    choices=["pause", "auto"],
                    help="Reset behavior: pause (wait) or auto (auto-start after delay)")
parser.add_argument("--reset-delay", type=float, default=3.0,
                    help="Delay before auto-start in auto reset mode (seconds)")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of environments")
parser.add_argument("--sensitivity", type=float, default=1.0,
                    help="Control sensitivity for keyboard/gamepad")
parser.add_argument("--preview", action="store_true", default=True,
                    help="Enable camera preview (default: True)")
parser.add_argument("--no-preview", action="store_true",
                    help="Disable camera preview")
parser.add_argument("--crf", type=int, default=23,
                    help="Video quality (0-51, lower=better, default: 23)")
parser.add_argument("--gop", type=str, default="auto",
                    help="GOP size / keyframe interval (default: auto, you can use 2 for LeRobot default)")
parser.add_argument("--randomize-light", action="store_true",
                    help="Randomize sun light intensity/color on each reset")
parser.add_argument("--diversity-keys", type=str, default="",
                    help="Comma-separated initial_state keys for spawn diversity check (e.g. cube_x,cube_y)")
parser.add_argument("--diversity-ratio", type=float, default=2.0,
                    help="Diversity re-roll threshold: cell is crowded if count >= mean * ratio (default: 2.0)")
parser.add_argument("--diversity-target", type=float, default=5.0,
                    help="Target points per hex cell under uniform distribution (default: 5.0)")

# Reward args
parser.add_argument("--reward-mode", type=str, default="none",
                    choices=["none", "success", "sim", "sim+success"],
                    help="Reward computation mode (default: none)")
parser.add_argument("--success-bonus", type=float, default=10.0,
                    help="Bonus on is_success() for sim+success mode")
parser.add_argument("--reward-weights", type=str, default="",
                    help="Sim reward weights as name=weight,... (e.g. 'drop_penalty=-10,time_penalty=-0.05'). "
                         "Unspecified terms get weight 1.0")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Validate frequency arguments
if args.recording_hz != args.policy_hz:
    print(f"\nERROR: recording-hz ({args.recording_hz}) must equal policy-hz ({args.policy_hz})!")
    sys.exit(1)

if args.physics_hz % args.policy_hz != 0:
    print(f"\nWARNING: physics-hz ({args.physics_hz}) is not divisible by policy-hz ({args.policy_hz})")
    print(f"Decimation will be: {args.physics_hz / args.policy_hz:.2f} (non-integer may cause issues)")

if args.physics_hz % args.render_hz != 0:
    print(f"\nWARNING: physics-hz ({args.physics_hz}) is not divisible by render-hz ({args.render_hz})")
    print(f"Render interval will be: {args.physics_hz / args.render_hz:.2f} (non-integer may cause issues)")

if args.render_hz > args.physics_hz:
    print(f"\nWARNING: render-hz ({args.render_hz}) > physics-hz ({args.physics_hz})")
    print("Render frequency will be capped at physics frequency")
    args.render_hz = args.physics_hz

if args.policy_hz > args.physics_hz:
    print(f"\nERROR: policy-hz ({args.policy_hz}) cannot exceed physics-hz ({args.physics_hz})!")
    sys.exit(1)

# Validate arguments
if args.teleop_device == "so101leader" and args.recalibrate and args.calibration_file == "so101_leader.json":
    print("\nERROR: When using --recalibrate, you must specify a unique --calibration-file name!")
    sys.exit(1)

# Handle --no-preview flag
if args.no_preview:
    args.preview = False

# Interactive prompt for output directory (before Isaac Sim launch)
if args.output != "data/recordings/dataset":
    # --output was explicitly provided
    print(f"\n[OUTPUT] Using: {args.output}")
elif sys.stdin.isatty():
    # Interactive prompt for default value
    print(f"\nOutput directory [{args.output}]: ", end="", flush=True)
    try:
        custom_output = input().strip()
        if custom_output:
            args.output = custom_output
    except EOFError:
        print()  # Newline after prompt
        pass  # Use default
else:
    # Non-interactive (piped input)
    print(f"\n[OUTPUT] Using default: {args.output}")

# Enable cameras
args.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Disable rate limiting for maximum simulation speed
from so101_lab.utils import disable_rate_limiting
disable_rate_limiting()

# Imports after Isaac Sim launch
import os
import time
from pathlib import Path

import numpy as np

from so101_lab.data.collector import RecordingManager
from so101_lab.data.lerobot_dataset import LeRobotDatasetWriter
from so101_lab.tasks import get_task
from so101_lab.utils.shm_preview import (
    cleanup_command_file,
    cleanup_shm,
    launch_viewer,
    read_command,
    stop_viewer,
    write_camera_to_shm,
    write_status_to_shm,
)


class State(Enum):
    IDLE = "IDLE"
    TELEOP = "TELEOP"
    RECORDING = "RECORDING"


# ═══════════════════════════════════════════════════════════════════════════════
# Reward helpers (only used when --reward-mode is not "none")
# ═══════════════════════════════════════════════════════════════════════════════

def discover_reward_functions(env_name: str) -> dict[str, callable] | None:
    """Try to import reward functions from env's rl/mdp/rewards module.

    Returns dict of {name: func} for plain functions (not classes),
    or None if the module doesn't exist.
    """
    import importlib
    import inspect

    try:
        module = importlib.import_module(f"so101_lab.tasks.{env_name}.rl.mdp.rewards")
    except (ImportError, ModuleNotFoundError):
        return None

    funcs = {}
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("_"):
            continue
        # Accept functions defined in any so101_lab.tasks.* rewards module
        obj_module = getattr(obj, "__module__", "") or ""
        if "rl.mdp.rewards" in obj_module:
            funcs[name] = obj
    return funcs if funcs else None


def compute_sim_metrics(env, reward_funcs: dict[str, callable]) -> dict[str, float]:
    """Call discovered reward functions, return raw (unweighted) values."""
    metrics = {}
    for name, func in reward_funcs.items():
        try:
            value = func(env)
            metrics[name] = value[0].item()
        except Exception:
            pass
    return metrics


def parse_reward_weights(weights_str: str) -> dict[str, float]:
    """Parse 'name=weight,name=weight,...' string."""
    if not weights_str.strip():
        return {}
    weights = {}
    for pair in weights_str.split(","):
        pair = pair.strip()
        if "=" in pair:
            name, val = pair.split("=", 1)
            weights[name.strip()] = float(val.strip())
    return weights


def compute_reward(
    metrics: dict[str, float],
    success: bool,
    reward_mode: str,
    weights: dict[str, float],
    success_bonus: float,
) -> float:
    """Compute scalar reward from metrics and mode."""
    if reward_mode == "success":
        return 1.0 if success else 0.0

    sim_reward = sum(
        weights.get(name, 1.0) * value
        for name, value in metrics.items()
    )

    if reward_mode == "sim":
        return sim_reward
    elif reward_mode == "sim+success":
        return sim_reward + (success_bonus if success else 0.0)

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Environment setup
# ═══════════════════════════════════════════════════════════════════════════════

def setup_environment(args) -> tuple:
    """Setup Isaac Lab environment and teleop device.

    Returns:
        (env, device)
    """
    EnvClass, EnvCfgClass = get_task(args.env)

    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length

    # ════════════════════════════════════════════════════════════
    # АВТОМАТИЧЕСКАЯ КОНФИГУРАЦИЯ ЧАСТОТ
    # ════════════════════════════════════════════════════════════

    # Physics timestep
    env_cfg.sim.dt = 1.0 / args.physics_hz

    # Decimation (сколько physics steps на один policy step)
    env_cfg.decimation = args.physics_hz // args.policy_hz

    # Render interval (сколько physics steps на один render)
    env_cfg.sim.render_interval = args.physics_hz // args.render_hz

    # Camera update period (должна совпадать с render frequency)
    camera_update_period = 1.0 / args.render_hz
    if hasattr(env_cfg.scene, 'top'):
        env_cfg.scene.top.update_period = camera_update_period
    if hasattr(env_cfg.scene, 'wrist'):
        env_cfg.scene.wrist.update_period = camera_update_period

    # Light randomization
    if hasattr(env_cfg, 'randomize_light'):
        env_cfg.randomize_light = args.randomize_light

    # ════════════════════════════════════════════════════════════

    env_cfg.use_teleop_device(args.teleop_device)

    env = EnvClass(cfg=env_cfg)

    if args.teleop_device == "keyboard":
        from so101_lab.devices import SO101Keyboard
        device = SO101Keyboard(env, sensitivity=args.sensitivity)
    elif args.teleop_device == "gamepad":
        from so101_lab.devices import SO101Gamepad
        device = SO101Gamepad(env, sensitivity=args.sensitivity)
    elif args.teleop_device == "so101leader":
        from so101_lab.devices import SO101Leader
        device = SO101Leader(
            env,
            port=args.port,
            recalibrate=args.recalibrate,
            calibration_file_name=args.calibration_file
        )

    return env, device


def main():
    """Main recording loop."""
    use_rewards = args.reward_mode != "none"

    env, device = setup_environment(args)

    # Get default task description from env or use --task arg
    if hasattr(env, "get_task_description"):
        default_task = env.get_task_description()
    else:
        default_task = args.task if args.task != "default task" else "manipulation task"

    # Parse GOP: "auto" -> None, otherwise int
    gop_value = None if args.gop == "auto" else int(args.gop)

    # Setup rewards if enabled
    reward_funcs = None
    reward_weights = {}
    if use_rewards:
        reward_funcs = discover_reward_functions(args.env)
        reward_weights = parse_reward_weights(args.reward_weights)
        if args.reward_mode in ("sim", "sim+success") and reward_funcs is None:
            print(f"\nERROR: --reward-mode={args.reward_mode} requires rl/mdp/rewards.py in {args.env} task")
            sys.exit(1)
        # sim_rewards.pt is always saved when reward_funcs are available, regardless of reward_mode

    # Create dataset writer
    extra_features = {}
    if use_rewards:
        extra_features["next.reward"] = ((), np.float32)

    dataset = LeRobotDatasetWriter(
        args.output, fps=args.policy_hz, task=default_task, crf=args.crf, gop=gop_value,
        extra_features=extra_features if extra_features else None,
    )
    recorder = RecordingManager(dataset, env)

    if use_rewards:
        recorder.load_sim_rewards(Path(args.output) / "sim_rewards.pt")

    # Spawn diversity checker
    diversity = None
    if args.diversity_keys:
        from so101_lab.utils.spawn_diversity import SpawnDiversityChecker
        keys = [k.strip() for k in args.diversity_keys.split(",")]
        diversity = SpawnDiversityChecker(args.output, keys, max_ratio=args.diversity_ratio, target_per_cell=args.diversity_target)
        print(diversity.stats())

    cleanup_command_file()

    # Launch viewer if preview enabled
    viewer_proc = None
    preview_active = args.preview
    if preview_active:
        viewer_proc = launch_viewer()
        preview_active = viewer_proc is not None

    # State machine
    state = State.IDLE
    waiting_for_auto_start = False
    auto_start_time = 0.0
    episode_frame_count = 0
    episode_success_achieved = False

    # Pending commands from Isaac Sim keyboard callbacks
    pending_record = False
    pending_save = False
    pending_debug = False
    pending_quit = False
    pending_rerecord = False

    # Register callbacks for N, F, T, R, Escape keys in Isaac Sim window
    def on_record_key():
        nonlocal pending_record
        pending_record = True

    def on_save_key():
        nonlocal pending_save
        pending_save = True

    def on_debug_key():
        nonlocal pending_debug
        pending_debug = True

    def on_quit_key():
        nonlocal pending_quit
        pending_quit = True

    def on_rerecord_key():
        nonlocal pending_rerecord
        pending_rerecord = True

    device.add_callback("N", on_record_key)
    device.add_callback("F", on_save_key)
    device.add_callback("T", on_debug_key)
    device.add_callback("R", on_rerecord_key)
    device.add_callback("ESCAPE", on_quit_key)

    # Preview throttling (только для preview, не для записи!)
    preview_dt = 1.0 / args.preview_hz if args.preview_hz > 0 else 0
    last_preview_time = 0.0

    # Episode seed and initial state tracking
    current_episode_seed = None
    current_initial_state = None

    def reset_env_with_seed():
        """Reset environment with a new random seed."""
        nonlocal current_episode_seed, current_initial_state
        max_attempts = (diversity.max_rerolls + 1) if diversity else 1
        for attempt in range(max_attempts):
            current_episode_seed = int(np.random.default_rng().integers(0, 2**31))
            result = env.reset(seed=current_episode_seed)
            current_initial_state = env.get_initial_state() if hasattr(env, "get_initial_state") else None
            if current_initial_state is None or diversity is None:
                break
            if not diversity.should_reroll(current_initial_state):
                if attempt > 0:
                    print(f"[DIVERSITY] Accepted after {attempt} re-roll(s)")
                break
            if attempt == max_attempts - 1:
                print(f"[DIVERSITY] Forced accept after {attempt} re-roll(s) (all crowded)")
                break
            x = current_initial_state[diversity.coord_keys[0]]
            y = current_initial_state[diversity.coord_keys[1]]
            cell = diversity._cell(x, y)
            count = diversity.grid.get(cell, 0)
            print(f"[DIVERSITY] re-roll #{attempt+1}: cell {cell} has {count} (threshold {diversity.mean_count * diversity.max_ratio:.1f})")
            diversity.reroll_count += 1
        return result

    def reset_env_same_seed():
        """Reset environment with the same seed (for re-recording)."""
        nonlocal current_initial_state
        result = env.reset(seed=current_episode_seed)
        current_initial_state = env.get_initial_state() if hasattr(env, "get_initial_state") else None
        return result

    def discard_recording():
        """Discard current recording, clearing reward data if needed."""
        if use_rewards:
            recorder._sim_rewards_episode.clear()
        recorder.on_episode_end(success=False)

    def save_recording():
        """Save current recording, flushing reward data if needed."""
        if use_rewards:
            recorder.flush_sim_rewards()
        recorder.on_episode_end(success=True)
        if diversity and current_initial_state is not None:
            diversity.accept(current_initial_state)

    obs_dict, _ = reset_env_with_seed()

    # Print controls
    print("\n" + "=" * 60)
    title = "Recording to LeRobot Format"
    if use_rewards:
        title += " (with Rewards)"
    print(title)
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Output: {args.output}")
    print(f"Device: {args.teleop_device}")
    if use_rewards:
        print(f"Reward mode: {args.reward_mode}")
        if reward_funcs:
            print(f"  Sim reward terms: {', '.join(reward_funcs.keys())}")
            if reward_weights:
                print(f"  Weights: {reward_weights}")
            else:
                print(f"  Weights: all 1.0 (default)")
        else:
            print(f"  Sim rewards: not available (no rl/mdp/rewards.py)")
        if args.reward_mode == "sim+success":
            print(f"  success-bonus: {args.success_bonus}")
    print("-" * 60)
    print("Frequencies:")
    print(f"  Physics:   {args.physics_hz} Hz (dt={1/args.physics_hz:.4f}s)")
    print(f"  Policy:    {args.policy_hz} Hz (decimation={args.physics_hz//args.policy_hz})")
    print(f"  Render:    {args.render_hz} Hz (interval={args.physics_hz//args.render_hz})")
    print(f"  Recording: {args.recording_hz} Hz (= policy frequency)")
    if args.preview_hz > 0:
        print(f"  Preview:   {args.preview_hz} Hz (throttled)")
    else:
        print(f"  Preview:   No throttling")
    print(f"Episode length: {args.episode_length}s")
    print(f"Reset mode: {args.reset_mode}", end="")
    if args.reset_mode == "auto":
        print(f" (delay: {args.reset_delay}s)")
    else:
        print()
    print("-" * 60)
    print("Controls (Isaac Sim window or camera_viewer.py):")
    print("  Space  - Start teleop")
    print("  N      - Start recording")
    print("  F      - Finish (save + reset)")
    print("  R      - Restart (reset to same scene)")
    print("  X      - Discard (reset without save)")
    print("  T      - Debug info")
    print("  Escape - Quit")
    print("-" * 60)
    print("Semantic: Recording (observation_before, action) pairs")
    print("-" * 60)
    if preview_active:
        print("Preview: ENABLED (camera_viewer.py launched)")
    else:
        print("Preview: DISABLED")
    print("=" * 60 + "\n")

    # Also display device-specific controls
    device.display_controls()

    print("\n[INFO] Starting main loop...")

    try:
        while simulation_app.is_running():
            current_time = time.time()

            # ─── PREVIEW THROTTLING ───────────────────────────────────
            should_update_preview = (
                args.preview_hz == 0 or
                (current_time - last_preview_time) >= preview_dt
            )
            if should_update_preview:
                last_preview_time = current_time

            # ─── GET CAMERA IMAGES ────────────────────────────────────
            obs = obs_dict["policy"]
            top_img = obs["top"][0].cpu().numpy() if "top" in obs else np.zeros((480, 640, 3))
            wrist_img = obs["wrist"][0].cpu().numpy() if "wrist" in obs else np.zeros((480, 640, 3))

            # ─── AUTO-START COUNTDOWN ─────────────────────────────────
            status_text = ""
            if waiting_for_auto_start:
                remaining = args.reset_delay - (current_time - auto_start_time)
                if remaining > 0:
                    status_text = f"Auto-start in {remaining:.1f}s"
                else:
                    # Auto-start: go directly to RECORDING
                    state = State.RECORDING
                    waiting_for_auto_start = False
                    episode_frame_count = 0
                    episode_success_achieved = False
                    device._started = True
                    # Set per-episode task description if environment supports it
                    if hasattr(env, "get_task_description"):
                        recorder.set_task(env.get_task_description())
                    # Set episode metadata (seed + initial state)
                    if current_episode_seed is not None:
                        recorder.set_episode_seed(current_episode_seed)
                    if current_initial_state is not None:
                        recorder.set_episode_initial_state(current_initial_state)
                    recorder.on_reset(obs_dict)
                    print(f"[AUTO] Recording started (Episode {dataset.total_episodes})")

            # ─── SHARED MEMORY (PREVIEW) ──────────────────────────────
            if preview_active and should_update_preview:
                write_camera_to_shm("top", top_img)
                write_camera_to_shm("wrist", wrist_img)
                write_status_to_shm({
                    "state": state.value,
                    "recording": state == State.RECORDING,
                    "teleop": state in (State.TELEOP, State.RECORDING),
                    "episode": int(dataset.total_episodes),
                    "frame": int(episode_frame_count),
                    "status_text": str(status_text),
                    "timestamp": time.time(),
                })

            # ─── READ COMMANDS ────────────────────────────────────────
            cmd = read_command() if preview_active else None

            # Check pending commands from Isaac Sim keyboard
            if pending_quit:
                cmd = "quit"
                pending_quit = False
            if pending_record:
                cmd = "record"
                pending_record = False
            if pending_save:
                cmd = "save"
                pending_save = False
            if pending_rerecord:
                cmd = "rerecord"
                pending_rerecord = False
            if pending_debug:
                pending_debug = False
                print("\n" + "=" * 60)
                print("DEBUG INFO")
                print("=" * 60)
                print(f"State: {state.value}")
                print(f"Episode: {dataset.total_episodes}, Frame: {episode_frame_count}")
                if use_rewards:
                    print(f"Reward mode: {args.reward_mode}, Success: {episode_success_achieved}")
                joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

                # Sim robot positions
                robot_joints = env.scene["robot"].data.joint_pos[0]
                print("Sim robot (radians):")
                for i, name in enumerate(joint_names):
                    rad = robot_joints[i].item()
                    print(f"  {name:15s}: {rad:7.3f} rad / {rad * 57.3:7.2f}°")

                # Leader positions (if SO101Leader device)
                if hasattr(device, 'get_device_state'):
                    leader_state = device.get_device_state()
                    print("\nLeader arm (normalized [-100, 100]):")
                    for name in joint_names:
                        if name in leader_state:
                            val = leader_state[name]
                            print(f"  {name:15s}: {val:7.2f}")
                print("=" * 60 + "\n")

            # ─── HANDLE COMMANDS (STATE MACHINE) ──────────────────────
            # Check device state changes (Space/X keys)
            if device._reset_state:
                # X key pressed - treat as discard
                cmd = "discard"
                device._reset_state = False

            # Handle commands based on current state
            if cmd == "quit":
                print("\n[INFO] Quit requested")
                break

            elif cmd == "teleop":
                if state == State.IDLE:
                    state = State.TELEOP
                    device._started = True
                    waiting_for_auto_start = False
                    print("[TELEOP] Teleop started")

            elif cmd == "record":
                if state in (State.IDLE, State.TELEOP):
                    state = State.RECORDING
                    device._started = True
                    waiting_for_auto_start = False
                    episode_frame_count = 0
                    episode_success_achieved = False
                    # Set per-episode task description if environment supports it
                    if hasattr(env, "get_task_description"):
                        recorder.set_task(env.get_task_description())
                    # Set episode metadata (seed + initial state)
                    if current_episode_seed is not None:
                        recorder.set_episode_seed(current_episode_seed)
                    if current_initial_state is not None:
                        recorder.set_episode_initial_state(current_initial_state)
                    recorder.on_reset(obs_dict)
                    print(f"[RECORD] Recording started (Episode {dataset.total_episodes}, seed={current_episode_seed})")

            elif cmd == "save":
                if state == State.RECORDING:
                    save_recording()
                    print(f"[SAVE] Episode {dataset.total_episodes - 1} saved ({episode_frame_count} frames)")
                    state = State.IDLE
                    device._started = False
                    obs_dict, _ = reset_env_with_seed()
                    episode_frame_count = 0
                    episode_success_achieved = False

                    if args.reset_mode == "auto":
                        waiting_for_auto_start = True
                        auto_start_time = current_time
                        print(f"[AUTO] Will auto-start in {args.reset_delay}s...")
                    else:
                        print("[IDLE] Press Space for teleop, N to record")

            elif cmd == "discard":
                if state in (State.TELEOP, State.RECORDING):
                    if state == State.RECORDING:
                        discard_recording()
                        print(f"[DISCARD] Episode discarded ({episode_frame_count} frames)")
                    else:
                        print("[RESET] Teleop reset (X key)")
                    state = State.IDLE
                    device._started = False
                    obs_dict, _ = reset_env_with_seed()
                    episode_frame_count = 0
                    episode_success_achieved = False

                    if args.reset_mode == "auto":
                        waiting_for_auto_start = True
                        auto_start_time = current_time
                        print(f"[AUTO] Will auto-start in {args.reset_delay}s...")
                    else:
                        print("[IDLE] Press Space for teleop, N to record")

            elif cmd == "rerecord":
                if state in (State.TELEOP, State.RECORDING):
                    if state == State.RECORDING:
                        discard_recording()
                        print(f"[RESTART] Episode discarded ({episode_frame_count} frames), same scene")
                    else:
                        print(f"[RESTART] Resetting to same scene (seed={current_episode_seed})")
                    # Reset with same seed, go to IDLE
                    obs_dict, _ = reset_env_same_seed()
                    state = State.IDLE
                    device._started = False
                    waiting_for_auto_start = False
                    episode_frame_count = 0
                    episode_success_achieved = False
                    print("[IDLE] Press Space for teleop, N to record")

            # Handle Space key (device._started becomes True)
            if device._started and state == State.IDLE and not waiting_for_auto_start:
                state = State.TELEOP
                print("[TELEOP] Teleop started")

            # ─── STEP ENVIRONMENT ─────────────────────────────────────
            if state in (State.TELEOP, State.RECORDING):
                action_dict = device.advance()
                if action_dict and not action_dict.get("reset"):
                    action = env.cfg.preprocess_device_action(action_dict, device)

                    # ЗАПИСЬ ПЕРЕД STEP - правильная семантика (obs_before, action)
                    if state == State.RECORDING:
                        recorder.on_step(obs_dict, action)
                        episode_frame_count += 1

                    # Применение action и получение нового состояния
                    obs_dict, reward, terminated, truncated, info = env.step(action)

                    # Check for task success (if environment supports it)
                    if hasattr(env, 'is_success') and state in (State.TELEOP, State.RECORDING):
                        current_success = env.is_success()[0].item()
                        if current_success and not episode_success_achieved:
                            episode_success_achieved = True
                            print(f"[SUCCESS] Task completed at frame {episode_frame_count}!")

                    # Compute and attach reward AFTER step (if enabled)
                    if use_rewards and state == State.RECORDING:
                        # Always compute sim metrics if available (saved to sim_rewards.pt)
                        metrics = compute_sim_metrics(env, reward_funcs) if reward_funcs else {}
                        if reward_funcs:
                            recorder.add_sim_rewards(metrics)

                        reward_value = compute_reward(
                            metrics, episode_success_achieved, args.reward_mode,
                            reward_weights, args.success_bonus,
                        )
                        recorder.set_last_reward(reward_value)

                    # Check if Isaac Lab auto-reset happened (terminated or truncated)
                    if terminated[0] or truncated[0]:
                        if state == State.RECORDING:
                            save_recording()
                            print(f"[AUTO-RESET] Episode {dataset.total_episodes - 1} saved ({episode_frame_count} frames)")
                        else:
                            # Debug: check what env thinks about episode length
                            env_episode_len = env.episode_length_buf[0].item()
                            max_len = env.max_episode_length
                            time_out_check = env_episode_len >= max_len - 1
                            print(f"[AUTO-RESET] Isaac Lab triggered reset:")
                            print(f"  terminated={terminated[0].item()}, truncated={truncated[0].item()}")
                            print(f"  episode_length_buf={env_episode_len}, max={max_len}, timeout_check={time_out_check}")
                            print(f"  frame_count={episode_frame_count}")
                        state = State.IDLE
                        device._started = False
                        episode_frame_count = 0
                        episode_success_achieved = False
                        # obs_dict already updated by env.step() auto-reset
                        print("[IDLE] Press Space for teleop, N to record")

                    # Check episode timeout (manual handling if needed)
                    elif env.episode_length_buf[0] >= env.max_episode_length:
                        if state == State.RECORDING:
                            save_recording()
                            print(f"[TIMEOUT] Episode {dataset.total_episodes - 1} saved ({episode_frame_count} frames)")

                        state = State.IDLE
                        device._started = False
                        obs_dict, _ = reset_env_with_seed()
                        episode_frame_count = 0
                        episode_success_achieved = False

                        if args.reset_mode == "auto":
                            waiting_for_auto_start = True
                            auto_start_time = current_time
                            print(f"[AUTO] Will auto-start in {args.reset_delay}s...")
                        else:
                            print("[IDLE] Press Space for teleop, N to record")
                else:
                    env.sim.render()

            # ─── IDLE STATE ───────────────────────────────────────────
            else:
                if "top" in env.scene.sensors:
                    env.scene["top"].update(dt=env.step_dt)
                if "wrist" in env.scene.sensors:
                    env.scene["wrist"].update(dt=env.step_dt)
                env.sim.render()

                obs_dict = env._get_observations()

    # ─── CLEANUP ──────────────────────────────────────────────────
    finally:
        print("\n" + "-" * 60)
        print("Cleaning up...")

        # Stop viewer subprocess
        stop_viewer(viewer_proc)

        if state == State.RECORDING:
            discard_recording()
            print(f"[DISCARD] Incomplete episode discarded ({episode_frame_count} frames)")

        # Save sim rewards side-file if using rewards
        if use_rewards:
            output_path = Path(args.output)
            recorder.save_sim_rewards(output_path / "sim_rewards.pt")

        dataset.close()

        print("\n" + "=" * 60)
        print("Recording Summary")
        print("=" * 60)
        print(f"Total episodes: {dataset.total_episodes}")
        print(f"Total frames: {dataset.total_frames}")
        if use_rewards:
            print(f"Reward mode: {args.reward_mode}")
            output_path = Path(args.output)
            if (output_path / "sim_rewards.pt").exists():
                print(f"Sim rewards saved to: {output_path / 'sim_rewards.pt'}")
        print(f"Dataset saved to: {args.output}")
        if diversity:
            print(diversity.stats())
        print("=" * 60)

        if preview_active:
            cleanup_shm()
            cleanup_command_file()

        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
