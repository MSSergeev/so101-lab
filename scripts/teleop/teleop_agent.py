#!/usr/bin/env python3
# Copyright (c) 2025, SO-101 Lab Project
# Adapted from: leisaac (https://github.com/isaac-sim/IsaacLab)
# Original license: BSD-3-Clause

"""Teleoperation for SO-101 with multiple device types.

Supports keyboard, gamepad, and SO101Leader (real robot) devices.
For recording demonstrations, use record_episodes.py instead.

Frequency Architecture (v2):
    --physics-hz: PhysX simulation frequency (default: 120 Hz)
    --policy-hz:  Control/action frequency (default: 30 Hz)
    --render-hz:  Rendering frequency (default: 30 Hz)

    By default, simulation runs at maximum speed (faster than realtime).
    Use --realtime flag to limit to real-time speed for comfortable human control.

Usage:
    # Keyboard (default, max speed)
    python scripts/teleop/teleop_agent.py

    # Keyboard with realtime limiting (comfortable for human control)
    python scripts/teleop/teleop_agent.py --realtime

    # Gamepad
    python scripts/teleop/teleop_agent.py --teleop-device=gamepad

    # SO101Leader (real robot)
    python scripts/teleop/teleop_agent.py --teleop-device=so101leader --port=/dev/ttyACM0
    python scripts/teleop/teleop_agent.py --teleop-device=so101leader --recalibrate --calibration-file=my_leader.json

    # High-frequency control
    python scripts/teleop/teleop_agent.py --physics-hz 240 --policy-hz 60 --render-hz 60

Controls vary by device - see docs/TELEOP_GUIDE.md for details.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teleoperation for SO-101")
parser.add_argument("--task", type=str, default="template",
                    help="Task environment (template, pick_cube, figure_shape_placement)")
parser.add_argument("--teleop-device", type=str, default="keyboard",
                    choices=["keyboard", "gamepad", "so101leader"],
                    help="Teleoperation device type")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Control sensitivity multiplier")
parser.add_argument("--physics-hz", type=int, default=120,
                    help="Physics simulation frequency (PhysX timestep)")
parser.add_argument("--policy-hz", type=int, default=30,
                    help="Policy/control frequency (how often actions are updated)")
parser.add_argument("--render-hz", type=int, default=30,
                    help="Rendering frequency (scene + cameras)")
parser.add_argument("--realtime", action="store_true",
                    help="Limit simulation to real-time speed (useful for human control)")
parser.add_argument("--port", type=str, default="/dev/ttyACM0",
                    help="Serial port for SO101Leader device")
parser.add_argument("--recalibrate", action="store_true",
                    help="Force recalibration of SO101Leader motors")
parser.add_argument("--calibration-file", type=str, default="so101_leader.json",
                    help="Calibration file name for SO101Leader")
parser.add_argument("--episode-length", type=float, default=25.0,
                    help="Maximum episode length in seconds")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# ════════════════════════════════════════════════════════════
# VALIDATE FREQUENCY ARGUMENTS
# ════════════════════════════════════════════════════════════

if args.policy_hz > args.physics_hz:
    print(f"\nERROR: policy-hz ({args.policy_hz}) cannot exceed physics-hz ({args.physics_hz})!")
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

if args.teleop_device == "so101leader" and args.recalibrate and args.calibration_file == "so101_leader.json":
    print("\nERROR: When using --recalibrate, specify a unique --calibration-file name!")
    sys.exit(1)

args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main():
    """Main teleoperation loop."""
    import time

    from so101_lab.tasks import get_task

    EnvClass, EnvCfgClass = get_task(args.task)

    env_cfg = EnvCfgClass()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = args.episode_length
    env_cfg.use_teleop_device(args.teleop_device)

    # ════════════════════════════════════════════════════════════
    # AUTOMATIC FREQUENCY CONFIGURATION
    # ════════════════════════════════════════════════════════════

    # Physics timestep
    env_cfg.sim.dt = 1.0 / args.physics_hz

    # Decimation (how many physics steps per policy step)
    env_cfg.decimation = args.physics_hz // args.policy_hz

    # Render interval (how many physics steps per render)
    env_cfg.sim.render_interval = args.physics_hz // args.render_hz

    # Camera update period (should match render frequency)
    camera_update_period = 1.0 / args.render_hz
    if hasattr(env_cfg.scene, 'top'):
        env_cfg.scene.top.update_period = camera_update_period
    if hasattr(env_cfg.scene, 'wrist'):
        env_cfg.scene.wrist.update_period = camera_update_period

    # ════════════════════════════════════════════════════════════

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

    # Print configuration
    print("\n" + "=" * 60)
    print("Teleoperation")
    print("=" * 60)
    print(f"Environment: {args.task}")
    print(f"Device: {args.teleop_device}")
    print("-" * 60)
    print("Frequencies:")
    print(f"  Physics:   {args.physics_hz} Hz (dt={1/args.physics_hz:.4f}s)")
    print(f"  Policy:    {args.policy_hz} Hz (decimation={args.physics_hz//args.policy_hz})")
    print(f"  Render:    {args.render_hz} Hz (interval={args.physics_hz//args.render_hz})")
    if args.realtime:
        print(f"  Realtime:  ENABLED (limited to {args.policy_hz} Hz)")
    else:
        print(f"  Realtime:  DISABLED (max speed)")
    print(f"Episode length: {args.episode_length}s")
    print("-" * 60)
    print("Controls:")
    print("  Space  - Start teleop")
    print("  X      - Reset environment")
    print("  T      - Debug info")
    print("  Escape - Quit")
    print("=" * 60 + "\n")

    device.display_controls()

    env.reset()

    print("\n[INFO] Environment ready. Press Space to start, Escape to exit.")

    import carb.input
    import omni.appwindow
    debug_pressed = False

    # Realtime limiting (optional)
    policy_dt = 1.0 / args.policy_hz

    while simulation_app.is_running():
        loop_start_time = time.time()

        appwindow = omni.appwindow.get_default_app_window()
        input_iface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()

        if input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.ESCAPE):
            print("\n[INFO] Escape pressed, exiting...")
            break

        # Debug info (T key)
        t_pressed = input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.T)
        if t_pressed and not debug_pressed:
            debug_pressed = True
            print("\n" + "=" * 60)
            print("DEBUG INFO")
            print("=" * 60)
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
        elif not t_pressed:
            debug_pressed = False

        action_dict = device.advance()

        if action_dict and action_dict.get("reset"):
            print("[INFO] Resetting environment...")
            env.reset()
        elif action_dict:
            action = env.cfg.preprocess_device_action(action_dict, device)
            env.step(action)

            if env.episode_length_buf[0] >= env.max_episode_length:
                print("[INFO] Episode timeout, resetting...")
                env.reset()
        else:
            env.sim.render()

        # Realtime limiting (only if --realtime flag is set)
        if args.realtime:
            elapsed = time.time() - loop_start_time
            sleep_time = policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
