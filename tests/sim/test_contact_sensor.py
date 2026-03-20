#!/usr/bin/env python3
"""Test contact detection on SO-101 gripper with keyboard teleop.

Tests R2/R3 from COMPOSITE_REWARD_TASKS.md:
- R2: Contact detection gripper↔table, gripper↔platform, gripper↔cube
- R3: Grasp detection via contact forces on both jaws

Uses Isaac Lab ContactSensorCfg (GPU tensor API).
- net_forces_w: total force on each jaw (any contact)
- force_matrix_w: per-filter forces (table=primitive works, mesh filters may not)

Usage:
    python tests/sim/test_contact_sensor.py
    python tests/sim/test_contact_sensor.py --realtime
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test contact detection on SO-101 gripper")
parser.add_argument("--realtime", action="store_true",
                    help="Limit to real-time speed for comfortable control")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ════════════════════════════════════════════════════════════════
# Imports after AppLauncher
# ════════════════════════════════════════════════════════════════

import time

import torch
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from so101_lab.tasks.figure_shape_placement.env import FigureShapePlacementEnv
from so101_lab.tasks.figure_shape_placement.env_cfg import (
    FigureShapePlacementEnvCfg,
    FigureShapePlacementSceneCfg,
)

# ════════════════════════════════════════════════════════════════
# Scene config — ContactSensor on both gripper jaws
# ════════════════════════════════════════════════════════════════

FILTER_NAMES = ["Table", "Cube", "Platform"]


@configclass
class ContactTestSceneCfg(FigureShapePlacementSceneCfg):
    """Scene with ContactSensor on gripper bodies."""

    gripper_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Scene/Table",
            "{ENV_REGEX_NS}/Cube",
            "{ENV_REGEX_NS}/Platform",
        ],
        update_period=0.0,
    )
    jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Scene/Table",
            "{ENV_REGEX_NS}/Cube",
            "{ENV_REGEX_NS}/Platform",
        ],
        update_period=0.0,
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.activate_contact_sensors = True


@configclass
class ContactTestEnvCfg(FigureShapePlacementEnvCfg):
    scene: ContactTestSceneCfg = ContactTestSceneCfg(num_envs=1, env_spacing=10.0)


# ════════════════════════════════════════════════════════════════
# Display
# ════════════════════════════════════════════════════════════════

FORCE_THRESHOLD = 0.5   # minimum force to display
GRASP_THRESHOLD = 1.0   # minimum net force on BOTH jaws for grasp


def print_contacts(env: FigureShapePlacementEnv, step_count: int):
    """Read ContactSensor data and print contact report."""
    gripper_sensor = env.scene["gripper_contact"]
    jaw_sensor = env.scene["jaw_contact"]

    # net_forces_w: (N, B, 3) — total force on body from any contact
    g_net = gripper_sensor.data.net_forces_w[0, 0].norm().item()
    j_net = jaw_sensor.data.net_forces_w[0, 0].norm().item()

    # force_matrix_w: (N, B, M, 3) — per-filter forces
    # M=3: [Table, Cube, Platform]
    g_matrix = gripper_sensor.data.force_matrix_w[0, 0]  # (M, 3)
    j_matrix = jaw_sensor.data.force_matrix_w[0, 0]      # (M, 3)

    # Replace NaN with 0 (NaN = no contact for that filter)
    g_matrix = torch.nan_to_num(g_matrix, nan=0.0)
    j_matrix = torch.nan_to_num(j_matrix, nan=0.0)

    # Per-filter force magnitudes
    g_forces = g_matrix.norm(dim=1)  # (M,)
    j_forces = j_matrix.norm(dim=1)  # (M,)

    # Skip if nothing above threshold
    if g_net < FORCE_THRESHOLD and j_net < FORCE_THRESHOLD:
        return

    # Build display
    parts = [f"Net: G={g_net:.1f} J={j_net:.1f}"]
    for i, name in enumerate(FILTER_NAMES):
        gf = g_forces[i].item()
        jf = j_forces[i].item()
        if gf > 0.1 or jf > 0.1:
            parts.append(f"{name}: G={gf:.1f} J={jf:.1f}")

    # Grasp: both jaws have significant net force
    grasped = g_net > GRASP_THRESHOLD and j_net > GRASP_THRESHOLD
    status = " >>> GRASPED <<<" if grasped else ""

    print(f"[{step_count:5d}] {' | '.join(parts)}{status}")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    env_cfg = ContactTestEnvCfg()
    env_cfg.use_teleop_device("keyboard")

    # Standard frequencies
    env_cfg.sim.dt = 1.0 / 120.0
    env_cfg.decimation = 4  # 30 Hz control
    env_cfg.sim.render_interval = 4

    env = FigureShapePlacementEnv(cfg=env_cfg)

    from so101_lab.devices import SO101Keyboard
    device = SO101Keyboard(env, sensitivity=1.0)

    print("\n" + "=" * 70)
    print("Contact Sensor Test (GPU ContactSensor)")
    print("=" * 70)
    print("API:     Isaac Lab ContactSensorCfg (RigidContactView)")
    print("Bodies:  gripper_link (G), moving_jaw (J)")
    print("Filters: Table, Cube, Platform")
    print("  Table  = primitive collider (should work)")
    print("  Cube   = convex hull (may not work as filter)")
    print("  Platform = SDF mesh (may not work as filter)")
    print(f"Thresholds: display={FORCE_THRESHOLD}, grasp={GRASP_THRESHOLD}")
    print("-" * 70)
    print("net_forces_w  = total force from ANY contact (always works)")
    print("force_matrix_w = per-filter force (only for supported colliders)")
    print("-" * 70)
    print("Controls: Space=start, X=reset, Escape=quit")
    print("=" * 70 + "\n")

    device.display_controls()
    env.reset()

    print("\n[INFO] Environment ready. Press Space to start, Escape to exit.")
    print("[INFO] Contact forces will be printed when detected.\n")

    import carb.input
    import omni.appwindow

    policy_dt = 1.0 / 30.0
    step_count = 0

    while simulation_app.is_running():
        loop_start = time.time()

        appwindow = omni.appwindow.get_default_app_window()
        input_iface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()

        if input_iface.get_keyboard_value(keyboard, carb.input.KeyboardInput.ESCAPE):
            print("\n[INFO] Exiting...")
            break

        action_dict = device.advance()

        if action_dict and action_dict.get("reset"):
            print("[INFO] Resetting environment...")
            env.reset()
            step_count = 0
        elif action_dict:
            action = env.cfg.preprocess_device_action(action_dict, device)
            env.step(action)
            step_count += 1
            print_contacts(env, step_count)
        else:
            env.sim.render()

        # Realtime limiting
        if args.realtime:
            elapsed = time.time() - loop_start
            sleep_time = policy_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
