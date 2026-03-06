#!/usr/bin/env python3
"""Interactive camera positioning tool for template scene.

Can be adapted for other scenes by changing camera search pattern in code.

Usage:
    python scripts/utils/interactive_camera_setup.py

Instructions:
    1. Scene loads with robot and cameras
    2. Select camera in Stage panel, adjust Transform in Property panel
    3. Copy and run code from terminal in Script Editor to get values
    4. Copy to so101_lab/tasks/template/env_cfg.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni
from pxr import Gf, UsdGeom

from so101_lab.tasks.template.env import TemplateEnv
from so101_lab.tasks.template.env_cfg import TemplateEnvCfg


def print_camera_transforms():
    """Print current camera transforms for copying to config.

    Run in Isaac Sim Script Editor (Window > Script Editor):
        from scripts.utils.interactive_camera_setup import print_camera_transforms
        print_camera_transforms()
    """
    stage = omni.usd.get_context().get_stage()
    env_path = "/World/envs/env_0"

    cameras = {
        "top": f"{env_path}/top_camera",
        "wrist": f"{env_path}/Robot/gripper_link/wrist_camera",
    }

    print("\n=== Camera Transforms ===")
    for name, prim_path in cameras.items():
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            xformable = UsdGeom.Xformable(prim)
            local_transform = xformable.GetLocalTransformation()

            translation = local_transform.ExtractTranslation()
            rotation = local_transform.ExtractRotation().GetQuat()

            print(f"\n{name} camera:")
            print(f"  pos=({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}),")
            print(f"  rot=({rotation.real:.5f}, {rotation.imaginary[0]:.5f}, {rotation.imaginary[1]:.5f}, {rotation.imaginary[2]:.5f}),")
            print(f"  convention='{'ros' if name == 'wrist' else 'world'}',")
        else:
            print(f"\n{name}: not found at {prim_path}")

    print("\n=== Copy to so101_lab/tasks/template/env_cfg.py ===\n")


def main():
    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 1
    env = TemplateEnv(cfg)
    env.reset()

    top_path = env.scene["top"].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    wrist_path = env.scene["wrist"].cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")

    print("\n" + "=" * 60)
    print("Interactive Camera Setup (Template Scene)")
    print("=" * 60)
    print(f"\nCameras found:")
    print(f"  {top_path}")
    print(f"  {wrist_path}")
    print("\n1. Select camera in Stage panel")
    print("2. Adjust Transform in Property panel")
    print("3. In Script Editor (Window > Script Editor), paste and run:")
    print("\n" + "-" * 60)
    print("""
import omni
from pxr import UsdGeom

stage = omni.usd.get_context().get_stage()

# Find all cameras in scene
cameras = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera):
        path = str(prim.GetPath())
        if "/World/envs/" in path:
            cameras.append(path)

print(f"\\n=== Found {len(cameras)} cameras ===")
for path in cameras:
    prim = stage.GetPrimAtPath(path)
    xform = UsdGeom.Xformable(prim)
    mat = xform.GetLocalTransformation()
    t = mat.ExtractTranslation()
    q = mat.ExtractRotation().GetQuat()

    print(f"\\nprim_path: {path}")
    print(f"  pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}),")
    print(f"  rot=({q.real:.5f}, {q.imaginary[0]:.5f}, {q.imaginary[1]:.5f}, {q.imaginary[2]:.5f}),")

print("\\n=== Copy to so101_lab/tasks/template/env_cfg.py ===")
print("Note: values shown in opengl convention (same as Isaac Sim GUI Transform)")
print("Use convention='opengl' in config for these values")
""")
    print("-" * 60)
    print("\nPress Ctrl+C to exit")
    print("=" * 60 + "\n")

    while simulation_app.is_running():
        env.sim.step()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
