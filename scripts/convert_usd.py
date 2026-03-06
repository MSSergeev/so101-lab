#!/usr/bin/env python3
"""Convert USDA to USD using Isaac Sim."""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

from pxr import Usd

usda_path = "assets/robots/so101/usd/so101.usda"
usd_path = "assets/robots/so101/usd/so101.usd"

print(f"Converting {usda_path} to {usd_path}...")
stage = Usd.Stage.Open(usda_path)
stage.Export(usd_path)
print("✓ Conversion complete")

simulation_app.close()
