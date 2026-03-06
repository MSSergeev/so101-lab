"""SO-101 robot configuration for Isaac Lab."""

from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_USD_DIR = Path(__file__).resolve().parents[3] / "assets" / "robots" / "so101" / "usd"

# so101_w_cam_simp.usd - with camera mount, simplified collision meshes (default)
# so101_w_cam.usd - with camera mount, full meshes
SO101_USD_PATH = _USD_DIR / "so101_w_cam_simp.usd"

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(SO101_USD_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.254, 0.278, 0.752),  # On top of table at specific position
        rot=(0.707107, 0.0, 0.0, -0.707107),  # -90° around Z axis
        joint_pos={
            "shoulder_pan": 0.0,      # 0°
            "shoulder_lift": -1.733,  # -99.32° (limit: -100°)
            "elbow_flex": 1.587,      # 90.93° (limit: 96.83°)
            "wrist_flex": 0.992,      # 56.82°
            "wrist_roll": 0.0,        # 0°
            "gripper": -0.174,        # -10° (limit: -10°)
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=5,
            velocity_limit_sim=5,
            stiffness=10.8,
            damping=0.9,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=4,
            velocity_limit_sim=5,
            stiffness=10.8,
            damping=0.9,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# Joint limits from USD file (verified in Isaac Sim GUI)
SO101_USD_JOINT_LIMITS = {
    "shoulder_pan": (-110.0, 110.0),  # [-1.91986, +1.91986] rad
    "shoulder_lift": (-100.0, 100.0), # [-1.74533, +1.74533] rad
    "elbow_flex": (-96.8, 96.8),      # [-1.68977, +1.68977] rad
    "wrist_flex": (-95.0, 95.0),      # [-1.65806, +1.65806] rad
    "wrist_roll": (-157.0, 162.0),    # [-2.74017, +2.82743] rad (asymmetric)
    "gripper": (-10.0, 100.0),        # [-0.17453, +1.74533] rad
}

# Normalized motor limits as returned by SO101Leader after calibration
SO101_LEADER_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}
