"""IK-based scripted policy for figure shape placement rollout collection."""

from __future__ import annotations

import math
from enum import IntEnum
from pathlib import Path

import torch
import yaml

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    matrix_from_quat, quat_apply, quat_from_euler_xyz, quat_inv, quat_mul,
    subtract_frame_transforms,
)

from so101_lab.policies.rl.neural_ik import NeuralIK
from so101_lab.tasks.figure_shape_placement.rl.mdp.observations import slot_pos_w


class Phase(IntEnum):
    NEUTRAL = 0    # fixed safe position in front of robot
    APPROACH = 1   # near cube, low height, XY offset toward robot base
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    TRANSIT = 5
    DESCEND_SLOT = 6
    RELEASE = 7
    DONE = 8


_DEFAULT_CFG = Path(__file__).resolve().parents[3] / "configs" / "collect_rollouts.yaml"


class WaypointInterpolator:
    """Per-env EE position interpolator with trapezoidal velocity profile.

    Moves a commanded EE position toward the target at limited speed with smooth
    acceleration and deceleration. Ensures IK always receives a nearby target.
    """

    def __init__(self, num_envs: int, max_speed: float, accel: float, dt: float, device):
        self.max_speed = max_speed
        self.accel = accel
        self.dt = dt
        self._pos = torch.zeros(num_envs, 3, device=device)
        self._vel = torch.zeros(num_envs, device=device)

    def reset(self, env_ids: torch.Tensor, start_pos: torch.Tensor) -> None:
        self._pos[env_ids] = start_pos
        self._vel[env_ids] = 0.0

    def step(self, target_pos: torch.Tensor) -> torch.Tensor:
        """Advance commanded position toward target. Returns (N, 3)."""
        diff = target_pos - self._pos
        dist = diff.norm(dim=1).clamp(min=1e-6)
        direction = diff / dist.unsqueeze(1)

        brake_dist = self._vel.pow(2) / (2.0 * self.accel + 1e-8)
        should_brake = brake_dist >= dist

        new_vel = self._vel.clone()
        acc_mask = ~should_brake & (self._vel < self.max_speed)
        new_vel[acc_mask] = (self._vel[acc_mask] + self.accel * self.dt).clamp(max=self.max_speed)
        new_vel[should_brake] = (self._vel[should_brake] - self.accel * self.dt).clamp(min=0.0)

        step = (new_vel * self.dt).clamp(max=dist)
        self._pos = self._pos + direction * step.unsqueeze(1)
        self._vel = new_vel
        return self._pos.clone()


class IKScriptedPolicy:
    """State-machine pick-and-place policy.

    IK backend: "dls" (DifferentialIKController) or "neural" (NeuralIK MLP).
    EE position interpolated with trapezoidal velocity profile for smooth motion.
    Works with raw ManagerBasedRLEnv (num_envs >= 1).
    Output: (num_envs, 6) joint targets in radians.
    """

    ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    def __init__(self, env, cfg_path: str | Path | None = None):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs

        with open(cfg_path or _DEFAULT_CFG) as f:
            cfg = yaml.safe_load(f)["ik"]

        neutral_raw = cfg.get("neutral_pos", [0.0, 0.20, 0.15])
        self.neutral_pos = torch.tensor(neutral_raw, dtype=torch.float32, device=self.device)
        self.approach_height = cfg.get("approach_height", 0.07)
        self.grasp_height = cfg["grasp_height"]
        self.slot_approach_height = cfg["slot_approach_height"]
        self.slot_descent_height = cfg["slot_descent_height"]
        self.pos_threshold = cfg["position_threshold"]
        self.approach_xy_threshold = cfg.get("approach_xy_threshold", 0.020)
        self.z_threshold = cfg.get("transition_z_threshold", 0.015)
        self.phase_timeout = cfg["phase_timeout_steps"]
        self.grasp_hold_steps = cfg["grasp_hold_steps"]
        self.gripper_open = cfg["gripper_open_rad"]
        self.gripper_closed = cfg["gripper_closed_rad"]
        self.grasp_noise_xy_std = cfg.get("grasp_noise_xy_std", cfg.get("noise_xy_std", 0.0))
        self.slot_noise_xy_std = cfg.get("slot_noise_xy_std", 0.0)
        self.grasp_side_offset = cfg["grasp_side_offset"]

        # Resolve arm joints
        self._arm_cfg = SceneEntityCfg(
            "robot",
            joint_names=self.ARM_JOINTS,
            body_names=["gripper_frame_link"],
        )
        self._arm_cfg.resolve(env.scene)
        self._ee_jacobi_idx = self._arm_cfg.body_ids[0] - 1  # fixed-base offset

        robot = env.scene["robot"]
        self._gripper_idx = robot.joint_names.index("gripper")

        # IK backend
        self._ik_method = cfg.get("method", "dls")
        if self._ik_method == "neural":
            checkpoint = cfg.get("neural_ik_checkpoint", "outputs/neural_ik")
            iters = cfg.get("neural_ik_iters", 1)
            self._neural_ik = NeuralIK(checkpoint, device=self.device, iters=iters)
            self._ik = None
        else:
            ik_cfg = DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
            )
            self._ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)
            self._neural_ik = None

        # Waypoint interpolator: trapezoidal velocity in EE space
        dt = float(env.step_dt)
        self._interp = WaypointInterpolator(
            self.num_envs,
            max_speed=cfg.get("max_ee_speed", 0.10),
            accel=cfg.get("ee_accel", 0.50),
            dt=dt,
            device=self.device,
        )
        self._interp_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # R_y(π): maps gripper Z → cube's local -Z (approach perpendicular to cube's top face).
        self._approach_offset = quat_from_euler_xyz(
            torch.tensor([0.0], device=self.device),
            torch.tensor([math.pi], device=self.device),
            torch.tensor([0.0], device=self.device),
        )  # (1, 4)

        # 4 yaw variants (90° apart) to handle cube 4-fold symmetry.
        self._rz_offsets = torch.stack([
            quat_from_euler_xyz(
                torch.tensor([0.0], device=self.device),
                torch.tensor([0.0], device=self.device),
                torch.tensor([k * math.pi / 2], device=self.device),
            ).squeeze(0)
            for k in range(4)
        ], dim=0)  # (4, 4)

        # Per-env state
        self._phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grasp_noise = torch.zeros(self.num_envs, 2, device=self.device)
        self._slot_noise = torch.zeros(self.num_envs, 2, device=self.device)
        # cube_pos_w - ee_pos_w captured at GRASP→LIFT; compensates slot target XY
        # so cube center lands at slot center regardless of grasp imprecision.
        self._grasp_cube_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self._gripper_target = torch.full((self.num_envs,), self.gripper_open, device=self.device)
        # Cached initial EE position (world frame). Set on first compute() after env.reset()
        # so data is guaranteed fresh. Used to init interpolator after resets — ee_frame data
        # is stale right after env._reset_idx() until PhysX recomputes FK on next sim step.
        self._initial_ee_pos_w: torch.Tensor | None = None

    def reset(self, env_ids: torch.Tensor) -> None:
        self._phase[env_ids] = Phase.NEUTRAL
        self._phase_steps[env_ids] = 0
        self._gripper_target[env_ids] = self.gripper_open
        n = len(env_ids)
        gn = torch.zeros(n, 2, device=self.device)
        gn[:, 0].normal_(0, self.grasp_noise_xy_std)
        gn[:, 1].normal_(0, self.grasp_noise_xy_std)
        self._grasp_noise[env_ids] = gn
        sn = torch.zeros(n, 2, device=self.device)
        sn[:, 0].normal_(0, self.slot_noise_xy_std)
        sn[:, 1].normal_(0, self.slot_noise_xy_std)
        self._slot_noise[env_ids] = sn
        self._grasp_cube_offset[env_ids] = 0.0
        if self._ik is not None:
            self._ik.reset(env_ids)
        # Interpolator will init from actual EE pos on next compute()
        self._interp_initialized[env_ids] = False

    def compute(self) -> torch.Tensor:
        """Returns (num_envs, 6) joint targets in radians."""
        robot = self.env.scene["robot"]
        ee_frame = self.env.scene["ee_frame"]

        ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]   # (N, 3) world
        ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]  # (N, 4) world
        root_pose_w = robot.data.root_pose_w               # (N, 7)

        # Cache initial EE position on first call (scene data is fresh after env.reset())
        if self._initial_ee_pos_w is None:
            self._initial_ee_pos_w = ee_pos_w.clone()

        # Init interpolator for newly reset envs from cached initial EE position.
        # ee_frame data is stale after env._reset_idx() — PhysX only recomputes FK
        # on the next sim step, so reading ee_pos_w here gives the end-of-episode position.
        uninit = ~self._interp_initialized
        if uninit.any():
            ids = uninit.nonzero(as_tuple=False).squeeze(1)
            self._interp.reset(ids, self._initial_ee_pos_w[ids])
            self._interp_initialized[ids] = True

        # EE in base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pos_w, ee_quat_w
        )

        # Final waypoints (for interpolator target + phase transition check)
        final_target_w = self._get_targets(ee_quat_w)

        # Advance interpolator → commanded EE position this step
        cmd_pos_w = self._interp.step(final_target_w[:, :3])
        cmd_target_w = torch.cat([cmd_pos_w, final_target_w[:, 3:]], dim=-1)

        # Target in base frame
        target_b = self._world_to_base(cmd_target_w, root_pose_w)

        joint_pos_arm = robot.data.joint_pos[:, self._arm_cfg.joint_ids]  # (N, 5)

        if self._ik_method == "neural":
            arm_target = self._neural_ik.compute(target_b[:, :3], target_b[:, 3:], joint_pos_arm)
        else:
            jacobian_full = robot.root_physx_view.get_jacobians()
            jac_w = jacobian_full[:, self._ee_jacobi_idx, :, :][:, :, self._arm_cfg.joint_ids]
            base_rot_mat = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jac_b = jac_w.clone()
            jac_b[:, :3, :] = torch.bmm(base_rot_mat, jac_w[:, :3, :])
            jac_b[:, 3:, :] = torch.bmm(base_rot_mat, jac_w[:, 3:, :])
            self._ik.set_command(target_b, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
            arm_target = self._ik.compute(ee_pos_b, ee_quat_b, jac_b, joint_pos_arm)

        # Build full action; hold phases freeze arm joints to avoid IK drift
        hold = (self._phase == Phase.GRASP) | (self._phase == Phase.RELEASE)
        arm_target[hold] = joint_pos_arm[hold]
        action = robot.data.joint_pos.clone()
        action[:, self._arm_cfg.joint_ids] = arm_target
        action[:, self._gripper_idx] = self._gripper_target

        # Update phase (check against FINAL target, not interpolated)
        changed = self._update_phase(ee_pos_w, final_target_w[:, :3])
        if len(changed) > 0:
            lift_envs = changed[self._phase[changed] == Phase.LIFT]
            if lift_envs.numel() > 0:
                cube = self.env.scene["cube"]
                self._grasp_cube_offset[lift_envs] = (
                    cube.data.root_pos_w[lift_envs] - ee_pos_w[lift_envs])
            self._interp.reset(changed, ee_pos_w[changed])

        return action

    def is_done(self) -> torch.Tensor:
        """Returns (num_envs,) bool."""
        return self._phase >= Phase.DONE

    # ------------------------------------------------------------------

    def _get_targets(self, ee_quat_w: torch.Tensor) -> torch.Tensor:
        """Compute final waypoint targets in world frame. Returns (N, 7) pose."""
        cube = self.env.scene["cube"]
        cube_pos_w = cube.data.root_pos_w   # (N, 3)
        cube_quat_w = cube.data.root_quat_w  # (N, 4)

        slot_local = slot_pos_w(self.env)
        slot_w = slot_local + self.env.scene.env_origins

        gnoise = self._grasp_noise   # (N, 2) — grasp-side diversity
        snoise = self._slot_noise    # (N, 2) — slot-side diversity

        z_local = torch.zeros(self.num_envs, 3, device=self.device)
        z_local[:, 2] = 1.0
        cube_normal_w = quat_apply(cube_quat_w, z_local)  # (N, 3)

        x_local = torch.zeros(self.num_envs, 3, device=self.device)
        x_local[:, 0] = 1.0

        offset = self._approach_offset.expand(self.num_envs, -1)
        base_quat = quat_mul(cube_quat_w, offset)

        rz = self._rz_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        candidates = quat_mul(
            base_quat.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 4),
            rz.reshape(-1, 4),
        ).reshape(self.num_envs, 4, 4)

        dots = (candidates * ee_quat_w.unsqueeze(1)).sum(dim=-1).abs()
        best = dots.argmax(dim=1)
        cube_approach_quat = candidates[torch.arange(self.num_envs, device=self.device), best]

        cube_x_w = quat_apply(cube_approach_quat, x_local)
        ee_x_w = quat_apply(ee_quat_w, x_local)

        targets_pos = torch.zeros(self.num_envs, 3, device=self.device)
        targets_quat = ee_quat_w.clone()

        m = self._phase == Phase.NEUTRAL
        if m.any():
            # neutral_pos = [x, y, z] in robot base frame (local).
            # XY rotated by robot yaw; Z added directly (no rotation).
            rp = self.env.scene["robot"].data.root_pose_w
            n_xy = torch.stack([self.neutral_pos[0], self.neutral_pos[1],
                                 torch.zeros(1, device=self.device).squeeze()])
            root_rot = matrix_from_quat(rp[m, 3:7])  # (M, 3, 3)
            n_w = torch.bmm(root_rot, n_xy.expand(m.sum(), -1).unsqueeze(-1)).squeeze(-1)
            targets_pos[m] = rp[m, :3] + n_w
            targets_pos[m, 2] = rp[m, 2] + self.neutral_pos[2]  # rp[z] ≈ table level
            targets_quat[m] = cube_approach_quat[m]

        m = self._phase == Phase.APPROACH
        if m.any():
            targets_pos[m] = (cube_pos_w[m] + cube_normal_w[m] * self.approach_height
                              + cube_x_w[m] * self.grasp_side_offset)
            targets_pos[m, 0] += gnoise[m, 0]
            targets_pos[m, 1] += gnoise[m, 1]
            targets_quat[m] = cube_approach_quat[m]

        for phase in (Phase.DESCEND, Phase.GRASP):
            m = self._phase == phase
            if not m.any():
                continue
            targets_pos[m] = (cube_pos_w[m] + cube_normal_w[m] * self.grasp_height
                              + cube_x_w[m] * self.grasp_side_offset)
            targets_pos[m, 0] += gnoise[m, 0]
            targets_pos[m, 1] += gnoise[m, 1]
            targets_quat[m] = cube_approach_quat[m]

        m = self._phase == Phase.LIFT
        if m.any():
            rp = self.env.scene["robot"].data.root_pose_w
            n_xy = torch.stack([self.neutral_pos[0], self.neutral_pos[1],
                                 torch.zeros(1, device=self.device).squeeze()])
            root_rot = matrix_from_quat(rp[m, 3:7])
            n_w = torch.bmm(root_rot, n_xy.expand(m.sum(), -1).unsqueeze(-1)).squeeze(-1)
            targets_pos[m] = rp[m, :3] + n_w
            targets_pos[m, 2] = rp[m, 2] + self.neutral_pos[2]

        m = self._phase == Phase.TRANSIT
        if m.any():
            off = self._grasp_cube_offset[m]
            targets_pos[m, :2] = slot_w[m, :2] - off[:, :2]
            targets_pos[m, 2] = slot_w[m, 2] + self.slot_approach_height

        m = (self._phase == Phase.DESCEND_SLOT) | (self._phase == Phase.RELEASE)
        if m.any():
            off = self._grasp_cube_offset[m]
            targets_pos[m, :2] = slot_w[m, :2] - off[:, :2]
            targets_pos[m, 2] = slot_w[m, 2] + self.slot_descent_height

        m = self._phase >= Phase.DONE
        if m.any():
            targets_pos[m] = self.env.scene["ee_frame"].data.target_pos_w[m, 0, :]

        return torch.cat([targets_pos, targets_quat], dim=-1)  # (N, 7)

    def _world_to_base(self, targets_w: torch.Tensor, root_pose_w: torch.Tensor) -> torch.Tensor:
        pos_b, quat_b = subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7],
            targets_w[:, :3], targets_w[:, 3:],
        )
        return torch.cat([pos_b, quat_b], dim=-1)

    def _update_phase(self, ee_pos_w: torch.Tensor, target_pos_w: torch.Tensor) -> torch.Tensor:
        """Update state machine. Returns tensor of env_ids that changed phase."""
        dist_3d = (ee_pos_w - target_pos_w).norm(dim=1)
        dist_xy = (ee_pos_w[:, :2] - target_pos_w[:, :2]).norm(dim=1)
        dist_z = (ee_pos_w[:, 2] - target_pos_w[:, 2]).abs()
        self._phase_steps += 1
        changed = []

        for i in range(self.num_envs):
            phase = int(self._phase[i])
            steps = int(self._phase_steps[i])
            timeout = (self.phase_timeout > 0) and (steps >= self.phase_timeout)

            if phase == Phase.DONE:
                continue

            # APPROACH: XY-only check — ensure alignment before descent
            # DESCEND: separate XY/Z check — prevent gripper closing above cube
            if phase == Phase.APPROACH:
                close = dist_xy[i].item() < self.approach_xy_threshold
            elif phase == Phase.DESCEND:
                close = (dist_xy[i].item() < self.pos_threshold
                         and dist_z[i].item() < self.z_threshold)
            else:
                close = dist_3d[i].item() < self.pos_threshold

            if phase == Phase.NEUTRAL:
                if close or timeout:
                    self._phase[i] = Phase.APPROACH
                    self._phase_steps[i] = 0
                    changed.append(i)
            elif phase == Phase.APPROACH:
                if close or timeout:
                    self._phase[i] = Phase.DESCEND
                    self._phase_steps[i] = 0
                    changed.append(i)
            elif phase == Phase.DESCEND:
                if close or timeout:
                    self._phase[i] = Phase.GRASP
                    self._phase_steps[i] = 0
                    self._gripper_target[i] = self.gripper_closed
                    changed.append(i)
            elif phase == Phase.GRASP:
                if steps >= self.grasp_hold_steps:
                    self._phase[i] = Phase.LIFT
                    self._phase_steps[i] = 0
                    changed.append(i)
            elif phase == Phase.LIFT:
                if close or timeout:
                    self._phase[i] = Phase.TRANSIT
                    self._phase_steps[i] = 0
                    changed.append(i)
            elif phase == Phase.TRANSIT:
                if close or timeout:
                    self._phase[i] = Phase.DESCEND_SLOT
                    self._phase_steps[i] = 0
                    changed.append(i)
            elif phase == Phase.DESCEND_SLOT:
                if close or timeout:
                    self._phase[i] = Phase.RELEASE
                    self._phase_steps[i] = 0
                    self._gripper_target[i] = self.gripper_open
                    changed.append(i)
            elif phase == Phase.RELEASE:
                if steps >= self.grasp_hold_steps:
                    self._phase[i] = Phase.DONE
                    self._phase_steps[i] = 0
                    changed.append(i)

        return (torch.tensor(changed, dtype=torch.long, device=self.device)
                if changed else torch.empty(0, dtype=torch.long, device=self.device))
