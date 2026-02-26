# humanoid_locomotion/common/observations_noisy.py
# SPDX-License-Identifier: BSD-3-Clause

"""
Observation definitions for humanoid locomotion with explicit, reproducible noise injection.

This module implements a conservative-yet-practical observation noise model tailored to:
- control rate: 100 Hz (dt = 0.01 s)
- simulation rate: 200 Hz

Noise model:
- Continuous observations: element-wise uniform additive noise  U(-eps, +eps)
- Binary foot contacts: bit-flip noise with probability p per foot per step
- Projected gravity: additive noise followed by vector renormalization
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Dict, List

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


# =============================================================================
# Noise configuration (reasonable defaults for 100 Hz control)
# =============================================================================

# Commands: [vx, vy, wz]
NOISE_CMD_LIN_MPS: float = 0.02     # m/s
NOISE_CMD_YAW_RPS: float = 0.02     # rad/s

# Base velocities (body frame)
NOISE_BASE_LIN_VEL_MPS: float = 0.04   # m/s
NOISE_BASE_ANG_VEL_RPS: float = 0.025  # rad/s

# Projected gravity (unit vector in body frame)
NOISE_GRAVITY_UNIT: float = 0.015

# Joint observations
NOISE_DOF_POS_RAD: float = 0.004    # rad
NOISE_DOF_VEL_RPS: float = 0.02     # rad/s

# Previous action (assumed normalized in [-1, 1]; if yours is in radians, set ~0.004 rad)
NOISE_LAST_ACTION_NORM: float = 0.015

# Foot speed in world frame (xy)
NOISE_FOOT_SPEED_MPS: float = 0.04  # m/s

# Binary contact bit flips
NOISE_CONTACT_FLIP_PROB: float = 0.01  # 1% per foot per step

# Contact threshold for net_contact_forces_w z component
CONTACT_FZ_THRESHOLD_N: float = 1.0


# =============================================================================
# Noise utilities
# =============================================================================

def _is_training_mode(env) -> bool:
    """Best-effort check to apply noise during training but not during play/eval."""
    # Many IsaacLab workflows set one of these flags; we support several.
    if hasattr(env, "training"):
        try:
            return bool(env.training)
        except Exception:
            pass
    if hasattr(env, "is_play"):
        try:
            return not bool(env.is_play)
        except Exception:
            pass
    # Default: assume training (safe for training scripts).
    return True


def _add_uniform_noise(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Element-wise uniform additive noise: x + U(-eps, +eps)."""
    if eps <= 0.0:
        return x
    return x + (2.0 * torch.rand_like(x) - 1.0) * float(eps)


def _renormalize_vector(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Renormalize vectors along the last dimension to unit length."""
    n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / n


def _flip_bits_01(x01: torch.Tensor, p: float) -> torch.Tensor:
    """
    Flip {0,1} entries with probability p (element-wise).
    x01 is a float tensor taking values in {0,1}.
    """
    if p <= 0.0:
        return x01
    flip = (torch.rand_like(x01) < float(p)).to(x01.dtype)
    # XOR for {0,1}: x' = x + flip - 2*x*flip
    return x01 + flip - 2.0 * x01 * flip


# =============================================================================
# Observation terms (with optional noise)
# =============================================================================

def obs_commands_vxy_wz(env) -> torch.Tensor:
    """(N,3) = [vx, vy, wz] from command manager, with conservative command noise."""
    if hasattr(env, "command_manager"):
        try:
            cmd = env.command_manager.get_command("base_velocity")[:, :3]
            cmd = cmd.to(device=env.device, dtype=torch.float32)
        except Exception:
            cmd = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    else:
        cmd = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)

    if _is_training_mode(env):
        cmd_xy = _add_uniform_noise(cmd[:, 0:2], NOISE_CMD_LIN_MPS)
        cmd_wz = _add_uniform_noise(cmd[:, 2:3], NOISE_CMD_YAW_RPS)
        cmd = torch.cat([cmd_xy, cmd_wz], dim=-1)
    return cmd


def obs_base_linear_velocity_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(N,3) base linear velocity in body frame, with additive noise."""
    robot = env.scene[asset_cfg.name]
    x = robot.data.root_lin_vel_b.to(torch.float32)
    if _is_training_mode(env):
        x = _add_uniform_noise(x, NOISE_BASE_LIN_VEL_MPS)
    return x


def obs_base_angular_velocity_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(N,3) base angular velocity in body frame, with additive noise."""
    robot = env.scene[asset_cfg.name]
    x = robot.data.root_ang_vel_b.to(torch.float32)
    if _is_training_mode(env):
        x = _add_uniform_noise(x, NOISE_BASE_ANG_VEL_RPS)
    return x


@torch.jit.script
def _quat_rotate_inverse_wxyz(q_wxyz: torch.Tensor, v_w: torch.Tensor) -> torch.Tensor:
    """Rotate world vector v_w into body frame given quaternion q (wxyz)."""
    w = q_wxyz[:, 0:1]
    xyz = q_wxyz[:, 1:4]
    t = 2.0 * torch.cross(xyz, v_w, dim=1)
    return v_w + w * t + torch.cross(xyz, t, dim=1)


def obs_projected_gravity_b(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    (N,3) gravity direction expressed in body frame (projected gravity).
    Noise is applied then vector is renormalized.
    """
    robot = env.scene[asset_cfg.name]
    q = robot.data.root_quat_w
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)
    g_b = _quat_rotate_inverse_wxyz(q, g_w).to(torch.float32)

    if _is_training_mode(env):
        g_b = _add_uniform_noise(g_b, NOISE_GRAVITY_UNIT)
        g_b = _renormalize_vector(g_b)
    return g_b


def obs_dof_position_offset(env, asset_cfg: SceneEntityCfg, default_joint_angles: Dict[str, float]) -> torch.Tensor:
    """(N,DoF) joint position offsets (q - q0), with additive noise."""
    q = mdp.joint_pos(env, asset_cfg=asset_cfg).to(torch.float32)
    q0_vals = [float(default_joint_angles.get(j, 0.0)) for j in asset_cfg.joint_names]
    q0 = torch.tensor(q0_vals, device=env.device, dtype=torch.float32)[None, :].repeat(env.num_envs, 1)
    x = q - q0

    if _is_training_mode(env):
        x = _add_uniform_noise(x, NOISE_DOF_POS_RAD)
    return x


def obs_dof_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(N,DoF) joint velocities, with additive noise."""
    x = mdp.joint_vel(env, asset_cfg=asset_cfg).to(torch.float32)
    if _is_training_mode(env):
        x = _add_uniform_noise(x, NOISE_DOF_VEL_RPS)
    return x


def obs_previous_action_fixed_dim(env, action_dim: int) -> torch.Tensor:
    """(N,action_dim) previous action, padded/truncated to fixed size, with additive noise."""
    a = mdp.last_action(env).to(torch.float32)
    if a.shape[-1] > action_dim:
        a = a[..., :action_dim]
    elif a.shape[-1] < action_dim:
        pad = torch.zeros((env.num_envs, action_dim - a.shape[-1]), device=env.device, dtype=torch.float32)
        a = torch.cat([a, pad], dim=-1)

    if _is_training_mode(env):
        a = _add_uniform_noise(a, NOISE_LAST_ACTION_NORM)
    return a


def _body_index(robot, body_name: str) -> int:
    """Return the rigid-body index for a given body name, or -1 if not found."""
    if hasattr(robot.data, "body_names"):
        names = list(robot.data.body_names)
        if body_name in names:
            return names.index(body_name)
    if hasattr(robot, "body_names"):
        names = list(robot.body_names)
        if body_name in names:
            return names.index(body_name)
    return -1


def obs_feet_contact_binary(env, left_foot_body: str, right_foot_body: str) -> torch.Tensor:
    """
    (N,2) binary foot contact state derived from net_contact_forces_w, with bit-flip noise.
    """
    robot = env.scene["robot"]
    if not hasattr(robot.data, "net_contact_forces_w"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    forces = robot.data.net_contact_forces_w
    li = _body_index(robot, left_foot_body)
    ri = _body_index(robot, right_foot_body)
    if li < 0 or ri < 0:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    fz_l = forces[:, li, 2]
    fz_r = forces[:, ri, 2]
    x = torch.stack(
        [(fz_l > CONTACT_FZ_THRESHOLD_N).to(torch.float32),
         (fz_r > CONTACT_FZ_THRESHOLD_N).to(torch.float32)],
        dim=-1
    )

    if _is_training_mode(env):
        x = _flip_bits_01(x, NOISE_CONTACT_FLIP_PROB)
    return x


def obs_foot_speed_xy(env, foot_body: str) -> torch.Tensor:
    """(N,2) foot linear speed in world xy, with additive noise."""
    robot = env.scene["robot"]
    if not hasattr(robot.data, "body_lin_vel_w"):
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    vel = robot.data.body_lin_vel_w
    idx = _body_index(robot, foot_body)
    if idx < 0:
        return torch.zeros((env.num_envs, 2), device=env.device, dtype=torch.float32)

    x = vel[:, idx, :2].to(torch.float32)
    if _is_training_mode(env):
        x = _add_uniform_noise(x, NOISE_FOOT_SPEED_MPS)
    return x


# =============================================================================
# Observation config (Policy/Critic)
# =============================================================================

@configclass
class ObservationsCfg:
    policy: ObsGroup = MISSING
    critic: ObsGroup = MISSING

    @classmethod
    def from_robot_cfg(cls, robot_cfg: Dict) -> "ObservationsCfg":
        cfg = cls()

        action_joints: List[str] = list(robot_cfg["action_joints"])
        action_dim = len(action_joints)
        defaults = dict(robot_cfg["default_joint_angles"])
        lf = str(robot_cfg["left_foot_body"])
        rf = str(robot_cfg["right_foot_body"])

        @configclass
        class PolicyCfg(ObsGroup):
            commands: ObsTerm = ObsTerm(func=obs_commands_vxy_wz)
            base_lin_vel: ObsTerm = ObsTerm(func=obs_base_linear_velocity_b, params={"asset_cfg": SceneEntityCfg("robot")})
            base_ang_vel: ObsTerm = ObsTerm(func=obs_base_angular_velocity_b, params={"asset_cfg": SceneEntityCfg("robot")})
            base_projected_gravity: ObsTerm = ObsTerm(func=obs_projected_gravity_b, params={"asset_cfg": SceneEntityCfg("robot")})
            dof_pos_offset: ObsTerm = ObsTerm(
                func=obs_dof_position_offset,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=action_joints),
                    "default_joint_angles": defaults,
                },
            )
            dof_vel: ObsTerm = ObsTerm(func=obs_dof_velocity, params={"asset_cfg": SceneEntityCfg("robot", joint_names=action_joints)})
            actions: ObsTerm = ObsTerm(func=obs_previous_action_fixed_dim, params={"action_dim": action_dim})

            def __post_init__(self):
                # Keep corruption enabled; we inject explicit noise in each obs function.
                self.enable_corruption = True
                self.concatenate_terms = True

        @configclass
        class CriticCfg(ObsGroup):
            commands: ObsTerm = ObsTerm(func=obs_commands_vxy_wz)
            base_lin_vel: ObsTerm = ObsTerm(func=obs_base_linear_velocity_b, params={"asset_cfg": SceneEntityCfg("robot")})
            base_ang_vel: ObsTerm = ObsTerm(func=obs_base_angular_velocity_b, params={"asset_cfg": SceneEntityCfg("robot")})
            base_projected_gravity: ObsTerm = ObsTerm(func=obs_projected_gravity_b, params={"asset_cfg": SceneEntityCfg("robot")})
            dof_pos_offset: ObsTerm = ObsTerm(
                func=obs_dof_position_offset,
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=action_joints),
                    "default_joint_angles": defaults,
                },
            )
            dof_vel: ObsTerm = ObsTerm(func=obs_dof_velocity, params={"asset_cfg": SceneEntityCfg("robot", joint_names=action_joints)})

            feet_contact: ObsTerm = ObsTerm(func=obs_feet_contact_binary, params={"left_foot_body": lf, "right_foot_body": rf})
            actions: ObsTerm = ObsTerm(func=obs_previous_action_fixed_dim, params={"action_dim": action_dim})

            lf_speed_xy: ObsTerm = ObsTerm(func=obs_foot_speed_xy, params={"foot_body": lf})
            rf_speed_xy: ObsTerm = ObsTerm(func=obs_foot_speed_xy, params={"foot_body": rf})

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        cfg.policy = PolicyCfg()
        cfg.critic = CriticCfg()
        return cfg
