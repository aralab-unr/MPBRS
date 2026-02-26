# humanoid_locomotion/common/rewards.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Dict, Optional, List

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv, mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


# -----------------------------------------------------------------------------
# Small robust helpers for IsaacLab version differences
# -----------------------------------------------------------------------------

def _scene_entity_with_joints(name: str, joints: List[str]) -> SceneEntityCfg:
    """Create SceneEntityCfg for joints, supporting different IsaacLab field names."""
    try:
        return SceneEntityCfg(name, joint_names=list(joints))
    except TypeError:
        pass
    try:
        return SceneEntityCfg(name, joint_names_expr=list(joints))
    except TypeError as e:
        raise TypeError(
            "SceneEntityCfg does not accept joint_names or joint_names_expr in this IsaacLab version."
        ) from e


def _get_asset_cfg_joint_list(asset_cfg: SceneEntityCfg) -> List[str]:
    """Read the joint list from SceneEntityCfg, whichever field exists."""
    for attr in ("joint_names", "joint_names_expr"):
        v = getattr(asset_cfg, attr, None)
        if v:
            return list(v)
    return []


def _zero_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Always returns zeros (safe placeholder reward)."""
    return torch.zeros((env.num_envs,), device=env.device, dtype=torch.float32)


def _robot_has_joint(robot_cfg: Dict, joint_name: Optional[str]) -> bool:
    if not joint_name:
        return False
    all_joints = robot_cfg.get("all_joints", None)
    if all_joints is None:
        return False
    return str(joint_name) in list(all_joints)


# -----------------------------------------------------------------------------
# Reset mask helper (PBRS needs this)
# -----------------------------------------------------------------------------

def _get_reset_mask(env: ManagerBasedRLEnv) -> Optional[torch.Tensor]:
    """Best-effort per-env reset mask."""
    if hasattr(env, "reset_buf"):
        return env.reset_buf
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf == 0
    return None


# -----------------------------------------------------------------------------
# Potential-based reward shaping (PBRS)
# -----------------------------------------------------------------------------

def pbrs(
    env: ManagerBasedRLEnv,
    potential_func,
    cache_key: str,
    potential_kwargs: Dict,
    gamma: float = 1.0,
    z_des: float | None = None,   # accept leaked z_des safely
) -> torch.Tensor:
    """
    Potential-based reward shaping:
        r = gamma * phi_now - phi_prev
    """
    # z_des is intentionally unused here; it may be injected by older Hydra configs.
    _ = z_des

    phi_now = potential_func(env, **potential_kwargs)  # (N,) or (N,1)
    if phi_now.ndim == 2:
        phi_now = phi_now.squeeze(-1)

    phi_now_detached = phi_now.detach()
    key = f"pbrs:{cache_key}"

    # First call: initialize cache, no shaping yet.
    if key not in env.extras:
        env.extras[key] = phi_now_detached.clone()
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    phi_prev = env.extras[key]
    shaped = float(gamma) * phi_now - phi_prev

    # Zero shaping on reset steps + refresh cache
    reset_mask = _get_reset_mask(env)
    if reset_mask is not None:
        shaped = torch.where(reset_mask, torch.zeros_like(shaped), shaped)
        phi_prev = torch.where(reset_mask, phi_now_detached, phi_prev)

    env.extras[key] = phi_now_detached.clone()
    return shaped.to(torch.float32)


# -----------------------------------------------------------------------------
# Sensor helpers
# -----------------------------------------------------------------------------

def _feet_contact_from_sensor(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Return (N,2) float contact flags from ContactSensor forces threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = None
    for attr in ("net_forces_w", "net_contact_forces_w", "forces_w"):
        if hasattr(contact_sensor.data, attr):
            forces = getattr(contact_sensor.data, attr)
            break

    if forces is None and hasattr(contact_sensor.data, "net_forces_w_history"):
        forces = contact_sensor.data.net_forces_w_history.max(dim=1)[0]

    if forces is None:
        return torch.zeros(
            (env.num_envs, len(sensor_cfg.body_ids)),
            device=env.device,
            dtype=torch.float32,
        )

    fz = forces[:, sensor_cfg.body_ids, 2]
    return (fz > threshold).to(torch.float32)


def _base_height_from_rays(env: ManagerBasedRLEnv, ray_sensor_name: str = "height_scanner") -> torch.Tensor:
    """Returns (N,1) distance from base_link to ground using RayCaster hits."""
    ray = env.scene.sensors[ray_sensor_name]
    data = ray.data

    if hasattr(data, "ray_dists"):
        d = data.ray_dists  # (N, R)
        d = torch.where(torch.isfinite(d), d, torch.tensor(10.0, device=env.device, dtype=d.dtype))
        return torch.min(d, dim=1, keepdim=True)[0].to(torch.float32)

    if hasattr(data, "ray_hits_w"):
        hits = data.ray_hits_w  # (N, R, 3)
        z_hit = hits[..., 2]
        z_hit = torch.where(torch.isfinite(z_hit), z_hit, torch.tensor(-1e6, device=env.device, dtype=z_hit.dtype))
        ground_z = torch.max(z_hit, dim=1, keepdim=True)[0]
        base_z = env.scene["robot"].data.root_pos_w[:, 2:3]
        return (base_z - ground_z).to(torch.float32)

    return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Core reward terms (tracking + posture)
# -----------------------------------------------------------------------------

def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / (float(std) ** 2))


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / (float(std) ** 2))


def rew_upright_from_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Upright reward based on gravity alignment in body frame (absolute, not PBRS)."""
    robot = env.scene[asset_cfg.name]
    q = robot.data.root_quat_w
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)

    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, g_w, dim=1)
    g_b = g_w + w * t + torch.cross(xyz, t, dim=1)

    err = torch.abs(g_b[:, 2] + 1.0)
    return torch.exp(-4.0 * err)


def rew_pose_similarity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    default_joint_angles: Dict[str, float],
) -> torch.Tensor:
    """Keep the specified joints near provided default angles."""
    q = mdp.joint_pos(env, asset_cfg=asset_cfg)

    jnames = _get_asset_cfg_joint_list(asset_cfg)
    if len(jnames) == 0:
        return _zero_reward(env)

    q0_vals = [float(default_joint_angles.get(j, 0.0)) for j in jnames]
    q0 = torch.tensor(q0_vals, device=env.device, dtype=torch.float32)[None, :].repeat(env.num_envs, 1)

    err = torch.mean(torch.abs(q - q0), dim=1)
    return torch.exp(-1.0 * err)


def rew_waist_default(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    default_joint_angles: Dict[str, float],
    waist_name: str,
) -> torch.Tensor:
    """Keep waist near default. Requires asset_cfg to select exactly one joint."""
    q = mdp.joint_pos(env, asset_cfg=asset_cfg)  # (N,1)
    q0 = float(default_joint_angles.get(waist_name, 0.0))
    err = torch.abs(q[:, 0] - q0)
    return torch.exp(-8.0 * err)


# -----------------------------------------------------------------------------
# Potentials for PBRS: upright, height, feet-air-time, joint-reg
# -----------------------------------------------------------------------------

def phi_upright_from_gravity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Upright potential based on gravity alignment in body frame.
    Returns bounded potential in (0, 1], higher is more upright.
    """
    robot = env.scene[asset_cfg.name]
    q = robot.data.root_quat_w
    g_w = torch.tensor([0.0, 0.0, -1.0], device=env.device, dtype=torch.float32).repeat(env.num_envs, 1)

    w = q[:, 0:1]
    xyz = q[:, 1:4]
    t = 2.0 * torch.cross(xyz, g_w, dim=1)
    g_b = g_w + w * t + torch.cross(xyz, t, dim=1)

    err = torch.abs(g_b[:, 2] + 1.0)
    phi = torch.exp(-4.0 * err)
    return float(scale) * phi.to(torch.float32)


def phi_height_from_rays(
    env: ManagerBasedRLEnv,
    z_des: float = 0.68,
    sigma_h: float = 0.08,
    ray_sensor_name: str = "height_scanner",
    scale: float = 1.0,
) -> torch.Tensor:
    h = _base_height_from_rays(env, ray_sensor_name=ray_sensor_name)  # (N,1)
    err2 = (h.squeeze(-1) - float(z_des)) ** 2
    phi = torch.exp(-err2 / (float(sigma_h) ** 2))
    return float(scale) * phi


def phi_feet_air_time_biped(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    scale: float = 1.0,
) -> torch.Tensor:
    cs: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = cs.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = cs.data.current_contact_time[:, sensor_cfg.body_ids]

    in_contact = contact_time > 0.0
    single_stance = (in_contact.int().sum(dim=1) == 1)

    swing = (~in_contact).float()
    phi = (air_time * swing).sum(dim=1)
    phi = torch.clamp(phi, max=float(threshold))
    phi = phi * single_stance.float()
    return float(scale) * phi


def phi_joint_reg_paper(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma_q: float = 0.5,
    joint_names: Optional[Dict[str, str]] = None,
    scale: float = 1.0,
    pair_mode: Optional[Dict[str, str]] = None,
) -> torch.Tensor:
    """
    Joint regularization potential robust to mirrored left/right sign conventions.

    For some humanoids, a symmetric pose satisfies qL ~= qR ("same" mode).
    For others (mirrored joint axes), symmetric pose satisfies qL ~= -qR ("opposite" mode).

    We auto-detect per pair (ankle_roll, ankle_pitch) by comparing mean |qL-qR| vs |qL+qR|
    and cache the decision in env.extras. You can also override with pair_mode.
    """
    if joint_names is None:
        raise ValueError("phi_joint_reg_paper requires joint_names mapping dict")

    required_keys = [
        "l_ankle_roll",
        "r_ankle_roll",
        "l_ankle_pitch",
        "r_ankle_pitch",
        "l_hip_yaw",
        "r_hip_yaw",
    ]
    missing_keys = [k for k in required_keys if k not in joint_names]
    if missing_keys:
        raise KeyError(f"phi_joint_reg_paper joint_names missing keys: {missing_keys}")

    robot: Articulation = env.scene[asset_cfg.name]

    needed_names = [str(joint_names[k]) for k in required_keys]
    cache_key = "joint_reg:ids:" + "|".join(needed_names)

    if cache_key not in env.extras:
        jnames = list(getattr(robot, "joint_names", None) or getattr(robot.data, "joint_names", None) or [])
        name_to_id = {n: i for i, n in enumerate(jnames)}

        missing = [n for n in needed_names if n not in name_to_id]
        if missing:
            raise KeyError(f"phi_joint_reg_paper missing joints in articulation: {missing}")

        env.extras[cache_key] = torch.tensor([name_to_id[n] for n in needed_names], device=env.device, dtype=torch.long)

    ids = env.extras[cache_key]
    q = robot.data.joint_pos

    qLa = q[:, ids[0]]  # left ankle roll
    qRa = q[:, ids[1]]  # right ankle roll
    qLp = q[:, ids[2]]  # left ankle pitch
    qRp = q[:, ids[3]]  # right ankle pitch
    qLy = q[:, ids[4]]  # left hip yaw
    qRy = q[:, ids[5]]  # right hip yaw

    def _normalize_mode(m: Optional[str]) -> str:
        if m is None:
            return "auto"
        m = str(m).strip().lower()
        if m in ("auto", "same", "opposite"):
            return m
        return "auto"

    pair_mode = pair_mode or {}
    m_roll = _normalize_mode(pair_mode.get("ankle_roll", None))
    m_pitch = _normalize_mode(pair_mode.get("ankle_pitch", None))

    def _infer_pair_sign_key(pair_name: str, qL: torch.Tensor, qR: torch.Tensor) -> int:
        """
        Returns +1 for 'same' (use qL - qR), -1 for 'opposite' (use qL + qR).
        Cache result in env.extras per robot/joint-pair.
        """
        k = f"{cache_key}:pair_sign:{pair_name}"
        if k in env.extras:
            return int(env.extras[k])

        err_same = torch.mean(torch.abs(qL - qR)).detach()
        err_opp = torch.mean(torch.abs(qL + qR)).detach()

        sign = -1 if err_opp < err_same else +1
        env.extras[k] = int(sign)
        return sign

    def _pair_error2(pair_name: str, qL: torch.Tensor, qR: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "same":
            e = (qL - qR)
        elif mode == "opposite":
            e = (qL + qR)
        else:
            sign = _infer_pair_sign_key(pair_name, qL, qR)
            e = (qL - qR) if sign == +1 else (qL + qR)
        return e * e

    s = float(sigma_q)

    # Ankle roll
    e2_roll = _pair_error2("ankle_roll", qLa, qRa, m_roll)
    phi = torch.exp(-(e2_roll) / s)

    # Ankle pitch
    e2_pitch = _pair_error2("ankle_pitch", qLp, qRp, m_pitch)
    phi = phi + torch.exp(-(e2_pitch) / s)

    # Hip yaw terms (paper-style): keep each near zero
    phi = phi + torch.exp(-(qLy ** 2) / s)
    phi = phi + torch.exp(-(qRy ** 2) / s)

    return float(scale) * phi


# -----------------------------------------------------------------------------
# Action penalties
# -----------------------------------------------------------------------------

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action), dim=1)


# -----------------------------------------------------------------------------
# Reward config
# -----------------------------------------------------------------------------

@configclass
class RewardsCfg:
    terminating: RewTerm = MISSING
    track_lin_vel_xy_exp: RewTerm = MISSING
    track_ang_vel_z_exp: RewTerm = MISSING
    waist_default: RewTerm = MISSING

    upright: RewTerm = MISSING
    height: RewTerm = MISSING
    joint_reg: RewTerm = MISSING

    action_l2_penalty: RewTerm = MISSING
    action_rate_penalty: RewTerm = MISSING

    @classmethod
    def from_robot_cfg(cls, robot_cfg: Dict) -> "RewardsCfg":
        cfg = cls()

        action_joints = list(robot_cfg["action_joints"])
        defaults = dict(robot_cfg["default_joint_angles"])
        lf = str(robot_cfg["left_foot_body"])
        rf = str(robot_cfg["right_foot_body"])

        waist_joint = robot_cfg.get("waist_joint", None)
        has_waist = _robot_has_joint(robot_cfg, str(waist_joint) if waist_joint is not None else None)

        cfg.terminating = RewTerm(func=mdp.is_terminated, weight=-100.0)

        cfg.track_lin_vel_xy_exp = RewTerm(
            func=track_lin_vel_xy_yaw_frame_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        cfg.track_ang_vel_z_exp = RewTerm(
            func=track_ang_vel_z_world_exp,
            weight=0.3,
            params={"command_name": "base_velocity", "std": 0.5},
        )
        if has_waist:
            wname = str(waist_joint)
            cfg.waist_default = RewTerm(
                func=rew_waist_default,
                weight=0.05,
                params={
                    "asset_cfg": _scene_entity_with_joints("robot", [wname]),
                    "default_joint_angles": defaults,
                    "waist_name": wname,
                },
            )
        else:
            cfg.waist_default = RewTerm(func=_zero_reward, weight=0.0)

        cfg.action_l2_penalty = RewTerm(func=action_l2, weight=-0.0005)
        cfg.action_rate_penalty = RewTerm(func=action_rate_l2, weight=-0.005)
        #direct
        # cfg.upright = RewTerm(
        #     func=phi_upright_from_gravity,
        #     weight=0.5,  
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "scale": 1.0,
        #     },
        # )

        # cfg.height = RewTerm(
        #     func=phi_height_from_rays,
        #     weight=0.75, 
        #     params={
        #         "z_des": float(robot_cfg.get("height_z_des", 0.68)),
        #         "sigma_h": 0.08,
        #         "ray_sensor_name": "height_scanner",
        #         "scale": 1.0,
        #     },
        # )

        # cfg.joint_reg = RewTerm(
        #     func=phi_joint_reg_paper,
        #     weight=0.25,  
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot"),
        #         "sigma_q": 0.5,
        #         "joint_names": dict(robot_cfg["joint_reg_names"]),
        #         "pair_mode": dict(robot_cfg.get("joint_reg_pair_mode", {})),
        #         "scale": 1.0,
        #     },
        # )
        # pbrs
        cfg.upright = RewTerm(
            func=pbrs,
            weight=1.0, 
            params={
                "potential_func": phi_upright_from_gravity,
                "cache_key": "upright",
                "potential_kwargs": {
                    "asset_cfg": SceneEntityCfg("robot"),
                    "scale": 1.0,
                },
                "gamma": 1.0,
            },
        )

        cfg.height = RewTerm(
            func=pbrs,
            weight=1.0,
            params={
                "potential_func": phi_height_from_rays,
                "cache_key": "height",
                "potential_kwargs": {
                    "z_des": float(robot_cfg.get("height_z_des", 0.68)),
                    "sigma_h": 0.08,
                    "ray_sensor_name": "height_scanner",
                    "scale": 1.0,
                },
                "gamma": 1.0,
            },
        )

        if "joint_reg_names" not in robot_cfg:
            raise KeyError(
                "RewardsCfg.from_robot_cfg requires robot_cfg['joint_reg_names'] for phi_joint_reg_paper. "
                "Keys: l_ankle_roll, r_ankle_roll, l_ankle_pitch, r_ankle_pitch, l_hip_yaw, r_hip_yaw."
            )

        cfg.joint_reg = RewTerm(
            func=pbrs,
            weight=1.0,
            params={
                "potential_func": phi_joint_reg_paper,
                "cache_key": "joint_reg",
                "potential_kwargs": {
                    "asset_cfg": SceneEntityCfg("robot"),
                    "sigma_q": 0.5,
                    "joint_names": dict(robot_cfg["joint_reg_names"]),
                    "pair_mode": dict(robot_cfg.get("joint_reg_pair_mode", {})),
                    "scale": 1.0,
                },
                "gamma": 1.00,
            },
        )
        return cfg