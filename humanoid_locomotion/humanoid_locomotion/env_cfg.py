# humanoid_locomotion/env_cfg.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from humanoid_locomotion.common.scene import SceneCfg, TerrainParams, HeightScannerParams
from humanoid_locomotion.common.terminations import TerminationsCfg, TerminationParams
from humanoid_locomotion.common.observations import ObservationsCfg
from humanoid_locomotion.common.events import EventsCfg
from humanoid_locomotion.common.rewards import RewardsCfg
from humanoid_locomotion.common.commands import CommandsCfg
from humanoid_locomotion.common.curriculum import CurriculumCfg
from humanoid_locomotion.common.actions import ActionsCfg


# -----------------------------------------------------------------------------
# RobotSpec
# -----------------------------------------------------------------------------
@dataclass
class RobotSpec:
    name: str
    usd_path: str
    base_link: str
    left_foot_body: str
    right_foot_body: str

    action_joints: List[str]
    all_joints: List[str]
    held_joints: List[str]

    default_joint_angles: Dict[str, float]
    joint_limits: Dict[str, Dict[str, float]]

    joint_reg_names: Dict[str, str]
    joint_reg_pair_mode: Optional[Dict[str, str]] = None
    waist_joint: Optional[str] = None

    init_pos_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.75)
    init_rot_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    height_z_des: Optional[float] = None

    # optional per-joint PD overrides (if you use effort-based actuator builder)
    kp_by_joint: Optional[Dict[str, float]] = None
    kd_by_joint: Optional[Dict[str, float]] = None

    # optional full actuator override dict (e.g., grouped regex actuators)
    actuators_override: Optional[Dict[str, ImplicitActuatorCfg]] = None


# -----------------------------------------------------------------------------
# PD gains from effort (torque limit)
# -----------------------------------------------------------------------------
def pd_from_effort(
    effort: float,
    *,
    kp_per_effort: float = 1.9,
    kd_per_effort: float = 0.105,
    kp_min: float = 10.0,
    kd_min: float = 0.5,
    kp_max: float = 100.0,
    kd_max: float = 10.0,
) -> Tuple[float, float]:
    e = max(1e-6, float(effort))
    kp = float(kp_per_effort) * e
    kd = float(kd_per_effort) * e
    kp = float(max(kp_min, min(kp_max, kp)))
    kd = float(max(kd_min, min(kd_max, kd)))
    return kp, kd


def build_actuators_for_spec(spec: RobotSpec) -> Dict[str, ImplicitActuatorCfg]:
    """Default per-joint actuator builder using effort->(kp,kd)."""
    actuators: Dict[str, ImplicitActuatorCfg] = {}
    kp_over = spec.kp_by_joint or {}
    kd_over = spec.kd_by_joint or {}

    for j in spec.all_joints:
        lim = spec.joint_limits.get(j, {})
        effort = float(lim.get("effort", 100.0))
        vel = float(lim.get("vel", 50.0))

        kp, kd = pd_from_effort(
            effort,
            kp_per_effort=1.9,
            kd_per_effort=0.105,
            kp_min=10.0,
            kd_min=0.5,
            kp_max=200.0,
            kd_max=20.0,
        )

        kp = float(kp_over.get(j, kp))
        kd = float(kd_over.get(j, kd))

        actuators[f"act_{j}"] = ImplicitActuatorCfg(
            joint_names_expr=[j],
            stiffness=kp,
            damping=kd,
            effort_limit_sim=effort,
            velocity_limit_sim=vel,
        )

    return actuators


# -----------------------------------------------------------------------------
# Normalize spec -> robot_cfg dict (used by Scene/Obs/Actions/Rewards/etc.)
# -----------------------------------------------------------------------------
def normalize_spec(spec: RobotSpec) -> Dict:
    # ordered action min/max
    action_q_min = [float(spec.joint_limits[j]["lower"]) for j in spec.action_joints]
    action_q_max = [float(spec.joint_limits[j]["upper"]) for j in spec.action_joints]

    # per-joint: offset and scale
    # requested rule: scale_j = 0.5 * (upper_j - lower_j)
    action_offset_by_joint: Dict[str, float] = {}
    action_scale_by_joint: Dict[str, float] = {}
    action_lower_by_joint: Dict[str, float] = {}
    action_upper_by_joint: Dict[str, float] = {}

    for j in spec.action_joints:
        q0 = float(spec.default_joint_angles.get(j, 0.0))
        jmin = float(spec.joint_limits[j]["lower"])
        jmax = float(spec.joint_limits[j]["upper"])

        scale = 0.5 * (jmax - jmin)
        scale = float(max(1e-6, scale))

        action_offset_by_joint[j] = q0
        action_scale_by_joint[j] = scale
        action_lower_by_joint[j] = jmin
        action_upper_by_joint[j] = jmax

    # torque limits
    tau_limit_by_joint = {
        j: float(spec.joint_limits.get(j, {}).get("effort", 300.0))
        for j in spec.all_joints
    }

    # held joint targets
    held_pos_target_by_joint = {
        j: float(spec.default_joint_angles.get(j, 0.0))
        for j in spec.held_joints
    }

    # actuators (override if provided)
    actuators = spec.actuators_override if spec.actuators_override is not None else build_actuators_for_spec(spec)

    robot_cfg: Dict = {
        "name": spec.name,
        "usd_path": spec.usd_path,
        "base_link": spec.base_link,
        "left_foot_body": spec.left_foot_body,
        "right_foot_body": spec.right_foot_body,

        "action_joints": list(spec.action_joints),
        "all_joints": list(spec.all_joints),
        "held_joints": list(spec.held_joints),

        "default_joint_angles": dict(spec.default_joint_angles),

        "action_q_min": list(action_q_min),
        "action_q_max": list(action_q_max),

        "action_scale_by_joint": dict(action_scale_by_joint),
        "action_offset_by_joint": dict(action_offset_by_joint),
        "action_lower_by_joint": dict(action_lower_by_joint),
        "action_upper_by_joint": dict(action_upper_by_joint),

        "tau_limit_by_joint": dict(tau_limit_by_joint),

        "joint_reg_names": dict(spec.joint_reg_names),
        "joint_reg_pair_mode": dict(getattr(spec, "joint_reg_pair_mode", None) or {}),
        "waist_joint": spec.waist_joint,

        "init_pos_xyz": tuple(map(float, spec.init_pos_xyz)),
        "init_rot_xyzw": tuple(map(float, spec.init_rot_xyzw)),
        "height_z_des": spec.height_z_des,

        "actuators": actuators,
        "held_pos_target_by_joint": dict(held_pos_target_by_joint),
    }
    return robot_cfg


# =============================================================================
# Robot registry
# =============================================================================
ROBOT_SPECS: Dict[str, RobotSpec] = {}


# =============================================================================
# N1
# =============================================================================
_N1_USD = os.path.join(os.environ["HOME"], "humanoid_locomotion/robots/N1/N1_raw.usd")
if not os.path.isfile(_N1_USD):
    raise FileNotFoundError(f"N1 USD not found: {_N1_USD}")

N1_LEFT_LEG_6: List[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
]
N1_RIGHT_LEG_6: List[str] = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
]
N1_WAIST_1: List[str] = ["waist_yaw_joint"]

N1_LEFT_ARM_5: List[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
]
N1_RIGHT_ARM_5: List[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
]

N1_ACTION_JOINTS_13: List[str] = N1_LEFT_LEG_6 + N1_RIGHT_LEG_6 + N1_WAIST_1
N1_JOINTS_23: List[str] = N1_ACTION_JOINTS_13 + N1_LEFT_ARM_5 + N1_RIGHT_ARM_5
assert len(N1_ACTION_JOINTS_13) == 13
assert len(N1_JOINTS_23) == 23

N1_LEFT_FOOT_BODY = "left_foot_pitch_link"
N1_RIGHT_FOOT_BODY = "right_foot_pitch_link"

N1_ACTIONS_MAX_13: List[float] = [
    2.618, 1.571, 1.571, 2.356, 0.436, 0.785,
    2.618, 0.262, 1.571, 2.356, 0.436, 0.785,
    2.618,
]
N1_ACTIONS_MIN_13: List[float] = [
    -2.618, -0.262, -1.571, -0.087, -0.436, -0.785,
    -2.618, -1.571, -1.571, -0.087, -0.436, -0.785,
    -2.618,
]
assert len(N1_ACTIONS_MAX_13) == 13
assert len(N1_ACTIONS_MIN_13) == 13

N1_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    # left leg
    "left_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_pitch_joint": float(+np.deg2rad(29.5)),
    "left_ankle_roll_joint": 0.0,
    "left_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    # right leg
    "right_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_pitch_joint": float(+np.deg2rad(29.5)),
    "right_ankle_roll_joint": 0.0,
    "right_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    # waist
    "waist_yaw_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": float(np.deg2rad(60.0)),
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": float(np.deg2rad(60.0)),
    "right_wrist_yaw_joint": 0.0,
}
N1_DEFAULT_JOINT_ANGLES = {k: float(v) for k, v in N1_DEFAULT_JOINT_ANGLES.items()}

N1_TAU_LIMIT_23: List[float] = [
    95, 54, 54, 95, 30, 30,
    95, 54, 54, 95, 30, 30,
    54,
    54, 30, 30, 30, 30,
    54, 30, 30, 30, 30,
]

def kp_kd_for_joint(j: str) -> Tuple[float, float]:
    if "hip_pitch" in j:
        return 180.0, 10.0
    if "hip_roll" in j:
        return 120.0, 10.0
    if "hip_yaw" in j:
        return 90.0, 8.0
    if "knee" in j:
        return 120.0, 8.0
    if "ankle" in j:
        return 60.0, 3.0
    if "waist" in j:
        return 90.0, 8.0
    if "shoulder_pitch" in j:
        return 90.0, 8.0
    return 45.0, 2.5

N1_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_pitch_joint$",
            ".*_hip_roll_joint$",
            ".*_hip_yaw_joint$",
            ".*_knee_pitch_joint$",
        ],
        effort_limit_sim=94,
        stiffness={
            ".*_hip_pitch_joint$": kp_kd_for_joint("hip_pitch")[0],
            ".*_hip_roll_joint$":  kp_kd_for_joint("hip_roll")[0],
            ".*_hip_yaw_joint$":   kp_kd_for_joint("hip_yaw")[0],
            ".*_knee_pitch_joint$": kp_kd_for_joint("knee")[0],
        },
        damping={
            ".*_hip_pitch_joint$": kp_kd_for_joint("hip_pitch")[1],
            ".*_hip_roll_joint$":  kp_kd_for_joint("hip_roll")[1],
            ".*_hip_yaw_joint$":   kp_kd_for_joint("hip_yaw")[1],
            ".*_knee_pitch_joint$": kp_kd_for_joint("knee")[1],
        },
    ),

    "feet": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_ankle_roll_joint$",
            ".*_ankle_pitch_joint$",
        ],
        effort_limit_sim=54,
        stiffness={
            ".*_ankle_roll_joint$":  kp_kd_for_joint("ankle")[0],
            ".*_ankle_pitch_joint$": kp_kd_for_joint("ankle")[0],
        },
        damping={
            ".*_ankle_roll_joint$":  kp_kd_for_joint("ankle")[1],
            ".*_ankle_pitch_joint$": kp_kd_for_joint("ankle")[1],
        },
    ),

    "waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint$"],
        effort_limit_sim=54,
        stiffness={"waist_yaw_joint$": kp_kd_for_joint("waist")[0]},
        damping={"waist_yaw_joint$": kp_kd_for_joint("waist")[1]},
    ),

    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint$",
            ".*_shoulder_roll_joint$",
            ".*_shoulder_yaw_joint$",
            ".*_elbow_pitch_joint$",
            ".*_wrist_yaw_joint$",
        ],
        effort_limit_sim=54,
        stiffness={
            # special-case shoulder_pitch
            ".*_shoulder_pitch_joint$": kp_kd_for_joint("shoulder_pitch")[0],

            # everything else uses default (45, 2.5) per your function
            ".*_shoulder_roll_joint$":  kp_kd_for_joint("shoulder_roll")[0],
            ".*_shoulder_yaw_joint$":   kp_kd_for_joint("shoulder_yaw")[0],
            ".*_elbow_pitch_joint$":    kp_kd_for_joint("elbow")[0],
            ".*_wrist_yaw_joint$":      kp_kd_for_joint("wrist_yaw")[0],
        },
        damping={
            ".*_shoulder_pitch_joint$": kp_kd_for_joint("shoulder_pitch")[1],
            ".*_shoulder_roll_joint$":  kp_kd_for_joint("shoulder_roll")[1],
            ".*_shoulder_yaw_joint$":   kp_kd_for_joint("shoulder_yaw")[1],
            ".*_elbow_pitch_joint$":    kp_kd_for_joint("elbow")[1],
            ".*_wrist_yaw_joint$":      kp_kd_for_joint("wrist_yaw")[1],
        },
    ),
}

assert len(N1_TAU_LIMIT_23) == 23
N1_TAU_LIMIT_BY_JOINT: Dict[str, float] = {j: float(tau) for j, tau in zip(N1_JOINTS_23, N1_TAU_LIMIT_23)}

N1_JOINT_LIMITS: Dict[str, Dict[str, float]] = {}
for j, lo, hi in zip(N1_ACTION_JOINTS_13, N1_ACTIONS_MIN_13, N1_ACTIONS_MAX_13):
    N1_JOINT_LIMITS[j] = {
        "lower": float(lo),
        "upper": float(hi),
        "effort": float(N1_TAU_LIMIT_BY_JOINT.get(j, 300.0)),
        "vel": 50.0,
    }
N1_HELD_JOINTS: List[str] = N1_LEFT_ARM_5 + N1_RIGHT_ARM_5
for j in N1_HELD_JOINTS:
    N1_JOINT_LIMITS[j] = {
        "lower": -3.14159,
        "upper": 3.14159,
        "effort": float(N1_TAU_LIMIT_BY_JOINT.get(j, 300.0)),
        "vel": 50.0,
    }

N1_JOINT_REG_NAMES: Dict[str, str] = {
    "l_ankle_roll":  "left_ankle_roll_joint",
    "r_ankle_roll":  "right_ankle_roll_joint",
    "l_ankle_pitch": "left_ankle_pitch_joint",
    "r_ankle_pitch": "right_ankle_pitch_joint",
    "l_hip_yaw":     "left_hip_yaw_joint",
    "r_hip_yaw":     "right_hip_yaw_joint",
}

N1_JOINT_REG_PAIR_MODE: Dict[str, str] = {
    "ankle_roll": "same",
    "ankle_pitch": "same",
    "hip_yaw": "same",
}

ROBOT_SPECS["n1"] = RobotSpec(
    name="n1",
    usd_path=_N1_USD,
    base_link="base_link",
    left_foot_body=N1_LEFT_FOOT_BODY,
    right_foot_body=N1_RIGHT_FOOT_BODY,
    action_joints=list(N1_ACTION_JOINTS_13),
    all_joints=list(N1_JOINTS_23),
    held_joints=list(N1_HELD_JOINTS),
    default_joint_angles=dict(N1_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(N1_JOINT_LIMITS),
    joint_reg_names=dict(N1_JOINT_REG_NAMES),
    joint_reg_pair_mode=dict(N1_JOINT_REG_PAIR_MODE),
    waist_joint="waist_yaw_joint",
    init_pos_xyz=(0.0, 0.0, 0.75),
    height_z_des=0.68,
    actuators_override=N1_ACTUATORS_OVERRIDE,
)


# =============================================================================
# Unitree H1  (with grouped actuator override you requested)
# =============================================================================
H1_USD = os.path.join(os.environ["HOME"], "humanoid_locomotion/robots/unitree_h1/h1.usd")
if not os.path.isfile(H1_USD):
    raise FileNotFoundError(f"H1 USD not found: {H1_USD}")

H1_LEFT_LEG_5: List[str] = [
    "left_hip_yaw_joint",
    "left_hip_roll_joint",
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_joint",
]
H1_RIGHT_LEG_5: List[str] = [
    "right_hip_yaw_joint",
    "right_hip_roll_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_joint",
]
H1_TORSO_1: List[str] = ["torso_joint"]
H1_LEFT_ARM_4: List[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
]
H1_RIGHT_ARM_4: List[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

H1_ACTION_JOINTS_11: List[str] = H1_LEFT_LEG_5 + H1_RIGHT_LEG_5 + H1_TORSO_1
H1_JOINTS_19: List[str] = H1_ACTION_JOINTS_11 + H1_LEFT_ARM_4 + H1_RIGHT_ARM_4
assert len(H1_ACTION_JOINTS_11) == 11
assert len(H1_JOINTS_19) == 19

H1_LEFT_FOOT_BODY = "left_ankle_link"
H1_RIGHT_FOOT_BODY = "right_ankle_link"
H1_BASE_LINK = "pelvis"

H1_URDF_LIMITS: Dict[str, Dict[str, float]] = {
    # left leg
    "left_hip_yaw_joint":   {"lower": -0.43, "upper":  0.43, "effort": 200.0, "vel": 23.0},
    "left_hip_roll_joint":  {"lower": -0.43, "upper":  0.43, "effort": 200.0, "vel": 23.0},
    "left_hip_pitch_joint": {"lower": -3.14, "upper":  2.53, "effort": 200.0, "vel": 23.0},
    "left_knee_joint":      {"lower": -0.26, "upper":  2.05, "effort": 300.0, "vel": 14.0},
    "left_ankle_joint":     {"lower": -0.87, "upper":  0.52, "effort":  40.0, "vel":  9.0},
    # right leg
    "right_hip_yaw_joint":   {"lower": -0.43, "upper":  0.43, "effort": 200.0, "vel": 23.0},
    "right_hip_roll_joint":  {"lower": -0.43, "upper":  0.43, "effort": 200.0, "vel": 23.0},
    "right_hip_pitch_joint": {"lower": -3.14, "upper":  2.53, "effort": 200.0, "vel": 23.0},
    "right_knee_joint":      {"lower": -0.26, "upper":  2.05, "effort": 300.0, "vel": 14.0},
    "right_ankle_joint":     {"lower": -0.87, "upper":  0.52, "effort":  40.0, "vel":  9.0},
    # torso
    "torso_joint": {"lower": -2.35, "upper": 2.35, "effort": 200.0, "vel": 23.0},
    # left arm
    "left_shoulder_pitch_joint": {"lower": -2.87, "upper":  2.87, "effort": 40.0, "vel":  9.0},
    "left_shoulder_roll_joint":  {"lower": -0.34, "upper":  3.11, "effort": 40.0, "vel":  9.0},
    "left_shoulder_yaw_joint":   {"lower": -1.30, "upper":  4.45, "effort": 18.0, "vel": 20.0},
    "left_elbow_joint":          {"lower": -1.25, "upper":  2.61, "effort": 18.0, "vel": 20.0},
    # right arm
    "right_shoulder_pitch_joint": {"lower": -2.87, "upper":  2.87, "effort": 40.0, "vel":  9.0},
    "right_shoulder_roll_joint":  {"lower": -3.11, "upper":  0.34, "effort": 40.0, "vel":  9.0},
    "right_shoulder_yaw_joint":   {"lower": -4.45, "upper":  1.30, "effort": 18.0, "vel": 20.0},
    "right_elbow_joint":          {"lower": -1.25, "upper":  2.61, "effort": 18.0, "vel": 20.0},
}

_missing = [j for j in H1_JOINTS_19 if j not in H1_URDF_LIMITS]
assert not _missing, f"[unitree_h1] missing limits for joints: {_missing}"

H1_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    # left leg
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "left_knee_joint": float(+np.deg2rad(29.5)),
    "left_ankle_joint": float(-np.deg2rad(13.7)),
    # right leg
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "right_knee_joint": float(+np.deg2rad(29.5)),
    "right_ankle_joint": float(-np.deg2rad(13.7)),
    # torso
    "torso_joint": 0.0,
    # left arm
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": float(np.deg2rad(60.0)),
    # right arm
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": float(np.deg2rad(60.0)),
}
H1_DEFAULT_JOINT_ANGLES = {k: float(v) for k, v in H1_DEFAULT_JOINT_ANGLES.items()}

H1_JOINT_LIMITS: Dict[str, Dict[str, float]] = {
    j: {
        "lower": float(H1_URDF_LIMITS[j]["lower"]),
        "upper": float(H1_URDF_LIMITS[j]["upper"]),
        "effort": float(H1_URDF_LIMITS[j]["effort"]),
        "vel": float(H1_URDF_LIMITS[j]["vel"]),
    }
    for j in H1_JOINTS_19
}

H1_HELD_JOINTS: List[str] = list(H1_LEFT_ARM_4 + H1_RIGHT_ARM_4)

H1_JOINT_REG_NAMES: Dict[str, str] = {
    "l_ankle_roll":  "left_hip_roll_joint",
    "r_ankle_roll":  "right_hip_roll_joint",
    "l_ankle_pitch": "left_ankle_joint",
    "r_ankle_pitch": "right_ankle_joint",
    "l_hip_yaw":     "left_hip_yaw_joint",
    "r_hip_yaw":     "right_hip_yaw_joint",
}
H1_JOINT_REG_PAIR_MODE: Dict[str, str] = {
    "ankle_roll": "same",
    "ankle_pitch": "same",
    "hip_yaw": "same",
}

# ---- grouped actuators override (exactly what you provided) ----
H1_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint$",
            ".*_hip_roll_joint$",
            ".*_hip_pitch_joint$",
            ".*_knee_joint$",
            "torso_joint$",
        ],
        effort_limit_sim=300,
        stiffness={
            ".*_hip_yaw_joint$": 150.0,
            ".*_hip_roll_joint$": 150.0,
            ".*_hip_pitch_joint$": 200.0,
            ".*_knee_joint$": 200.0,
            "torso_joint$": 200.0,
        },
        damping={
            ".*_hip_yaw_joint$": 5.0,
            ".*_hip_roll_joint$": 5.0,
            ".*_hip_pitch_joint$": 5.0,
            ".*_knee_joint$": 5.0,
            "torso_joint$": 5.0,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[".*_ankle_joint$"],
        effort_limit_sim=100,
        stiffness={".*_ankle_joint$": 20.0},
        damping={".*_ankle_joint$": 4.0},
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint$",
            ".*_shoulder_roll_joint$",
            ".*_shoulder_yaw_joint$",
            ".*_elbow_joint$",
        ],
        effort_limit_sim=300,
        stiffness={
            ".*_shoulder_pitch_joint$": 40.0,
            ".*_shoulder_roll_joint$": 40.0,
            ".*_shoulder_yaw_joint$": 40.0,
            ".*_elbow_joint$": 40.0,
        },
        damping={
            ".*_shoulder_pitch_joint$": 10.0,
            ".*_shoulder_roll_joint$": 10.0,
            ".*_shoulder_yaw_joint$": 10.0,
            ".*_elbow_joint$": 10.0,
        },
    ),
}

ROBOT_SPECS["unitree_h1"] = RobotSpec(
    name="unitree_h1",
    usd_path=H1_USD,
    base_link=H1_BASE_LINK,
    left_foot_body=H1_LEFT_FOOT_BODY,
    right_foot_body=H1_RIGHT_FOOT_BODY,
    action_joints=list(H1_ACTION_JOINTS_11),
    all_joints=list(H1_JOINTS_19),
    held_joints=list(H1_HELD_JOINTS),
    default_joint_angles=dict(H1_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(H1_JOINT_LIMITS),
    joint_reg_names=dict(H1_JOINT_REG_NAMES),
    joint_reg_pair_mode=dict(H1_JOINT_REG_PAIR_MODE),
    waist_joint="torso_joint",
    init_pos_xyz=(0.0, 0.0, 1.05),
    height_z_des=0.95,
    actuators_override=H1_ACTUATORS_OVERRIDE,
)


# =============================================================================
# Unitree G1
# =============================================================================
_G1_USD = os.path.join(os.environ["HOME"], "humanoid_locomotion/robots/unitree_g1/g1.usd")
if not os.path.isfile(_G1_USD):
    raise FileNotFoundError(f"Unitree G1 USD not found: {_G1_USD}")

G1_LEFT_LEG_6: List[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
G1_RIGHT_LEG_6: List[str] = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
G1_WAIST_YAW_1: List[str] = ["waist_yaw_joint"]
G1_WAIST_HELD_2: List[str] = ["waist_roll_joint", "waist_pitch_joint"]

G1_LEFT_ARM_7: List[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
G1_RIGHT_ARM_7: List[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_ACTION_JOINTS_13: List[str] = G1_LEFT_LEG_6 + G1_RIGHT_LEG_6 + G1_WAIST_YAW_1
G1_JOINTS_29: List[str] = G1_ACTION_JOINTS_13 + G1_WAIST_HELD_2 + G1_LEFT_ARM_7 + G1_RIGHT_ARM_7
assert len(G1_ACTION_JOINTS_13) == 13
assert len(G1_JOINTS_29) == 29

G1_LEFT_FOOT_BODY = "left_ankle_roll_link"
G1_RIGHT_FOOT_BODY = "right_ankle_roll_link"
G1_BASE_LINK = "pelvis"

G1_URDF_LIMITS: Dict[str, Dict[str, float]] = {
    # left leg
    "left_hip_pitch_joint":   {"lower": -2.5307,   "upper":  2.8798,   "effort":  88.0, "vel": 32.0},
    "left_hip_roll_joint":    {"lower": -0.5236,   "upper":  2.9671,   "effort": 139.0, "vel": 20.0},
    "left_hip_yaw_joint":     {"lower": -2.7576,   "upper":  2.7576,   "effort":  88.0, "vel": 32.0},
    "left_knee_joint":        {"lower": -0.087267, "upper":  2.8798,   "effort": 139.0, "vel": 20.0},
    "left_ankle_pitch_joint": {"lower": -0.87267,  "upper":  0.5236,   "effort":  50.0, "vel": 37.0},
    "left_ankle_roll_joint":  {"lower": -0.2618,   "upper":  0.2618,   "effort":  50.0, "vel": 37.0},
    # right leg
    "right_hip_pitch_joint":   {"lower": -2.5307,   "upper":  2.8798,  "effort":  88.0, "vel": 32.0},
    "right_hip_roll_joint":    {"lower": -2.9671,   "upper":  0.5236,  "effort": 139.0, "vel": 20.0},
    "right_hip_yaw_joint":     {"lower": -2.7576,   "upper":  2.7576,  "effort":  88.0, "vel": 32.0},
    "right_knee_joint":        {"lower": -0.087267, "upper":  2.8798,  "effort": 139.0, "vel": 20.0},
    "right_ankle_pitch_joint": {"lower": -0.87267,  "upper":  0.5236,  "effort":  50.0, "vel": 37.0},
    "right_ankle_roll_joint":  {"lower": -0.2618,   "upper":  0.2618,  "effort":  50.0, "vel": 37.0},
    # waist
    "waist_yaw_joint":   {"lower": -2.618, "upper": 2.618, "effort": 88.0, "vel": 32.0},
    "waist_roll_joint":  {"lower": -0.52,  "upper": 0.52,  "effort": 50.0, "vel": 37.0},
    "waist_pitch_joint": {"lower": -0.52,  "upper": 0.52,  "effort": 50.0, "vel": 37.0},
    # left arm
    "left_shoulder_pitch_joint": {"lower": -3.0892,      "upper":  2.6704,      "effort": 25.0, "vel": 37.0},
    "left_shoulder_roll_joint":  {"lower": -1.5882,      "upper":  2.2515,      "effort": 25.0, "vel": 37.0},
    "left_shoulder_yaw_joint":   {"lower": -2.618,       "upper":  2.618,       "effort": 25.0, "vel": 37.0},
    "left_elbow_joint":          {"lower": -1.0472,      "upper":  2.0944,      "effort": 25.0, "vel": 37.0},
    "left_wrist_roll_joint":     {"lower": -1.972222054, "upper":  1.972222054, "effort": 25.0, "vel": 37.0},
    "left_wrist_pitch_joint":    {"lower": -1.614429558, "upper":  1.614429558, "effort":  5.0, "vel": 22.0},
    "left_wrist_yaw_joint":      {"lower": -1.614429558, "upper":  1.614429558, "effort":  5.0, "vel": 22.0},
    # right arm
    "right_shoulder_pitch_joint": {"lower": -3.0892,      "upper":  2.6704,      "effort": 25.0, "vel": 37.0},
    "right_shoulder_roll_joint":  {"lower": -2.2515,      "upper":  1.5882,      "effort": 25.0, "vel": 37.0},
    "right_shoulder_yaw_joint":   {"lower": -2.618,       "upper":  2.618,       "effort": 25.0, "vel": 37.0},
    "right_elbow_joint":          {"lower": -1.0472,      "upper":  2.0944,      "effort": 25.0, "vel": 37.0},
    "right_wrist_roll_joint":     {"lower": -1.972222054, "upper":  1.972222054, "effort": 25.0, "vel": 37.0},
    "right_wrist_pitch_joint":    {"lower": -1.614429558, "upper":  1.614429558, "effort":  5.0, "vel": 22.0},
    "right_wrist_yaw_joint":      {"lower": -1.614429558, "upper":  1.614429558, "effort":  5.0, "vel": 22.0},
}

_missing = [j for j in G1_JOINTS_29 if j not in G1_URDF_LIMITS]
assert not _missing, f"[unitree_g1] missing limits for joints: {_missing}"

G1_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    # left leg
    "left_hip_pitch_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": float(+np.deg2rad(29.5)),
    "left_ankle_roll_joint": 0.0,
    "left_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    # right leg
    "right_hip_pitch_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": float(+np.deg2rad(29.5)),
    "right_ankle_roll_joint": 0.0,
    "right_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": float(np.deg2rad(60.0)),
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": float(np.deg2rad(60.0)),
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}
G1_DEFAULT_JOINT_ANGLES = {k: float(v) for k, v in G1_DEFAULT_JOINT_ANGLES.items()}

G1_JOINT_LIMITS: Dict[str, Dict[str, float]] = {
    j: {
        "lower": float(G1_URDF_LIMITS[j]["lower"]),
        "upper": float(G1_URDF_LIMITS[j]["upper"]),
        "effort": float(G1_URDF_LIMITS[j]["effort"]),
        "vel": float(G1_URDF_LIMITS[j]["vel"]),
    }
    for j in G1_JOINTS_29
}

G1_HELD_JOINTS: List[str] = list(G1_WAIST_HELD_2 + G1_LEFT_ARM_7 + G1_RIGHT_ARM_7)

G1_JOINT_REG_NAMES: Dict[str, str] = {
    "l_ankle_roll":  "left_ankle_roll_joint",
    "r_ankle_roll":  "right_ankle_roll_joint",
    "l_ankle_pitch": "left_ankle_pitch_joint",
    "r_ankle_pitch": "right_ankle_pitch_joint",
    "l_hip_yaw":     "left_hip_yaw_joint",
    "r_hip_yaw":     "right_hip_yaw_joint",
}
G1_JOINT_REG_PAIR_MODE: Dict[str, str] = {
    "ankle_roll": "same",
    "ankle_pitch": "same",
    "hip_yaw": "same",
}


G1_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint$",
            ".*_hip_roll_joint$",
            ".*_hip_pitch_joint$",
            ".*_knee_joint$",
        ],
        effort_limit_sim=300,
        stiffness={
            ".*_hip_yaw_joint$": 100.0,
            ".*_hip_roll_joint$": 100.0,
            ".*_hip_pitch_joint$": 125.0,
            ".*_knee_joint$": 125.0,
        },
        damping={
            ".*_hip_yaw_joint$": 4.0,
            ".*_hip_roll_joint$": 4.0,
            ".*_hip_pitch_joint$": 4.0,
            ".*_knee_joint$": 4.0,
        },
    ),

    "feet": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_ankle_pitch_joint$",
            ".*_ankle_roll_joint$",
        ],
        effort_limit_sim=100,
        stiffness={
            ".*_ankle_pitch_joint$": 20.0,
            ".*_ankle_roll_joint$": 20.0,
        },
        damping={
            ".*_ankle_pitch_joint$": 2.0,
            ".*_ankle_roll_joint$": 2.0,
        },
    ),

    "waist": ImplicitActuatorCfg(
        joint_names_expr=[
            "waist_yaw_joint$",
            "waist_roll_joint$",
            "waist_pitch_joint$",
        ],
        effort_limit_sim=300,
        stiffness={
            "waist_yaw_joint$": 100.0,
            "waist_roll_joint$": 100.0,
            "waist_pitch_joint$": 100.0,
        },
        damping={
            "waist_yaw_joint$": 5.0,
            "waist_roll_joint$": 5.0,
            "waist_pitch_joint$": 5.0,
        },
    ),

    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint$",
            ".*_shoulder_roll_joint$",
            ".*_shoulder_yaw_joint$",
            ".*_elbow_joint$",
            ".*_wrist_.*_joint$",
        ],
        effort_limit_sim=100,
        stiffness={
            ".*_shoulder_pitch_joint$": 40.0,
            ".*_shoulder_roll_joint$": 40.0,
            ".*_shoulder_yaw_joint$": 40.0,
            ".*_elbow_joint$": 40.0,
            ".*_wrist_.*_joint$": 40.0,
        },
        damping={
            ".*_shoulder_pitch_joint$": 10.0,
            ".*_shoulder_roll_joint$": 10.0,
            ".*_shoulder_yaw_joint$": 10.0,
            ".*_elbow_joint$": 10.0,
            ".*_wrist_.*_joint$": 10.0,
        },
    ),
}

ROBOT_SPECS["unitree_g1"] = RobotSpec(
    name="unitree_g1",
    usd_path=_G1_USD,
    base_link=G1_BASE_LINK,
    left_foot_body=G1_LEFT_FOOT_BODY,
    right_foot_body=G1_RIGHT_FOOT_BODY,
    action_joints=list(G1_ACTION_JOINTS_13),
    all_joints=list(G1_JOINTS_29),
    held_joints=list(G1_HELD_JOINTS),
    default_joint_angles=dict(G1_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(G1_JOINT_LIMITS),
    joint_reg_names=dict(G1_JOINT_REG_NAMES),
    joint_reg_pair_mode=dict(G1_JOINT_REG_PAIR_MODE),
    waist_joint="waist_yaw_joint",
    init_pos_xyz=(0.0, 0.0, 0.85),
    height_z_des=0.835,
    actuators_override=G1_ACTUATORS_OVERRIDE,
)

# -----------------------------
# Booster T1 
# -----------------------------
_T1_USD = os.path.join(os.environ["HOME"], "humanoid_locomotion/robots/booster_t1/T1_serial.usd")
if not os.path.isfile(_T1_USD):
    raise FileNotFoundError(f"Booster T1 USD not found: {_T1_USD}")
T1_HEAD_2: List[str] = [
    "AAHead_yaw",
    "Head_pitch",
]
T1_WAIST_1: List[str] = [
    "Waist",
]
T1_LEFT_ARM_4: List[str] = [
    "Left_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
]
T1_RIGHT_ARM_4: List[str] = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]
T1_LEFT_LEG_6: List[str] = [
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
]
T1_RIGHT_LEG_6: List[str] = [
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

T1_ACTION_JOINTS_13: List[str] = T1_LEFT_LEG_6 + T1_RIGHT_LEG_6 + T1_WAIST_1
T1_JOINTS_23: List[str] = T1_ACTION_JOINTS_13 + T1_HEAD_2 + T1_LEFT_ARM_4 + T1_RIGHT_ARM_4
assert len(T1_ACTION_JOINTS_13) == 13
assert len(T1_JOINTS_23) == 23
T1_LEFT_FOOT_BODY = "left_foot_link"
T1_RIGHT_FOOT_BODY = "right_foot_link"
T1_BASE_LINK = "Trunk"
T1_URDF_LIMITS: Dict[str, Dict[str, float]] = {
    # head
    "AAHead_yaw":   {"lower": -1.57, "upper": 1.57, "effort": 7.0,  "vel": 12.56},
    "Head_pitch":   {"lower": -0.35, "upper": 1.22, "effort": 7.0,  "vel": 12.56},
    # arms (left)
    "Left_Shoulder_Pitch": {"lower": -3.31, "upper":  1.22, "effort": 18.0, "vel": 18.84},
    "Left_Shoulder_Roll":  {"lower": -1.74, "upper":  1.57, "effort": 18.0, "vel": 18.84},
    "Left_Elbow_Pitch":    {"lower": -2.27, "upper":  2.27, "effort": 18.0, "vel": 18.84},
    "Left_Elbow_Yaw":      {"lower": -2.44, "upper":  0.0,  "effort": 18.0, "vel": 18.84},
    # arms (right)
    "Right_Shoulder_Pitch": {"lower": -3.31, "upper":  1.22, "effort": 18.0, "vel": 18.84},
    "Right_Shoulder_Roll":  {"lower": -1.57, "upper":  1.74, "effort": 18.0, "vel": 18.84},
    "Right_Elbow_Pitch":    {"lower": -2.27, "upper":  2.27, "effort": 18.0, "vel": 18.84},
    "Right_Elbow_Yaw":      {"lower":  0.0,  "upper":  2.44, "effort": 18.0, "vel": 18.84},
    # waist
    "Waist": {"lower": -1.57, "upper": 1.57, "effort": 30.0, "vel": 10.88},
    # left leg
    "Left_Hip_Pitch":  {"lower": -1.8,  "upper": 1.57, "effort": 45.0, "vel": 12.5},
    "Left_Hip_Roll":   {"lower": -0.2,  "upper": 1.57, "effort": 30.0, "vel": 10.9},
    "Left_Hip_Yaw":    {"lower": -1.0,  "upper": 1.0,  "effort": 30.0, "vel": 10.9},
    "Left_Knee_Pitch": {"lower":  0.0,  "upper": 2.34, "effort": 60.0, "vel": 11.7},
    "Left_Ankle_Pitch":{"lower": -0.87, "upper": 0.35, "effort": 20.0, "vel": 18.8},
    "Left_Ankle_Roll": {"lower": -0.44, "upper": 0.44, "effort": 15.0, "vel": 12.4},
    # right leg
    "Right_Hip_Pitch":  {"lower": -1.8,  "upper": 1.57, "effort": 45.0, "vel": 12.5},
    "Right_Hip_Roll":   {"lower": -1.57, "upper": 0.2,  "effort": 30.0, "vel": 10.9},
    "Right_Hip_Yaw":    {"lower": -1.0,  "upper": 1.0,  "effort": 30.0, "vel": 10.9},
    "Right_Knee_Pitch": {"lower":  0.0,  "upper": 2.34, "effort": 60.0, "vel": 11.7},
    "Right_Ankle_Pitch":{"lower": -0.87, "upper": 0.35, "effort": 20.0, "vel": 18.8},
    "Right_Ankle_Roll": {"lower": -0.44, "upper": 0.44, "effort": 15.0, "vel": 12.4},
}

_missing = [j for j in T1_JOINTS_23 if j not in T1_URDF_LIMITS]
assert not _missing, f"[booster_t1] missing limits for joints: {_missing}"

T1_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
        # left leg 
        "Left_Hip_Pitch": 0.0, 
        "Left_Hip_Roll":  0.0,
        "Left_Hip_Yaw":   0.0,
        "Left_Knee_Pitch": float(+np.deg2rad(40.0)),
        "Left_Ankle_Pitch": 0.0,
        "Left_Ankle_Roll": 0.0,
        # right leg
        "Right_Hip_Pitch": 0.0, 
        "Right_Hip_Roll":  0.0,
        "Right_Hip_Yaw":   0.0,
        "Right_Knee_Pitch": float(+np.deg2rad(40.0)),
        "Right_Ankle_Pitch": 0.0,
        "Right_Ankle_Roll": 0.0,
        # waist
        "Waist": 0.0,
        # head
        "AAHead_yaw": 0.0,
        "Head_pitch": 0.0,
        # left arm
        "Left_Shoulder_Pitch": 0.0,
        "Left_Shoulder_Roll":  float(-np.deg2rad(80.0)),
        "Left_Elbow_Pitch":    0.0,
        "Left_Elbow_Yaw":      float(-np.deg2rad(30.0)),
        # right arm
        "Right_Shoulder_Pitch": 0.0,
        "Right_Shoulder_Roll":  float(np.deg2rad(80.0)),
        "Right_Elbow_Pitch":    0.0,
        "Right_Elbow_Yaw":      float(np.deg2rad(30.0)),
}
T1_DEFAULT_JOINT_ANGLES = {k: float(v) for k, v in T1_DEFAULT_JOINT_ANGLES.items()}
T1_TAU_LIMIT_23: List[float] = [float(T1_URDF_LIMITS[j]["effort"]) for j in T1_JOINTS_23]
assert len(T1_TAU_LIMIT_23) == 23
T1_TAU_LIMIT_BY_JOINT: Dict[str, float] = {j: float(tau) for j, tau in zip(T1_JOINTS_23, T1_TAU_LIMIT_23)}
T1_JOINT_LIMITS: Dict[str, Dict[str, float]] = {}
for j in T1_JOINTS_23:
    lim = T1_URDF_LIMITS[j]
    T1_JOINT_LIMITS[j] = {
        "lower": float(lim["lower"]),
        "upper": float(lim["upper"]),
        "effort": float(lim["effort"]),
        "vel": float(lim["vel"]),
    }
T1_HELD_JOINTS: List[str] = [j for j in T1_JOINTS_23 if j not in T1_ACTION_JOINTS_13]
T1_JOINT_REG_NAMES: Dict[str, str] = {
    "l_ankle_roll":  "Left_Ankle_Roll",
    "r_ankle_roll":  "Right_Ankle_Roll",
    "l_ankle_pitch": "Left_Ankle_Pitch",
    "r_ankle_pitch": "Right_Ankle_Pitch",
    "l_hip_yaw":     "Left_Hip_Yaw",
    "r_hip_yaw":     "Right_Hip_Yaw",
}
T1_JOINT_REG_PAIR_MODE: Dict[str, str] = {
    "ankle_roll": "same",
    "ankle_pitch": "same",
    "hip_yaw": "same",
}
T1_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {
    "Hip_Pitch": ImplicitActuatorCfg(
        joint_names_expr=[
            "Left_Hip_Pitch$",
            "Right_Hip_Pitch$",
        ],
        stiffness=100.0,
        damping=4.0,
        effort_limit_sim=45.0,
        # velocity_limit_sim can be omitted (IsaacLab default) or set if you want
        armature=0.0,
        friction=0.1,
    ),
    "Hip_Roll_Yaw": ImplicitActuatorCfg(
        joint_names_expr=[
            "Left_Hip_Roll$",
            "Right_Hip_Roll$",
            "Left_Hip_Yaw$",
            "Right_Hip_Yaw$",
        ],
        stiffness=100.0,
        damping=4.0,
        effort_limit_sim=30.0,
        armature=0.0,
        friction=0.1,
    ),
    "Knee": ImplicitActuatorCfg(
        joint_names_expr=[
            "Left_Knee_Pitch$",
            "Right_Knee_Pitch$",
        ],
        stiffness=120.0,
        damping=5.0,
        effort_limit_sim=60.0,
        armature=0.0,
        friction=0.1,
    ),
    "Ankle_Pitch": ImplicitActuatorCfg(
        joint_names_expr=[
            "Left_Ankle_Pitch$",
            "Right_Ankle_Pitch$",
        ],
        stiffness=20.0,
        damping=1.0,
        effort_limit_sim=24.0,
        armature=0.0,
        friction=0.1,
    ),
    "Ankle_Roll": ImplicitActuatorCfg(
        joint_names_expr=[
            "Left_Ankle_Roll$",
            "Right_Ankle_Roll$",
        ],
        stiffness=20.0,
        damping=1.0,
        effort_limit_sim=15.0,
        armature=0.0,
        friction=0.1,
    ),
    "Waist": ImplicitActuatorCfg(
        joint_names_expr=[
            "Waist$",
        ],
        stiffness=100.0,
        damping=3.0,
        effort_limit_sim=15.0,
        armature=0.0,
        friction=0.1,
    ),
    "Head": ImplicitActuatorCfg(
        joint_names_expr=[
            "AAHead_yaw$",
            "Head_pitch$",
        ],
        stiffness=50.0,
        damping=1.0,
        effort_limit_sim=15.0,
        armature=0.0,
        friction=0.1,
    ),
    "Arm": ImplicitActuatorCfg(
        joint_names_expr=[
            "Left_Shoulder_Pitch$",
            "Left_Shoulder_Roll$",
            "Left_Elbow_Pitch$",
            "Left_Elbow_Yaw$",
            "Right_Shoulder_Pitch$",
            "Right_Shoulder_Roll$",
            "Right_Elbow_Pitch$",
            "Right_Elbow_Yaw$",
        ],
        stiffness=50.0,
        damping=1.0,
        effort_limit_sim=15.0,
        armature=0.0,
        friction=0.1,
    ),
}
ROBOT_SPECS["booster_t1"] = RobotSpec(
    name="booster_t1",
    usd_path=_T1_USD,
    base_link=T1_BASE_LINK,
    left_foot_body=T1_LEFT_FOOT_BODY,
    right_foot_body=T1_RIGHT_FOOT_BODY,
    action_joints=list(T1_ACTION_JOINTS_13),
    all_joints=list(T1_JOINTS_23),
    held_joints=list(T1_HELD_JOINTS),
    default_joint_angles=dict(T1_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(T1_JOINT_LIMITS),
    joint_reg_names=dict(T1_JOINT_REG_NAMES),
    joint_reg_pair_mode=dict(T1_JOINT_REG_PAIR_MODE), 
    waist_joint="Waist",
    init_pos_xyz=(0.0, 0.0, 0.7),
    height_z_des=0.68,
    actuators_override=T1_ACTUATORS_OVERRIDE,
)

# -----------------------
# Fourier GR3V2 
# -----------------------
_GR3V2_USD = os.path.join(os.environ["HOME"], "humanoid_locomotion/robots/fourier_gr3v2/gr3v2_1_1.usd")
if not os.path.isfile(_GR3V2_USD):
    raise FileNotFoundError(f"GR3V2 USD not found: {_GR3V2_USD}")
GR3V2_LEFT_LEG_6: List[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
]
GR3V2_RIGHT_LEG_6: List[str] = [
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
GR3V2_WAIST_YAW_1: List[str] = ["waist_yaw_joint"]
GR3V2_WAIST_HELD_2: List[str] = ["waist_roll_joint", "waist_pitch_joint"]
GR3V2_HEAD_2: List[str] = ["head_yaw_joint", "head_pitch_joint"]
GR3V2_RIGHT_ARM_7: List[str] = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]
GR3V2_LEFT_ARM_7: List[str] = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
]
GR3V2_ACTION_JOINTS_13: List[str] = GR3V2_LEFT_LEG_6 + GR3V2_RIGHT_LEG_6 + GR3V2_WAIST_YAW_1
GR3V2_JOINTS_31: List[str] = (
    GR3V2_ACTION_JOINTS_13
    + GR3V2_WAIST_HELD_2
    + GR3V2_HEAD_2
    + GR3V2_LEFT_ARM_7
    + GR3V2_RIGHT_ARM_7
)

assert len(GR3V2_ACTION_JOINTS_13) == 13
assert len(GR3V2_JOINTS_31) == 31
GR3V2_LEFT_FOOT_BODY = "left_foot_roll_link"
GR3V2_RIGHT_FOOT_BODY = "right_foot_roll_link"
GR3V2_BASE_LINK = "base_link"
GR3V2_URDF_LIMITS: Dict[str, Dict[str, float]] = {
    # waist
    "waist_yaw_joint":   {"lower": -2.618,   "upper": 2.618,   "effort": 140.4, "vel": 12.985},
    "waist_roll_joint":  {"lower": -0.17453, "upper": 0.17453, "effort": 108.6, "vel": 14.77},
    "waist_pitch_joint": {"lower": -0.20944, "upper": 0.5236,  "effort": 108.6, "vel": 14.77},
    # head
    "head_yaw_joint":   {"lower": -1.3963, "upper": 1.3963, "effort": 17.4, "vel": 9.21},
    "head_pitch_joint": {"lower": -0.5236, "upper": 0.1483, "effort": 17.4, "vel": 9.21},
    # right arm
    "right_shoulder_pitch_joint": {"lower": -2.9671, "upper": 2.9671, "effort": 74.4, "vel": 7.75},
    "right_shoulder_roll_joint":  {"lower": -1.9199, "upper": 0.2618, "effort": 74.4, "vel": 7.75},
    "right_shoulder_yaw_joint":   {"lower": -1.8326, "upper": 1.8326, "effort": 42.9, "vel": 6.28},
    "right_elbow_pitch_joint":    {"lower": -2.2689, "upper": 0.087266, "effort": 42.9, "vel": 6.28},
    "right_wrist_yaw_joint":      {"lower": -1.8326, "upper": 1.8326, "effort": 42.9, "vel": 6.28},
    "right_wrist_pitch_joint":    {"lower": -0.87266, "upper": 1.309,  "effort": 17.4, "vel": 9.2153},
    "right_wrist_roll_joint":     {"lower": -1.2217,  "upper": 1.0472, "effort": 17.4, "vel": 9.2153},
    # left arm
    "left_shoulder_pitch_joint": {"lower": -2.9671, "upper": 2.9671, "effort": 74.4, "vel": 7.75},
    "left_shoulder_roll_joint":  {"lower": -0.2618, "upper": 1.9199, "effort": 74.4, "vel": 7.75},
    "left_shoulder_yaw_joint":   {"lower": -1.8326, "upper": 1.8326, "effort": 42.9, "vel": 6.28},
    "left_elbow_pitch_joint":    {"lower": -2.2689, "upper": 0.087266, "effort": 42.9, "vel": 6.28},
    "left_wrist_yaw_joint":      {"lower": -1.8326, "upper": 1.8326, "effort": 42.9, "vel": 6.28},
    "left_wrist_pitch_joint":    {"lower": -0.87266, "upper": 1.309,  "effort": 17.4, "vel": 9.2153},
    "left_wrist_roll_joint":     {"lower": -1.0472,  "upper": 1.2217, "effort": 17.4, "vel": 9.2153},
    # right leg
    "right_hip_pitch_joint":   {"lower": -2.618,   "upper": 2.618,   "effort": 366.0, "vel": 6.49},
    "right_hip_roll_joint":    {"lower": -1.5708,  "upper": 0.17453, "effort": 140.4, "vel": 12.985},
    "right_hip_yaw_joint":     {"lower": -1.5708,  "upper": 0.69813, "effort": 140.4, "vel": 12.985},
    "right_knee_pitch_joint":  {"lower":  0.0,     "upper": 2.3562,  "effort": 366.0, "vel": 6.49},
    "right_ankle_pitch_joint": {"lower": -0.7854,  "upper": 0.7854,  "effort": 59.4,  "vel": 16.76},
    "right_ankle_roll_joint":  {"lower": -0.43633, "upper": 0.43633, "effort": 59.4,  "vel": 16.76},
    # left leg
    "left_hip_pitch_joint":   {"lower": -2.618,   "upper": 2.618,   "effort": 366.0, "vel": 6.49},
    "left_hip_roll_joint":    {"lower": -0.17453, "upper": 1.5708,  "effort": 140.4, "vel": 12.985},
    "left_hip_yaw_joint":     {"lower": -0.69813, "upper": 1.5708,  "effort": 140.4, "vel": 12.985},
    "left_knee_pitch_joint":  {"lower":  0.0,     "upper": 2.3562,  "effort": 366.0, "vel": 6.49},
    "left_ankle_pitch_joint": {"lower": -0.7854,  "upper": 0.7854,  "effort": 59.4,  "vel": 16.76},
    "left_ankle_roll_joint":  {"lower": -0.43633, "upper": 0.43633, "effort": 59.4,  "vel": 16.76},
}
_missing = [j for j in GR3V2_JOINTS_31 if j not in GR3V2_URDF_LIMITS]
assert not _missing, f"[fourier_gr3v2] missing limits for joints: {_missing}"
GR3V2_ACTIONS_MIN_13: List[float] = [float(GR3V2_URDF_LIMITS[j]["lower"]) for j in GR3V2_ACTION_JOINTS_13]
GR3V2_ACTIONS_MAX_13: List[float] = [float(GR3V2_URDF_LIMITS[j]["upper"]) for j in GR3V2_ACTION_JOINTS_13]
assert len(GR3V2_ACTIONS_MIN_13) == 13
assert len(GR3V2_ACTIONS_MAX_13) == 13
GR3V2_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    # left leg
    "left_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "left_hip_roll_joint":  0.0,
    "left_hip_yaw_joint":   0.0,
    "left_knee_pitch_joint": float(+np.deg2rad(29.5)),
    "left_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    "left_ankle_roll_joint": 0.0,
    # right leg
    "right_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "right_hip_roll_joint":  0.0,
    "right_hip_yaw_joint":   0.0,
    "right_knee_pitch_joint": float(+np.deg2rad(29.5)),
    "right_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    "right_ankle_roll_joint": 0.0,
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": float(np.deg2rad(14.0)),
    # head
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    # left arm
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint":  0.0,
    "left_shoulder_yaw_joint":   0.0,
    "left_elbow_pitch_joint":    0.0,
    "left_wrist_yaw_joint":      0.0,
    "left_wrist_pitch_joint":    0.0,
    "left_wrist_roll_joint":     0.0,
    # right arm
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint":  0.0,
    "right_shoulder_yaw_joint":   0.0,
    "right_elbow_pitch_joint":    0.0,
    "right_wrist_yaw_joint":      0.0,
    "right_wrist_pitch_joint":    0.0,
    "right_wrist_roll_joint":     0.0,
}
GR3V2_DEFAULT_JOINT_ANGLES = {k: float(v) for k, v in GR3V2_DEFAULT_JOINT_ANGLES.items()}
GR3V2_TAU_LIMIT_31: List[float] = [float(GR3V2_URDF_LIMITS[j]["effort"]) for j in GR3V2_JOINTS_31]
assert len(GR3V2_TAU_LIMIT_31) == 31
GR3V2_TAU_LIMIT_BY_JOINT: Dict[str, float] = {j: float(tau) for j, tau in zip(GR3V2_JOINTS_31, GR3V2_TAU_LIMIT_31)}
GR3V2_JOINT_LIMITS: Dict[str, Dict[str, float]] = {}
for j, lo, hi in zip(GR3V2_ACTION_JOINTS_13, GR3V2_ACTIONS_MIN_13, GR3V2_ACTIONS_MAX_13):
    lim = GR3V2_URDF_LIMITS[j]
    GR3V2_JOINT_LIMITS[j] = {
        "lower": float(lo),
        "upper": float(hi),
        "effort": float(GR3V2_TAU_LIMIT_BY_JOINT.get(j, lim.get("effort", 300.0))),
        "vel": float(lim.get("vel", 50.0)),
    }
GR3V2_HELD_JOINTS: List[str] = list(GR3V2_WAIST_HELD_2 + GR3V2_HEAD_2 + GR3V2_LEFT_ARM_7 + GR3V2_RIGHT_ARM_7)
for j in GR3V2_HELD_JOINTS:
    lim = GR3V2_URDF_LIMITS[j]
    GR3V2_JOINT_LIMITS[j] = {
        "lower": float(lim["lower"]),
        "upper": float(lim["upper"]),
        "effort": float(GR3V2_TAU_LIMIT_BY_JOINT.get(j, lim.get("effort", 300.0))),
        "vel": float(lim.get("vel", 50.0)),
    }
GR3V2_JOINT_REG_NAMES: Dict[str, str] = {
    "l_ankle_roll":  "left_ankle_roll_joint",
    "r_ankle_roll":  "right_ankle_roll_joint",
    "l_ankle_pitch": "left_ankle_pitch_joint",
    "r_ankle_pitch": "right_ankle_pitch_joint",
    "l_hip_yaw":     "left_hip_yaw_joint",
    "r_hip_yaw":     "right_hip_yaw_joint",
}
GR3V2_JOINT_REG_PAIR_MODE: Dict[str, str] = {
    "ankle_roll": "same",
    "ankle_pitch": "same",
    "hip_yaw": "opposite",
}
GR3V2_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint$",
            ".*_hip_roll_joint$",
            ".*_hip_pitch_joint$",
            ".*_knee_pitch_joint$",
        ],
        effort_limit_sim=300,
        stiffness={
            ".*_hip_yaw_joint$": 200.0,
            ".*_hip_roll_joint$": 200.0,
            ".*_hip_pitch_joint$": 300.0,
            ".*_knee_pitch_joint$": 300.0,
        },
        damping={
            ".*_hip_yaw_joint$": 10.0,
            ".*_hip_roll_joint$": 10.0,
            ".*_hip_pitch_joint$": 10.0,
            ".*_knee_pitch_joint$": 10.0,
        },
    ),

    "feet": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_ankle_pitch_joint$",
            ".*_ankle_roll_joint$",
        ],
        effort_limit_sim=100,
        stiffness={
            ".*_ankle_pitch_joint$": 20.0,
            ".*_ankle_roll_joint$": 20.0,
        },
        damping={
            ".*_ankle_pitch_joint$": 8.0,
            ".*_ankle_roll_joint$": 8.0,
        },
    ),

    "waist": ImplicitActuatorCfg(
        joint_names_expr=[
            "waist_yaw_joint$",
            "waist_roll_joint$",
            "waist_pitch_joint$",
        ],
        effort_limit_sim=300,
        stiffness={
            "waist_yaw_joint$": 200.0,
            "waist_roll_joint$": 200.0,
            "waist_pitch_joint$": 200.0,
        },
        damping={
            "waist_yaw_joint$": 8.0,
            "waist_roll_joint$": 8.0,
            "waist_pitch_joint$": 8.0,
        },
    ),

    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint$",
            ".*_shoulder_roll_joint$",
            ".*_shoulder_yaw_joint$",
            ".*_elbow_pitch_joint$",
            ".*_wrist_yaw_joint$",
            ".*_wrist_pitch_joint$",
            ".*_wrist_roll_joint$",
        ],
        effort_limit_sim=100,
        stiffness={
            ".*_shoulder_pitch_joint$": 40.0,
            ".*_shoulder_roll_joint$": 40.0,
            ".*_shoulder_yaw_joint$": 40.0,
            ".*_elbow_pitch_joint$": 40.0,
            ".*_wrist_yaw_joint$": 40.0,
            ".*_wrist_pitch_joint$": 40.0,
            ".*_wrist_roll_joint$": 40.0,
        },
        damping={
            ".*_shoulder_pitch_joint$": 10.0,
            ".*_shoulder_roll_joint$": 10.0,
            ".*_shoulder_yaw_joint$": 10.0,
            ".*_elbow_pitch_joint$": 10.0,
            ".*_wrist_yaw_joint$": 10.0,
            ".*_wrist_pitch_joint$": 10.0,
            ".*_wrist_roll_joint$": 10.0,
        },
    ),

    "head": ImplicitActuatorCfg(
        joint_names_expr=[
            "head_yaw_joint$",
            "head_pitch_joint$",
        ],
        effort_limit_sim=20,
        stiffness={
            "head_yaw_joint$": 20.0,
            "head_pitch_joint$": 20.0,
        },
        damping={
            "head_yaw_joint$": 2.0,
            "head_pitch_joint$": 2.0,
        },
    ),
}
ROBOT_SPECS["fourier_gr3v2"] = RobotSpec(
    name="fourier_gr3v2",
    usd_path=_GR3V2_USD,
    base_link=GR3V2_BASE_LINK,
    left_foot_body=GR3V2_LEFT_FOOT_BODY,
    right_foot_body=GR3V2_RIGHT_FOOT_BODY,
    action_joints=list(GR3V2_ACTION_JOINTS_13),
    all_joints=list(GR3V2_JOINTS_31),
    held_joints=list(GR3V2_HELD_JOINTS),
    default_joint_angles=dict(GR3V2_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(GR3V2_JOINT_LIMITS),
    joint_reg_names=dict(GR3V2_JOINT_REG_NAMES),
    joint_reg_pair_mode=dict(GR3V2_JOINT_REG_PAIR_MODE), 
    waist_joint="waist_yaw_joint",
    init_pos_xyz=(0.0, 0.0, 0.925),
    height_z_des=0.925,
    actuators_override=GR3V2_ACTUATORS_OVERRIDE,
)

# -----------------------
# hightorque_hi 
# -----------------------
_HI_USD = os.path.join(os.environ["HOME"], "humanoid_locomotion/robots/hightorque_hi/hi_25dof.usd")
if not os.path.isfile(_HI_USD):
    raise FileNotFoundError(f"hightorque_hi USD not found: {_HI_USD}")
HI_LEFT_LEG_6: List[str] = [
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_thigh_joint",       
    "l_calf_joint",        
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
]
HI_RIGHT_LEG_6: List[str] = [
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_thigh_joint",       
    "r_calf_joint",        
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]
HI_WAIST_YAW_1: List[str] = ["waist_joint"]
HI_HEAD_2: List[str] = ["head_yaw_joint", "head_pitch_joint"]
HI_LEFT_ARM_5: List[str] = [
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_upper_arm_joint",   
    "l_elbow_joint",
    "l_wrist_joint",
]
HI_RIGHT_ARM_5: List[str] = [
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_upper_arm_joint",   
    "r_elbow_joint",
    "r_wrist_joint",
]
HI_ACTION_JOINTS_13: List[str] = HI_LEFT_LEG_6 + HI_RIGHT_LEG_6 + HI_WAIST_YAW_1
HI_JOINTS_25: List[str] = (
    HI_ACTION_JOINTS_13
    + HI_HEAD_2
    + HI_LEFT_ARM_5
    + HI_RIGHT_ARM_5
)

assert len(HI_ACTION_JOINTS_13) == 13
assert len(HI_JOINTS_25) == 25
HI_BASE_LINK = "base_link"
HI_LEFT_FOOT_BODY = "l_ankle_roll_link"
HI_RIGHT_FOOT_BODY = "r_ankle_roll_link"
HI_URDF_LIMITS: Dict[str, Dict[str, float]] = {
    j: {"lower": -3.14, "upper": 3.14, "effort": 21.0, "vel": 21.0}
    for j in HI_JOINTS_25
}
HI_ACTIONS_MIN_13: List[float] = [float(HI_URDF_LIMITS[j]["lower"]) for j in HI_ACTION_JOINTS_13]
HI_ACTIONS_MAX_13: List[float] = [float(HI_URDF_LIMITS[j]["upper"]) for j in HI_ACTION_JOINTS_13]
assert len(HI_ACTIONS_MIN_13) == 13
assert len(HI_ACTIONS_MAX_13) == 13
HI_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    # left leg
    "l_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "l_hip_roll_joint":  0.0,
    "l_thigh_joint":     0.0,
    "l_calf_joint":      float(+np.deg2rad(29.5)),   
    "l_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    "l_ankle_roll_joint":  0.0,
    # right leg
    "r_hip_pitch_joint": float(-np.deg2rad(14.0)),
    "r_hip_roll_joint":  0.0,
    "r_thigh_joint":     0.0,
    "r_calf_joint":      float(+np.deg2rad(29.5)),   
    "r_ankle_pitch_joint": float(-np.deg2rad(13.7)),
    "r_ankle_roll_joint":  0.0,
    # waist
    "waist_joint": 0.0,
    # head
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    # left arm
    "l_shoulder_pitch_joint": 0.0,
    "l_shoulder_roll_joint":  0.0,
    "l_upper_arm_joint":      0.0,
    "l_elbow_joint":          0.0,
    "l_wrist_joint":          0.0,
    # right arm
    "r_shoulder_pitch_joint": 0.0,
    "r_shoulder_roll_joint":  0.0,
    "r_upper_arm_joint":      0.0,
    "r_elbow_joint":          0.0,
    "r_wrist_joint":          0.0,
}
HI_DEFAULT_JOINT_ANGLES = {k: float(v) for k, v in HI_DEFAULT_JOINT_ANGLES.items()}
HI_TAU_LIMIT_25: List[float] = [float(HI_URDF_LIMITS[j]["effort"]) for j in HI_JOINTS_25]
assert len(HI_TAU_LIMIT_25) == 25
HI_TAU_LIMIT_BY_JOINT: Dict[str, float] = {j: float(tau) for j, tau in zip(HI_JOINTS_25, HI_TAU_LIMIT_25)}
HI_JOINT_LIMITS: Dict[str, Dict[str, float]] = {}
for j, lo, hi in zip(HI_ACTION_JOINTS_13, HI_ACTIONS_MIN_13, HI_ACTIONS_MAX_13):
    lim = HI_URDF_LIMITS[j]
    HI_JOINT_LIMITS[j] = {
        "lower": float(lo),
        "upper": float(hi),
        "effort": float(HI_TAU_LIMIT_BY_JOINT.get(j, lim.get("effort", 50.0))),
        "vel": float(lim.get("vel", 50.0)),
    }
HI_HELD_JOINTS: List[str] = list(HI_HEAD_2 + HI_LEFT_ARM_5 + HI_RIGHT_ARM_5)
for j in HI_HELD_JOINTS:
    lim = HI_URDF_LIMITS[j]
    HI_JOINT_LIMITS[j] = {
        "lower": float(lim["lower"]),
        "upper": float(lim["upper"]),
        "effort": float(HI_TAU_LIMIT_BY_JOINT.get(j, lim.get("effort", 50.0))),
        "vel": float(lim.get("vel", 50.0)),
    }
HI_JOINT_REG_NAMES: Dict[str, str] = {
    "l_ankle_roll":  "l_ankle_roll_joint",
    "r_ankle_roll":  "r_ankle_roll_joint",
    "l_ankle_pitch": "l_ankle_pitch_joint",
    "r_ankle_pitch": "r_ankle_pitch_joint",
    "l_hip_yaw":     "l_thigh_joint",
    "r_hip_yaw":     "r_thigh_joint",
}
HI_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {

    "Waist": ImplicitActuatorCfg(
        joint_names_expr=["waist_joint$"],
        stiffness=50.0,
        damping=1.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Head": ImplicitActuatorCfg(
        joint_names_expr=[
            "head_yaw_joint$",
            "head_pitch_joint$",
        ],
        stiffness=50.0,
        damping=1.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Hip_Pitch": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_hip_pitch_joint$",
            "r_hip_pitch_joint$",
        ],
        stiffness=100.0,
        damping=4.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Hip_Roll": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_hip_roll_joint$",
            "r_hip_roll_joint$",
        ],
        stiffness=100.0,
        damping=4.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Hip_Yaw": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_thigh_joint$",
            "r_thigh_joint$",
        ],
        stiffness=100.0,
        damping=4.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Knee": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_calf_joint$",
            "r_calf_joint$",
        ],
        stiffness=125.0,
        damping=5.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Ankle_Pitch": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_ankle_pitch_joint$",
            "r_ankle_pitch_joint$",
        ],
        stiffness=25.0,
        damping=1.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Ankle_Roll": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_ankle_roll_joint$",
            "r_ankle_roll_joint$",
        ],
        stiffness=25.0,
        damping=1.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Shoulders": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_shoulder_pitch_joint$",
            "l_shoulder_roll_joint$",
            "r_shoulder_pitch_joint$",
            "r_shoulder_roll_joint$",
        ],
        stiffness=40.0,
        damping=1.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Upper_Arm_Yaw": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_upper_arm_joint$",
            "r_upper_arm_joint$",
        ],
        stiffness=40.0,
        damping=1.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Elbow": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_elbow_joint$",
            "r_elbow_joint$",
        ],
        stiffness=40.0,
        damping=1.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
    "Wrist": ImplicitActuatorCfg(
        joint_names_expr=[
            "l_wrist_joint$",
            "r_wrist_joint$",
        ],
        stiffness=40.0,
        damping=1.0,
        effort_limit_sim=21.0,
        armature=0.0,
        friction=0.1,
    ),
}
ROBOT_SPECS["hightorque_hi"] = RobotSpec(
    name="hightorque_hi",
    usd_path=_HI_USD,
    base_link=HI_BASE_LINK,
    left_foot_body=HI_LEFT_FOOT_BODY,
    right_foot_body=HI_RIGHT_FOOT_BODY,
    action_joints=list(HI_ACTION_JOINTS_13),
    all_joints=list(HI_JOINTS_25),
    held_joints=list(HI_HELD_JOINTS),
    default_joint_angles=dict(HI_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(HI_JOINT_LIMITS),
    joint_reg_names=dict(HI_JOINT_REG_NAMES),
    waist_joint="waist_joint",
    init_pos_xyz=(0.0, 0.0, 0.70),   
    height_z_des=0.68,
    actuators_override=HI_ACTUATORS_OVERRIDE,              
)

# ---------------------------------------------------------------------
# KUAVO_S45 
# ---------------------------------------------------------------------
S45_USD = os.path.join(os.environ["HOME"], "humanoid_locomotion/robots/kuavo_s45/biped_s45.usd")
if not os.path.isfile(S45_USD):
    raise FileNotFoundError(f"kuavo_s45 USD not found: {S45_USD}")
S45_BASE_LINK = "base_link"
S45_LEFT_FOOT_BODY = "leg_l6_link"
S45_RIGHT_FOOT_BODY = "leg_r6_link"
S45_LEFT_TOE  = "l_foot_toe"
S45_LEFT_HEEL = "l_foot_heel"
S45_RIGHT_TOE  = "r_foot_toe"
S45_RIGHT_HEEL = "r_foot_heel"
S45_LEFT_LEG_6: List[str] = [
    "leg_l1_joint",  
    "leg_l2_joint", 
    "leg_l3_joint",  
    "leg_l4_joint", 
    "leg_l5_joint",  
    "leg_l6_joint",  
]
S45_RIGHT_LEG_6: List[str] = [
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r3_joint",
    "leg_r4_joint",
    "leg_r5_joint",
    "leg_r6_joint",
]
S45_LEFT_ARM_7: List[str] = [
    "zarm_l1_joint",
    "zarm_l2_joint",
    "zarm_l3_joint",
    "zarm_l4_joint",
    "zarm_l5_joint",
    "zarm_l6_joint",
    "zarm_l7_joint",
]
S45_RIGHT_ARM_7: List[str] = [
    "zarm_r1_joint",
    "zarm_r2_joint",
    "zarm_r3_joint",
    "zarm_r4_joint",
    "zarm_r5_joint",
    "zarm_r6_joint",
    "zarm_r7_joint",
]
S45_HEAD_2: List[str] = [
    "zhead_1_joint",  
    "zhead_2_joint", 
]

S45_ACTION_JOINTS_12: List[str] = S45_LEFT_LEG_6 + S45_RIGHT_LEG_6
S45_HELD_JOINTS_16: List[str] = S45_HEAD_2 + S45_LEFT_ARM_7 + S45_RIGHT_ARM_7
S45_JOINTS_28: List[str] = S45_ACTION_JOINTS_12 + S45_HELD_JOINTS_16
assert len(S45_ACTION_JOINTS_12) == 12
assert len(S45_HELD_JOINTS_16) == 16
assert len(S45_JOINTS_28) == 28
S45_JOINT_LIMITS_URDF: Dict[str, Dict[str, float]] = {
    # left leg
    "leg_l1_joint": {"lower": -0.314159265358979, "upper": 0.663225115757845, "effort": 180.0, "vel": 23.55},
    "leg_l2_joint": {"lower": -0.872664625997165, "upper": 0.785398163397448, "effort": 100.0, "vel": 41.3433333333333},
    "leg_l3_joint": {"lower": -2.00712863979348, "upper": 1.5707963267949,   "effort": 100.0, "vel": 41.3433333333333},
    "leg_l4_joint": {"lower": 0.0,                "upper": 2.53072741539178, "effort": 180.0, "vel": 23.55},
    "leg_l5_joint": {"lower": -0.872664625997165, "upper": 0.349065850398866,"effort": 36.0,  "vel": 23.0266666666667},
    "leg_l6_joint": {"lower": -0.261799387799149, "upper": 0.261799387799149,"effort": 36.0,  "vel": 23.0266666666667},
    # right leg
    "leg_r1_joint": {"lower": -0.663225115757845, "upper": 0.314159265358979,"effort": 180.0, "vel": 23.55},
    "leg_r2_joint": {"lower": -0.785398163397448, "upper": 0.872664625997165,"effort": 100.0, "vel": 41.3433333333333},
    "leg_r3_joint": {"lower": -2.00712863979348,  "upper": 1.5707963267949,  "effort": 100.0, "vel": 41.3433333333333},
    "leg_r4_joint": {"lower": 0.0,                "upper": 2.53072741539178, "effort": 180.0, "vel": 23.55},
    "leg_r5_joint": {"lower": -0.872664625997165, "upper": 0.349065850398866,"effort": 36.0,  "vel": 23.0266666666667},
    "leg_r6_joint": {"lower": -0.261799387799149, "upper": 0.261799387799149,"effort": 36.0,  "vel": 23.0266666666667},
    # left arm
    "zarm_l1_joint": {"lower": -3.14159265358979, "upper": 1.5707963267949,  "effort": 100.0, "vel": 41.3433333333333},
    "zarm_l2_joint": {"lower": -0.349065850398866,"upper": 2.0943951023932,  "effort": 50.0,  "vel": 19.154},
    "zarm_l3_joint": {"lower": -1.5707963267949,  "upper": 1.5707963267949,  "effort": 39.0,  "vel": 18.84},
    "zarm_l4_joint": {"lower": -2.61799387799149, "upper": 0.0,              "effort": 50.0,  "vel": 18.84},
    "zarm_l5_joint": {"lower": -1.5707963267949,  "upper": 1.5707963267949,  "effort": 12.0,  "vel": 21.666},
    "zarm_l6_joint": {"lower": -1.30899693899575, "upper": 0.698131700797732,"effort": 12.0,  "vel": 21.666},
    "zarm_l7_joint": {"lower": -0.698131700797732,"upper": 0.698131700797732,"effort": 12.0,  "vel": 21.666},
    # right arm
    "zarm_r1_joint": {"lower": -3.14159265358979, "upper": 1.5707963267949,  "effort": 100.0, "vel": 41.3433333333333},
    "zarm_r2_joint": {"lower": -2.0943951023932,  "upper": 0.349065850398866,"effort": 50.0,  "vel": 19.154},
    "zarm_r3_joint": {"lower": -1.5707963267949,  "upper": 1.5707963267949,  "effort": 39.0,  "vel": 18.84},
    "zarm_r4_joint": {"lower": -2.61799387799149, "upper": 0.0,              "effort": 50.0,  "vel": 18.84},
    "zarm_r5_joint": {"lower": -1.5707963267949,  "upper": 1.5707963267949,  "effort": 12.0,  "vel": 21.666},
    "zarm_r6_joint": {"lower": -0.698131700797732,"upper": 1.30899693899575, "effort": 12.0,  "vel": 21.666},
    "zarm_r7_joint": {"lower": -0.698131700797732,"upper": 0.698131700797732,"effort": 12.0,  "vel": 21.666},
    # head
    "zhead_1_joint": {"lower": -1.5707963267949,  "upper": 1.5707963267949,  "effort": 1.5,   "vel": 5.23333333333333},
    "zhead_2_joint": {"lower": -0.523598775598299,"upper": 0.523598775598299,"effort": 12.0,  "vel": 5.23333333333333},
}

S45_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    "leg_l1_joint":  0.02,
    "leg_r1_joint": -0.02,
    "leg_l2_joint":  0.00,
    "leg_r2_joint":  0.00,
    "leg_l3_joint": -0.15,
    "leg_r3_joint": -0.15,
    "leg_l4_joint":  0.5,
    "leg_r4_joint":  0.5,
    "leg_l5_joint": -0.10,
    "leg_r5_joint": -0.10,
    "leg_l6_joint":  0.00,
    "leg_r6_joint":  0.00,
    "zarm_l1_joint": -0.0,
    "zarm_l2_joint":  0.20,
    "zarm_l3_joint":  0.00,
    "zarm_l4_joint": -0.0,
    "zarm_l5_joint":  0.00,
    "zarm_l6_joint":  0.00,
    "zarm_l7_joint":  0.00,
    "zarm_r1_joint": -0.0,
    "zarm_r2_joint": -0.0,
    "zarm_r3_joint":  0.00,
    "zarm_r4_joint": -0.0,
    "zarm_r5_joint":  0.00,
    "zarm_r6_joint":  0.00,
    "zarm_r7_joint":  0.00,
    "zhead_1_joint": 0.0,
    "zhead_2_joint": 0.0,
}

for j, v in list(S45_DEFAULT_JOINT_ANGLES.items()):
    lim = S45_JOINT_LIMITS_URDF[j]
    S45_DEFAULT_JOINT_ANGLES[j] = float(np.clip(v, lim["lower"], lim["upper"]))
S45_JOINT_LIMITS: Dict[str, Dict[str, float]] = {
    j: {
        "lower": float(S45_JOINT_LIMITS_URDF[j]["lower"]),
        "upper": float(S45_JOINT_LIMITS_URDF[j]["upper"]),
        "effort": float(S45_JOINT_LIMITS_URDF[j]["effort"]),
        "vel": float(S45_JOINT_LIMITS_URDF[j]["vel"]),
    }
    for j in S45_JOINTS_28
}

S45_JOINT_REG_NAMES: Dict[str, str] = {
    "l_hip_yaw":   "leg_l2_joint",
    "l_ankle_pitch":"leg_l5_joint",
    "l_ankle_roll":"leg_l6_joint",
    "r_hip_yaw":   "leg_r2_joint",
    "r_ankle_pitch":"leg_r5_joint",
    "r_ankle_roll":"leg_r6_joint",
}

S45_JOINT_REG_PAIR_MODE: Dict[str, str] = {
    "ankle_roll": "opposite",
    "ankle_pitch": "opposite",
    "hip_yaw": "opposite",
}
KUAVO_S45_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            "leg_[lr][1-4]_joint$", 
        ],
        effort_limit_sim=300,
        stiffness={
            "leg_[lr]1_joint$": 150.0,  
            "leg_[lr]2_joint$": 120.0,  
            "leg_[lr]3_joint$": 180.0,  
            "leg_[lr]4_joint$": 180.0, 
        },
        damping={
            "leg_[lr]1_joint$": 5.0,
            "leg_[lr]2_joint$": 4.0,
            "leg_[lr]3_joint$": 5.0,
            "leg_[lr]4_joint$": 5.0,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[
            "leg_[lr][5-6]_joint$", 
        ],
        effort_limit_sim=100,
        stiffness={
            "leg_[lr]5_joint$": 25.0,
            "leg_[lr]6_joint$": 20.0,
        },
        damping={
            "leg_[lr]5_joint$": 2.5,
            "leg_[lr]6_joint$": 2.0,
        },
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            "zarm_[lr][1-4]_joint$",   
            "zarm_[lr][5-7]_joint$",   
        ],
        effort_limit_sim=150,
        stiffness={
            "zarm_[lr]1_joint$": 20.0,
            "zarm_[lr]2_joint$": 15.0,
            "zarm_[lr]3_joint$": 20.0,
            "zarm_[lr]4_joint$": 20.0,
            "zarm_[lr]5_joint$": 15.0,
            "zarm_[lr]6_joint$": 10.0,
            "zarm_[lr]7_joint$": 10.0,
        },
        damping={
            "zarm_[lr]1_joint$": 8.0,
            "zarm_[lr]2_joint$": 6.0,
            "zarm_[lr]3_joint$": 6.0,
            "zarm_[lr]4_joint$": 6.0,
            "zarm_[lr]5_joint$": 3.0,
            "zarm_[lr]6_joint$": 2.5,
            "zarm_[lr]7_joint$": 2.5,
        },
    ),
    "head": ImplicitActuatorCfg(
        joint_names_expr=[
            "zhead_1_joint$",
            "zhead_2_joint$",
        ],
        effort_limit_sim=20,
        stiffness={
            "zhead_1_joint$": 10.0,
            "zhead_2_joint$": 15.0,
        },
        damping={
            "zhead_1_joint$": 1.0,
            "zhead_2_joint$": 1.5,
        },
    ),
}
ROBOT_SPECS["kuavo_s45"] = RobotSpec(
    name="kuavo_s45",
    usd_path=S45_USD,
    base_link=S45_BASE_LINK,
    left_foot_body=S45_LEFT_FOOT_BODY,
    right_foot_body=S45_RIGHT_FOOT_BODY,
    action_joints=list(S45_ACTION_JOINTS_12),
    all_joints=list(S45_JOINTS_28),
    held_joints=list(S45_HELD_JOINTS_16),
    default_joint_angles=dict(S45_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(S45_JOINT_LIMITS),
    joint_reg_names=dict(S45_JOINT_REG_NAMES),
    joint_reg_pair_mode=dict(S45_JOINT_REG_PAIR_MODE), 
    waist_joint=None,
    init_pos_xyz=(0.0, 0.0, 0.875),
    height_z_des=0.85,
    actuators_override=KUAVO_S45_ACTUATORS_OVERRIDE,
)
# -----------------------------
# TienKung
# -----------------------------
TIENKUNG2_USD = os.path.join(
    os.environ["HOME"],
    "humanoid_locomotion/robots/tienkung2/tienkung2_lite.usd",
)
if not os.path.isfile(TIENKUNG2_USD):
    raise FileNotFoundError(f"tienkung2 USD not found: {TIENKUNG2_USD}")
TIENKUNG2_BASE_LINK = "pelvis"
TIENKUNG2_LEFT_FOOT_BODY = "ankle_roll_l_link"
TIENKUNG2_RIGHT_FOOT_BODY = "ankle_roll_r_link"
TIENKUNG2_LEFT_TOE = None
TIENKUNG2_LEFT_HEEL = None
TIENKUNG2_RIGHT_TOE = None
TIENKUNG2_RIGHT_HEEL = None
TIENKUNG2_LEFT_LEG_6: List[str] = [
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
]
TIENKUNG2_RIGHT_LEG_6: List[str] = [
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
]
TIENKUNG2_LEFT_ARM_4: List[str] = [
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
]
TIENKUNG2_RIGHT_ARM_4: List[str] = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]
TIENKUNG2_HEAD_0: List[str] = []
TIENKUNG2_ACTION_JOINTS_12: List[str] = TIENKUNG2_LEFT_LEG_6 + TIENKUNG2_RIGHT_LEG_6
TIENKUNG2_HELD_JOINTS_8: List[str] = TIENKUNG2_LEFT_ARM_4 + TIENKUNG2_RIGHT_ARM_4
TIENKUNG2_JOINTS_20: List[str] = TIENKUNG2_ACTION_JOINTS_12 + TIENKUNG2_HELD_JOINTS_8
assert len(TIENKUNG2_ACTION_JOINTS_12) == 12
assert len(TIENKUNG2_HELD_JOINTS_8) == 8
assert len(TIENKUNG2_JOINTS_20) == 20

TIENKUNG2_JOINT_LIMITS_URDF: Dict[str, Dict[str, float]] = {
    # left leg
    "hip_roll_l_joint":   {"lower": -0.79,   "upper":  0.79,   "effort": 180.0, "vel": 15.6},
    "hip_pitch_l_joint":  {"lower": -2.79,   "upper":  2.09,   "effort": 300.0, "vel": 15.6},
    "hip_yaw_l_joint":    {"lower": -1.04,   "upper":  1.04,   "effort": 180.0, "vel": 15.6},
    "knee_pitch_l_joint": {"lower":  0.0,    "upper":  2.39,   "effort": 300.0, "vel": 15.6},
    "ankle_pitch_l_joint":{"lower": -1.22,   "upper":  0.5236, "effort":  60.0, "vel": 12.8},
    "ankle_roll_l_joint": {"lower": -0.4363, "upper":  0.4363, "effort":  30.0, "vel":  7.8},
    # right leg
    "hip_roll_r_joint":   {"lower": -0.79,   "upper":  0.79,   "effort": 180.0, "vel": 15.6},
    "hip_pitch_r_joint":  {"lower": -2.79,   "upper":  2.09,   "effort": 300.0, "vel": 15.6},
    "hip_yaw_r_joint":    {"lower": -1.04,   "upper":  1.04,   "effort": 180.0, "vel": 15.6},
    "knee_pitch_r_joint": {"lower":  0.0,    "upper":  2.39,   "effort": 300.0, "vel": 15.6},
    "ankle_pitch_r_joint":{"lower": -1.22,   "upper":  0.5236, "effort":  60.0, "vel": 12.8},
    "ankle_roll_r_joint": {"lower": -0.4363, "upper":  0.4363, "effort":  30.0, "vel":  7.8},
    # left arm
    "shoulder_pitch_l_joint": {"lower": -2.96,   "upper": 2.96,  "effort": 52.5, "vel": 14.1},
    "shoulder_roll_l_joint":  {"lower": -0.2618, "upper": 3.4,   "effort": 52.5, "vel": 14.1},
    "shoulder_yaw_l_joint":   {"lower": -2.96,   "upper": 2.96,  "effort": 52.5, "vel": 14.1},
    "elbow_pitch_l_joint":    {"lower": -2.61,   "upper": 0.261, "effort": 52.5, "vel": 14.1},
    # right arm
    "shoulder_pitch_r_joint": {"lower": -2.96,  "upper": 2.96,  "effort": 52.5, "vel": 14.1},
    "shoulder_roll_r_joint":  {"lower": -3.4,   "upper": 0.2618,"effort": 52.5, "vel": 14.1},
    "shoulder_yaw_r_joint":   {"lower": -2.96,  "upper": 2.96,  "effort": 52.5, "vel": 14.1},
    "elbow_pitch_r_joint":    {"lower": -2.61,  "upper": 0.261, "effort": 52.5, "vel": 14.1},
}

TIENKUNG2_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    # left leg
    "hip_roll_l_joint": 0.0,
    "hip_pitch_l_joint": 0.0,
    "hip_yaw_l_joint": 0.0,
    "knee_pitch_l_joint": 0.50,
    "ankle_pitch_l_joint": 0.0,
    "ankle_roll_l_joint": 0.0,
    # right leg
    "hip_roll_r_joint": 0.0,
    "hip_pitch_r_joint": 0.0,
    "hip_yaw_r_joint": 0.0,
    "knee_pitch_r_joint": 0.50,
    "ankle_pitch_r_joint": 0.0,
    "ankle_roll_r_joint": 0.0,
    # arms (held)
    "shoulder_pitch_l_joint": 0.0,
    "shoulder_roll_l_joint": 0.2,     
    "shoulder_yaw_l_joint": 0.0,
    "elbow_pitch_l_joint": -0.5,     
    "shoulder_pitch_r_joint": 0.0,
    "shoulder_roll_r_joint": -0.2,   
    "shoulder_yaw_r_joint": 0.0,
    "elbow_pitch_r_joint": -0.5,
}

for j, v in list(TIENKUNG2_DEFAULT_JOINT_ANGLES.items()):
    lim = TIENKUNG2_JOINT_LIMITS_URDF[j]
    TIENKUNG2_DEFAULT_JOINT_ANGLES[j] = float(np.clip(v, lim["lower"], lim["upper"]))
TIENKUNG2_JOINT_LIMITS: Dict[str, Dict[str, float]] = {
    j: {
        "lower": float(TIENKUNG2_JOINT_LIMITS_URDF[j]["lower"]),
        "upper": float(TIENKUNG2_JOINT_LIMITS_URDF[j]["upper"]),
        "effort": float(TIENKUNG2_JOINT_LIMITS_URDF[j]["effort"]),
        "vel": float(TIENKUNG2_JOINT_LIMITS_URDF[j]["vel"]),
    }
    for j in TIENKUNG2_JOINTS_20
}

TIENKUNG2_JOINT_REG_NAMES: Dict[str, str] = {
    "l_hip_yaw": "hip_yaw_l_joint",
    "l_ankle_pitch": "ankle_pitch_l_joint",
    "l_ankle_roll": "ankle_roll_l_joint",
    "r_hip_yaw": "hip_yaw_r_joint",
    "r_ankle_pitch": "ankle_pitch_r_joint",
    "r_ankle_roll": "ankle_roll_r_joint",
}
TIENKUNG2_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            "hip_roll_[lr]_joint$",
            "hip_pitch_[lr]_joint$",
            "hip_yaw_[lr]_joint$",
            "knee_pitch_[lr]_joint$",
        ],
        effort_limit_sim=300,
        stiffness={
            "hip_roll_[lr]_joint$": 100.0,
            "hip_pitch_[lr]_joint$": 125.0,
            "hip_yaw_[lr]_joint$": 100.0,
            "knee_pitch_[lr]_joint$": 125.0,
        },
        damping={
            "hip_roll_[lr]_joint$": 10.0,
            "hip_pitch_[lr]_joint$": 10.0,
            "hip_yaw_[lr]_joint$": 15.0,
            "knee_pitch_[lr]_joint$": 15.0,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[
            "ankle_pitch_[lr]_joint$",
            "ankle_roll_[lr]_joint$",
        ],
        effort_limit_sim=100,
        stiffness={
            "ankle_pitch_[lr]_joint$": 20.0,
            "ankle_roll_[lr]_joint$": 20.0,
        },
        damping={
            "ankle_pitch_[lr]_joint$": 2.0,
            "ankle_roll_[lr]_joint$": 2.0,
        },
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            "shoulder_pitch_[lr]_joint$",
            "shoulder_roll_[lr]_joint$",
            "shoulder_yaw_[lr]_joint$",
            "elbow_pitch_[lr]_joint$",
        ],
        effort_limit_sim=100,
        stiffness={
            "shoulder_pitch_[lr]_joint$": 40.0,
            "shoulder_roll_[lr]_joint$": 40.0,
            "shoulder_yaw_[lr]_joint$": 40.0,
            "elbow_pitch_[lr]_joint$": 40.0,
        },
        damping={
            "shoulder_pitch_[lr]_joint$": 10.0,
            "shoulder_roll_[lr]_joint$": 10.0,
            "shoulder_yaw_[lr]_joint$": 10.0,
            "elbow_pitch_[lr]_joint$": 10.0,
        },
    ),
}

ROBOT_SPECS["tienkung2"] = RobotSpec(
    name="tienkung2",
    usd_path=TIENKUNG2_USD,
    base_link=TIENKUNG2_BASE_LINK,
    left_foot_body=TIENKUNG2_LEFT_FOOT_BODY,
    right_foot_body=TIENKUNG2_RIGHT_FOOT_BODY,
    action_joints=list(TIENKUNG2_ACTION_JOINTS_12),
    all_joints=list(TIENKUNG2_JOINTS_20),
    held_joints=list(TIENKUNG2_HELD_JOINTS_8),
    default_joint_angles=dict(TIENKUNG2_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(TIENKUNG2_JOINT_LIMITS),
    joint_reg_names=dict(TIENKUNG2_JOINT_REG_NAMES),
    waist_joint=None,               
    init_pos_xyz=(0.0, 0.0, 1.0),    
    height_z_des=0.975,
    actuators_override=TIENKUNG2_ACTUATORS_OVERRIDE,                
)

# -----------------------
# berkeley_humanoid_lite
# -----------------------
_BHL_USD = os.path.join(
    os.environ["HOME"],
    "humanoid_locomotion/robots/berkeley_humanoid_lite/berkeley_humanoid_lite.usd",
)
if not os.path.isfile(_BHL_USD):
    raise FileNotFoundError(f"berkeley_humanoid_lite USD not found: {_BHL_USD}")
BHL_LEFT_ARM_5: List[str] = [
    "arm_left_shoulder_pitch_joint",
    "arm_left_shoulder_roll_joint",
    "arm_left_shoulder_yaw_joint",
    "arm_left_elbow_pitch_joint",
    "arm_left_elbow_roll_joint",
]
BHL_RIGHT_ARM_5: List[str] = [
    "arm_right_shoulder_pitch_joint",
    "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint",
    "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
]
BHL_LEFT_LEG_6: List[str] = [
    "leg_left_hip_roll_joint",
    "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint",
    "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint",
    "leg_left_ankle_roll_joint",
]
BHL_RIGHT_LEG_6: List[str] = [
    "leg_right_hip_roll_joint",
    "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint",
    "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint",
    "leg_right_ankle_roll_joint",
]
BHL_ACTION_JOINTS_12: List[str] = BHL_LEFT_LEG_6 + BHL_RIGHT_LEG_6
BHL_JOINTS_22: List[str] = BHL_ACTION_JOINTS_12 + BHL_LEFT_ARM_5 + BHL_RIGHT_ARM_5
assert len(BHL_ACTION_JOINTS_12) == 12
assert len(BHL_JOINTS_22) == 22
BHL_BASE_LINK = "base"
BHL_LEFT_FOOT_BODY = "leg_left_ankle_roll"
BHL_RIGHT_FOOT_BODY = "leg_right_ankle_roll"
BHL_URDF_LIMITS: Dict[str, Dict[str, float]] = {
    # ---------------- arm left ----------------
    "arm_left_shoulder_pitch_joint": {"lower": -1.5708,   "upper": 0.785398, "effort": 20.0, "vel": 15.0},
    "arm_left_shoulder_roll_joint":  {"lower": -0.261799, "upper": 1.309,    "effort": 20.0, "vel": 15.0},
    "arm_left_shoulder_yaw_joint":   {"lower": -0.785398, "upper": 0.785398, "effort": 20.0, "vel": 15.0},
    "arm_left_elbow_pitch_joint":    {"lower": 0.0,       "upper": 1.5708,   "effort": 20.0, "vel": 15.0},
    "arm_left_elbow_roll_joint":     {"lower": -0.785398, "upper": 0.785398, "effort": 20.0, "vel": 15.0},

    # ---------------- arm right ----------------
    "arm_right_shoulder_pitch_joint": {"lower": -0.785398, "upper": 1.5708,   "effort": 20.0, "vel": 15.0},
    "arm_right_shoulder_roll_joint":  {"lower": -1.309,    "upper": 0.261799, "effort": 20.0, "vel": 15.0},
    "arm_right_shoulder_yaw_joint":   {"lower": -0.785398, "upper": 0.785398, "effort": 20.0, "vel": 15.0},
    "arm_right_elbow_pitch_joint":    {"lower": -1.5708,   "upper": 0.0,      "effort": 20.0, "vel": 15.0},
    "arm_right_elbow_roll_joint":     {"lower": -0.785398, "upper": 0.785398, "effort": 20.0, "vel": 15.0},

    # ---------------- leg left ----------------
    "leg_left_hip_roll_joint":   {"lower": -0.174533, "upper": 1.5708,   "effort": 20.0, "vel": 15.0},
    "leg_left_hip_yaw_joint":    {"lower": -0.981748, "upper": 0.589049, "effort": 20.0, "vel": 15.0},
    "leg_left_hip_pitch_joint":  {"lower": -1.89805,  "upper": 0.981748, "effort": 20.0, "vel": 15.0},
    "leg_left_knee_pitch_joint": {"lower": 0.0,       "upper": 2.44346,  "effort": 20.0, "vel": 15.0},
    "leg_left_ankle_pitch_joint":{"lower": -0.785398, "upper": 0.785398, "effort": 20.0, "vel": 15.0},
    "leg_left_ankle_roll_joint": {"lower": -0.261799, "upper": 0.261799, "effort": 20.0, "vel": 15.0},

    # ---------------- leg right ----------------
    "leg_right_hip_roll_joint":   {"lower": -1.5708,   "upper": 0.174533, "effort": 20.0, "vel": 15.0},
    "leg_right_hip_yaw_joint":    {"lower": -0.589049, "upper": 0.981748, "effort": 20.0, "vel": 15.0},
    "leg_right_hip_pitch_joint":  {"lower": -1.89805,  "upper": 0.981748, "effort": 20.0, "vel": 15.0},
    "leg_right_knee_pitch_joint": {"lower": 0.0,       "upper": 2.44346,  "effort": 20.0, "vel": 15.0},
    "leg_right_ankle_pitch_joint":{"lower": -0.785398, "upper": 0.785398, "effort": 20.0, "vel": 15.0},
    "leg_right_ankle_roll_joint": {"lower": -0.261799, "upper": 0.261799, "effort": 20.0, "vel": 15.0},
}
_missing = [j for j in BHL_JOINTS_22 if j not in BHL_URDF_LIMITS]
assert not _missing, f"[berkeley_humanoid_lite] missing limits for joints: {_missing}"
BHL_ACTIONS_MIN_12: List[float] = [float(BHL_URDF_LIMITS[j]["lower"]) for j in BHL_ACTION_JOINTS_12]
BHL_ACTIONS_MAX_12: List[float] = [float(BHL_URDF_LIMITS[j]["upper"]) for j in BHL_ACTION_JOINTS_12]
assert len(BHL_ACTIONS_MIN_12) == 12
assert len(BHL_ACTIONS_MAX_12) == 12
BHL_DEFAULT_JOINT_ANGLES: Dict[str, float] = {
    # ---------------- left leg ----------------
    "leg_left_hip_roll_joint":   0.0,
    "leg_left_hip_yaw_joint":    0.0,
    "leg_left_hip_pitch_joint":  0.0,
    "leg_left_knee_pitch_joint": float(+np.deg2rad(32.0)),
    "leg_left_ankle_pitch_joint": 0.0,
    "leg_left_ankle_roll_joint": 0.0,
    # ---------------- right leg ----------------
    "leg_right_hip_roll_joint":   0.0,
    "leg_right_hip_yaw_joint":    0.0,
    "leg_right_hip_pitch_joint":  0.0,
    "leg_right_knee_pitch_joint": float(+np.deg2rad(32.0)),
    "leg_right_ankle_pitch_joint": 0.0,
    "leg_right_ankle_roll_joint": 0.0,
    # ---------------- left arm ----------------
    "arm_left_shoulder_pitch_joint": 0.0,
    "arm_left_shoulder_roll_joint":  0.0,
    "arm_left_shoulder_yaw_joint":   0.0,
    "arm_left_elbow_pitch_joint":    float(+np.deg2rad(20.0)),  
    "arm_left_elbow_roll_joint":     0.0,
    # ---------------- right arm ----------------
    "arm_right_shoulder_pitch_joint": 0.0,
    "arm_right_shoulder_roll_joint":  0.0,
    "arm_right_shoulder_yaw_joint":   0.0,
    "arm_right_elbow_pitch_joint":    float(-np.deg2rad(20.0)),
    "arm_right_elbow_roll_joint":     0.0,
}
BHL_DEFAULT_JOINT_ANGLES = {k: float(v) for k, v in BHL_DEFAULT_JOINT_ANGLES.items()}
BHL_TAU_LIMIT_22: List[float] = [float(BHL_URDF_LIMITS[j]["effort"]) for j in BHL_JOINTS_22]
assert len(BHL_TAU_LIMIT_22) == 22
BHL_TAU_LIMIT_BY_JOINT: Dict[str, float] = {j: float(tau) for j, tau in zip(BHL_JOINTS_22, BHL_TAU_LIMIT_22)}
BHL_JOINT_LIMITS: Dict[str, Dict[str, float]] = {}
for j, lo, hi in zip(BHL_ACTION_JOINTS_12, BHL_ACTIONS_MIN_12, BHL_ACTIONS_MAX_12):
    lim = BHL_URDF_LIMITS[j]
    BHL_JOINT_LIMITS[j] = {
        "lower": float(lo),
        "upper": float(hi),
        "effort": float(BHL_TAU_LIMIT_BY_JOINT.get(j, lim.get("effort", 300.0))),
        "vel": float(lim.get("vel", 50.0)),
    }
BHL_HELD_JOINTS: List[str] = list(BHL_LEFT_ARM_5 + BHL_RIGHT_ARM_5)
for j in BHL_HELD_JOINTS:
    lim = BHL_URDF_LIMITS[j]
    BHL_JOINT_LIMITS[j] = {
        "lower": float(lim["lower"]),
        "upper": float(lim["upper"]),
        "effort": float(BHL_TAU_LIMIT_BY_JOINT.get(j, lim.get("effort", 300.0))),
        "vel": float(lim.get("vel", 50.0)),
    }
BHL_JOINT_REG_NAMES: Dict[str, str] = {
    "l_ankle_roll":  "leg_left_ankle_roll_joint",
    "r_ankle_roll":  "leg_right_ankle_roll_joint",
    "l_ankle_pitch": "leg_left_ankle_pitch_joint",
    "r_ankle_pitch": "leg_right_ankle_pitch_joint",
    "l_hip_yaw":     "leg_left_hip_yaw_joint",
    "r_hip_yaw":     "leg_right_hip_yaw_joint",
}

BERKELEY_HUMANOID_LITE_ACTUATORS_OVERRIDE: Dict[str, ImplicitActuatorCfg] = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            "leg_(left|right)_hip_(roll|yaw|pitch)_joint$",
            "leg_(left|right)_knee_pitch_joint$",
        ],
        effort_limit_sim=50,
        stiffness={
            "leg_(left|right)_hip_roll_joint$": 60.0,
            "leg_(left|right)_hip_yaw_joint$": 45.0,
            "leg_(left|right)_hip_pitch_joint$": 90.0,
            "leg_(left|right)_knee_pitch_joint$": 60.0,
        },
        damping={
            "leg_(left|right)_hip_roll_joint$": 5.0,
            "leg_(left|right)_hip_yaw_joint$": 5.0,
            "leg_(left|right)_hip_pitch_joint$": 4.0,
            "leg_(left|right)_knee_pitch_joint$": 4.0,
        },
    ),
    "feet": ImplicitActuatorCfg(
        joint_names_expr=[
            "leg_(left|right)_ankle_(pitch|roll)_joint$",
        ],
        effort_limit_sim=40,
        stiffness={
            "leg_(left|right)_ankle_pitch_joint$": 25.0,
            "leg_(left|right)_ankle_roll_joint$": 20.0,
        },
        damping={
            "leg_(left|right)_ankle_pitch_joint$": 3.0,
            "leg_(left|right)_ankle_roll_joint$": 2.5,
        },
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=[
            "arm_(left|right)_shoulder_(pitch|roll|yaw)_joint$",
            "arm_(left|right)_elbow_(pitch|roll)_joint$",
        ],
        effort_limit_sim=40,
        stiffness={
            "arm_(left|right)_shoulder_pitch_joint$": 22.0,
            "arm_(left|right)_shoulder_roll_joint$": 20.0,
            "arm_(left|right)_shoulder_yaw_joint$": 20.0,
            "arm_(left|right)_elbow_pitch_joint$": 25.0,
            "arm_(left|right)_elbow_roll_joint$": 25.0,
        },
        damping={
            "arm_(left|right)_shoulder_pitch_joint$": 8.0,
            "arm_(left|right)_shoulder_roll_joint$": 7.0,
            "arm_(left|right)_shoulder_yaw_joint$": 7.0,
            "arm_(left|right)_elbow_pitch_joint$": 6.0,
            "arm_(left|right)_elbow_roll_joint$": 5.0,
        },
    ),
}
ROBOT_SPECS["berkeley_humanoid_lite"] = RobotSpec(
    name="berkeley_humanoid_lite",
    usd_path=_BHL_USD,
    base_link=BHL_BASE_LINK,
    left_foot_body=BHL_LEFT_FOOT_BODY,
    right_foot_body=BHL_RIGHT_FOOT_BODY,
    action_joints=list(BHL_ACTION_JOINTS_12),
    all_joints=list(BHL_JOINTS_22),
    held_joints=list(BHL_HELD_JOINTS),
    default_joint_angles=dict(BHL_DEFAULT_JOINT_ANGLES),
    joint_limits=dict(BHL_JOINT_LIMITS),
    joint_reg_names=dict(BHL_JOINT_REG_NAMES),
    waist_joint=None,                
    init_pos_xyz=(0.0, 0.0, 0.70),     
    height_z_des=0.67,
    actuators_override=BERKELEY_HUMANOID_LITE_ACTUATORS_OVERRIDE,                 
)

# =============================================================================
# Env-level params
# =============================================================================
@dataclass
class EnvParams:
    num_envs: int = 1024
    env_spacing: float = 2.5
    dt: float = 0.005
    decimation: int = 2
    episode_length_s: float = 20.0
    viewer_eye: Tuple[float, float, float] = (6.0, 0.0, 3.0)
    enable_external_forces_every_iteration: bool = False
    solver_position_iteration_count: int = 8
    solver_velocity_iteration_count: int = 4


# =============================================================================
# Robot selection
# =============================================================================
DEFAULT_ROBOT_KEY = "n1"

# =============================================================================
# Build env cfg from RobotSpec
# =============================================================================
@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg
    observations: ObservationsCfg
    actions: object
    events: EventsCfg
    rewards: RewardsCfg
    terminations: TerminationsCfg
    commands: CommandsCfg
    curriculum: CurriculumCfg

    @classmethod
    def from_robot_key(
        cls,
        robot_key: str,
        *,
        env: Optional[EnvParams] = None,
        terrain: Optional[TerrainParams] = None,
        height_scanner: Optional[HeightScannerParams] = None,
        terminations: Optional[TerminationParams] = None,
    ) -> "LocomotionEnvCfg":
        if robot_key not in ROBOT_SPECS:
            raise KeyError(f"Unknown robot_key={robot_key}. Available: {list(ROBOT_SPECS.keys())}")

        env = EnvParams() if env is None else env
        terrain = TerrainParams() if terrain is None else terrain
        height_scanner = HeightScannerParams() if height_scanner is None else height_scanner
        terminations = TerminationParams() if terminations is None else terminations

        spec = ROBOT_SPECS[robot_key]
        robot_cfg = normalize_spec(spec)

        cfg = cls()
        cfg.scene = SceneCfg.from_robot_cfg(
            robot_cfg,
            num_envs=int(env.num_envs),
            env_spacing=float(env.env_spacing),
            terrain_params=terrain,
            height_params=height_scanner,
            contact_sensor_name="contact_forces",
            height_scanner_name="height_scanner",
            init_pos_xyz=tuple(robot_cfg.get("init_pos_xyz", (0.0, 0.0, 0.75))),
            init_rot_xyzw=tuple(robot_cfg.get("init_rot_xyzw", (0.0, 0.0, 0.0, 1.0))),
        )
        cfg.actions = ActionsCfg.from_robot_cfg(robot_cfg)
        cfg.observations = ObservationsCfg.from_robot_cfg(robot_cfg)
        cfg.events = EventsCfg.from_robot_cfg(robot_cfg)
        cfg.rewards = RewardsCfg.from_robot_cfg(robot_cfg)

        if robot_cfg.get("height_z_des") is not None:
            try:
                cfg.rewards.height.params["z_des"] = float(robot_cfg["height_z_des"])
            except Exception:
                pass

        cfg.terminations = TerminationsCfg.from_robot_cfg(robot_cfg, params=terminations)
        cfg.commands = CommandsCfg.from_robot_cfg(robot_cfg)
        cfg.curriculum = CurriculumCfg.from_robot_cfg(robot_cfg)

        cfg.sim.dt = float(env.dt)
        cfg.decimation = int(env.decimation)
        cfg.episode_length_s = float(env.episode_length_s)
        cfg.sim.render_interval = cfg.decimation
        cfg.viewer.eye = tuple(map(float, env.viewer_eye))
        cfg.sim.physx.enable_external_forces_every_iteration = bool(env.enable_external_forces_every_iteration)
        cfg.sim.physx.solver_position_iteration_count = int(env.solver_position_iteration_count)
        cfg.sim.physx.solver_velocity_iteration_count = int(env.solver_velocity_iteration_count)

        return cfg


@configclass
class HumanoidLocomotionEnvCfg(LocomotionEnvCfg):
    def __post_init__(self) -> None:
        built = LocomotionEnvCfg.from_robot_key(
            DEFAULT_ROBOT_KEY,
            env=EnvParams(),
            terrain=TerrainParams(),
            height_scanner=HeightScannerParams(),
            terminations=TerminationParams(),
        )
        self.scene = built.scene
        self.actions = built.actions
        self.observations = built.observations
        self.events = built.events
        self.rewards = built.rewards
        self.terminations = built.terminations
        self.commands = built.commands
        self.curriculum = built.curriculum
        self.sim = built.sim
        self.viewer = built.viewer
        self.decimation = built.decimation
        self.episode_length_s = built.episode_length_s