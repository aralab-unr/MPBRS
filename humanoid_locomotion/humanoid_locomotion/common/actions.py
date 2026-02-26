# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Dict, List, Optional, Sequence, Any

import torch

from isaaclab.utils import configclass
from isaaclab.managers import ActionTerm, ActionTermCfg

def _pick_first(robot_cfg: Dict[str, Any], keys: Sequence[str], default=None):
    for k in keys:
        if k in robot_cfg and robot_cfg[k] is not None:
            return robot_cfg[k]
    return default


def _to_list(x) -> List:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    try:
        return list(x)
    except Exception:
        return [x]


def _to_float_dict(x) -> Dict[str, float]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return {str(k): float(v) for k, v in x.items()}
    try:
        return {str(k): float(v) for k, v in dict(x).items()}
    except Exception:
        return {}


# --------------------------------------------------------------------------------------
# Action term
# --------------------------------------------------------------------------------------

class JointPositionHoldActionTerm(ActionTerm):
    """
    Controls a subset of joints (action_joint_names) using position targets from the policy,
    while holding held_joint_names at their default angles every sim step.

    Raw actions are assumed in [-1, 1] and are transformed as:
        processed = raw * scale + offset

    - If use_delta=True: target = default + processed
    - Else:             target = processed (absolute)
    """

    def __init__(self, cfg: "JointPositionHoldActionCfg", env):
        super().__init__(cfg, env)

        self._asset_name: str = cfg.asset_name
        self._action_joints: List[str] = list(cfg.action_joint_names)
        self._held_joints: List[str] = list(cfg.held_joint_names)
        self._default_joint_angles: Dict[str, float] = dict(cfg.default_joint_angles)

        self._scale: float = float(cfg.scale)
        self._offset: float = float(cfg.offset)
        self._use_delta: bool = bool(cfg.use_delta)

        if len(self._action_joints) == 0:
            raise ValueError("[JointPositionHoldActionTerm] action_joint_names is empty!")

        self._robot = env.scene[self._asset_name]
        joint_names = list(self._robot.data.joint_names)
        self._name_to_id = {n: i for i, n in enumerate(joint_names)}
        missing = [j for j in (self._action_joints + self._held_joints) if j not in self._name_to_id]
        if missing:
            raise RuntimeError(
                f"[JointPositionHoldActionTerm] Missing joints in robot: {missing}\n"
                f"Available joints: {joint_names}"
            )

        self._action_ids = torch.tensor(
            [self._name_to_id[j] for j in self._action_joints],
            device=self.device,
            dtype=torch.long,
        )
        self._held_ids = (
            torch.tensor([self._name_to_id[j] for j in self._held_joints], device=self.device, dtype=torch.long)
            if self._held_joints
            else None
        )

        self._q_default_action = torch.tensor(
            [float(self._default_joint_angles.get(j, 0.0)) for j in self._action_joints],
            device=self.device,
            dtype=torch.float32,
        )[None, :].repeat(self.num_envs, 1)

        if self._held_joints:
            self._q_default_held = torch.tensor(
                [float(self._default_joint_angles.get(j, 0.0)) for j in self._held_joints],
                device=self.device,
                dtype=torch.float32,
            )[None, :].repeat(self.num_envs, 1)
        else:
            self._q_default_held = None

        self._raw_actions = torch.zeros(
            (self.num_envs, len(self._action_joints)), device=self.device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._raw_actions[env_ids_t] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def apply_actions(self) -> None:
        if self._use_delta:
            q_action_tgt = self._q_default_action + self._processed_actions
        else:
            q_action_tgt = self._processed_actions

        try:
            self._robot.set_joint_position_target(q_action_tgt, joint_ids=self._action_ids)
        except Exception:
            self._robot.write_joint_position_target_to_sim(q_action_tgt, joint_ids=self._action_ids)

        if self._held_ids is not None and self._q_default_held is not None:
            try:
                self._robot.set_joint_position_target(self._q_default_held, joint_ids=self._held_ids)
            except Exception:
                self._robot.write_joint_position_target_to_sim(self._q_default_held, joint_ids=self._held_ids)


# --------------------------------------------------------------------------------------
# Config for the term (THIS is the key fix: store fields directly, not in params dict)
# --------------------------------------------------------------------------------------

@configclass
class JointPositionHoldActionCfg(ActionTermCfg):
    class_type: type = JointPositionHoldActionTerm
    asset_name: str = "robot"
    action_joint_names: List[str] = MISSING
    held_joint_names: List[str] = field(default_factory=list)
    default_joint_angles: Dict[str, float] = field(default_factory=dict)
    scale: float = 0.5
    offset: float = 0.0
    use_delta: bool = True

# --------------------------------------------------------------------------------------
# ActionsCfg container (used by env_cfg)
# --------------------------------------------------------------------------------------

@configclass
class ActionsCfg:
    """Container for action terms."""
    joint_pos: ActionTermCfg = MISSING

    @classmethod
    def from_robot_cfg(cls, robot_cfg: Dict) -> "ActionsCfg":
        cfg = cls()

        action_joints = _to_list(
            _pick_first(
                robot_cfg,
                keys=("action_joints", "action_joint_names", "action_dof_names"),
                default=[],
            )
        )
        held_joints = _to_list(
            _pick_first(
                robot_cfg,
                keys=("held_joints", "held_joint_names", "arm_joints"),
                default=[],
            )
        )
        default_joint_angles = _to_float_dict(
            _pick_first(
                robot_cfg,
                keys=("default_joint_angles", "default_joint_pos"),
                default={},
            )
        )

        if len(action_joints) == 0:
            raise ValueError(
                "robot_cfg does not contain action joints. "
                "Expected one of: action_joints / action_joint_names / action_dof_names / action_joints_15"
            )

        # Optional tuning keys
        scale = float(robot_cfg.get("action_scale", 0.5))
        offset = float(robot_cfg.get("action_offset", 0.0))
        use_delta = bool(robot_cfg.get("action_use_delta", True))

        cfg.joint_pos = JointPositionHoldActionCfg(
            asset_name="robot",
            action_joint_names=action_joints,
            held_joint_names=held_joints,
            default_joint_angles=default_joint_angles,
            scale=scale,
            offset=offset,
            use_delta=use_delta,
        )
        return cfg
