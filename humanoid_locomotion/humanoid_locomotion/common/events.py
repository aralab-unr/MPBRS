# humanoid_locomotion/common/events.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Dict, List

import torch

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass


def _get_articulation_joint_names(asset) -> List[str]:
    for attr in ("joint_names", "dof_names"):
        if hasattr(asset, attr):
            return list(getattr(asset, attr))
        if hasattr(asset, "data") and hasattr(asset.data, attr):
            return list(getattr(asset.data, attr))
    raise AttributeError("Could not find joint/dof names on articulation.")


def spawn_random_action_joints(
    env,
    env_ids,
    *,
    action_joints: List[str],
    default_joint_angles: Dict[str, float],
    action_scale_by_joint: Dict[str, float],
    frac: float = 0.1,
):
    if not torch.is_tensor(env_ids):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    asset = env.scene["robot"]
    all_joint_names = _get_articulation_joint_names(asset)

    num_envs = int(env_ids.numel())
    dof_count = len(all_joint_names)

    qpos = torch.zeros((num_envs, dof_count), device=env.device, dtype=torch.float32)
    qvel = torch.zeros((num_envs, dof_count), device=env.device, dtype=torch.float32)

    for i, jname in enumerate(all_joint_names):
        qpos[:, i] = float(default_joint_angles.get(jname, 0.0))

    for i, jname in enumerate(all_joint_names):
        if jname in action_joints:
            s = float(action_scale_by_joint[jname])
            delta = (float(frac) * s) * (2.0 * torch.rand((num_envs,), device=env.device) - 1.0)
            qpos[:, i] = qpos[:, i] + delta

    if hasattr(asset, "write_joint_positions_to_sim"):
        asset.write_joint_positions_to_sim(qpos, env_ids=env_ids)
    elif hasattr(asset, "write_joint_state_to_sim"):
        asset.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)

    if hasattr(asset, "write_joint_velocities_to_sim"):
        asset.write_joint_velocities_to_sim(qvel, env_ids=env_ids)


def reset_scene_safe(env, env_ids=None):
    if hasattr(mdp, "reset_scene_to_default"):
        return mdp.reset_scene_to_default(env, env_ids)
    if hasattr(mdp, "reset_scene"):
        return mdp.reset_scene(env, env_ids)
    return None


@configclass
class EventsCfg:
    reset_scene: EventTerm = MISSING
    random_action_joints: EventTerm = MISSING

    @classmethod
    def from_robot_cfg(cls, robot_cfg: Dict) -> "EventsCfg":
        cfg = cls()

        cfg.reset_scene = EventTerm(func=reset_scene_safe, mode="reset")

        cfg.random_action_joints = EventTerm(
            func=spawn_random_action_joints,
            mode="reset",
            params={
                "action_joints": list(robot_cfg["action_joints"]),
                "default_joint_angles": dict(robot_cfg["default_joint_angles"]),
                "action_scale_by_joint": dict(robot_cfg["action_scale_by_joint"]),
                "frac": 0.10,
            },
        )
        return cfg


