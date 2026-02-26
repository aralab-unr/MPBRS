# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from dataclasses import dataclass
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# If your obs_projected_gravity_b lives in common.observations, import it:
from humanoid_locomotion.common.observations import obs_projected_gravity_b


# -----------------------------------------------------------------------------
# Params (optional knobs)
# -----------------------------------------------------------------------------
@dataclass
class TerminationParams:
    min_base_height_dist: float = 0.35
    bad_orientation_limit_angle: float = math.pi * 0.65


# -----------------------------------------------------------------------------
# Height helper (keep it HERE as you requested)
# -----------------------------------------------------------------------------
def _base_height_from_rays(env: ManagerBasedRLEnv, ray_sensor_name: str = "height_scanner") -> torch.Tensor:
    """Returns (N,1) distance from base_link to ground using RayCaster hits."""
    ray = env.scene.sensors[ray_sensor_name]
    data = ray.data

    # IsaacLab variants
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
# Termination functions
# -----------------------------------------------------------------------------
def done_base_height_distance_below(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, min_dist: float
) -> torch.Tensor:
    h = _base_height_from_rays(env, ray_sensor_name="height_scanner").squeeze(-1)
    return h < float(min_dist)


def done_bad_orientation_safe(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, limit_angle: float
) -> torch.Tensor:
    # Prefer built-in if present
    if hasattr(mdp, "bad_orientation"):
        return mdp.bad_orientation(env, asset_cfg=asset_cfg, limit_angle=limit_angle)

    # Fallback: use projected gravity (body frame), expecting z ≈ -1 when upright
    g_b = obs_projected_gravity_b(env, asset_cfg)
    err = torch.abs(g_b[:, 2] + 1.0)
    return err > 0.8


def done_self_collision(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Terminates when robot collides with itself.
    Uses IsaacLab built-in if available.
    """
    if hasattr(mdp, "self_collision"):
        return mdp.self_collision(env, asset_cfg=asset_cfg)

    # Fallback (rarely needed depending on IsaacLab version)
    robot = env.scene[asset_cfg.name]
    if hasattr(robot.data, "self_collision"):
        return robot.data.self_collision

    # If self-collision not supported, return no termination
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# -----------------------------------------------------------------------------
# Configclass
# -----------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    time_out: DoneTerm = DoneTerm(func=mdp.time_out, time_out=True)

    fallen: DoneTerm = DoneTerm(
        func=done_base_height_distance_below,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_dist": 0.35,
        },
    )

    bad_orientation: DoneTerm = DoneTerm(
        func=done_bad_orientation_safe,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "limit_angle": math.pi * 0.65,
        },
    )

    self_collision: DoneTerm = DoneTerm(
        func=done_self_collision,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    @classmethod
    def from_robot_cfg(
        cls,
        robot_cfg: dict,
        *,
        params: TerminationParams | None = None,
    ) -> "TerminationsCfg":
        """Optional builder if your env_cfg calls TerminationsCfg.from_robot_cfg(...)."""
        p = TerminationParams() if params is None else params

        return cls(
            fallen=DoneTerm(
                func=done_base_height_distance_below,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "min_dist": float(p.min_base_height_dist),
                },
            ),
            bad_orientation=DoneTerm(
                func=done_bad_orientation_safe,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "limit_angle": float(p.bad_orientation_limit_angle),
                },
            ),
            self_collision=DoneTerm(
                func=done_self_collision,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        )
