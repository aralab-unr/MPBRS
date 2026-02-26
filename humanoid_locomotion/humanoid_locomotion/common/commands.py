# humanoid_locomotion/common/commands.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Dict

from isaaclab.envs import mdp
from isaaclab.utils import configclass


@configclass
class CommandsCfg:
    base_velocity: mdp.UniformVelocityCommandCfg = MISSING

    @classmethod
    def from_robot_cfg(cls, robot_cfg: Dict) -> "CommandsCfg":
        cfg = cls()

        cfg.base_velocity = mdp.UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(5.0, 7.5),
            rel_standing_envs=0.05,
            rel_heading_envs=0.2,
            heading_command=True,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                # lin_vel_x=(0.0, 2.0), #walk command
                lin_vel_x=(2.5, 4.5), #run command
                # lin_vel_x=(3.5, 5.0), 
                lin_vel_y=(-0.2, 0.2),
                ang_vel_z=(-0.15, 0.15),
                heading=(-0.2, 0.2),
            ),
        )
        return cfg
