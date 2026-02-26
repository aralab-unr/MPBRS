# humanoid_locomotion/common/curriculum.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class CurriculumCfg:
    """No curriculum yet (kept for compatibility)."""

    @classmethod
    def from_robot_cfg(cls, robot_cfg: dict) -> "CurriculumCfg":
        return cls()
