# humanoid_locomotion/common/scene.py
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

# -----------------------------------------------------------------------------
# Empty spawn cfg compatibility (IsaacLab versions differ)
# -----------------------------------------------------------------------------
def empty_spawn_cfg():
    if hasattr(sim_utils, "EmptySpawnCfg"):
        return sim_utils.EmptySpawnCfg()
    if hasattr(sim_utils, "EmptyCfg"):
        return sim_utils.EmptyCfg()
    if hasattr(sim_utils, "XformCfg"):
        return sim_utils.XformCfg()
    return None


def make_disabled_asset(prim_path: str) -> AssetBaseCfg:
    spawn = empty_spawn_cfg()
    if spawn is None:
        if hasattr(sim_utils, "DomeLightCfg"):
            return AssetBaseCfg(prim_path=prim_path, spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0), intensity=0.0))
        return AssetBaseCfg(prim_path=prim_path, spawn=sim_utils.GroundPlaneCfg())
    return AssetBaseCfg(prim_path=prim_path, spawn=spawn)


# -----------------------------------------------------------------------------
# RayCaster + GridPatternCfg import 
# -----------------------------------------------------------------------------
try:
    from isaaclab.sensors import RayCasterCfg
except Exception:
    RayCasterCfg = None  

patterns_mod = None
GridPatternCfg = None
if RayCasterCfg is not None:
    for _cand in (
        "isaaclab.sensors.patterns",
        "isaaclab.sensors.ray_caster.patterns",
        "isaaclab.sensors.raycasters.patterns",
    ):
        try:
            patterns_mod = __import__(_cand, fromlist=["GridPatternCfg"])
            GridPatternCfg = getattr(patterns_mod, "GridPatternCfg", None)
            if GridPatternCfg is not None:
                break
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Small serializable params
# -----------------------------------------------------------------------------
@dataclass
class TerrainParams:
    """Normal ground plane params."""
    size_xy: Tuple[float, float] = (100.0, 100.0)
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0


@dataclass
class HeightScannerParams:
    """Ray-based height scanner params.

    Set enable=False to completely disable it.
    """
    enable: bool = True
    resolution: float = 0.15
    size_xy: Tuple[float, float] = (0.6, 0.6)
    debug_vis: bool = False


# -----------------------------------------------------------------------------
# SceneCfg 
# -----------------------------------------------------------------------------
@configclass
class SceneCfg(InteractiveSceneCfg):
    ground: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=""),
    )

    contact_forces: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=8,
        track_air_time=True,
        debug_vis=False,
    )

    height_scanner: object = (
        RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="yaw",
            pattern_cfg=GridPatternCfg(resolution=0.15, size=[0.6, 0.6]) if GridPatternCfg is not None else None,
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )
        if (RayCasterCfg is not None and GridPatternCfg is not None)
        else make_disabled_asset("/World/HeightScannerDisabled")
    )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    @classmethod
    def from_robot_cfg(
        cls,
        robot_cfg: Dict[str, Any],
        *,
        num_envs: int,
        env_spacing: float,
        terrain_params: Optional[TerrainParams] = None,
        height_params: Optional[HeightScannerParams] = None,
        contact_sensor_name: str = "contact_forces",
        height_scanner_name: str = "height_scanner",
        init_pos_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.70),
        init_rot_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ) -> "SceneCfg":
        t = terrain_params or TerrainParams()
        h = height_params or HeightScannerParams()

        scene = cls(num_envs=int(num_envs), env_spacing=float(env_spacing))
        scene.ground = AssetBaseCfg(
            prim_path="/World/ground",
            spawn=sim_utils.GroundPlaneCfg(
                size=(float(t.size_xy[0]), float(t.size_xy[1])),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=float(t.static_friction),
                    dynamic_friction=float(t.dynamic_friction),
                    restitution=float(t.restitution),
                ),
            ),
        )

        scene.robot = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(robot_cfg["usd_path"]),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    max_depenetration_velocity=5.0,
                    enable_gyroscopic_forces=True,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=4,
                    sleep_threshold=0.0,
                    stabilization_threshold=0.001,
                ),
                copy_from_source=False,
                activate_contact_sensors=True,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=tuple(map(float, init_pos_xyz)),
                rot=tuple(map(float, init_rot_xyzw)),
                joint_pos={k: float(v) for k, v in robot_cfg["default_joint_angles"].items()},
                joint_vel={k: 0.0 for k in robot_cfg["default_joint_angles"].keys()},
            ),
            actuators=dict(robot_cfg["actuators"]),
        )

        setattr(
            scene,
            contact_sensor_name,
            ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*",
                update_period=0.0,
                history_length=8,
                track_air_time=True,
                debug_vis=False,
            ),
        )

        if h.enable and RayCasterCfg is not None and GridPatternCfg is not None:
            base_link = str(robot_cfg.get("base_link", "base_link"))
            setattr(
                scene,
                height_scanner_name,
                RayCasterCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{base_link}",
                    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                    ray_alignment="yaw",
                    pattern_cfg=GridPatternCfg(
                        resolution=float(h.resolution),
                        size=[float(h.size_xy[0]), float(h.size_xy[1])],
                    ),
                    mesh_prim_paths=["/World/ground"],
                    debug_vis=bool(h.debug_vis),
                ),
            )
        else:
            setattr(scene, height_scanner_name, make_disabled_asset("/World/HeightScannerDisabled"))
        scene.dome_light = AssetBaseCfg(
            prim_path="/World/DomeLight",
            spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
        )

        return scene
