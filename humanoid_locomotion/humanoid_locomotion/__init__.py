# humanoid_locomotion/__init__.py
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Humanoid-Locomotion",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "humanoid_locomotion.env_cfg:HumanoidLocomotionEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HumanoidLocomotionCfg",
    },
)
