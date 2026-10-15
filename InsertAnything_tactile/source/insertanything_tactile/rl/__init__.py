"""Explicit opt-in registration of the six tactile-angle RL tasks."""
import gymnasium as gym
from . import config

TASKS = {
    "CircleHole-I": config.CircleHoleCfg, "SquareHole-II": config.SquareHoleCfg,
    "LHole-III": config.LHoleCfg, "TriangleHole-IV": config.TriangleHoleCfg,
    "HexagonHole-III": config.HexagonHoleIIICfg, "HexagonHole-IV": config.HexagonHoleIVCfg,
}
for suffix, cfg in TASKS.items():
    gym.register(id=f"InsertAnythingTactile6D-{suffix}-Direct-v0",
                 entry_point="insertanything_tactile.rl.env:Tactile6DEnv", disable_env_checker=True,
                 kwargs={"env_cfg_entry_point": cfg,
                         "rl_games_cfg_entry_point": "insertanything_tactile.rl.agent:agent_cfg"})
