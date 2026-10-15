"""Import after AppLauncher to register six isolated tactile tasks."""
import gymnasium as gym
from . import config

TASKS = {
    "CircleHole-I": (config.CircleHoleCfg, "circle"),
    "SquareHole-II": (config.SquareHoleCfg, "square"),
    "LHole-III": (config.LHoleCfg, "L"),
    "TriangleHole-IV": (config.TriangleHoleCfg, "triangle"),
    "HexagonHole-III": (config.HexagonHoleIIICfg, "hexagon"),
    "HexagonHole-IV": (config.HexagonHoleIVCfg, "hexagon"),
}
for suffix, (cfg, agent) in TASKS.items():
    gym.register(
        id=f"InsertAnythingTactile-{suffix}-Direct-v0",
        entry_point="insertanything_tactile.env:InsertAnythingTactileEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": cfg,
            "rl_games_cfg_entry_point": (
                "InsertAnything.tasks.direct.insertanything.agents:"
                f"rl_games_ppo_{agent}_cfg.yaml"
            ),
        },
    )
