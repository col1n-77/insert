"""Register six independent no-tactile angle-observation tasks after AppLauncher."""

import gymnasium as gym

from .env_cfg import (
    InsertAnythingAngleCircleHole_I_Cfg,
    InsertAnythingAngleHexagonHole_III_Cfg,
    InsertAnythingAngleHexagonHole_IV_Cfg,
    InsertAnythingAngleLHole_III_Cfg,
    InsertAnythingAngleSquareHole_II_Cfg,
    InsertAnythingAngleTriangleHole_IV_Cfg,
)

_TASKS = (
    ("CircleHole-I", InsertAnythingAngleCircleHole_I_Cfg, "circle"),
    ("SquareHole-II", InsertAnythingAngleSquareHole_II_Cfg, "square"),
    ("LHole-III", InsertAnythingAngleLHole_III_Cfg, "L"),
    ("TriangleHole-IV", InsertAnythingAngleTriangleHole_IV_Cfg, "triangle"),
    ("HexagonHole-III", InsertAnythingAngleHexagonHole_III_Cfg, "hexagon"),
    ("HexagonHole-IV", InsertAnythingAngleHexagonHole_IV_Cfg, "hexagon"),
)

for _suffix, _cfg, _agent in _TASKS:
    gym.register(
        id=f"InsertAnythingAngle-{_suffix}-Direct-v0",
        entry_point="insertanything_angle.env:InsertAnythingAngleEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _cfg,
            "rl_games_cfg_entry_point": (
                "InsertAnything.tasks.direct.insertanything.agents:"
                f"rl_games_ppo_{_agent}_cfg.yaml"
            ),
        },
    )
