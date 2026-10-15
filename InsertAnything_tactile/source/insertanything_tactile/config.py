"""Independent task configurations using UniVTAC/TacEx GelSight Mini optics."""
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from InsertAnything.tasks.direct.insertanything.insertanything_env_cfg import InsertAnythingEnvCfg
from InsertAnything.tasks.direct.insertanything.insertanything_tasks_cfg import (
    InsertAnythingCircleHole_I, InsertAnythingSquareHole_II, InsertAnythingLHole_III,
    InsertAnythingTriangleHole_IV, InsertAnythingHexagonHole_III, InsertAnythingHexagonHole_IV,
)
from .assets import ROBOT_ASSET


@configclass
class GelSightMiniCfg:
    sides: tuple[str, ...] = ("left", "right")
    resolution: tuple[int, int] = (320, 240)  # width, height
    debug_vis: bool = False
    clipping_range: tuple[float, float] = (0.024, 0.034)
    calibration_mode: str = "geometry_per_side"  # legacy_shared reproduces historical 28 mm optics


@configclass
class InsertAnythingTactileEnvCfg(InsertAnythingEnvCfg):
    tactile: GelSightMiniCfg = GelSightMiniCfg()
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, clone_in_fabric=False)
    rerender_on_reset: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = str(ROBOT_ASSET)
        self.scene.clone_in_fabric = False
        self.sim.render_interval = self.decimation
        self.task.show_visual_markers = False
        # Upstream's CSV logger writes into the original task directory.
        # Captures in this extension have their own output directory.
        self.task.contact_force["log_contact_force"] = False
        self.viewer.eye = (1.05, 0.6, 0.8)
        self.viewer.lookat = (0.55, 0.0, 0.18)


@configclass
class CircleHoleCfg(InsertAnythingTactileEnvCfg):
    task = InsertAnythingCircleHole_I()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class SquareHoleCfg(InsertAnythingTactileEnvCfg):
    task = InsertAnythingSquareHole_II()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class LHoleCfg(InsertAnythingTactileEnvCfg):
    task = InsertAnythingLHole_III()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class TriangleHoleCfg(InsertAnythingTactileEnvCfg):
    task = InsertAnythingTriangleHole_IV()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class HexagonHoleIIICfg(InsertAnythingTactileEnvCfg):
    task = InsertAnythingHexagonHole_III()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class HexagonHoleIVCfg(InsertAnythingTactileEnvCfg):
    task = InsertAnythingHexagonHole_IV()
    task_name = task.name
    episode_length_s = task.duration_s
