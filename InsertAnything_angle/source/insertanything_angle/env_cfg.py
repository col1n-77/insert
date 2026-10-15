"""A separate configuration family retaining the original task definitions."""

from isaaclab.utils import configclass

from InsertAnything.tasks.direct.insertanything.insertanything_env_cfg import (
    CtrlCfg,
    InsertAnythingEnvCfg,
)
from InsertAnything.tasks.direct.insertanything.insertanything_tasks_cfg import (
    InsertAnythingCircleHole_I,
    InsertAnythingHexagonHole_III,
    InsertAnythingHexagonHole_IV,
    InsertAnythingLHole_III,
    InsertAnythingSquareHole_II,
    InsertAnythingTask,
    InsertAnythingTriangleHole_IV,
)


@configclass
class AngleObservationCfg:
    noise_std_deg: float = 0.0
    scale_deg: float = 15.0
    noise_seed_offset: int = 161803
    critic_true_angle: bool = True


@configclass
class AxisAlignmentCfg:
    # Zero disables ONLY the added reward, retaining the same observations.
    reward_weight: float = 1.0
    sigma_deg: float = 5.0
    # r_axis = exp(-(alpha/sigma)^2)-1. False reproduces historical positive rewards.
    zero_at_alignment: bool = True
    strict_axis_deg: float = 3.0
    strict_min_depth_mm: float = -2.0


@configclass
class DepthAxisGateCfg:
    # Opt-in: historical models retain their original additive reward.
    enabled: bool = False
    tolerance_deg: float = 3.0
    sigma_deg: float = 5.0
    min_gate: float = 0.1


@configclass
class InsertAnythingAngleEnvCfg(InsertAnythingEnvCfg):
    initial_in_hand_tilt_deg: float = 10.0
    ctrl: CtrlCfg = CtrlCfg(lock_roll_pitch=False)
    angle_obs: AngleObservationCfg = AngleObservationCfg()
    axis_alignment: AxisAlignmentCfg = AxisAlignmentCfg()
    depth_axis_gate: DepthAxisGateCfg = DepthAxisGateCfg()


@configclass
class InsertAnythingAngleCircleHole_I_Cfg(InsertAnythingAngleEnvCfg):
    task: InsertAnythingTask = InsertAnythingCircleHole_I()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingAngleSquareHole_II_Cfg(InsertAnythingAngleEnvCfg):
    task: InsertAnythingTask = InsertAnythingSquareHole_II()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingAngleLHole_III_Cfg(InsertAnythingAngleEnvCfg):
    task: InsertAnythingTask = InsertAnythingLHole_III()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingAngleTriangleHole_IV_Cfg(InsertAnythingAngleEnvCfg):
    task: InsertAnythingTask = InsertAnythingTriangleHole_IV()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingAngleHexagonHole_III_Cfg(InsertAnythingAngleEnvCfg):
    task: InsertAnythingTask = InsertAnythingHexagonHole_III()
    task_name: str = task.name
    episode_length_s: float = task.duration_s


@configclass
class InsertAnythingAngleHexagonHole_IV_Cfg(InsertAnythingAngleEnvCfg):
    task: InsertAnythingTask = InsertAnythingHexagonHole_IV()
    task_name: str = task.name
    episode_length_s: float = task.duration_s
