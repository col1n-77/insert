"""Separate six-DOF RL task configs; original tactile/upstream tasks are unchanged."""
from pathlib import Path
from isaaclab.utils import configclass
from insertanything_tactile.config import InsertAnythingTactileEnvCfg
from InsertAnything.tasks.direct.insertanything.insertanything_tasks_cfg import (
    InsertAnythingCircleHole_I, InsertAnythingSquareHole_II, InsertAnythingLHole_III,
    InsertAnythingTriangleHole_IV, InsertAnythingHexagonHole_III, InsertAnythingHexagonHole_IV,
)

ROOT = Path(__file__).resolve().parents[3]


@configclass
class TactileAlignmentCfg:
    checkpoint: str = str(ROOT / "outputs/dynamic_angle_rl_model_20261012/best.pt")
    weight: float = 1.0
    sigma_deg: float = 5.0
    insertion_gate_sigma_deg: float = 10.0
    success_angle_deg: float = 3.0
    contact_loss_penalty: float = 0.25
    min_contact_pixels: int = 64
    min_contact_depth_mm: float = 0.02
    initial_tilt_deg: float = 10.0
    max_rotation_step_rad: float = 0.035


@configclass
class Tactile6DEnvCfg(InsertAnythingTactileEnvCfg):
    tactile_alignment: TactileAlignmentCfg = TactileAlignmentCfg()

    def __post_init__(self):
        super().__post_init__()
        # A keypoint reward would introduce a second, ground-truth tilt signal.
        self.task.use_decoupled_reward = True
        self.task.hand_init_orn_range = [0., 0., self.task.hand_init_orn_range[2]]
        self.initial_in_hand_tilt_deg = self.tactile_alignment.initial_tilt_deg
        self.ctrl.lock_roll_pitch = False
        self.task.held_asset_pos_range = [0.002, 0.002, 0.0]
        self.ctrl.pos_action_threshold = [0.003, 0.003, 0.003]
        self.ctrl.rot_action_threshold = [0.035, 0.035, 0.035]


@configclass
class CircleHoleCfg(Tactile6DEnvCfg):
    task = InsertAnythingCircleHole_I()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class SquareHoleCfg(Tactile6DEnvCfg):
    task = InsertAnythingSquareHole_II()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class LHoleCfg(Tactile6DEnvCfg):
    task = InsertAnythingLHole_III()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class TriangleHoleCfg(Tactile6DEnvCfg):
    task = InsertAnythingTriangleHole_IV()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class HexagonHoleIIICfg(Tactile6DEnvCfg):
    task = InsertAnythingHexagonHole_III()
    task_name = task.name
    episode_length_s = task.duration_s


@configclass
class HexagonHoleIVCfg(Tactile6DEnvCfg):
    task = InsertAnythingHexagonHole_IV()
    task_name = task.name
    episode_length_s = task.duration_s
