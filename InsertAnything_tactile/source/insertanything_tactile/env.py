"""GelSight Mini side channels without changing the pretrained policy interface."""
from tacex import GelSightSensor
from tacex_assets.sensors.gelsight_mini import GELSIGHT_MINI_TAXIM_CFG
from InsertAnything.tasks.direct.insertanything.insertanything_env import InsertAnythingEnv
from .assets import prepare_robot_asset
from tactile_calibration import geometry_zero_mm, height_from_depth_m, make_calibration


class CalibratedGelSightSensor(GelSightSensor):
    def __init__(self, cfg, zero_mm):
        self.optical_zero_mm = zero_mm
        super().__init__(cfg)

    def _get_height_map(self):
        # TacEx calls this before indentation and RGB rendering. Both outputs
        # therefore use the same calibrated reference, including contact flags.
        height = height_from_depth_m(self.camera.data.output["depth"][..., 0],
                                    self.optical_zero_mm, self.cfg.gelpad_dimensions.height * 1000,
                                    self.cfg.sensor_camera_cfg.clipping_range)
        self._data.output["height_map"].copy_(height)
        return self._data.output["height_map"]


class InsertAnythingTactileEnv(InsertAnythingEnv):
    def __init__(self, cfg, **kwargs):
        if not cfg.tactile.sides or any(side not in ("left", "right") for side in cfg.tactile.sides):
            raise ValueError("tactile.sides must contain left and/or right")
        if len(set(cfg.tactile.sides)) != len(cfg.tactile.sides):
            raise ValueError("Duplicate tactile sides")
        prepare_robot_asset()
        super().__init__(cfg, **kwargs)

    def _setup_scene(self):
        super()._setup_scene()
        self.tactile_sensors = {}
        mode = self.cfg.tactile.calibration_mode
        if mode not in ("geometry_per_side", "legacy_shared"):
            raise ValueError(f"Unknown tactile calibration mode: {mode}")
        gel_height_mm = GELSIGHT_MINI_TAXIM_CFG.gelpad_dimensions.height * 1000
        zero = {side: (geometry_zero_mm(self.sim.stage, "/World/envs/env_0/Robot", side)
                       if mode == "geometry_per_side" else
                       self.cfg.tactile.clipping_range[0] * 1000 + gel_height_mm)
                for side in self.cfg.tactile.sides}
        self.tactile_calibration = make_calibration(zero, self.cfg.tactile.clipping_range,
                                                   gel_height_mm, self.cfg.tactile.resolution, mode)
        for side in self.cfg.tactile.sides:
            sensor_cfg = GELSIGHT_MINI_TAXIM_CFG.copy()
            sensor_cfg.prim_path = f"/World/envs/env_.*/Robot/gelsight_mini_case_{side}"
            sensor_cfg.device = self.device
            sensor_cfg.debug_vis = self.cfg.tactile.debug_vis
            # Lazy data access samples once per caller/policy step. A zero
            # period also makes explicit refresh work without advancing time;
            # Isaac Lab's force_recompute alone does not mark a sensor stale.
            sensor_cfg.update_period = 0.0
            sensor_cfg.sensor_camera_cfg.resolution = self.cfg.tactile.resolution
            sensor_cfg.sensor_camera_cfg.update_period = sensor_cfg.update_period
            sensor_cfg.sensor_camera_cfg.clipping_range = self.cfg.tactile.clipping_range
            sensor_cfg.sensor_camera_cfg.update_latest_camera_pose = True
            sensor_cfg.data_types = ["tactile_rgb", "height_map"]
            sensor_cfg.optical_sim_cfg.tactile_img_res = self.cfg.tactile.resolution
            sensor_cfg.optical_sim_cfg.device = self.device
            sensor_cfg.optical_sim_cfg.with_shadow = False
            sensor_cfg.marker_motion_sim_cfg = None
            sensor = (CalibratedGelSightSensor(sensor_cfg, zero[side]) if mode == "geometry_per_side"
                      else GelSightSensor(sensor_cfg))
            self.tactile_sensors[side] = sensor
            self.scene.sensors[f"gsmini_{side}"] = sensor

    def get_tactile_observations(self, *, clone=False):
        """GPU tensors: RGB uint8 NHWC; geometric press depth float32 NHW in mm.

        PhysX rigid contact + depth-to-Taxim optics. No UIPC/FEM deformation,
        marker shear, or calibrated force estimate is represented here.
        Images are separate from the original policy/critic observations.
        """
        result = {}
        for side, sensor in self.tactile_sensors.items():
            data = sensor.data.output
            result[side] = {
                "rgb": data["tactile_rgb"],
                "press_depth_mm": (-data["height_map"]).clamp_min(0),
                "camera_depth_mm": sensor.camera.data.output["depth"][..., 0].nan_to_num(
                    nan=self.cfg.tactile.clipping_range[1], posinf=self.cfg.tactile.clipping_range[1],
                    neginf=self.cfg.tactile.clipping_range[1]
                ) * 1000.0,
            }
            if clone:
                result[side] = {key: value.clone() for key, value in result[side].items()}
        return result

    def refresh_tactile(self, render=True):
        """Refresh after a reset or a diagnostic pose write."""
        if render:
            self.sim.forward()
            for _ in range(2):
                self.sim.render()
        for sensor in self.tactile_sensors.values():
            sensor.update(0.0, force_recompute=True)
        return self.get_tactile_observations()
