"""Local USD override; all referenced upstream assets remain read-only."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_ASSET = PROJECT_ROOT / "assets" / "franka_gsmini_physx.usda"


def prepare_robot_asset() -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdUtils
    from tacex_assets import TACEX_ASSETS_DATA_DIR

    version = "2"
    if ROBOT_ASSET.exists() and Sdf.Layer.FindOrOpen(str(ROBOT_ASSET)).customLayerData.get("tactile_version") == version:
        return ROBOT_ASSET
    data = Path(TACEX_ASSETS_DATA_DIR)
    robot = data / "Robots/Franka/GelSight_Mini/Gripper/physx_rigid_gelpads.usd"
    gelpad = data / "Sensors/GelSight_Mini/Gelpad_mid_res.usd"
    for asset in (robot, gelpad):
        if not asset.is_file():
            raise FileNotFoundError(asset)
    ROBOT_ASSET.parent.mkdir(parents=True, exist_ok=True)
    # Copy the small composition layer, keeping meshes/materials as absolute
    # references. This also removes the stale low-res payload at its source.
    original = Sdf.Layer.FindOrOpen(str(robot))
    layer = Sdf.Layer.CreateAnonymous("franka_gsmini_physx.usda")
    layer.TransferContent(original)
    UsdUtils.ModifyAssetPaths(layer, lambda path: str(gelpad) if path.endswith("Gelpad_low_res.usd")
                             else Sdf.ComputeAssetPathRelativeToLayer(original, path))
    layer.customLayerData = {"tactile_version": version}
    stage = Usd.Stage.Open(layer)
    for side in ("left", "right"):
        prim = stage.GetPrimAtPath(f"/panda/gelpad_{side}")
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.005)
    tip = stage.GetPrimAtPath("/panda/panda_fingertip_centered")
    mass = UsdPhysics.MassAPI.Apply(tip)
    mass.CreateMassAttr(1e-6)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9))
    stage.GetRootLayer().Export(str(ROBOT_ASSET))
    return ROBOT_ASSET
