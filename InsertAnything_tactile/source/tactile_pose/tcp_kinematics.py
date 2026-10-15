"""Small serial-chain FK/IK extracted from the actual USD joint frames.

Read only: leaves the robot base and asset untouched. Verify FK against PhysX
before using its IK solution for a static scene pose write.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transform(position, quaternion_wxyz):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_quat(np.asarray(quaternion_wxyz)[[1, 2, 3, 0]]).as_matrix()
    t[:3, 3] = position
    return t


class TcpKinematics:
    def __init__(self, edges, world_base):
        self.edges = edges
        self.world_base = world_base

    @classmethod
    def from_usd(cls, stage, robot_path, joint_names, world_base):
        from pxr import Usd, UsdPhysics
        def local(joint, side):
            p = getattr(joint, f"GetLocalPos{side}Attr")().Get()
            q = getattr(joint, f"GetLocalRot{side}Attr")().Get()
            return transform(np.array(p), np.r_[q.GetReal(), np.array(q.GetImaginary())])
        graph = {}
        root = stage.GetPrimAtPath(robot_path)
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint): continue
            joint = UsdPhysics.Joint(prim)
            a, b = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if len(a) != 1 or len(b) != 1: continue
            axis, index = None, None
            if prim.IsA(UsdPhysics.RevoluteJoint):
                if prim.GetName() not in joint_names: continue
                index = joint_names.index(prim.GetName())
                axis = np.eye(3)["XYZ".index(UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get())]
            elif not prim.IsA(UsdPhysics.FixedJoint):
                continue
            edge = (local(joint, 0), np.linalg.inv(local(joint, 1)), axis, index)
            graph.setdefault(str(a[0]), []).append((str(b[0]), edge, False))
            graph.setdefault(str(b[0]), []).append((str(a[0]), edge, True))
        start = robot_path + "/panda_link0"
        goal = robot_path + "/panda_fingertip_centered"
        pending = [(start, [])]
        visited = set()
        while pending:
            node, path = pending.pop()
            if node == goal: return cls(path, world_base)
            if node in visited: continue
            visited.add(node)
            for other, edge, reverse in graph.get(node, []):
                if other not in visited: pending.append((other, path + [(edge, reverse)]))
        raise ValueError("No USD joint chain from panda_link0 to actual TCP")

    def forward(self, joints):
        result = self.world_base.copy()
        for (a, b, axis, index), reverse in self.edges:
            motion = np.eye(4)
            if index is not None:
                motion[:3, :3] = Rotation.from_rotvec(axis * joints[index]).as_matrix()
            relative = a @ motion @ b
            result = result @ (np.linalg.inv(relative) if reverse else relative)
        return result

    def solve(self, target_position, target_rotation, initial_joints, limits):
        # SciPy differentiates at float64-scale steps. A float32 copy here
        # would round away those perturbations and produce a zero Jacobian.
        reference = np.asarray(initial_joints, dtype=np.float64).copy()
        def residual(arm):
            joints = reference.copy()
            joints[:7] = arm
            actual = self.forward(joints)
            # 1 cm translation and 1 radian rotation have comparable residuals.
            return np.r_[100. * (actual[:3, 3] - target_position),
                          Rotation.from_matrix(actual[:3, :3].T @ target_rotation).as_rotvec()]
        lower, upper = limits[:7, 0] + 1e-5, limits[:7, 1] - 1e-5
        result = least_squares(residual, np.clip(reference[:7], lower, upper), bounds=(lower, upper),
                               method="dogbox", ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=100)
        reference[:7] = result.x
        actual = self.forward(reference)
        pe = float(np.linalg.norm(actual[:3, 3] - target_position))
        re = float(np.rad2deg(Rotation.from_matrix(actual[:3, :3].T @ target_rotation).magnitude()))
        if pe > 2e-6 or re > .001:
            raise RuntimeError(f"IK did not converge within tolerance: position={pe} m, rotation={re} deg")
        return reference, {"fk_position_error_m": pe, "fk_rotation_error_deg": re, "nfev": result.nfev}
