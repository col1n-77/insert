"""Tensor-only frame math for tactile alignment; quaternions use wxyz."""
import math
import torch
import torch.nn.functional as F


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.linalg.cross(av, bv)), dim=-1)


def quat_inverse(q):
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1) / q.square().sum(-1, keepdim=True).clamp_min(1e-12)


def quat_apply(q, v):
    q = F.normalize(q, dim=-1)
    uv = torch.linalg.cross(q[..., 1:], v)
    return v + 2*(q[..., :1]*uv + torch.linalg.cross(q[..., 1:], uv))


def rotation_vector_quat(v):
    angle = v.norm(dim=-1, keepdim=True)
    # sinc is finite, including exactly zero actions.
    return torch.cat((torch.cos(angle/2), .5*torch.sinc(angle/(2*math.pi))*v), dim=-1)


def world_delta_target(q_wt, delta_rotation_w, max_angle_rad):
    norm = delta_rotation_w.norm(dim=-1, keepdim=True)
    delta = delta_rotation_w * (max_angle_rad / norm.clamp_min(1e-12)).clamp_max(1.)
    return F.normalize(quat_mul(rotation_vector_quat(delta), q_wt), dim=-1)


def estimated_axis_w(q_wt, angle_rad):
    # RotY(theta) @ [0, 0, -1]. This is an oriented axis, not abs(dot).
    axis_t = torch.stack((-angle_rad.sin(), torch.zeros_like(angle_rad), -angle_rad.cos()), dim=-1)
    return F.normalize(quat_apply(q_wt, axis_t), dim=-1)


def local_z_w(q):
    z = torch.zeros_like(q[..., 1:]); z[..., 2] = 1
    return F.normalize(quat_apply(q, z), dim=-1)


def axis_angle(a, b):
    return torch.acos((F.normalize(a, dim=-1)*F.normalize(b, dim=-1)).sum(-1).clamp(-1., 1.))


def gaussian_alignment(alpha_rad, valid, sigma_deg):
    if sigma_deg <= 0:
        raise ValueError("Alignment sigma must be positive")
    return torch.exp(-(alpha_rad / math.radians(sigma_deg)).square()) * valid.float()


def measured_in_hand(q_wt, q_wo):
    """Simulation diagnostics only; never an estimator input or angle reward."""
    q_to = quat_mul(quat_inverse(q_wt), q_wo)
    zero_inv = q_to.new_tensor([0., 0., -1., 0.]).expand_as(q_to)
    delta = quat_mul(q_to, zero_inv)
    ex = delta.new_tensor([1., 0., 0.]).expand(delta.shape[:-1]+(3,))
    ez = delta.new_tensor([0., 0., 1.]).expand_as(ex)
    theta = torch.atan2(quat_apply(delta, ez)[..., 0], quat_apply(delta, ex)[..., 0])
    rv = torch.zeros_like(ex); rv[..., 1] = theta
    residual = quat_mul(quat_inverse(rotation_vector_quat(rv)), delta)
    off_axis = 2*torch.atan2(residual[..., 1:].norm(dim=-1), residual[..., 0].abs())
    return theta, off_axis
