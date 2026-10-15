"""Active, right-handed rotations with column vectors and wxyz quaternions."""
import torch
import torch.nn.functional as F


def quaternion_to_matrix(q):
    if torch.any(torch.linalg.vector_norm(q, dim=-1) < 1e-8):
        raise ValueError("A zero quaternion is not a rotation")
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
        2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
        2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(*q.shape[:-1], 3, 3)


def relative_pose(p_wg, q_wg, p_wo, q_wo):
    """Return p_GO, R_GO: transform object-coordinate vectors into gripper axes."""
    r_gw = quaternion_to_matrix(q_wg).transpose(-1, -2)
    return ((r_gw @ (p_wo - p_wg).unsqueeze(-1)).squeeze(-1),
            r_gw @ quaternion_to_matrix(q_wo))


def matrix_to_rotation_6d(matrix):
    """Concatenate the first two COLUMNS, not flattened first two rows."""
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(value):
    """Gram-Schmidt SO(3) decoder, with valid fallbacks for degenerate inputs."""
    a, b = value[..., :3], value[..., 3:]
    e1 = torch.zeros_like(a)
    e1[..., 0] = 1
    x = torch.where(torch.linalg.vector_norm(a, dim=-1, keepdim=True) > 1e-6,
                    F.normalize(a, dim=-1, eps=1e-6), e1)
    b = b - (x * b).sum(-1, keepdim=True) * x
    fallback = F.one_hot(x.abs().argmin(-1), 3).to(value.dtype)
    fallback = F.normalize(fallback - (fallback * x).sum(-1, keepdim=True) * x, dim=-1)
    y = torch.where(torch.linalg.vector_norm(b, dim=-1, keepdim=True) > 1e-6,
                    F.normalize(b, dim=-1, eps=1e-6), fallback)
    z = torch.linalg.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def geodesic_angle(prediction, target):
    """SO(3) angle in radians, stable at identity; atan2 also handles pi."""
    error = prediction.transpose(-1, -2) @ target
    skew = torch.stack((error[..., 2, 1] - error[..., 1, 2],
                        error[..., 0, 2] - error[..., 2, 0],
                        error[..., 1, 0] - error[..., 0, 1]), dim=-1)
    sine = 0.5 * torch.linalg.vector_norm(skew, dim=-1)
    cosine = (error.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5
    return torch.atan2(sine, cosine.clamp(-1, 1))


def mean_rotation(rotations):
    """Chordal mean, projected to SO(3). Fit on training labels only."""
    u, _, vh = torch.linalg.svd(rotations.mean(0))
    correction = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    correction[-1, -1] = torch.linalg.det(u @ vh)
    return u @ correction @ vh
