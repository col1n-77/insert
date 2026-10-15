"""Pure PyTorch angle geometry and isolated observation-noise state.

Quaternions are scalar-first (w, x, y, z). Object and hole insertion axes use
their local +Z directions. With the nominal down-facing TCP, object +Z is TCP
-Z; a positive in-hand angle rotates that vector about TCP +Y.
"""

from __future__ import annotations

import math

import torch


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    uv = torch.linalg.cross(q[..., 1:], v, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + torch.linalg.cross(q[..., 1:], uv, dim=-1))


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((q[..., :1], -q[..., 1:]), dim=-1)
    return quat_rotate(conjugate, v)


def angle_geometry(
    object_quat_w: torch.Tensor, tcp_quat_w: torch.Tensor, hole_quat_w: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Measure actual orientations, never the angle sampled at episode reset."""
    local_z = torch.zeros_like(object_quat_w[..., 1:])
    local_z[..., 2] = 1.0
    object_axis_w = quat_rotate(object_quat_w, local_z)
    hole_axis_w = quat_rotate(hole_quat_w, local_z)
    object_axis_tcp = quat_rotate_inverse(tcp_quat_w, object_axis_w)
    angle = torch.atan2(-object_axis_tcp[..., 0], -object_axis_tcp[..., 2])
    # atan2(cross norm, dot) is stable near zero and preserves directed axes.
    axis_error = torch.atan2(
        torch.linalg.vector_norm(torch.linalg.cross(object_axis_w, hole_axis_w, dim=-1), dim=-1),
        (object_axis_w * hole_axis_w).sum(dim=-1).clamp(-1.0, 1.0),
    )
    # Residual from the assumed TCP XZ plane; zero means the 1-DoF model fits.
    off_axis = torch.asin(object_axis_tcp[..., 1].abs().clamp(0.0, 1.0))
    return {
        "true_in_hand_angle_rad": angle,
        "axis_error_rad": axis_error,
        "off_axis_residual_rad": off_axis,
        "object_axis_w": object_axis_w,
        "hole_axis_w": hole_axis_w,
        "object_axis_tcp": object_axis_tcp,
    }


def gaussian_axis_reward(
    axis_error_rad: torch.Tensor, weight: float, sigma_deg: float, *, zero_at_alignment: bool = True
) -> torch.Tensor:
    """Weighted axis penalty, in [-weight, 0], with exactly zero at alignment.

    expm1 evaluates exp(x)-1 accurately near zero. The uncentered branch exists
    only to reproduce older runs whose saved configuration predates this flag.
    """
    exponent = -torch.square(axis_error_rad / math.radians(sigma_deg))
    shape = torch.expm1(exponent) if zero_at_alignment else torch.exp(exponent)
    return float(weight) * shape


def axis_depth_gate(
    axis_error_rad: torch.Tensor, tolerance_deg: float, sigma_deg: float, min_gate: float
) -> torch.Tensor:
    """Keep full depth reward inside tolerance, then smoothly approach a floor.

    The input is the true directed object/hole axis angle, not the in-hand
    angle or TCP tilt. Parameters use degrees; tensors use radians.
    """
    if not math.isfinite(tolerance_deg) or not 0 <= tolerance_deg < 180:
        raise ValueError("Depth gate tolerance must be finite and in [0, 180)")
    if not math.isfinite(sigma_deg) or sigma_deg <= 0:
        raise ValueError("Depth gate sigma must be finite and positive")
    if not math.isfinite(min_gate) or not 0 <= min_gate <= 1:
        raise ValueError("Depth gate floor must be finite and in [0, 1]")
    excess = (axis_error_rad - math.radians(tolerance_deg)).clamp_min(0.0)
    # This form yields exactly 1 inside the tolerance, including float32.
    return 1.0 + (1.0 - min_gate) * torch.expm1(-torch.square(excess / math.radians(sigma_deg)))


class AngleNoiseCache:
    """One independent Gaussian value per environment and observation boundary.

    A new control step invalidates every environment. A reset invalidates only
    the reset environments. Rewards, diagnostics, and repeated observation
    reads share the cached sample. The actual geometry is always re-evaluated.
    This generator never consumes the global PyTorch random stream used by the
    original task's resets, dynamics randomization, and observation noise.
    """

    def __init__(self, num_envs: int, device: str | torch.device, std_deg: float, seed: int):
        self.device = torch.device(device)
        self.std_rad = math.radians(float(std_deg))
        self.generator = torch.Generator(device=self.device)
        self.values = torch.zeros(num_envs, device=self.device)
        self.valid = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.sample_count = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed) % (2**63 - 1)
        self.generator.manual_seed(self.seed)
        self.valid.zero_()
        self.values.zero_()
        self.sample_count.zero_()

    def invalidate(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.valid.zero_()
        else:
            self.valid[env_ids] = False

    def sample(self) -> torch.Tensor:
        ids = (~self.valid).nonzero(as_tuple=False).flatten()
        if ids.numel():
            if self.std_rad == 0.0:
                self.values[ids] = 0.0
            else:
                self.values[ids] = torch.randn(
                    (ids.numel(),), generator=self.generator, device=self.device
                ) * self.std_rad
            self.sample_count[ids] += 1
            self.valid[ids] = True
        return self.values
