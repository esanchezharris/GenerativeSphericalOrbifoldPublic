r"""Cameras for rendering the tiled sphere.

The planar pipeline never needed these: ``main.py:init_renderer`` sets both ``mv`` and
``proj`` to the identity, which is the right call for a flat tiling viewed head-on. A sphere
has to be looked *at*, from several directions per optimisation step, so score distillation
sees the whole surface rather than one hemisphere.

Matrix conventions follow :mod:`escher.rendering.renderer_nvdiffrast`, which computes
``vert_hom @ mv^T @ proj^T`` and then negates clip-space ``y`` and ``z``. So ``mv`` and
``proj`` are ordinary column-vector matrices: ``clip = proj @ mv @ p``.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "fibonacci_sphere",
    "look_at",
    "orbit_views",
    "perspective",
    "random_views",
]


def look_at(
    eye: Tensor, target: Tensor | None = None, up: Tensor | None = None
) -> Tensor:
    """World-to-view matrix. ``eye`` is ``(..., 3)``; returns ``(..., 4, 4)``.

    Right-handed, looking down ``-z`` in view space, matching the OpenGL convention the
    rasteriser expects.
    """
    eye = torch.as_tensor(eye, dtype=torch.float32)
    if eye.ndim == 1:
        eye = eye.unsqueeze(0)
    batch = eye.shape[0]

    target = (
        torch.zeros_like(eye) if target is None else torch.as_tensor(target, dtype=torch.float32).expand_as(eye)
    )
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0]).expand_as(eye).clone()
    else:
        up = torch.as_tensor(up, dtype=torch.float32).expand_as(eye).clone()

    forward = target - eye
    forward = forward / forward.norm(dim=-1, keepdim=True)

    # A camera directly over the pole has `up` parallel to `forward`; nudge it so the cross
    # product stays well conditioned instead of producing NaNs.
    degenerate = (torch.cross(forward, up, dim=-1).norm(dim=-1) < 1e-6).unsqueeze(-1)
    up = torch.where(degenerate, torch.tensor([0.0, 1.0, 0.0]).expand_as(up), up)

    right = torch.cross(forward, up, dim=-1)
    right = right / right.norm(dim=-1, keepdim=True)
    true_up = torch.cross(right, forward, dim=-1)

    mv = torch.eye(4).repeat(batch, 1, 1)
    mv[:, 0, :3] = right
    mv[:, 1, :3] = true_up
    mv[:, 2, :3] = -forward
    mv[:, 0, 3] = -(right * eye).sum(-1)
    mv[:, 1, 3] = -(true_up * eye).sum(-1)
    mv[:, 2, 3] = (forward * eye).sum(-1)
    return mv


def perspective(
    fovy_deg: float = 40.0, aspect: float = 1.0, near: float = 0.1, far: float = 100.0
) -> Tensor:
    """Standard OpenGL perspective projection, ``(4, 4)``."""
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    proj = torch.zeros(4, 4)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = (far + near) / (near - far)
    proj[2, 3] = (2.0 * far * near) / (near - far)
    proj[3, 2] = -1.0
    return proj


def fibonacci_sphere(n: int, dtype=torch.float32) -> Tensor:
    """``(n, 3)`` near-uniformly spaced unit vectors, via the Fibonacci spiral.

    Preferred over independent uniform sampling when a *batch* of views should cover the
    sphere evenly: random directions clump, and a clumped batch wastes score-distillation
    steps re-supervising the same region.
    """
    if n < 1:
        raise ValueError(f"n must be positive, got {n}")
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = torch.acos(1.0 - 2.0 * i / n)
    theta = math.pi * (1.0 + 5.0**0.5) * i
    return torch.stack(
        [
            torch.cos(theta) * torch.sin(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(phi),
        ],
        dim=1,
    ).to(dtype)


def _views_from_directions(directions: Tensor, distance: float) -> Tensor:
    return look_at(directions * distance)


def random_views(
    n: int,
    distance: float = 3.0,
    generator: torch.Generator | None = None,
    jitter: float = 1.0,
) -> Tensor:
    """``(n, 4, 4)`` view matrices from directions spread over the sphere.

    Starts from a Fibonacci set for even coverage, then applies a random rotation so
    successive optimisation steps do not keep sampling the same directions.

    Args:
        jitter: 0 gives the deterministic Fibonacci set; 1 gives a full random rotation.
    """
    directions = fibonacci_sphere(n)
    if jitter > 0:
        q = torch.randn(4, generator=generator)
        q = q / q.norm()
        rot = _quaternion_to_matrix(q)
        if jitter < 1.0:
            rot = _slerp_to_identity(rot, jitter)
        directions = directions @ rot.T
    return _views_from_directions(directions, distance)


def orbit_views(n: int, distance: float = 3.0, elevation_deg: float = 20.0) -> Tensor:
    """``(n, 4, 4)`` views evenly spaced around one latitude -- for turntable videos."""
    az = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
    el = math.radians(elevation_deg)
    directions = torch.stack(
        [torch.cos(az) * math.cos(el), torch.sin(az) * math.cos(el), torch.full_like(az, math.sin(el))],
        dim=1,
    )
    return _views_from_directions(directions, distance)


def _quaternion_to_matrix(q: Tensor) -> Tensor:
    w, x, y, z = q
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _slerp_to_identity(rot: Tensor, t: float) -> Tensor:
    """Interpolate a rotation toward the identity, staying in SO(3) via a polar projection."""
    blended = (1.0 - t) * torch.eye(3) + t * rot
    u, _, vh = torch.linalg.svd(blended)
    out = u @ vh
    if torch.det(out) < 0:
        u[:, -1] *= -1
        out = u @ vh
    return out
