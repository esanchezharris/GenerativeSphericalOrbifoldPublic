"""An analytic, differentiable tile silhouette: project the boundary, soft-fill it.

Why not the rasterizer's alpha: measured with an isolated probe, nvdiffrast's alpha
channel -- and the full RGB+alpha composite the SDS silhouette pass used -- return
gradients of EXACTLY zero to the vertices under a uniform texture. ``dr.antialias`` is
the only supposed source of vertex gradients there, and whatever it contributes here is
identically zero; the interior is constant, so nothing else can carry them. Every
outline change in the SDS runs actually came from the textured main pass's
texture-slide gradients plus noise, which is consistent with the weak, chaotic outline
equilibria those runs measured.

The fix is to stop asking the rasterizer for something we hold analytically. Seen from
the tile-centric camera, the isolated tile's silhouette IS its projected boundary
polygon -- and the boundary vertices are exactly what the shape phase optimizes. So:
project the ordered boundary loop with the same camera conventions as the renderer,
compute an unsigned distance field to the polygon (differentiable), take the sign from
a crossing-number test (detached: it only changes when the boundary crosses a pixel
center), and squash through a sigmoid. Dense, exact gradients in a ``tau``-wide band
around the outline, no renderer in the loop, and the whole shape phase runs on CPU.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

__all__ = ["boundary_loop", "project_to_pixels", "soft_polygon_mask"]


def boundary_loop(mesh) -> np.ndarray:
    """Ordered vertex indices once around the tile boundary (not repeated at the end).

    Stitches the mesh's boundary chains by their shared endpoint indices, reversing
    chains as needed, so it depends on neither the chain order nor each chain's stored
    orientation. Works for any mesh exposing ``boundary_chains`` (lune or kite).
    """
    chains = [list(c) for c in mesh.boundary_chains]
    loop = chains.pop(0)
    while chains:
        for i, chain in enumerate(chains):
            if chain[0] == loop[-1]:
                loop += chain[1:]
            elif chain[-1] == loop[-1]:
                loop += chain[::-1][1:]
            else:
                continue
            chains.pop(i)
            break
        else:
            raise ValueError("boundary chains do not share endpoints; cannot stitch")
    if loop[-1] == loop[0]:
        loop = loop[:-1]
    return np.asarray(loop, dtype=np.int64)


def project_to_pixels(
    points: Tensor, mv: Tensor, proj: Tensor, height: int, width: int
) -> Tensor:
    """``(n, 3)`` world points -> ``(n, 2)`` pixel coordinates ``(col, row)``.

    Mirrors ``renderer_nvdiffrast`` exactly: clip = hom @ mv^T @ proj^T, then the
    y-negation it applies for image memory order. Validated against a rasterized alpha
    of the same mesh (IoU ~1 at the undeformed state).
    """
    mv = mv.reshape(4, 4).to(points.dtype)
    proj = proj.reshape(4, 4).to(points.dtype)
    hom = torch.cat([points, points.new_ones(points.shape[0], 1)], dim=-1)
    clip = hom @ mv.T @ proj.T
    w = clip[:, 3].clamp_min(1e-9)
    col = (clip[:, 0] / w + 1.0) * 0.5 * width
    row = (-clip[:, 1] / w + 1.0) * 0.5 * height
    return torch.stack([col, row], dim=-1)


def soft_polygon_mask(
    pts: Tensor, height: int, width: int, tau: float = 2.0, chunk: int = 65536
) -> Tensor:
    """``(n, 2)`` polygon in pixel space -> ``(H, W)`` soft-edged inside mask.

    ``sigmoid(sign * distance / tau)``: distance to the nearest polygon segment is the
    differentiable part; the inside/outside sign comes from a crossing-number test and
    is detached. ``tau`` is the soft-edge width in pixels -- gradients live in that
    band, and the pyramid loss on top extends their reach.
    """
    A = pts
    B = torch.roll(pts, -1, dims=0)
    AB = B - A
    ab2 = (AB * AB).sum(-1).clamp_min(1e-12)

    dtype, device = pts.dtype, pts.device
    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    py, px = torch.meshgrid(ys, xs, indexing="ij")
    pixels = torch.stack([px, py], dim=-1).reshape(-1, 2)

    rows = []
    for start in range(0, pixels.shape[0], chunk):
        P = pixels[start : start + chunk].unsqueeze(1)  # (c, 1, 2)
        AP = P - A.unsqueeze(0)  # (c, n, 2)
        t = ((AP * AB.unsqueeze(0)).sum(-1) / ab2).clamp(0.0, 1.0)
        D = AP - t.unsqueeze(-1) * AB.unsqueeze(0)
        dist = (D * D).sum(-1).clamp_min(1e-12).sqrt().min(-1).values  # (c,)

        with torch.no_grad():
            pyc = P[:, 0, 1].unsqueeze(1)  # (c, 1)
            pxc = P[:, 0, 0].unsqueeze(1)
            ay, by = A[:, 1].unsqueeze(0), B[:, 1].unsqueeze(0)
            crosses = (ay <= pyc) != (by <= pyc)
            denom = torch.where(crosses, (B[:, 1] - A[:, 1]).unsqueeze(0), torch.ones_like(ay))
            x_hit = A[:, 0].unsqueeze(0) + (pyc - ay) / denom * (B[:, 0] - A[:, 0]).unsqueeze(0)
            inside = ((crosses & (x_hit > pxc)).sum(-1) % 2) == 1
            sign = torch.where(inside, 1.0, -1.0).to(dtype)

        rows.append(torch.sigmoid(sign * dist / tau))
    return torch.cat(rows).reshape(height, width)
