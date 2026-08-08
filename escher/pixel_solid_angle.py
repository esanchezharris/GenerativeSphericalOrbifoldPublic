"""Per-pixel spherical area for the shape phase's fixed camera.

Pixel counts are not steradians: at the production (2,3,4) framing the spherical
area a pixel covers varies 2.4x between the tile's center and its rim. The tile's
area is PINNED at 4pi/|G| on the sphere, so matching the target's area in PIXELS
delivers a target ~3.2% oversized in the measure that binds -- a structural
hard-IoU ceiling of ~0.95 before optimization starts (the exact failure mode
MATCH_TILE_AREA was added to remove, reintroduced by the projection).

``pixel_solid_angles`` returns, for every pixel of the shape camera's frame, the
area ON THE UNIT SPHERE of that pixel's footprint (zero where the pixel misses the
sphere). Summing it over a mask measures the mask in steradians.

Convention-free by construction: rays are built by inverting the exact
``proj @ mv`` matrix used by ``soft_silhouette.project_to_pixels``, so any change
to the camera conventions flows through automatically.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["pixel_solid_angles"]


def _as_matrix(m) -> np.ndarray:
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    return np.asarray(m, dtype=np.float64).reshape(4, 4)


def _sphere_hits(pm_inv: np.ndarray, ndc_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NDC (x, y) grid -> unit-sphere front-hit points and a hit mask.

    Unprojects each NDC point at two clip depths through ``(proj @ mv)^-1`` to get
    a world-space ray, then takes the nearer intersection with the unit sphere.
    """
    n = ndc_xy.shape[0]
    ones = np.ones((n, 1))
    p0 = np.concatenate([ndc_xy, np.full((n, 1), 0.0), ones], axis=1) @ pm_inv.T
    p1 = np.concatenate([ndc_xy, np.full((n, 1), 0.5), ones], axis=1) @ pm_inv.T
    p0 = p0[:, :3] / p0[:, 3:4]
    p1 = p1[:, :3] / p1[:, 3:4]

    origin = p0
    direction = p1 - p0
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    b = (origin * direction).sum(axis=1)
    c = (origin * origin).sum(axis=1) - 1.0
    disc = b * b - c
    hit = disc >= 0.0
    t = -b - np.sqrt(np.where(hit, disc, 0.0))
    points = origin + t[:, None] * direction
    # Normalize the hits exactly onto the sphere; misses keep a placeholder.
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = np.where(hit[:, None], points / np.clip(norms, 1e-12, None), 0.0)
    return points, hit


def _triangle_solid_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Van Oosterom-Strackee: solid angle at the origin of unit-vector triangles."""
    triple = np.einsum("ij,ij->i", a, np.cross(b, c))
    denom = (
        1.0
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return np.abs(2.0 * np.arctan2(np.abs(triple), denom))


def pixel_solid_angles(mv, proj, height: int, width: int) -> np.ndarray:
    """``(H, W)`` spherical area (steradians) of each pixel's unit-sphere footprint.

    Zero for pixels whose footprint is not entirely on the sphere -- boundary
    pixels straddling the horizon are dropped rather than partially integrated,
    which is exact for any mask that stays inside the projected disc (every tile
    and target in the shape phase does).
    """
    pm = _as_matrix(proj) @ _as_matrix(mv)
    pm_inv = np.linalg.inv(pm)

    # Pixel CORNER grid in the exact convention of project_to_pixels:
    # col = (ndc_x + 1)/2 * W  ->  ndc_x = 2*col/W - 1;  row uses the y-negation.
    cols = np.arange(width + 1, dtype=np.float64)
    rows = np.arange(height + 1, dtype=np.float64)
    ndc_x = 2.0 * cols / width - 1.0
    ndc_y = -(2.0 * rows / height - 1.0)
    gx, gy = np.meshgrid(ndc_x, ndc_y, indexing="xy")  # (H+1, W+1)
    ndc = np.stack([gx.ravel(), gy.ravel()], axis=1)

    pts, hit = _sphere_hits(pm_inv, ndc)
    pts = pts.reshape(height + 1, width + 1, 3)
    hit = hit.reshape(height + 1, width + 1)

    p00 = pts[:-1, :-1].reshape(-1, 3)
    p01 = pts[:-1, 1:].reshape(-1, 3)
    p10 = pts[1:, :-1].reshape(-1, 3)
    p11 = pts[1:, 1:].reshape(-1, 3)
    all_hit = (hit[:-1, :-1] & hit[:-1, 1:] & hit[1:, :-1] & hit[1:, 1:]).reshape(-1)

    omega = _triangle_solid_angle(p00, p01, p11) + _triangle_solid_angle(
        p00, p11, p10
    )
    omega = np.where(all_hit, omega, 0.0)
    return omega.reshape(height, width)
