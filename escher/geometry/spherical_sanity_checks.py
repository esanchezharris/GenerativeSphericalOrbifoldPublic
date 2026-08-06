r"""Bijectivity certificates for a mesh embedded on the sphere.

The spherical counterpart of
:func:`~escher.geometry.sanity_checks.check_triangle_orientation`.

The useful measure is the **signed solid angle** of each triangle, via Van Oosterom &
Strackee:

.. math::

    \tan\!\left(\frac{\Omega}{2}\right)
      = \frac{A \cdot (B \times C)}{1 + A\!\cdot\!B + B\!\cdot\!C + C\!\cdot\!A}

evaluated with ``atan2`` so it stays correct across the full range. Being *signed*, a folded
triangle contributes negative area, so the sum of all faces is a genuine certificate rather
than a heuristic:

- every triangle positively oriented, **and**
- total solid angle over all tiles exactly :math:`4\pi`

together mean the tiling covers the sphere once -- no gaps and no overlaps. Counting occupied
bins cannot show this: with a distorted tile the vertices clump, and bins go empty even
though the surface is covered. Area is independent of how the vertices happen to be spread.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "boundary_arc_length",
    "boundary_arc_ratio",
    "count_flipped_faces",
    "signed_solid_angles",
    "total_solid_angle",
    "check_covers_sphere_once",
]


def boundary_arc_length(points: np.ndarray, chain: np.ndarray) -> float:
    """Geodesic length of the polyline through ``chain``, in radians."""
    p = _as_unit(np.asarray(points)[np.asarray(chain)])
    cos = np.clip(np.einsum("ij,ij->i", p[:-1], p[1:]), -1.0, 1.0)
    return float(np.arccos(cos).sum())


def boundary_arc_ratio(
    points: np.ndarray, reference: np.ndarray, chains
) -> float:
    """Tile perimeter relative to the undeformed domain's.

    **The discriminator for a real Escher tiling.** A figure-shaped outline is far longer
    than the smooth boundary it started from, so a ratio near 1.0 means the figure is merely
    painted onto an undeformed tile, however convincing the texture looks. Cheap enough to
    log every step, and it answers the question the renders cannot.
    """
    now = sum(boundary_arc_length(points, c) for c in chains)
    before = sum(boundary_arc_length(reference, c) for c in chains)
    return now / max(before, 1e-12)

FULL_SPHERE = 4.0 * np.pi


def _as_unit(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    norms = np.linalg.norm(points, axis=-1, keepdims=True)
    return points / np.clip(norms, 1e-300, None)


def signed_solid_angles(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(n_faces,)`` signed solid angle in steradians; negative where a triangle is folded."""
    p = _as_unit(points)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        1.0
        + np.einsum("ij,ij->i", a, b)
        + np.einsum("ij,ij->i", b, c)
        + np.einsum("ij,ij->i", c, a)
    )
    return 2.0 * np.arctan2(numerator, denominator)


def total_solid_angle(points: np.ndarray, faces: np.ndarray) -> float:
    """Sum of :func:`signed_solid_angles`. Equals ``4*pi`` for a bijective closed tiling."""
    return float(signed_solid_angles(points, faces).sum())


def count_flipped_faces(points: np.ndarray, faces: np.ndarray) -> int:
    """Triangles whose outward normal points into the sphere."""
    return int((signed_solid_angles(points, faces) <= 0.0).sum())


def check_covers_sphere_once(
    points: np.ndarray, faces: np.ndarray, rtol: float = 1e-6
) -> tuple[bool, str]:
    """Return ``(ok, message)`` describing whether the tiling is a valid bijection.

    Args:
        points: ``(n, 3)`` vertices of the **fully tiled** mesh.
        faces: ``(n_faces, 3)`` faces of the fully tiled mesh.
        rtol: relative tolerance on the total area against ``4*pi``.
    """
    areas = signed_solid_angles(points, faces)
    n_flipped = int((areas <= 0.0).sum())
    total = float(areas.sum())
    rel = abs(total - FULL_SPHERE) / FULL_SPHERE

    if n_flipped:
        return False, (
            f"{n_flipped}/{len(faces)} faces are folded "
            f"(most negative {areas.min():.3e} sr)"
        )
    if rel > rtol:
        excess = total / FULL_SPHERE
        verdict = "overlaps" if excess > 1 else "leaves gaps"
        return False, (
            f"total solid angle {total:.9f} sr is {excess:.6f}x the sphere's {FULL_SPHERE:.9f} "
            f"-- the tiling {verdict}"
        )
    return True, f"covers the sphere once ({total:.9f} sr, relative error {rel:.2e})"
