r"""Replicating a fundamental domain over the sphere by its orbifold symmetry group.

Port of the role played by ``Tiler.m`` / ``three_point_tile.m`` in the reference.

A spherical orbifold is :math:`S^2 / G` for a finite rotation group :math:`G`. The embedding
solver maps the cut mesh onto **one** fundamental domain; applying every element of
:math:`G` recovers the closed sphere. For the Escher pipeline this is also what makes the
tiling a tiling: each copy is the *same* tile, so one texture serves all of them.

The relevant groups are the rotation subgroups of the spherical triangle groups
:math:`(p, q, r)` with :math:`1/p + 1/q + 1/r > 1`:

===========  ==========================  =====
Cone orders  Group                       Order
===========  ==========================  =====
``(k,2,2)``  dihedral :math:`D_k`        ``2k``
``(2,3,3)``  tetrahedral :math:`T`       12
``(2,3,4)``  octahedral :math:`O`        24
``(2,3,5)``  icosahedral :math:`I`       60
===========  ==========================  =====

with :math:`|G| = 2 / (1/p + 1/q + 1/r - 1)` in every case.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "SphericalTiler",
    "expected_group_order",
    "generate_rotation_group",
    "rotation_matrix",
]


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Right-handed rotation by ``angle`` radians about ``axis`` (Rodrigues' formula)."""
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-14:
        raise ValueError("rotation axis must be non-zero")
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return c * np.eye(3) + s * cross + (1 - c) * np.outer([x, y, z], [x, y, z])


def expected_group_order(cone_orders: tuple[int, int, int]) -> int:
    r""":math:`|G| = 2 / (1/p + 1/q + 1/r - 1)`, the order of the rotation group.

    Raises:
        ValueError: if the orders do not describe a *spherical* orbifold
            (:math:`1/p + 1/q + 1/r \le 1` gives a Euclidean or hyperbolic one instead).
    """
    p, q, r = cone_orders
    excess = 1.0 / p + 1.0 / q + 1.0 / r - 1.0
    if excess <= 1e-12:
        raise ValueError(
            f"cone orders {cone_orders} give 1/p+1/q+1/r = {excess + 1:.4f} <= 1, which is a "
            "Euclidean or hyperbolic orbifold, not a spherical one"
        )
    order = 2.0 / excess
    if abs(order - round(order)) > 1e-9:
        raise ValueError(f"cone orders {cone_orders} give non-integral group order {order}")
    return int(round(order))


def generate_rotation_group(
    generators: list[np.ndarray], tol: float = 1e-9, max_size: int = 1000
) -> np.ndarray:
    """Close a set of rotations under multiplication.

    Args:
        generators: ``(3, 3)`` rotation matrices.
        tol: two elements within this Frobenius distance are the same group element.
        max_size: guard against a non-finite group from inexact generators.

    Returns:
        ``(|G|, 3, 3)`` array, identity first.
    """
    elements = [np.eye(3)]

    def index_of(m: np.ndarray) -> int:
        for i, e in enumerate(elements):
            if np.abs(e - m).max() < tol:
                return i
        return -1

    frontier = [np.eye(3)]
    while frontier:
        current = frontier.pop()
        for gen in generators:
            candidate = gen @ current
            if index_of(candidate) < 0:
                elements.append(candidate)
                frontier.append(candidate)
                if len(elements) > max_size:
                    raise ValueError(
                        f"group did not close within {max_size} elements; generators are "
                        "probably not exact rotations of finite order"
                    )
    return np.stack(elements)


@dataclass
class SphericalTiler:
    """A finite rotation group acting on points of the unit sphere.

    Construct via :meth:`dihedral` or :meth:`from_generators`.
    """

    rotations: np.ndarray  #: ``(|G|, 3, 3)``
    cone_orders: tuple[int, int, int] | None = None

    @property
    def order(self) -> int:
        return len(self.rotations)

    @classmethod
    def from_generators(
        cls, generators: list[np.ndarray], cone_orders: tuple[int, int, int] | None = None
    ) -> "SphericalTiler":
        group = generate_rotation_group(generators)
        if cone_orders is not None:
            expected = expected_group_order(cone_orders)
            if len(group) != expected:
                raise ValueError(
                    f"generators produced a group of order {len(group)} but cone orders "
                    f"{cone_orders} require {expected}"
                )
        return cls(rotations=group, cone_orders=cone_orders)

    @classmethod
    def dihedral(
        cls, k: int, principal_axis=(0.0, 0.0, 1.0), secondary_axis=(0.0, 1.0, 0.0)
    ) -> "SphericalTiler":
        r"""The :math:`(k,2,2)` dihedral group :math:`D_k`, of order ``2k``.

        Generated by a :math:`2\pi/k` rotation about ``principal_axis`` and a half-turn about
        ``secondary_axis``, which must be perpendicular to it.
        """
        principal = np.asarray(principal_axis, dtype=np.float64)
        secondary = np.asarray(secondary_axis, dtype=np.float64)
        cos = float(principal @ secondary) / (
            np.linalg.norm(principal) * np.linalg.norm(secondary)
        )
        if abs(cos) > 1e-9:
            raise ValueError(
                f"dihedral axes must be perpendicular (got cos = {cos:.3e})"
            )
        return cls.from_generators(
            [
                rotation_matrix(principal, 2 * np.pi / k),
                rotation_matrix(secondary, np.pi),
            ],
            cone_orders=(k, 2, 2),
        )

    # ------------------------------------------------------------------------ tiling
    def tile_points(self, points: np.ndarray) -> np.ndarray:
        """Apply every group element. ``(n, 3)`` -> ``(|G|, n, 3)``."""
        return np.einsum("gij,nj->gni", self.rotations, np.asarray(points, dtype=np.float64))

    def tile_mesh(
        self, points: np.ndarray, faces: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Replicate a mesh into the closed sphere.

        Returns ``(vertices, faces, tile_index)`` with vertices concatenated per copy, face
        indices offset accordingly, and ``tile_index`` labelling which group element produced
        each face -- useful for per-tile colouring.

        Copies produced by an orientation-reversing element would need their faces flipped;
        rotation groups contain none, so face winding carries over unchanged.
        """
        points = np.asarray(points, dtype=np.float64)
        faces = np.asarray(faces)
        n = len(points)

        tiled = self.tile_points(points).reshape(-1, 3)
        all_faces = np.concatenate([faces + g * n for g in range(self.order)])
        tile_index = np.repeat(np.arange(self.order), len(faces))
        return tiled, all_faces, tile_index

    def tile_attribute(self, values: np.ndarray) -> np.ndarray:
        """Repeat a per-vertex attribute once per group element: ``(n, d)`` -> ``(|G|*n, d)``.

        Used for texture coordinates. Because every copy repeats the *same* UVs, all ``|G|``
        tiles sample one shared texture -- which is exactly what makes the result a tiling of
        a single figure rather than independently textured patches.
        """
        return np.tile(np.asarray(values), (self.order, 1))

    def tile_edges(self, points: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Replicate an edge set. Returns ``(|G| * n_edges, 2, 3)`` line segments."""
        points = np.asarray(points, dtype=np.float64)
        edges = np.asarray(edges)
        return np.stack([(points @ R.T)[edges] for R in self.rotations]).reshape(-1, 2, 3)

    def coverage_fraction(self, points: np.ndarray, n_bins: int = 400) -> float:
        """Fraction of equal-area sphere bins containing at least one tiled point.

        A coarse but honest check that the copies really do cover the sphere. Keep
        ``n_bins`` well below the vertex count or the result is density-limited rather than
        coverage-limited.
        """
        tiled = self.tile_points(points).reshape(-1, 3)
        tiled = tiled / np.linalg.norm(tiled, axis=1, keepdims=True)
        n_lat = max(int(np.sqrt(n_bins / 2)), 1)
        n_lon = max(n_bins // n_lat, 1)
        # bin by cos(latitude) so every cell subtends the same solid angle
        ilat = np.clip(((1 - tiled[:, 2]) / 2 * n_lat).astype(int), 0, n_lat - 1)
        lon = np.arctan2(tiled[:, 1], tiled[:, 0]) % (2 * np.pi)
        ilon = np.clip((lon / (2 * np.pi) * n_lon).astype(int), 0, n_lon - 1)
        occupied = len(set(zip(ilat.tolist(), ilon.tolist())))
        return occupied / (n_lat * n_lon)
