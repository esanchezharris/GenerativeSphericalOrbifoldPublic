r"""Fundamental-domain meshes on the sphere.

The spherical counterpart of :func:`~escher.geometry.get_base_mesh.get_2d_square_mesh` and
:func:`~escher.geometry.get_base_mesh.get_hexagonal_mesh`. Those produce the planar base
domain that the wallpaper-group constraints deform; this produces the spherical one.

For a :math:`(k,2,2)` dihedral orbifold the fundamental domain is a **lune**: the wedge
between two meridians :math:`2\pi/k` apart, running from the pole down to the equator. Its
area is :math:`4\pi/2k`, so ``2k`` copies close the sphere.

The mesh starts as that exact lune. The solver then deforms it -- the orbifold conditions
only require the k-fold rotation to carry one side of the cut onto the other, so the boundary
is free to take any shape of the right angular width. Which is the point: that freedom is
what the generative pipeline searches over.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LuneMesh", "get_lune_mesh"]


@dataclass
class LuneMesh:
    """A disk-topology mesh covering one fundamental domain of a ``(k,2,2)`` orbifold.

    Boundary index arrays run pole -> equator for the sides, and left -> right along the
    bottom. Endpoints are shared between adjacent pieces.
    """

    points: np.ndarray  #: ``(n, 3)`` vertices on the unit sphere
    faces: np.ndarray  #: ``(n_faces, 3)``, counter-clockwise seen from outside
    left: np.ndarray  #: side at longitude ``-pi/k``, from pole to equator
    right: np.ndarray  #: side at longitude ``+pi/k``, from pole to equator
    bottom: np.ndarray  #: equator arc, from ``left[-1]`` to ``right[-1]``
    pole: int  #: index of the k-fold cone point
    bottom_mid: int  #: index of the 2-fold cone point at the middle of the equator arc
    uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    """``(n, 2)`` texture coordinates in ``[0, 1]^2``.

    Defined on the *fundamental domain*, and deliberately not recomputed after the solver
    deforms the mesh: every tile is the same copy of this domain, so replicating these UVs
    across the group makes all ``2k`` tiles sample one shared texture. That is what makes the
    result a tiling of a single figure rather than ``2k`` unrelated patches.
    """

    k: int = 4
    metadata: dict = field(default_factory=dict)

    @property
    def n_verts(self) -> int:
        return len(self.points)

    @property
    def edges(self) -> np.ndarray:
        """``(n_edges, 2)`` unique undirected edges with ``i < j``."""
        e = np.concatenate(
            [self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]]
        )
        e = np.sort(e, axis=1)
        return np.unique(e, axis=0)

    @property
    def bottom_left(self) -> int:
        return int(self.bottom[0])

    @property
    def bottom_right(self) -> int:
        return int(self.bottom[-1])

    @property
    def boundary_chains(self) -> list[np.ndarray]:
        """The tile boundary as ordered chains -- the mesh-type-agnostic accessor that
        ``boundary_loop``, the palette adjacency, and the perimeter metric consume."""
        return [self.left, self.bottom, self.right]


def _spherical_to_cartesian(colatitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    sin_t = np.sin(colatitude)
    return np.stack(
        [sin_t * np.cos(longitude), sin_t * np.sin(longitude), np.cos(colatitude)], axis=-1
    )


def get_lune_mesh(k: int = 4, n_theta: int = 24, n_phi: int = 16) -> LuneMesh:
    r"""Mesh the ``(k,2,2)`` fundamental domain.

    Args:
        k: order of the polar cone point; the lune spans :math:`2\pi/k` in longitude.
        n_theta: rings from the pole (exclusive) to the equator (inclusive).
        n_phi: vertices across the lune per ring; must be odd so a vertex lands exactly on
            the 2-fold cone at the middle of the equator arc.

    The pole is a single vertex rather than a collapsed ring, so the mesh is a genuine disk
    with no duplicate vertices.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    if n_theta < 2:
        raise ValueError(f"n_theta must be at least 2, got {n_theta}")
    if n_phi < 3:
        raise ValueError(f"n_phi must be at least 3, got {n_phi}")
    if n_phi % 2 == 0:
        raise ValueError(
            f"n_phi must be odd so that a vertex sits on the 2-fold cone point at the "
            f"centre of the equator arc, got {n_phi}"
        )

    half_width = np.pi / k
    longitudes = np.linspace(-half_width, half_width, n_phi)
    # rings 1..n_theta; ring n_theta is the equator
    colatitudes = np.linspace(0.0, np.pi / 2, n_theta + 1)[1:]

    points = [np.array([0.0, 0.0, 1.0])]  # pole
    # UV runs across the lune in u and from pole to equator in v. The pole is a single
    # vertex where every longitude coincides, so it takes the midpoint of the u range.
    uv = [np.array([0.5, 0.0])]
    ring_start = []
    for ring, theta in enumerate(colatitudes, start=1):
        ring_start.append(len(points))
        points.extend(_spherical_to_cartesian(np.full(n_phi, theta), longitudes))
        uv.extend(
            np.stack(
                [np.linspace(0.0, 1.0, n_phi), np.full(n_phi, ring / n_theta)], axis=1
            )
        )
    points = np.stack(points)
    uv = np.stack(uv)
    pole = 0

    faces = []
    # fan from the pole to the first ring
    first = ring_start[0]
    for j in range(n_phi - 1):
        faces.append([pole, first + j, first + j + 1])
    # quad strips between consecutive rings, split into two triangles each
    for r in range(len(ring_start) - 1):
        a, b = ring_start[r], ring_start[r + 1]
        for j in range(n_phi - 1):
            faces.append([a + j, b + j, a + j + 1])
            faces.append([a + j + 1, b + j, b + j + 1])
    faces = np.asarray(faces, dtype=np.int64)

    ring_start = np.asarray(ring_start)
    left = np.concatenate([[pole], ring_start])
    right = np.concatenate([[pole], ring_start + (n_phi - 1)])
    bottom = ring_start[-1] + np.arange(n_phi)
    bottom_mid = int(bottom[n_phi // 2])

    mesh = LuneMesh(
        points=points,
        faces=_orient_outward(points, faces),
        left=left,
        right=right,
        bottom=bottom,
        pole=pole,
        bottom_mid=bottom_mid,
        uv=uv,
        k=k,
        metadata={"n_theta": n_theta, "n_phi": n_phi, "half_width": half_width},
    )
    return mesh


def _orient_outward(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip any triangle whose normal points into the sphere.

    On the unit sphere the outward direction at a triangle is its centroid direction, so the
    test is just ``dot(normal, centroid) > 0``.
    """
    tri = points[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centroids = tri.mean(axis=1)
    flip = np.einsum("ij,ij->i", normals, centroids) < 0
    faces = faces.copy()
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return faces
