r"""The :math:`(k,2,2)` dihedral spherical orbifold.

Counterpart of the planar constraint classes in :mod:`escher.OTE.tilings` (``OrbifoldI`` and
friends), for the sphere. This is the family the reference dump uses: ``orbifold_type = 4``
with ``k = 4`` in ``script_orbifold_sphere.m``.

The fundamental domain is a lune from the pole to the equator, spanning :math:`2\pi/k` in
longitude, with three cone points:

- order ``k`` at the **pole**, where the ``k`` copies around the axis meet;
- order 2 at the **middle of the equator arc**, about which that arc folds onto itself;
- order 2 at the **equator corners**, which are one point in the orbifold since the k-fold
  rotation identifies them.

Two boundary identifications, matching the two distinct rotations found in the reference
``A`` matrix:

- the two sides of the cut, paired by :math:`R_z(2\pi/k)`;
- the equator arc onto itself, folded by the half-turn about the axis through its midpoint.

Endpoints are excluded from the paired ranges and pinned instead, exactly as the planar
classes do (``sides["left"][1:-1]``): constraining a vertex both by a rotation and by a pin
would make ``A`` rank-deficient and :class:`~escher.OTE.core.spherical.affine_space.AffineSpace`
rejects that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from escher.geometry.sphere_tiler import SphericalTiler, rotation_matrix
from escher.geometry.spherical_base_mesh import LuneMesh, get_lune_mesh

from .sparse_system_3d import SparseSystem3D

__all__ = ["DihedralOrbifold"]


@dataclass
class DihedralOrbifold:
    r"""Orbifold boundary conditions for the :math:`(k,2,2)` dihedral group.

    Args:
        mesh: the fundamental-domain mesh, from :func:`get_lune_mesh`.

    Example:
        >>> orb = DihedralOrbifold.from_resolution(k=4, n_theta=24, n_phi=17)
        >>> A, b = orb.A, orb.b
        >>> A.shape[1] == 3 * orb.mesh.n_verts
        True
    """

    mesh: LuneMesh
    A: sp.csr_matrix
    b: np.ndarray

    @property
    def k(self) -> int:
        return self.mesh.k

    @property
    def cone_orders(self) -> tuple[int, int, int]:
        return (self.k, 2, 2)

    @classmethod
    def from_mesh(cls, mesh: LuneMesh) -> "DihedralOrbifold":
        A, b = _build_constraints(mesh)
        return cls(mesh=mesh, A=A, b=b)

    @classmethod
    def from_resolution(
        cls, k: int = 4, n_theta: int = 24, n_phi: int = 17
    ) -> "DihedralOrbifold":
        """Build the mesh and its constraints in one step."""
        return cls.from_mesh(get_lune_mesh(k=k, n_theta=n_theta, n_phi=n_phi))

    def tiler(self) -> SphericalTiler:
        r"""The symmetry group :math:`D_k`, of order ``2k``."""
        return SphericalTiler.dihedral(self.k)

    def initial_guess(self) -> np.ndarray:
        """A feasible ``(3n,)`` starting point: the undeformed lune, projected onto ``Ax=b``.

        Mirrors ``Ab.projectOnto(X)`` in ``solve_bfgs_fast.m``.
        """
        from escher.OTE.core.spherical.affine_space import AffineSpace

        return AffineSpace(self.A, self.b).project(self.mesh.points.reshape(-1))


def _build_constraints(mesh: LuneMesh) -> tuple[sp.csr_matrix, np.ndarray]:
    k = mesh.k
    sp3 = SparseSystem3D(mesh.n_verts)

    # --- cut sides: R_z(2*pi/k) carries the left side onto the right side.
    # Drop the pole (shared, pinned) and the equator corners (pinned as 2-fold cones).
    Rz = rotation_matrix([0.0, 0.0, 1.0], 2.0 * np.pi / k)
    sp3.add_rotation_constraints(mesh.left[1:-1], mesh.right[1:-1], Rz)

    # --- equator arc folds onto itself under the half-turn about its midpoint.
    # The midpoint sits at longitude 0, so that axis is x and the rotation is
    # diag(1, -1, -1), sending longitude phi to -phi.
    Rx = rotation_matrix([1.0, 0.0, 0.0], np.pi)
    bottom = mesh.bottom
    half = len(bottom) // 2
    # pair inward from the ends, excluding the corners (pinned) and the midpoint (fixed by
    # the rotation itself, so pairing it with itself would be degenerate)
    lower = bottom[1:half]
    upper = bottom[half + 1 : -1][::-1]
    sp3.add_rotation_constraints(lower, upper, Rx)

    # --- cone points
    half_width = np.pi / k
    corner_left = np.array([np.cos(-half_width), np.sin(-half_width), 0.0])
    sp3.add_fixed_constraint(mesh.pole, [0.0, 0.0, 1.0])
    sp3.add_fixed_constraint(mesh.bottom_mid, [1.0, 0.0, 0.0])
    sp3.add_fixed_constraint(mesh.bottom_left, corner_left)
    sp3.add_fixed_constraint(mesh.bottom_right, Rz @ corner_left)

    return sp3.build()
