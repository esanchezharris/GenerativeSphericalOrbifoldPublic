r"""Weights-mode constraint system for the three-cone spherical orbifolds.

The counterpart of :class:`DihedralOrbifold` on the kite fundamental domain: the
constraint matrix pairs the two cut chains by their cone rotations and pins the four
cone corners, and the tile SHAPE is then steered indirectly through the edge weights of
the Karcher orbifold-Tutte solve -- the full-mesh (GEM) parameterization, with the
solve free to move every interior vertex.

Named for its headline case -- ``(2,3,4)``, the octahedral group, 24 tiles -- but any
spherical triple works: ``(2,3,3)`` tetrahedral and ``(2,3,5)`` icosahedral come free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from escher.geometry.sphere_tiler import SphericalTiler
from escher.geometry.spherical_kite_mesh import KiteMesh, get_kite_mesh

from .boundary_explicit_triple import cut_rotations
from .sparse_system_3d import SparseSystem3D

__all__ = ["OctahedralOrbifold"]


@dataclass
class OctahedralOrbifold:
    """Orbifold boundary conditions for a ``(p,q,r)`` rotation group, weights mode.

    Interface-compatible with :class:`DihedralOrbifold` where the driver touches it:
    ``mesh``, ``A``, ``b``, ``initial_guess``, ``tiler``.
    """

    mesh: KiteMesh
    A: sp.csr_matrix
    b: np.ndarray
    R1: np.ndarray  #: about cone 1, carries left1 onto right1
    R2: np.ndarray  #: about cone 3, carries left2 onto right2

    @property
    def cone_orders(self) -> tuple[int, int, int]:
        return self.mesh.cone_orders

    @classmethod
    def from_mesh(cls, mesh: KiteMesh) -> "OctahedralOrbifold":
        R1, R2 = cut_rotations(mesh)
        sp3 = SparseSystem3D(mesh.n_verts)
        # Cut chains paired by the cone rotations; endpoints excluded (pinned below --
        # constraining a vertex both ways would make A rank-deficient, exactly the
        # planar classes' sides[1:-1] convention).
        sp3.add_rotation_constraints(mesh.left1[1:-1], mesh.right1[1:-1], R1)
        sp3.add_rotation_constraints(mesh.left2[1:-1], mesh.right2[1:-1], R2)
        for v in (mesh.cone1, mesh.cone2a, mesh.cone2b, mesh.cone3):
            sp3.add_fixed_constraint(v, mesh.points[v])
        A, b = sp3.build()
        return cls(mesh=mesh, A=A, b=b, R1=R1, R2=R2)

    @classmethod
    def from_resolution(
        cls, cone_orders: tuple[int, int, int] = (2, 3, 4), n: int = 20
    ) -> "OctahedralOrbifold":
        return cls.from_mesh(get_kite_mesh(cone_orders, n=n))

    def tiler(self) -> SphericalTiler:
        return SphericalTiler.from_generators(
            [self.R1, self.R2], cone_orders=self.mesh.cone_orders
        )

    def initial_guess(self) -> np.ndarray:
        """A feasible ``(3n,)`` start: the undeformed kite, projected onto ``Ax=b``."""
        from escher.OTE.core.spherical.affine_space import AffineSpace

        return AffineSpace(self.A, self.b).project(self.mesh.points.reshape(-1))
