r"""Boundary-explicit parameterization of the three-cone spherical orbifolds.

The ``(k,2,2)`` dihedral version (``boundary_explicit.py``) proved the recipe: free the
interior of one side of the cut, generate the other side by the symmetry group, pin the
whole boundary, and solve the interior with clamped cotangent weights. This is the same
recipe on the KITE fundamental domain of ``(p,q,r)`` -- for ``(2,3,4)`` that means the
octahedral group and **24 tiles**, three times the figure density of the ``k=4``
dihedral runs and the count the target image shows.

Free parameters: the interiors of the two left cut chains (cone1 -> cone2a -> cone3).
Their twins are *generated* -- ``right1 = R1 @ left1`` with ``R1`` the rotation about
cone 1 by :math:`2\pi/p_1`, and ``right2 = R2 @ left2`` about cone 3 by
:math:`2\pi/p_3` (the reference derives the same rotations by Procrustes from the
endpoint positions; here they are analytic and verified at construction). The four kite
corners are the cones, pinned. Same honest caveat as the dihedral mode: prescribed
boundaries void the orbifold-Tutte bijectivity theorem, so validity is *checked* --
fold rejection, the margin hinge, and the certificate do that work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

from escher.geometry.sphere_tiler import SphericalTiler, rotation_matrix
from escher.geometry.spherical_kite_mesh import KiteMesh, get_kite_mesh

from .sparse_system_3d import SparseSystem3D

__all__ = ["BoundaryExplicitTriple", "clamped_cotan_weights", "cut_rotations"]


def cut_rotations(mesh) -> tuple[np.ndarray, np.ndarray]:
    """The two cut rotations of a kite mesh, signs resolved numerically and VERIFIED.

    ``R1`` about cone 1 by :math:`2\\pi/p_1` maps ``left1 -> right1``; ``R2`` about
    cone 3 by :math:`2\\pi/p_3` maps ``left2 -> right2``. The reference derives the same
    rotations by Procrustes from the endpoint positions (``CutBoundaryPiece.setR``);
    here they are analytic, with the same exactness assertion. Shared by the
    boundary-explicit and weights-mode (``OctahedralOrbifold``) constraint systems.
    """
    corners = mesh.metadata["corners"]
    c1, c2a, c3, c2b = corners
    orders = mesh.cone_orders

    def one(axis, order):
        for sign in (1.0, -1.0):
            R = rotation_matrix(axis, sign * 2 * np.pi / order)
            if np.allclose(R @ c2a, c2b, atol=1e-9):
                return R
        raise AssertionError("no cone rotation maps cone2a to cone2b")

    R1 = one(c1, orders[0])
    R2 = one(c3, orders[2])
    assert np.abs(mesh.points[mesh.left1] @ R1.T - mesh.points[mesh.right1]).max() < 1e-9
    assert np.abs(mesh.points[mesh.left2] @ R2.T - mesh.points[mesh.right2]).max() < 1e-9
    return R1, R2


def clamped_cotan_weights(mesh) -> np.ndarray:
    r"""Clamped cotangent weights aligned with ``mesh.edges`` (``Solver.m``'s scheme).

    Uniform interior weights were the root cause of the B-series fold storm (init min
    area ratio 0.105 vs 0.998 with cotangent); every fixed-boundary solve uses these.
    """
    import igl

    L = igl.cotmatrix(mesh.points, mesh.faces.astype(np.int64))
    w = np.asarray(L[mesh.edges[:, 0], mesh.edges[:, 1]]).ravel()
    w[w < 1e-3] = 1e-3
    return w


@dataclass
class BoundaryExplicitTriple:
    """Fixed-boundary ``(p,q,r)`` orbifold problem with the left cut as parameters.

    Interface-compatible with :class:`BoundaryExplicitDihedral` where the drivers
    touch it: ``mesh``, ``A``, ``pin_order``, ``boundary_b``, ``initial_free_points``,
    ``cotan_edge_weights``, ``tiler``, ``anchored_chains``, ``n_free``.
    """

    mesh: KiteMesh
    A: sp.csr_matrix
    pin_order: np.ndarray
    R1: np.ndarray  #: about cone 1, maps left1 -> right1
    R2: np.ndarray  #: about cone 3, maps left2 -> right2
    _roles: list = field(repr=False, default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_resolution(
        cls, cone_orders: tuple[int, int, int] = (2, 3, 4), n: int = 20
    ) -> "BoundaryExplicitTriple":
        mesh = get_kite_mesh(cone_orders, n=n)
        R1, R2 = cut_rotations(mesh)

        pin_order: list[int] = []
        roles: list[tuple[str, int]] = []

        def add(idx, role, src=-1):
            pin_order.append(int(idx))
            roles.append((role, int(src)))

        add(mesh.cone1, "cone1")
        add(mesh.cone2a, "cone2a")
        add(mesh.cone2b, "cone2b")
        add(mesh.cone3, "cone3")
        free1 = mesh.left1[1:-1]
        free2 = mesh.left2[1:-1]
        for i, v in enumerate(free1):
            add(v, "free_1", i)
        for i, v in enumerate(free2):
            add(v, "free_2", i)
        for i, v in enumerate(mesh.right1[1:-1]):
            add(v, "gen_1", i)  # image of free_1[i] under R1
        for i, v in enumerate(mesh.right2[1:-1]):
            add(v, "gen_2", i)  # image of free_2[i] under R2

        sp3 = SparseSystem3D(mesh.n_verts)
        for v in pin_order:
            sp3.add_fixed_constraint(v, mesh.points[v])
        A, _ = sp3.build()

        return cls(
            mesh=mesh, A=A, pin_order=np.asarray(pin_order), R1=R1, R2=R2, _roles=roles
        )

    # -------------------------------------------------------------------- parameters
    @property
    def n_free_1(self) -> int:
        return sum(1 for r, _ in self._roles if r == "free_1")

    @property
    def n_free(self) -> int:
        return sum(1 for r, _ in self._roles if r.startswith("free"))

    def initial_free_points(self) -> Tensor:
        idx = [
            self.pin_order[i]
            for i, (r, _) in enumerate(self._roles)
            if r.startswith("free")
        ]
        return torch.as_tensor(self.mesh.points[idx], dtype=torch.float64)

    def boundary_b(self, free_points: Tensor) -> Tensor:
        """The pin right-hand side from the free parameters, differentiably."""
        if free_points.shape != (self.n_free, 3):
            raise ValueError(
                f"expected ({self.n_free}, 3) free points, got {tuple(free_points.shape)}"
            )
        p = free_points / free_points.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        f1 = p[: self.n_free_1]
        f2 = p[self.n_free_1 :]
        r1 = torch.as_tensor(self.R1, dtype=torch.float64)
        r2 = torch.as_tensor(self.R2, dtype=torch.float64)

        cones = {
            role: torch.as_tensor(self.mesh.points[idx], dtype=torch.float64)
            for role, idx in (
                ("cone1", self.mesh.cone1),
                ("cone2a", self.mesh.cone2a),
                ("cone2b", self.mesh.cone2b),
                ("cone3", self.mesh.cone3),
            )
        }
        rows = []
        for role, src in self._roles:
            if role in cones:
                rows.append(cones[role])
            elif role == "free_1":
                rows.append(f1[src])
            elif role == "free_2":
                rows.append(f2[src])
            elif role == "gen_1":
                rows.append(r1 @ f1[src])
            elif role == "gen_2":
                rows.append(r2 @ f2[src])
            else:  # pragma: no cover
                raise AssertionError(role)
        return torch.stack(rows).reshape(-1)

    def anchored_chains(self, p: Tensor) -> list[Tensor]:
        """The free chains with their cone anchors, for the spacing/smoothness regs."""
        cones = {
            name: torch.as_tensor(self.mesh.points[idx], dtype=p.dtype)
            for name, idx in (
                ("c1", self.mesh.cone1),
                ("c2a", self.mesh.cone2a),
                ("c3", self.mesh.cone3),
            )
        }
        n1 = self.n_free_1
        return [
            torch.cat([cones["c1"][None], p[:n1], cones["c2a"][None]]),
            torch.cat([cones["c2a"][None], p[n1:], cones["c3"][None]]),
        ]

    def cotan_edge_weights(self) -> np.ndarray:
        return clamped_cotan_weights(self.mesh)

    def tiler(self) -> SphericalTiler:
        return SphericalTiler.from_generators(
            [self.R1, self.R2], cone_orders=self.mesh.cone_orders
        )

    def feasible_x0(self, b: np.ndarray) -> np.ndarray:
        x0 = self.mesh.points.copy()
        x0[self.pin_order] = np.asarray(b, dtype=np.float64).reshape(-1, 3)
        return x0.reshape(-1)
