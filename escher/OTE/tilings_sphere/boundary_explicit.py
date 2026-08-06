r"""Boundary-explicit parameterization of the ``(k,2,2)`` spherical orbifold.

The weight-based pipeline steers the tile outline *indirectly*: SDS pushes edge weights, the
solve responds, and the boundary moves as a side effect. Six measured runs showed the limits
of that route -- outline growth and local area collapse are coupled in the weight space, so
barriers either fail (spread 632, 203) or smother the shape (perimeter regressing to 1.04).

The reference implementation contains the alternative in plain sight:
``FixedBoundaryConditions.m`` ("classic Tutte" position pins, :math:`b` = prescribed
coordinates) and ``embed_from_boundary.m`` (given a boundary object, embed the interior).
Boundary positions enter **only through** :math:`b`, and our port kept that structure.

So: parameterize the **free half of the cut** directly --

- the left cut chain's interior points (the pole and the equator corner stay pinned),
- the left half of the equator arc (the 2-fold cone at its midpoint stays pinned),

generate the mates by the symmetry group (``right = R_z(2\pi/k) @ left``, right half of the
arc = half-turn image of the left half), pin the **entire** boundary, and solve only the
interior with uniform Tutte weights. Consequences:

- neighbouring tiles interlock **by construction** -- the mate is generated, not constrained;
- SDS's silhouette gradient lands on the outline as first-class parameters (~35 points
  instead of 1522 weights);
- the interior no longer needs extreme weights, removing the degeneracy mechanism the
  barriers were fighting.

The price, stated honestly: prescribed boundaries void the orbifold-Tutte bijectivity
guarantee (the reference uses fixed pieces for *convex* spherical polygons). Validity is
therefore *checked*, not assumed -- the certificate and boundary regularizers do that work.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor

from escher.geometry.sphere_tiler import SphericalTiler, rotation_matrix
from escher.geometry.spherical_base_mesh import LuneMesh, get_lune_mesh

from .sparse_system_3d import SparseSystem3D

__all__ = ["BoundaryExplicitDihedral"]


@dataclass
class BoundaryExplicitDihedral:
    """Fixed-boundary orbifold problem with the free half of the cut as parameters.

    Build via :meth:`from_resolution`. ``A`` pins every boundary vertex; :meth:`boundary_b`
    turns the free parameters into the matching right-hand side, differentiably.
    """

    mesh: LuneMesh
    A: sp.csr_matrix
    #: boundary vertex indices in the exact row order of ``A`` (3 rows each)
    pin_order: np.ndarray
    #: index into :attr:`pin_order` -> ("free"|"pole"|"mid"|"corner_l"|"corner_r", src)
    _roles: list = field(repr=False, default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_resolution(
        cls, k: int = 4, n_theta: int = 28, n_phi: int = 19
    ) -> "BoundaryExplicitDihedral":
        mesh = get_lune_mesh(k=k, n_theta=n_theta, n_phi=n_phi)

        # Free parameters: left chain interior + left half of the bottom arc.
        left_free = mesh.left[1:-1]  # exclude pole and bottom-left corner
        half = len(mesh.bottom) // 2
        bottom_free = mesh.bottom[1:half]  # exclude corner and 2-fold midpoint

        # Generated mates: right chain from the left, right half-arc from the left half.
        right_gen = mesh.right[1:-1]
        bottom_gen = mesh.bottom[half + 1 : -1][::-1]  # pairs with bottom_free reversed

        pin_order: list[int] = []
        roles: list[tuple[str, int]] = []

        def add(idx, role, src=-1):
            pin_order.append(int(idx))
            roles.append((role, int(src)))

        add(mesh.pole, "pole")
        add(mesh.bottom_mid, "mid")
        add(mesh.bottom_left, "corner_l")
        add(mesh.bottom_right, "corner_r")
        for i, v in enumerate(left_free):
            add(v, "free_left", i)
        for i, v in enumerate(bottom_free):
            add(v, "free_bottom", i)
        for i, v in enumerate(right_gen):
            add(v, "gen_right", i)  # image of free_left[i] under R_z
        for i, v in enumerate(bottom_gen):
            add(v, "gen_bottom", i)  # image of free_bottom[i] under the half-turn

        sp3 = SparseSystem3D(mesh.n_verts)
        for v in pin_order:
            # placeholder positions; the real b comes from boundary_b() at solve time
            sp3.add_fixed_constraint(v, mesh.points[v])
        A, _ = sp3.build()

        return cls(mesh=mesh, A=A, pin_order=np.asarray(pin_order), _roles=roles)

    # -------------------------------------------------------------------- parameters
    @property
    def n_free_left(self) -> int:
        return sum(1 for r, _ in self._roles if r == "free_left")

    @property
    def n_free_bottom(self) -> int:
        return sum(1 for r, _ in self._roles if r == "free_bottom")

    @property
    def n_free(self) -> int:
        return self.n_free_left + self.n_free_bottom

    def initial_free_points(self) -> Tensor:
        """``(n_free, 3)`` starting positions: the undeformed lune's own boundary."""
        idx = [
            self.pin_order[i]
            for i, (r, _) in enumerate(self._roles)
            if r in ("free_left", "free_bottom")
        ]
        return torch.as_tensor(self.mesh.points[idx], dtype=torch.float64)

    def rotations(self) -> tuple[Tensor, Tensor]:
        """``(R_z(2pi/k), R_half_turn)`` as float64 tensors."""
        rz = rotation_matrix([0.0, 0.0, 1.0], 2.0 * np.pi / self.mesh.k)
        rx = rotation_matrix([1.0, 0.0, 0.0], np.pi)
        return (
            torch.as_tensor(rz, dtype=torch.float64),
            torch.as_tensor(rx, dtype=torch.float64),
        )

    def boundary_b(self, free_points: Tensor) -> Tensor:
        r"""Assemble the pin right-hand side from the free parameters, differentiably.

        Args:
            free_points: ``(n_free, 3)``; normalised onto the sphere here, so callers may
                hold raw unconstrained parameters.

        The mates are *generated* (``right = R_z @ left`` etc.), which is what makes the
        tiling interlock by construction: gradients on a generated vertex flow back through
        the rotation onto its free source automatically via autograd.
        """
        if free_points.shape != (self.n_free, 3):
            raise ValueError(
                f"expected ({self.n_free}, 3) free points, got {tuple(free_points.shape)}"
            )
        p = free_points / free_points.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        left = p[: self.n_free_left]
        bottom = p[self.n_free_left :]
        rz, rx = self.rotations()

        fixed = {
            "pole": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
            "mid": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
            "corner_l": torch.as_tensor(
                self.mesh.points[self.mesh.bottom_left], dtype=torch.float64
            ),
            "corner_r": torch.as_tensor(
                self.mesh.points[self.mesh.bottom_right], dtype=torch.float64
            ),
        }

        rows = []
        for role, src in self._roles:
            if role in fixed:
                rows.append(fixed[role])
            elif role == "free_left":
                rows.append(left[src])
            elif role == "free_bottom":
                rows.append(bottom[src])
            elif role == "gen_right":
                rows.append(rz @ left[src])
            elif role == "gen_bottom":
                rows.append(rx @ bottom[src])
            else:  # pragma: no cover
                raise AssertionError(role)
        return torch.stack(rows).reshape(-1)

    def anchored_chains(self, p: Tensor) -> list[Tensor]:
        """The free chains with their pinned anchors, for the spacing/smoothness regs.

        Same contract as :meth:`BoundaryExplicitTriple.anchored_chains`, so the driver
        stays orbifold-agnostic.
        """
        nl = self.n_free_left
        pole = torch.tensor([0.0, 0.0, 1.0], dtype=p.dtype)
        mid = torch.tensor([1.0, 0.0, 0.0], dtype=p.dtype)
        corner = torch.as_tensor(self.mesh.points[self.mesh.bottom_left], dtype=p.dtype)
        return [
            torch.cat([pole[None], p[:nl], corner[None]]),
            torch.cat([corner[None], p[nl:], mid[None]]),
        ]

    def cotan_edge_weights(self) -> np.ndarray:
        r"""Clamped cotangent weights of the base mesh, aligned with ``mesh.edges``.

        The reference solver is built on these (``Solver.m``: ``L = cotmatrix(V,T)``, with
        negatives clamped to 1e-3). They matter more than "just" fidelity: with **uniform**
        weights the fixed-boundary solve of the undeformed lune already crushes the pole-fan
        faces to a **0.105** minimum area ratio -- inside the fold margin before optimisation
        even starts, which silently folded run B1 and made runs B2/B3 reject ~100% of steps.
        With cotangent weights the same solve reproduces the lune to min ratio **0.998**.
        """
        import igl

        L = igl.cotmatrix(self.mesh.points, self.mesh.faces.astype(np.int64))
        w = np.asarray(L[self.mesh.edges[:, 0], self.mesh.edges[:, 1]]).ravel()
        w[w < 1e-3] = 1e-3  # Solver.m's clamp for the occasional negative cotangent
        return w

    def tiler(self) -> SphericalTiler:
        return SphericalTiler.dihedral(self.mesh.k)

    def feasible_x0(self, b: np.ndarray) -> np.ndarray:
        """A start satisfying ``A x = b``: mesh points with boundary rows overwritten."""
        x0 = self.mesh.points.copy()
        x0[self.pin_order] = np.asarray(b, dtype=np.float64).reshape(-1, 3)
        return x0.reshape(-1)
