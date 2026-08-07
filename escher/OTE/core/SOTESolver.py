r"""Spherical Orbifold Tutte Embedding solver.

Counterpart to :class:`~escher.OTE.core.OTESolver.OTESolver`: that one solves the *planar*
orbifold Tutte embedding as a single KKT linear system, this one solves the *spherical* case,
which is a nonlinear minimisation of the Karcher energy under the same kind of linear
orbifold constraints.

.. rubric:: History

The previous version of this file minimised the **Euclidean** Dirichlet energy
:math:`x^\top (W \otimes I_3) x`. That is the wrong functional: its gradient has a large
radial component (mean :math:`|\cos|` against the radial direction measured at 0.856 on the
reference problem), so minimising it drags vertices toward the origin instead of sliding them
along the sphere. It also carried a sign flip between its initialiser and its closure, built a
dense ``kron(w, I3)`` -- 58M entries at n=2554 -- and omitted the reference's two-stage
preconditioner schedule.

The correct functional is the **Karcher (geodesic)** energy, whose gradient is tangent to the
sphere (same measurement: :math:`1.4 \times 10^{-12}`). See
:mod:`escher.OTE.core.spherical.karcher`.

The implementation lives in :mod:`escher.OTE.core.spherical`; this module is the front door.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from escher.OTE.core.spherical.affine_space import AffineSpace
from escher.OTE.core.spherical.karcher import karcher_energy, karcher_energy_and_grad
from escher.OTE.core.spherical.lbfgs import ProjectedLBFGS
from escher.OTE.core.spherical.precond import PrecondFixed, PrecondIdentity
from escher.OTE.core.spherical.solver import (
    SphericalEmbeddingResult,
    make_karcher_objective,
    solve_spherical_embedding,
)

__all__ = [
    "AffineSpace",
    "PrecondFixed",
    "PrecondIdentity",
    "ProjectedLBFGS",
    "SOTESolver",
    "SphericalEmbeddingResult",
    "karcher_energy",
    "karcher_energy_and_grad",
    "make_karcher_objective",
    "solve_spherical_embedding",
]


class SOTESolver:
    """Solve a spherical orbifold Tutte embedding.

    Args:
        edges: ``(n_edges, 2)`` vertex index pairs of the cut mesh.
        laplacian: ``(n, n)`` weighted Laplacian (``+w`` off-diagonal, ``-sum(w)`` on the
            diagonal), used to build the preconditioner.
        A, b: the orbifold boundary equations.
        x0: ``(3n,)`` colStacked initial guess satisfying ``A x0 = b`` -- typically the planar
            :class:`~escher.OTE.core.OTESolver.OTESolver` result lifted onto the sphere.
    """

    def __init__(
        self,
        edges: np.ndarray,
        laplacian: sp.spmatrix,
        A: sp.spmatrix,
        b: np.ndarray,
        x0: np.ndarray,
        *,
        memory: int = 3,
        device: str = "cpu",
    ):
        self.edges = np.asarray(edges)
        self.laplacian = sp.csr_matrix(laplacian)
        self.A = sp.csr_matrix(A)
        self.b = np.asarray(b, dtype=np.float64).ravel()
        self.x0 = np.asarray(x0, dtype=np.float64).ravel()
        self.memory = memory
        self.device = device

        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must be (n_edges, 2), got {self.edges.shape}")
        n = self.laplacian.shape[0]
        if n * 3 != self.x0.size:
            raise ValueError(
                f"laplacian is {n}x{n} but x0 has {self.x0.size} entries "
                f"(expected {n * 3})"
            )

    def solve(
        self,
        weights: np.ndarray,
        *,
        x0: np.ndarray | None = None,
        tol_x: float = 1e-10,
        max_iter: int = 10_000,
    ) -> SphericalEmbeddingResult:
        """Embed with the given per-edge weights.

        Args:
            weights: ``(n_edges,)`` strictly positive weights aligned with ``edges``.
            x0: optional warm start overriding the constructor's. During SDS training the
                weights move only slightly between steps, so passing the previous solution
                cuts the iteration count sharply.
        """
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if weights.shape != (len(self.edges),):
            raise ValueError(
                f"weights must have one entry per edge ({len(self.edges)}), "
                f"got {weights.shape}"
            )
        if (weights <= 0).any():
            raise ValueError(
                "weights must be strictly positive for the Tutte embedding to be bijective"
            )

        return solve_spherical_embedding(
            edges=self.edges,
            weights=weights,
            laplacian=self.laplacian,
            A=self.A,
            b=self.b,
            x0=self.x0 if x0 is None else np.asarray(x0, dtype=np.float64).ravel(),
            memory=self.memory,
            tol_x=tol_x,
            max_iter=max_iter,
            device=self.device,
        )
