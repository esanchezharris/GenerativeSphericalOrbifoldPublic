r"""Assembling the orbifold constraint system :math:`Ax = b` in 3D.

The spherical analogue of :class:`~escher.OTE.core.SparseSystem.SparseSystem`, which builds
the same thing for planar tilings in 2D.

Two constraint kinds cover every spherical orbifold, matching what the reference dump
contains (51 rotation pairs plus 4 position pins, 165 rows total):

- **rotation** -- :math:`R u_i - u_j = 0`, three rows per vertex pair, identifying two
  boundary arcs under a symmetry;
- **fixed** -- :math:`u_i = p`, three rows, pinning a cone point.

Coordinates are colStacked, so vertex ``v``'s components live at ``3v``, ``3v+1``, ``3v+2``
-- the same layout as ``x0.mat``.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["SparseSystem3D"]


class SparseSystem3D:
    """Accumulate constraint rows, then emit ``(A, b)``.

    Args:
        n_vertices: vertex count of the mesh being constrained.
        drop_tol: coefficients with magnitude below this are not stored. The MATLAB dump
            keeps structural zeros (its rows have 4 nonzeros where 2 carry information);
            dropping them changes nothing mathematically and keeps ``A`` tighter.
    """

    def __init__(self, n_vertices: int, drop_tol: float = 1e-14):
        self.n_vertices = n_vertices
        self.drop_tol = drop_tol
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._vals: list[float] = []
        self._b: list[float] = []
        self._n_rows = 0

    def _add_row(self, cols, vals, rhs: float) -> None:
        kept = [(c, v) for c, v in zip(cols, vals) if abs(v) > self.drop_tol]
        if not kept:
            if abs(rhs) > self.drop_tol:
                raise ValueError(f"inconsistent empty constraint row with rhs {rhs}")
            return  # 0 == 0 carries no information and would make A rank-deficient
        for c, v in kept:
            self._rows.append(self._n_rows)
            self._cols.append(int(c))
            self._vals.append(float(v))
        self._b.append(float(rhs))
        self._n_rows += 1

    @property
    def n_rows(self) -> int:
        return self._n_rows

    def add_rotation_constraints(
        self, source: np.ndarray, target: np.ndarray, R: np.ndarray
    ) -> None:
        r"""Require :math:`R\,u_{source_i} = u_{target_i}` for each ``i``.

        Args:
            source, target: equal-length vertex index arrays, corresponding elementwise.
            R: ``(3, 3)`` rotation.
        """
        source = np.asarray(source, dtype=np.int64).ravel()
        target = np.asarray(target, dtype=np.int64).ravel()
        if source.shape != target.shape:
            raise ValueError(
                f"source and target must be the same length, got {source.shape} and "
                f"{target.shape}"
            )
        if np.intersect1d(source, target).size:
            raise ValueError(
                "a vertex appears in both source and target of the same rotation "
                "constraint, which would over-determine it; exclude shared endpoints"
            )
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"R must be (3, 3), got {R.shape}")

        for si, ti in zip(source.tolist(), target.tolist()):
            for axis in range(3):
                cols = [3 * si, 3 * si + 1, 3 * si + 2, 3 * ti + axis]
                vals = [R[axis, 0], R[axis, 1], R[axis, 2], -1.0]
                self._add_row(cols, vals, 0.0)

    def add_fixed_constraint(self, vertex: int, position) -> None:
        """Pin ``vertex`` to ``position``."""
        position = np.asarray(position, dtype=np.float64).ravel()
        if position.shape != (3,):
            raise ValueError(f"position must have 3 components, got {position.shape}")
        for axis in range(3):
            self._add_row([3 * int(vertex) + axis], [1.0], float(position[axis]))

    def build(self) -> tuple[sp.csr_matrix, np.ndarray]:
        """Emit ``(A, b)`` with ``A`` of shape ``(n_rows, 3 * n_vertices)``."""
        A = sp.csr_matrix(
            (self._vals, (self._rows, self._cols)),
            shape=(self._n_rows, 3 * self.n_vertices),
        )
        return A, np.asarray(self._b, dtype=np.float64)
