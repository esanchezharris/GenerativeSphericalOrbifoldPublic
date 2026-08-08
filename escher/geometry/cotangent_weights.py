r"""Clamped cotangent edge weights -- the configuration the reference solver is built on.

``@Solver/Solver.m`` constructs its system from ``L = cotmatrix(V,T)`` and clamps the
occasional negative off-diagonal to ``1e-3``. Those weights matter beyond fidelity: they
are the weights at which the base mesh is (approximately) its own harmonic fixed point,
so a Tutte solve using them reproduces the fundamental domain that was designed rather
than a distortion of it.

Kept here, shared, because it was previously a private method on one orbifold class while
the mode the pipeline actually ships used uniform weights instead -- the two conventions
drifting apart is exactly the bug this module exists to prevent.
"""

from __future__ import annotations

import numpy as np

__all__ = ["cotan_edge_weights"]


def cotan_edge_weights(
    points: np.ndarray, faces: np.ndarray, edges: np.ndarray, clamp: float = 1e-3
) -> np.ndarray:
    """``(n_edges,)`` positive weights aligned with ``edges``.

    Note the sign convention: igl's ``cotmatrix`` already carries the POSITIVE cotangent
    weights off-diagonal (the negative sum sits on the diagonal). Negating them makes
    every weight negative, the clamp then flattens all of them to one value, and because
    the Karcher energy is scale-invariant in the weights that is indistinguishable from
    uniform -- a silent no-op rather than an error.
    """
    import igl

    L = igl.cotmatrix(np.asarray(points, dtype=np.float64), np.asarray(faces, dtype=np.int64))
    w = np.asarray(L[edges[:, 0], edges[:, 1]]).ravel().astype(np.float64)
    w[w < clamp] = clamp
    return w
