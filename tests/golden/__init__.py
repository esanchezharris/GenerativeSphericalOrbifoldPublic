"""Loader for the MATLAB reference dumps.

See README.md in this directory for what each file holds and which are trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

GOLDEN_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Golden:
    """One dump of the MATLAB reference solver's state.

    Coordinate vectors are ``colStack``ed as ``[x1 y1 z1 x2 y2 z2 ...]``; use
    :func:`as_points` / :func:`as_flat` to convert.
    """

    A: sp.csr_matrix  #: (165, 3n) orbifold boundary equations
    b: np.ndarray  #: (165,) right-hand side
    N: sp.csr_matrix  #: (3n, 3n-165) orthonormal sparse null-space basis of A
    x0: np.ndarray  #: (3n,) lifted initial guess, on the unit sphere
    wmat: sp.csr_matrix  #: (n, n) Laplacian, negative cotan weights clamped to 1e-3
    L: sp.csr_matrix  #: (n, n) raw cotangent Laplacian
    V: np.ndarray  #: (n_uncut, 3) source mesh vertices, uncut
    F: np.ndarray  #: (n_faces, 3) faces, converted to 0-based

    @property
    def n_verts(self) -> int:
        """Vertex count of the *cut* mesh, which is what the solver operates on."""
        return self.wmat.shape[0]

    @property
    def edges(self) -> np.ndarray:
        """(n_edges, 2) unique undirected edges ``i < j`` taken from the Laplacian pattern."""
        coo = sp.triu(self.wmat, k=1).tocoo()
        return np.stack([coo.row, coo.col], axis=1)

    @property
    def edge_weights(self) -> np.ndarray:
        """(n_edges,) positive weight per edge, aligned with :attr:`edges`.

        The Laplacian is stored with ``+w`` off-diagonal and ``-sum(w)`` on the diagonal,
        so off-diagonal entries are already the weights.
        """
        return sp.triu(self.wmat, k=1).tocoo().data.copy()


def as_points(x: np.ndarray) -> np.ndarray:
    """colStacked ``(3n,)`` -> ``(n, 3)``."""
    return np.asarray(x).reshape(-1, 3)


def as_flat(p: np.ndarray) -> np.ndarray:
    """``(n, 3)`` -> colStacked ``(3n,)``."""
    return np.asarray(p).reshape(-1)


@lru_cache(maxsize=1)
def load_golden() -> Golden:
    """Load and normalise the reference dumps. Cached; the result is treated as immutable."""

    def mat(name: str, key: str):
        return scipy.io.loadmat(str(GOLDEN_DIR / name))[key]

    return Golden(
        A=mat("A.mat", "A").tocsr(),
        b=np.asarray(mat("b.mat", "b")).ravel(),
        N=mat("N.mat", "N").tocsr(),
        x0=np.asarray(mat("x0.mat", "x0")).ravel(),
        wmat=mat("wMat.mat", "wmat").tocsr(),
        L=mat("L.mat", "L").tocsr(),
        V=np.asarray(mat("V.mat", "V"), dtype=np.float64),
        # MATLAB faces are 1-based; store 0-based so they index numpy arrays directly.
        F=np.asarray(mat("F.mat", "F"), dtype=np.int64) - 1,
    )
