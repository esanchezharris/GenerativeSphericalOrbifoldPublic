"""Generate spherical orbifold tilings from scratch and render them.

No MATLAB data involved: the mesh, the orbifold constraints, the solve and the tiling are all
built here. Random edge weights stand in for what score-distillation sampling will drive in
Phase 2 -- each weight vector is one tile shape.

Run from the repo root::

    python escher/sanity_checks/demo_spherical_tiling.py

Writes ``output/spherical_tilings.png``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from escher.OTE.core.spherical.solver import laplacian_from_edges, solve_spherical_embedding
from escher.OTE.tilings_sphere import DihedralOrbifold
from escher.geometry.spherical_sanity_checks import (
    check_covers_sphere_once,
    count_flipped_faces,
)
from escher.rendering.render_sphere_matplotlib import draw_sphere_faces

OUTPUT = Path(os.environ.get("OUTPUT_DIR", "output"))

# (k, weight seed, log10 weight spread). seed=None means uniform weights.
CASES = [
    (4, None, 0.0),
    (3, 11, 1.0),
    (4, 3, 1.0),
    (6, 5, 1.0),
]


def solve_case(k: int, seed: int | None, spread: float):
    orb = DihedralOrbifold.from_resolution(k=k, n_theta=28, n_phi=19)
    edges = orb.mesh.edges

    if seed is None:
        weights = np.ones(len(edges))
        label = "uniform weights"
    else:
        rng = np.random.default_rng(seed)
        weights = 10.0 ** rng.uniform(-spread, spread, size=len(edges))
        label = f"random weights (seed {seed})"

    result = solve_spherical_embedding(
        edges=edges,
        weights=weights,
        laplacian=laplacian_from_edges(edges, weights, orb.mesh.n_verts),
        A=orb.A,
        b=orb.b,
        x0=orb.initial_guess(),
    )
    return orb, result, label


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(CASES), figsize=(4.0 * len(CASES), 5.0))
    fig.suptitle(
        "Spherical orbifold tilings generated from scratch  —  "
        "each sphere is one tile shape, repeated by its symmetry group",
        fontsize=13,
        y=1.02,
    )

    for ax, (k, seed, spread) in zip(np.atleast_1d(axes), CASES):
        orb, result, label = solve_case(k, seed, spread)
        tiler = orb.tiler()
        verts, faces, tile_index = tiler.tile_mesh(result.points, orb.mesh.faces)

        ok, message = check_covers_sphere_once(verts, faces)
        flipped = count_flipped_faces(result.points, orb.mesh.faces)
        print(
            f"k={k:<2} {label:<26} energy {result.energy:8.4f}  "
            f"|Ax-b| {result.constraint_violation:.1e}  flipped {flipped}  -> {message}"
        )

        draw_sphere_faces(ax, verts, faces, tile_index, elev=22, azim=48)
        ax.set_title(
            f"$({k},2,2)$ — $D_{{{k}}}$, {tiler.order} tiles\n{label}",
            fontsize=10,
        )

    out = OUTPUT / "spherical_tilings.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
