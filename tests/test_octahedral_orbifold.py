"""Validate the weights-mode octahedral constraint system (tilings_sphere/octahedral.py).

Same gates as the dihedral weights mode (test_differentiable.py) on the (2,3,4) kite:
exact feasibility at init, group-image exactness after a non-uniform solve, the
finite-difference gradient gate through the implicit solve, fold-freeness across two
orders of weight magnitude, and the 24-tile certificate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.OTE.core.spherical.differentiable import SphericalEmbedder
from escher.OTE.tilings_sphere import OctahedralOrbifold
from escher.geometry.spherical_sanity_checks import (
    check_covers_sphere_once,
    count_flipped_faces,
)


@pytest.fixture(scope="module")
def orb():
    return OctahedralOrbifold.from_resolution((2, 3, 4), n=6)


def make_embedder(orb, **kw):
    return SphericalEmbedder(orb.mesh.edges, orb.A, orb.b, orb.initial_guess(), **kw)


def random_weights(orb, span=0.3, seed=1):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(
        10.0 ** rng.uniform(-span, span, size=len(orb.mesh.edges)), dtype=torch.float64
    )


# ------------------------------------------------------------------------- structure
def test_init_is_exactly_feasible(orb):
    x0 = orb.mesh.points.reshape(-1)
    assert np.abs(orb.A @ x0 - orb.b).max() < 1e-9, "the undeformed kite satisfies A x = b"
    xp = orb.initial_guess()
    assert np.abs(orb.A @ xp - orb.b).max() < 1e-9


def test_octahedral_group_has_24_tiles(orb):
    assert orb.tiler().order == 24


# -------------------------------------------------------------------------- solving
def test_solved_chains_are_exact_group_images_and_cones_pinned(orb):
    emb = make_embedder(orb, warm_start=False)
    with torch.no_grad():
        pts = emb(random_weights(orb, seed=2)).numpy()
    m = orb.mesh
    assert np.abs(pts[m.left1] @ orb.R1.T - pts[m.right1]).max() < 1e-6
    assert np.abs(pts[m.left2] @ orb.R2.T - pts[m.right2]).max() < 1e-6
    for idx in (m.cone1, m.cone2a, m.cone2b, m.cone3):
        assert np.allclose(pts[idx], m.points[idx], atol=1e-6)


def test_fold_free_across_two_orders_of_weight_magnitude(orb):
    """The GEM premise on the kite: extreme weights, no folds."""
    emb = make_embedder(orb, warm_start=False)
    for seed in range(3):
        with torch.no_grad():
            pts = emb(random_weights(orb, span=1.0, seed=seed)).numpy()
        assert count_flipped_faces(pts, orb.mesh.faces) == 0, f"seed {seed}"


def test_24_tile_certificate(orb):
    emb = make_embedder(orb, warm_start=False)
    with torch.no_grad():
        pts = emb(random_weights(orb, seed=3)).numpy()
    verts, faces, _ = orb.tiler().tile_mesh(pts, orb.mesh.faces)
    ok, msg = check_covers_sphere_once(verts, faces)
    assert ok, msg


# ------------------------------------------------------------------ the actual gate
def test_implicit_gradient_matches_finite_differences(orb):
    """dL/dw through the implicit Karcher solve on the kite -- the same discipline as
    the dihedral gate: tight solver tolerance, generous h."""
    emb = make_embedder(orb, warm_start=False, tol_x=1e-14)
    n_edges = len(orb.mesh.edges)

    rng = np.random.default_rng(4)
    w0 = torch.as_tensor(
        10.0 ** rng.uniform(-0.3, 0.3, size=n_edges), dtype=torch.float64
    )
    target = torch.as_tensor(
        rng.standard_normal((orb.mesh.n_verts, 3)), dtype=torch.float64
    )

    def loss_of(w):
        return (emb(w) * target).sum()

    w = w0.clone().requires_grad_(True)
    loss_of(w).backward()
    analytic = w.grad.clone()
    assert torch.isfinite(analytic).all()
    assert analytic.abs().max() > 1e-8

    h = 1e-4
    rel_errors = []
    for e in rng.choice(n_edges, size=12, replace=False):
        plus, minus = w0.clone(), w0.clone()
        plus[e] += h
        minus[e] -= h
        fd = (loss_of(plus) - loss_of(minus)).item() / (2 * h)
        a = analytic[e].item()
        rel_errors.append(abs(fd - a) / max(abs(a), 1e-12))
        assert fd == pytest.approx(a, rel=1e-3, abs=1e-9), f"edge {e}"
    assert float(np.median(rel_errors)) < 1e-5


# ------------------------------------------------------------------ driver plumbing
def test_sphere_escher_runs_weights_mode_on_the_kite(tmp_path):
    from pathlib import Path

    from omegaconf import OmegaConf

    from escher.main_sphere import PATH, SphereEscher

    a = OmegaConf.load(PATH / "configs/sphere.yaml")
    a.PARAM_MODE = "weights"
    a.ORBIFOLD_CONES = [2, 3, 4]
    a.KITE_N = 6
    a.DEVICE = "cpu"
    a.OUTPUT_DIR = str(tmp_path)

    e = SphereEscher.__new__(SphereEscher)
    e.args = a
    e.device = torch.device("cpu")
    e.output_dir = Path(tmp_path)
    e._init_geometry()
    e._init_parameters()

    assert e.tiler.order == 24
    pts = e.solve_points()
    pts.sum().backward()
    assert e.shape_param is e.W
    assert e.W.grad is not None and e.W.grad.abs().sum() > 0

    e.apply_shape_freeze()
    assert e.optimizer.param_groups[0]["lr"] == 0.0
