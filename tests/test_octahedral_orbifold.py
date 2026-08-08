"""Validate the weights-mode octahedral constraint system (tilings_sphere/octahedral.py).

Same gates as the dihedral weights mode (test_differentiable.py) on the (2,3,4) kite:
exact feasibility at init, group-image exactness after a non-uniform solve, the
finite-difference gradient gate through the implicit solve, fold-freeness across two
orders of weight magnitude, and the 24-tile certificate.

The structural gates additionally run over every spherical triple -- (2,3,3), (2,3,4),
(2,3,5) -- since the class is octahedral only in name.
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


# ------------------------------------------------------- the other spherical triples
# The class is named for (2,3,4) but nothing in it is octahedral-specific, so the
# tetrahedral and icosahedral groups should pass the same structural gates unchanged.
# (2,3,5) is the one that matters in practice: 60 tiles is the figure density of the
# concept image, where (4,2,2)'s 8 tiles read as continents rather than figures.
@pytest.mark.parametrize("triple, order", [((2, 3, 3), 12), ((2, 3, 4), 24), ((2, 3, 5), 60)])
def test_every_spherical_triple_tiles_its_group_exactly_once(triple, order):
    orb = OctahedralOrbifold.from_resolution(triple, n=6)
    assert np.abs(orb.A @ orb.mesh.points.reshape(-1) - orb.b).max() < 1e-9
    assert np.abs(orb.A @ orb.initial_guess() - orb.b).max() < 1e-9

    tiler = orb.tiler()
    assert tiler.order == order

    emb = make_embedder(orb, warm_start=False)
    with torch.no_grad():
        pts = emb(random_weights(orb, seed=3)).numpy()
    m = orb.mesh
    assert np.abs(pts[m.left1] @ orb.R1.T - pts[m.right1]).max() < 1e-6
    assert np.abs(pts[m.left2] @ orb.R2.T - pts[m.right2]).max() < 1e-6

    for seed in range(3):
        with torch.no_grad():
            extreme = emb(random_weights(orb, span=1.0, seed=seed)).numpy()
        assert count_flipped_faces(extreme, m.faces) == 0, f"{triple} seed {seed}"

    verts, faces, _ = tiler.tile_mesh(pts, m.faces)
    ok, msg = check_covers_sphere_once(verts, faces)
    assert ok, f"{triple}: {msg}"


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


@pytest.mark.parametrize("cones", [None, [2, 3, 4], [2, 3, 5]])
def test_cotangent_relative_weights_start_at_the_designed_domain(tmp_path, cones):
    """W = 0 must reproduce the fundamental domain we designed, not a distortion of it.

    With the absolute (upstream) convention W = 0 means uniform weights, and the solve
    drifts away from the base mesh: measured 0.143 on (2,3,4) with per-face areas spread
    82x, which is also the texel-density range the texture phase has to paint through.
    Cotangent-relative weights make W = 0 the reference configuration exactly.
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    from escher.OTE.core.spherical.regularizers import spherical_face_areas
    from escher.main_sphere import PATH, SphereEscher

    def build(relative):
        a = OmegaConf.load(PATH / "configs/sphere.yaml")
        a.PARAM_MODE = "weights"
        a.ORBIFOLD_CONES = cones
        a.KITE_N = 6
        a.MESH_N_PHI = 9
        a.MESH_N_THETA = 10
        a.DEVICE = "cpu"
        a.OUTPUT_DIR = str(tmp_path)
        a.COTANGENT_RELATIVE_WEIGHTS = relative
        e = SphereEscher.__new__(SphereEscher)
        e.args = a
        e.device = torch.device("cpu")
        e.output_dir = Path(tmp_path)
        e._init_geometry()
        e._init_parameters()
        with torch.no_grad():
            return e, e.solve_points().numpy()

    e_rel, pts_rel = build(True)
    _, pts_abs = build(False)
    design = e_rel.mesh.points

    # Near-, not exact-, fixed point: cotangent weights make a mesh harmonic for the
    # EUCLIDEAN Dirichlet energy, while this solve minimizes the geodesic Karcher energy.
    # The residual is the curvature difference -- ~4e-5 on a unit sphere, against 0.14 for
    # uniform weights on the same domain, so the ratio is what carries the claim.
    drift_rel = float(np.abs(pts_rel - design).max())
    drift_abs = float(np.abs(pts_abs - design).max())
    # 5e-3 accommodates the coarse test meshes; the residual shrinks under refinement
    # (full-resolution lune 8e-4, kites 4e-5). The ratio below is what carries the claim.
    assert drift_rel < 5e-3, f"cotangent-relative should ~reproduce the design, got {drift_rel}"
    assert drift_abs > 50 * drift_rel, (
        f"uniform weights should drift far more: {drift_abs:.4f} vs {drift_rel:.2e}"
    )

    faces = torch.as_tensor(e_rel.mesh.faces, dtype=torch.long)
    spread = lambda p: float(  # noqa: E731
        (lambda a: a.max() / a.min())(
            spherical_face_areas(torch.as_tensor(p, dtype=torch.float64), faces)
        )
    )
    assert spread(pts_rel) < spread(pts_abs), "relative weights must not be more distorted"

    # A global scale of the weights is a no-op for the Karcher energy, so the mapping
    # must be about RATIOS -- this is why the fixed 39:1 span of the absolute map could
    # not represent cotangent weights needing 107:1 on (2,3,4).
    with torch.no_grad():
        w0 = e_rel.edge_weights()
    assert float(w0.min()) > 0.0
    assert torch.allclose(w0, e_rel._cotan_weights, rtol=1e-9)


@pytest.mark.parametrize("cones", [None, [2, 3, 4], [2, 3, 5]])
def test_area_barrier_is_silent_at_the_starting_state(tmp_path, cones):
    """The drift barrier must charge nothing before anything has drifted.

    ``ref_areas`` used to come from the raw fundamental-domain mesh, but the run starts
    at the uniform-weight SOLVE, and those are different configurations -- on (2,3,5) the
    solve rescales faces by 0.054x-9.27x. The barrier therefore opened with a large
    phantom penalty (measured: 2856 of 7535 total loss on the shipped (2,3,4) carve; on
    (2,3,5) it was 6960 against a mask term of 6080 and the carve could not move at all,
    IoU pinned at 0.6169 with perimeter exactly 1.0000x).
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    from escher.OTE.core.spherical.regularizers import area_tail_loss, equal_area_loss
    from escher.main_sphere import PATH, SphereEscher

    a = OmegaConf.load(PATH / "configs/sphere.yaml")
    a.PARAM_MODE = "weights"
    a.ORBIFOLD_CONES = cones
    a.KITE_N = 6
    a.MESH_N_PHI = 9
    a.MESH_N_THETA = 10
    a.DEVICE = "cpu"
    a.OUTPUT_DIR = str(tmp_path)

    e = SphereEscher.__new__(SphereEscher)
    e.args = a
    e.device = torch.device("cpu")
    e.output_dir = Path(tmp_path)
    e._init_geometry()
    e._init_parameters()

    with torch.no_grad():
        pts = e.solve_points()
    assert float(equal_area_loss(pts, e.faces_t, e.ref_areas)) < 1e-12
    assert float(area_tail_loss(pts, e.faces_t, e.ref_areas, fraction=0.05)) < 1e-12
