"""Validate the three-cone boundary-explicit parameterization end to end on CPU.

Same gates as the dihedral suite (test_boundary_explicit.py), on the (2,3,4) octahedral
kite: structure, exact group images, validity of the cotangent solve, the
finite-difference gradient gate through solve + adjoint + group maps, and the 24-tile
certificate.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.OTE.core.spherical.differentiable import BoundaryEmbedder
from escher.OTE.tilings_sphere.boundary_explicit_triple import BoundaryExplicitTriple
from escher.geometry.spherical_sanity_checks import (
    check_covers_sphere_once,
    count_flipped_faces,
)


@pytest.fixture(scope="module")
def small():
    orb = BoundaryExplicitTriple.from_resolution((2, 3, 4), n=6)
    emb = BoundaryEmbedder(
        orb.mesh.edges, orb.A, orb.pin_order, orb.mesh.points,
        weights=orb.cotan_edge_weights(), warm_start=False,
    )
    return orb, emb


def wiggled(orb, scale=0.04, seed=0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    P = orb.initial_free_points().numpy()
    return torch.as_tensor(P + scale * rng.standard_normal(P.shape), dtype=torch.float64)


# ------------------------------------------------------------------------- structure
def test_pins_cover_the_whole_boundary(small):
    orb, _ = small
    boundary = set(np.concatenate(orb.mesh.boundary_chains).tolist())
    assert set(orb.pin_order.tolist()) == boundary
    assert orb.A.shape[0] == 3 * len(boundary)


def test_octahedral_group_has_24_tiles(small):
    orb, _ = small
    assert orb.tiler().order == 24


def test_initial_b_reproduces_the_kite_boundary(small):
    orb, _ = small
    b = orb.boundary_b(orb.initial_free_points()).numpy().reshape(-1, 3)
    assert np.abs(b - orb.mesh.points[orb.pin_order]).max() < 1e-12


def test_generated_mates_are_exact_group_images(small):
    """Interlocking by construction: the right chains ARE the cone-rotation images."""
    orb, emb = small
    with torch.no_grad():
        pts = emb(orb.boundary_b(wiggled(orb, seed=1))).numpy()
    m = orb.mesh
    assert np.abs(pts[m.left1] @ orb.R1.T - pts[m.right1]).max() < 1e-9
    assert np.abs(pts[m.left2] @ orb.R2.T - pts[m.right2]).max() < 1e-9


# ------------------------------------------------------------------------ validity
def test_initial_solve_stays_near_the_kite(small):
    orb, emb = small
    with torch.no_grad():
        pts = emb(orb.boundary_b(orb.initial_free_points())).numpy()
    assert count_flipped_faces(pts, orb.mesh.faces) == 0
    assert np.abs(pts - orb.mesh.points).max() < 0.15


def test_cotan_weights_keep_init_areas_healthy():
    """The uniform-weights trap bit the lune at min ratio 0.105; pin the kite's health."""
    from escher.OTE.core.spherical.regularizers import spherical_face_areas

    orb = BoundaryExplicitTriple.from_resolution((2, 3, 4), n=10)
    ref = spherical_face_areas(
        torch.as_tensor(orb.mesh.points, dtype=torch.float64),
        torch.as_tensor(orb.mesh.faces, dtype=torch.long),
    ).detach()
    emb = BoundaryEmbedder(
        orb.mesh.edges, orb.A, orb.pin_order, orb.mesh.points,
        weights=orb.cotan_edge_weights(), warm_start=False,
    )
    with torch.no_grad():
        pts = emb(orb.boundary_b(orb.initial_free_points()))
    areas = spherical_face_areas(pts, torch.as_tensor(orb.mesh.faces, dtype=torch.long))
    assert float((areas / ref).min()) > 0.8


def test_wiggled_boundary_yields_a_valid_24_tile_sphere(small):
    orb, emb = small
    with torch.no_grad():
        pts = emb(orb.boundary_b(wiggled(orb, seed=2))).numpy()
    assert count_flipped_faces(pts, orb.mesh.faces) == 0
    verts, faces, _ = orb.tiler().tile_mesh(pts, orb.mesh.faces)
    ok, msg = check_covers_sphere_once(verts, faces)
    assert ok, msg


def test_cone_points_stay_pinned_regardless_of_parameters(small):
    orb, emb = small
    with torch.no_grad():
        pts = emb(orb.boundary_b(wiggled(orb, seed=3))).numpy()
    m = orb.mesh
    for idx in (m.cone1, m.cone2a, m.cone2b, m.cone3):
        assert np.allclose(pts[idx], m.points[idx], atol=1e-9)


# ------------------------------------------------------------------ the actual gate
def test_boundary_gradient_matches_finite_differences(small):
    """dL/dP through solve + adjoint + group maps vs central differences; same
    discipline as ever -- tight solver tolerance, generous h."""
    orb, _ = small
    emb = BoundaryEmbedder(
        orb.mesh.edges, orb.A, orb.pin_order, orb.mesh.points,
        weights=orb.cotan_edge_weights(), warm_start=False, tol_x=1e-14,
    )
    rng = np.random.default_rng(4)
    P0 = wiggled(orb, scale=0.03, seed=4)
    target = torch.as_tensor(
        rng.standard_normal((orb.mesh.n_verts, 3)), dtype=torch.float64
    )

    def loss_of(P):
        return (emb(orb.boundary_b(P)) * target).sum()

    P = P0.clone().requires_grad_(True)
    loss_of(P).backward()
    analytic = P.grad.clone()
    assert torch.isfinite(analytic).all()
    assert analytic.abs().max() > 1e-8

    h = 1e-4
    rel_errors = []
    picks = [(i, k) for i in rng.choice(orb.n_free, 6, replace=False) for k in range(3)]
    for i, k in picks:
        plus, minus = P0.clone(), P0.clone()
        plus[i, k] += h
        minus[i, k] -= h
        fd = (loss_of(plus) - loss_of(minus)).item() / (2 * h)
        a = analytic[i, k].item()
        rel_errors.append(abs(fd - a) / max(abs(a), 1e-9))
        assert fd == pytest.approx(a, rel=1e-3, abs=1e-8), f"free point {i} axis {k}"
    assert float(np.median(rel_errors)) < 1e-5


def test_gradient_reaches_free_points_through_generated_mates(small):
    orb, emb = small
    P = orb.initial_free_points().clone().requires_grad_(True)
    pts = emb(orb.boundary_b(P))
    loss = pts[orb.mesh.right1[1:-1]].sum()  # touches ONLY a generated chain
    loss.backward()
    assert P.grad[: orb.n_free_1].abs().sum() > 0, "gradient failed to flow through R1"


# ------------------------------------------------------------------ driver plumbing
def test_sphere_escher_runs_the_kite_domain(tmp_path):
    """The ORBIFOLD_CONES branch: geometry, chains regularizer, freeze -- no diffusion."""
    from pathlib import Path

    from omegaconf import OmegaConf

    from escher.main_sphere import PATH, SphereEscher

    a = OmegaConf.load(PATH / "configs/sphere.yaml")
    a.PARAM_MODE = "boundary"
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
    reg = e.boundary_chain_regularizers()
    (pts.sum() + reg).backward()
    assert e.shape_param.grad is not None and e.shape_param.grad.abs().sum() > 0
    assert e.boundary_ratio(pts) == pytest.approx(1.0, abs=1e-6)

    _, flips, reverted = e.ensure_valid_shape()
    assert flips == 0 and not reverted

    e.apply_shape_freeze()
    assert e.optimizer.param_groups[0]["lr"] == 0.0


def test_revert_clears_adam_state(tmp_path):
    """The revert deadlock: Adam's exp_avg kept pointing into the fold, so every step
    after the first revert proposed the same fold (394 consecutive reverts measured).
    Reverting must also drop the shape parameter's optimizer state."""
    from pathlib import Path

    from omegaconf import OmegaConf

    from escher.main_sphere import PATH, SphereEscher

    a = OmegaConf.load(PATH / "configs/sphere.yaml")
    a.PARAM_MODE = "boundary"
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

    # one real optimizer step so Adam has state for P
    e.solve_points().sum().backward()
    e.optimizer.step()
    assert len(e.optimizer.state.get(e.shape_param, {})) > 0

    e.reset_shape_optimizer_state()
    assert len(e.optimizer.state.get(e.shape_param, {})) == 0

    # and the texture group's state must survive the reset
    e.optimizer.zero_grad()
    (e.texture.sum() * 0.001).backward()
    e.optimizer.step()
    assert len(e.optimizer.state.get(e.texture, {})) > 0
    e.reset_shape_optimizer_state()
    assert len(e.optimizer.state.get(e.texture, {})) > 0
