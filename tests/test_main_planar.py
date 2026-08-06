"""CPU validation of the planar pipeline hardening (escher/main.py).

Uses the same ``__new__``-based construction the spherical CPU tests rely on: geometry
and solver only, no renderer, no guidance. The planar Tutte solve is one dense linear
solve, so everything here runs offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

igl = pytest.importorskip("igl")

from escher.main import Escher, load_planar_args  # noqa: E402
from escher.geometry.sanity_checks import check_triangle_orientation  # noqa: E402


def tiny_escher(tmp_path: Path, **overrides) -> Escher:
    args = load_planar_args(OmegaConf.create({}))
    args.TILING_TYPE = "OrbifoldI"
    args.MESH_RESOLUTION = 10
    args.DEVICE = "cpu"
    args.OUTPUT_DIR = str(tmp_path)
    args.PROMPT = ["a gingerbread man"]  # init_mesh_and_solver reads len(PROMPT)
    for k, v in overrides.items():
        args[k] = v

    e = Escher.__new__(Escher)
    e.args = args
    e.device = torch.device("cpu")
    e.init_mesh_and_solver()
    return e


# --------------------------------------------------------------------- config layering
def test_cli_overrides_base_defaults():
    args = load_planar_args(OmegaConf.create({"SEED": 999}))
    assert args.SEED == 999
    assert "N_STEPS" in args  # base.yaml defaults survive


def test_conf_file_is_an_overlay_not_a_replacement():
    # sphere_texture.yaml restates only its deltas; loading it as a PLANAR overlay is
    # nonsense semantically but proves the mechanics: base keys it never mentions
    # (e.g. MESH_RESOLUTION) must survive the merge.
    args = load_planar_args(OmegaConf.create({"CONF_FILE": "configs/sphere_texture.yaml"}))
    assert "MESH_RESOLUTION" in args
    assert args.FREEZE_SHAPE_AFTER == 0  # and the overlay's own keys land


# ------------------------------------------------------------------------ weight map
def test_map_weights_range_and_uniform_start(tmp_path):
    e = tiny_escher(tmp_path)
    with torch.no_grad():
        e.W.zero_()
    w = e.map_weights()
    assert torch.allclose(w, torch.full_like(w, 0.5))  # expit(0) -> mid-range

    r = (1 - e.args.W_RANGE) / 2
    with torch.no_grad():
        e.W.fill_(40.0)
    assert e.map_weights().max() <= 1 - r + 1e-6
    with torch.no_grad():
        e.W.fill_(-40.0)
    assert e.map_weights().min() >= r - 1e-6


def test_uniform_weights_solve_is_fold_free(tmp_path):
    e = tiny_escher(tmp_path)
    with torch.no_grad():
        e.W.zero_()
    mapped, _, success = e.solver.solve(e.map_weights())
    assert success
    check_triangle_orientation(mapped, e.faces)  # raises on any flipped triangle

    loop = igl.boundary_loop(e.faces_npy)
    assert np.array_equal(loop, e.bdry), "stored boundary loop must match igl's"


# ----------------------------------------------------------------------- W_INIT_PATH
def test_w_init_path_round_trip(tmp_path):
    donor = tiny_escher(tmp_path)
    with torch.no_grad():
        donor.W.normal_(generator=torch.Generator().manual_seed(7))
    ckpt = tmp_path / "shape.pt"
    torch.save({"iteration": 5, "W": donor.W.detach().clone()}, ckpt)

    e = tiny_escher(tmp_path, W_INIT_PATH=str(ckpt))
    assert torch.allclose(e.W.detach(), donor.W.detach())


def test_w_init_path_rejects_mismatched_mesh(tmp_path):
    donor = tiny_escher(tmp_path)
    ckpt = tmp_path / "shape.pt"
    torch.save({"iteration": 5, "W": donor.W.detach().clone()}, ckpt)
    with pytest.raises(ValueError, match="must match"):
        tiny_escher(tmp_path, W_INIT_PATH=str(ckpt), MESH_RESOLUTION=12)


# ------------------------------------------------------------------- texture init
def test_texture_init_color(tmp_path):
    e = tiny_escher(tmp_path, TEXTURE_INIT_COLOR=[0.76, 0.60, 0.42])
    e.init_optimizer()
    tan = torch.tensor([0.76, 0.60, 0.42])
    assert torch.allclose(e.color_parameters.detach()[0, 0, 0], tan)
    assert torch.allclose(e.color_parameters.detach()[0, -1, -1], tan)
