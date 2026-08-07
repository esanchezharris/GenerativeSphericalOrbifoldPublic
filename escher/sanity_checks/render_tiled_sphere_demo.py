"""Render a textured, tiled sphere and check the full gradient chain end to end.

Run from the repo root (needs a CUDA GPU)::

    python escher/sanity_checks/render_tiled_sphere_demo.py

Verifies the chain that score distillation depends on:

    edge weights -> [implicit solve] -> domain vertices -> [group tiling]
                 -> [rasteriser] -> image

If a gradient reaches the weights from a loss on the image, every link holds.
Writes ``output/tiled_sphere_render.png``.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from escher.OTE.core.spherical.differentiable import SphericalEmbedder
from escher.OTE.tilings_sphere import DihedralOrbifold
from escher.rendering.camera import orbit_views
from escher.rendering.render_sphere_nvdiffrast import build_tiled_sphere, render_tiled_sphere

OUTPUT = Path(os.environ.get("OUTPUT_DIR", "output"))
K = 4


def make_test_texture(size: int = 256) -> torch.Tensor:
    """A deliberately asymmetric, easily recognised figure.

    Asymmetry matters: a symmetric blob would look identical under the group's rotations, so
    it could not reveal a UV or orientation mistake in the tiling.
    """
    v, u = np.meshgrid(
        np.linspace(0, 1, size), np.linspace(0, 1, size), indexing="ij"
    )
    tex = np.ones((size, size, 3), dtype=np.float32) * 0.94

    # a thick diagonal stripe
    stripe = np.abs((u + v) - 1.0) < 0.14
    tex[stripe] = [0.20, 0.35, 0.70]

    # an off-centre disc, so orientation is unambiguous
    disc = ((u - 0.30) ** 2 + (v - 0.30) ** 2) < 0.018
    tex[disc] = [0.90, 0.45, 0.10]

    # a border, so tile seams are visible
    border = (u < 0.03) | (u > 0.97) | (v < 0.03) | (v > 0.97)
    tex[border] = [0.10, 0.10, 0.12]
    return torch.from_numpy(tex)


def main() -> None:
    device = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("this demo needs a CUDA GPU")

    orb = DihedralOrbifold.from_resolution(k=K, n_theta=24, n_phi=17)
    mesh = orb.mesh
    embedder = SphericalEmbedder(mesh.edges, orb.A, orb.b, orb.initial_guess())

    rng = np.random.default_rng(3)
    weights = torch.tensor(
        10.0 ** rng.uniform(-0.7, 0.7, size=len(mesh.edges)),
        dtype=torch.float64,
        requires_grad=True,
    )

    points = embedder(weights)
    print(f"solved: energy {embedder.last_result.energy:.6f}, "
          f"|Ax-b| {embedder.last_result.constraint_violation:.2e}, "
          f"stationarity {embedder.last_stationarity:.2e}")

    tiler = orb.tiler()
    sphere = build_tiled_sphere(
        points.to(device).float(), mesh.faces, mesh.uv, tiler
    )
    print(f"tiled: {sphere.n_tiles} tiles, {len(sphere.vertices)} verts, "
          f"{len(sphere.faces)} faces")

    texture = make_test_texture().to(device)
    mv = orbit_views(4, distance=3.0, elevation_deg=18.0)
    images, alpha = render_tiled_sphere(sphere, texture, mv=mv, image_size=512)
    print(f"rendered: {tuple(images.shape)}, coverage "
          f"{float(alpha.mean()):.3f} of frame")

    # ---- the gradient chain ------------------------------------------------------
    loss = images.mean()
    loss.backward()
    grad = weights.grad
    print()
    print("gradient chain image -> weights:")
    print(f"  finite      : {bool(torch.isfinite(grad).all())}")
    print(f"  nonzero     : {int((grad != 0).sum())}/{len(grad)} entries")
    print(f"  |grad|       : {float(grad.norm()):.6e}")
    assert torch.isfinite(grad).all() and grad.norm() > 0, "gradient did not reach the weights"

    # ---- figure ------------------------------------------------------------------
    OUTPUT.mkdir(parents=True, exist_ok=True)
    imgs = images.detach().cpu().numpy()
    a = alpha.detach().cpu().numpy()
    composited = imgs * a + 1.0 * (1 - a)  # white background

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    axes[0].imshow(make_test_texture().numpy())
    axes[0].set_title("shared texture\n(one tile's UV space)", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for i in range(4):
        axes[i + 1].imshow(np.clip(composited[i], 0, 1))
        axes[i + 1].set_axis_off()
        axes[i + 1].set_title(f"view {i + 1} of {sphere.n_tiles}-tile sphere", fontsize=10)

    fig.suptitle(
        f"Textured spherical orbifold tiling — $({K},2,2)$, $D_{{{K}}}$, "
        f"{sphere.n_tiles} tiles all sampling one texture",
        fontsize=13,
    )
    out = OUTPUT / "tiled_sphere_render.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
