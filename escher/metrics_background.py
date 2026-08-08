"""Measure how much of the rendered sphere reads as background ("filler").

The render gate for the zero-filler work: over N orbit views of the tinted
deliverable, count sphere-covered pixels that look like background -- near-white
low-chroma (what the SDS model paints for negative space) or near the texture's
flat init color (gutter bleed through the mip chain) -- and report the fraction.

Tint is applied because the deliverable is tinted, and the hue rotation is exactly
what achromatic pixels defeat: they are fixed points of a rotation about the RGB
gray axis (escher/rendering/palette.py), so background paint renders identically in
every tile and reads as one continuous field crossing tile boundaries. The metric
must see what the viewer sees. Shading is left OFF: it multiplies every pixel by a
lambert term, which would push genuine background below the luminance threshold on
the dark side of the sphere.

Usage (inside the WSL venv)::

    python escher/metrics_background.py output/texF_fish/checkpoint.pt
    python escher/metrics_background.py <ckpt> N_VIEWS=30 TINT=0

Writes ``background_metrics.json`` and ``background_audit.png`` beside the
checkpoint. The audit PNG highlights background-like pixels in magenta -- the
false-color view is what makes a 2% number actionable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from escher.render_final import load_run
from escher.rendering.camera import orbit_views
from escher.rendering.render_sphere_nvdiffrast import (
    build_tiled_sphere,
    render_tiled_sphere,
)

# Background-likeness thresholds. "White-like": low chroma AND bright -- the color
# family SDS deposits for negative space, and the one the tint rotation cannot
# recolor. "Init-like": within an L2 ball of the flat texture init -- texels no
# triangle ever painted, leaking into minified views through the mips.
CHROMA_MAX = 0.10
LUMA_MIN = 0.85
INIT_L2 = 0.12
DEFAULT_INIT = (0.5, 0.5, 0.5)  # texture init when TEXTURE_INIT_COLOR is unset


def measure(
    checkpoint: Path,
    n_views: int = 30,
    tint: bool = True,
    audit_views: int = 4,
    gutter: bool = True,
) -> dict:
    escher, iteration = load_run(checkpoint)

    # Measure what the deliverable renders: render_final gutter-fills the
    # unsampled texels by default, so the gate does too (GUTTER=0 to measure the
    # raw checkpoint -- that is what the 3.078% shipped baseline was).
    if gutter:
        from escher.rendering.texture_mask import gutter_fill, uv_valid_mask

        valid = uv_valid_mask(
            escher.mesh.uv, escher.mesh.faces, int(escher.args.TEXTURE_RESOLUTION)
        )
        with torch.no_grad():
            filled = gutter_fill(escher.texture.detach().cpu().numpy(), valid)
            escher.texture.data.copy_(
                torch.as_tensor(filled, device=escher.texture.device)
            )

    tint_mtx = None
    if tint:
        from escher.rendering.palette import tile_color_matrices

        hues = list(escher.args.get("PALETTE_HUES_DEG", [0.0, 120.0, -120.0]))
        tint_mtx = tile_color_matrices(escher.tiler, escher.mesh, hues)

    init_color = escher.args.get("TEXTURE_INIT_COLOR", None)
    init = np.asarray(
        DEFAULT_INIT if init_color is None else list(init_color), dtype=np.float32
    )

    covered = 0
    n_white = 0
    n_init = 0
    n_bg = 0
    panels: list[tuple[np.ndarray, np.ndarray]] = []

    with torch.no_grad():
        points = escher.solve_points()
        sphere = build_tiled_sphere(
            points.to(escher.device).float(),
            escher.mesh.faces,
            escher.mesh.uv,
            escher.tiler,
        )
        views = orbit_views(
            n_views,
            distance=escher.args.get("PREVIEW_DISTANCE", 3.0),
            elevation_deg=15.0,
        )
        for i in range(0, n_views, 4):  # small batches to bound VRAM
            mv = views[i : i + 4]
            images, alpha = render_tiled_sphere(
                sphere,
                escher.texture,
                mv=mv,
                image_size=escher.args.RENDER_SIZE,
                tile_color_matrices=tint_mtx,
                shade_ambient=None,
            )
            img = images.clamp(0, 1).cpu().numpy()
            on_sphere = alpha.cpu().numpy()[..., 0] > 0.5

            chroma = img.max(-1) - img.min(-1)
            luma = img.mean(-1)
            white_like = (chroma < CHROMA_MAX) & (luma > LUMA_MIN) & on_sphere
            init_like = (
                np.linalg.norm(img - init, axis=-1) < INIT_L2
            ) & on_sphere
            bg = white_like | init_like

            covered += int(on_sphere.sum())
            n_white += int(white_like.sum())
            n_init += int(init_like.sum())
            n_bg += int(bg.sum())

            for v in range(img.shape[0]):
                if len(panels) < audit_views:
                    panels.append((img[v], bg[v]))

    result = {
        "checkpoint": str(checkpoint),
        "iteration": iteration,
        "n_views": n_views,
        "tinted": bool(tint),
        "gutter_filled": bool(gutter),
        "thresholds": {
            "chroma_max": CHROMA_MAX,
            "luma_min": LUMA_MIN,
            "init_l2": INIT_L2,
            "init_color": init.tolist(),
        },
        "covered_px": covered,
        "bg_px": n_bg,
        "bg_frac": n_bg / max(covered, 1),
        "bg_white_frac": n_white / max(covered, 1),
        "bg_init_frac": n_init / max(covered, 1),
    }

    out_dir = checkpoint.parent
    (out_dir / "background_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.6))
    for ax, (img, bg) in zip(np.atleast_1d(axes), panels):
        marked = img.copy()
        marked[bg] = (1.0, 0.0, 1.0)
        ax.imshow(marked)
        ax.set_axis_off()
    fig.suptitle(
        f"background-like pixels (magenta): {100 * result['bg_frac']:.2f}% of sphere"
        f" -- white-like {100 * result['bg_white_frac']:.2f}%,"
        f" init-like {100 * result['bg_init_frac']:.2f}%",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "background_audit.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(
        f"bg_frac {100 * result['bg_frac']:.3f}% of {covered} sphere pixels "
        f"(white-like {100 * result['bg_white_frac']:.3f}%, "
        f"init-like {100 * result['bg_init_frac']:.3f}%) "
        f"-> {out_dir / 'background_metrics.json'}"
    )
    return result


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <checkpoint.pt> [N_VIEWS=30] [TINT=0]")
    checkpoint = Path(sys.argv[1])
    n_views = 30
    tint = True
    gutter = True
    for arg in sys.argv[2:]:
        if arg.startswith("N_VIEWS="):
            n_views = int(arg.split("=", 1)[1])
        elif arg == "TINT=0":
            tint = False
        elif arg == "GUTTER=0":
            gutter = False
    measure(checkpoint, n_views=n_views, tint=tint, gutter=gutter)


if __name__ == "__main__":
    main()
