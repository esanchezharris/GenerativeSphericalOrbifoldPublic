"""Turn a finished run into deliverables: a turntable video and an exported mesh.

Usage::

    python escher/render_final.py output/sphere_gingerbread/checkpoint.pt

Writes, alongside the checkpoint:

- ``turntable.mp4`` -- the tiled sphere rotating
- ``tiling.obj`` + ``tiling.png`` -- the closed mesh with its shared texture, loadable in
  any 3D viewer
- ``final.png`` -- a contact sheet
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from escher.geometry.spherical_sanity_checks import check_covers_sphere_once
from escher.main_sphere import SphereEscher
from escher.rendering.camera import orbit_views
from escher.rendering.render_sphere_nvdiffrast import build_tiled_sphere, render_tiled_sphere


def load_run(checkpoint: Path) -> tuple[SphereEscher, int]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = OmegaConf.create(state["config"])

    # Skip loading the diffusion model: rendering needs none of it.
    escher = SphereEscher.__new__(SphereEscher)
    escher.args = args
    escher.device = torch.device(args.DEVICE)
    escher.output_dir = Path(args.OUTPUT_DIR)
    escher._init_geometry()
    escher._init_parameters()
    iteration = escher.load_checkpoint(checkpoint)
    return escher, iteration


def export_mesh(escher: SphereEscher, out_dir: Path) -> bool:
    """Write the tiled sphere as OBJ + MTL + texture; returns the certificate."""
    with torch.no_grad():
        points = escher.solve_points()
    sphere = build_tiled_sphere(
        points.to(escher.device).float(), escher.mesh.faces, escher.mesh.uv, escher.tiler
    )
    verts = sphere.vertices.detach().cpu().numpy()
    faces = sphere.faces.detach().cpu().numpy()
    uv = sphere.uv.detach().cpu().numpy()

    ok, message = check_covers_sphere_once(verts, faces)
    print(f"geometry: {message}")

    texture = escher.texture.detach().clamp(0, 1).cpu().numpy()
    imageio.imwrite(out_dir / "tiling.png", (texture * 255).astype(np.uint8))

    with open(out_dir / "tiling.mtl", "w") as f:
        f.write("newmtl tile\nKa 1 1 1\nKd 1 1 1\nmap_Kd tiling.png\n")

    with open(out_dir / "tiling.obj", "w") as f:
        f.write("mtllib tiling.mtl\nusemtl tile\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in uv:
            # OBJ's v axis points up, the opposite of image row order
            f.write(f"vt {t[0]:.6f} {1.0 - t[1]:.6f}\n")
        for tri in faces + 1:  # OBJ is 1-based
            f.write(f"f {tri[0]}/{tri[0]} {tri[1]}/{tri[1]} {tri[2]}/{tri[2]}\n")
    print(f"wrote {out_dir/'tiling.obj'} ({len(verts)} verts, {len(faces)} faces)")
    return ok


def render_turntable(
    escher: SphereEscher,
    out_dir: Path,
    n_frames: int = 120,
    tint: torch.Tensor | None = None,
    shade: float | None = None,
) -> None:
    with torch.no_grad():
        points = escher.solve_points()
        sphere = build_tiled_sphere(
            points.to(escher.device).float(), escher.mesh.faces, escher.mesh.uv, escher.tiler
        )
        frames = []
        views = orbit_views(
            n_frames, distance=escher.args.get("PREVIEW_DISTANCE", 3.0), elevation_deg=15.0
        )
        for i in range(0, n_frames, 4):  # render in small batches to bound VRAM
            mv = views[i : i + 4]
            images, alpha = render_tiled_sphere(
                sphere,
                escher.texture,
                mv=mv,
                image_size=escher.args.RENDER_SIZE,
                tile_color_matrices=tint,
                shade_ambient=shade,
            )
            comp = (images * alpha + 1.0 * (1 - alpha)).clamp(0, 1).cpu().numpy()
            frames.extend((f * 255).astype(np.uint8) for f in comp)

    path = out_dir / "turntable.mp4"
    imageio.mimwrite(path, frames, fps=30, quality=8, macro_block_size=1)
    print(f"wrote {path} ({len(frames)} frames)")

    fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
    axes[0].imshow(escher.texture.detach().clamp(0, 1).cpu().numpy())
    axes[0].set_title("shared texture", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for i in range(4):
        axes[i + 1].imshow(frames[i * (len(frames) // 4)])
        axes[i + 1].set_axis_off()
    fig.suptitle(f'"{escher.args.PROMPT}"', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "final.png", dpi=140, bbox_inches="tight")
    print(f"wrote {out_dir/'final.png'}")


def finalize(
    checkpoint: str | Path,
    *,
    tint: bool | None = None,
    shade: bool = True,
    out_dir: str | Path | None = None,
    turntable: bool = True,
    gutter: bool = True,
) -> dict:
    """Turn a finished checkpoint into deliverables; callable by a driver.

    ``tint=None`` keeps the run config's TILE_TINT choice; ``out_dir=None`` writes
    beside the checkpoint (historical behavior); ``turntable=False`` skips the
    nvdiffrast video (export_mesh is CPU-safe, which is what dry runs use).
    ``gutter`` fills the never-rasterized ~60% of texels with their nearest
    sampled color before rendering, so the mip chain stops leaking the flat init
    color into minified views (measured 1.17% of sphere pixels on the shipped
    fish). Returns artifact paths plus ``geometry_ok`` -- the 4pi certificate.
    """
    checkpoint = Path(checkpoint)
    escher, iteration = load_run(checkpoint)
    print(f"loaded step {iteration} from {checkpoint}")

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
        print(f"gutter-filled {int((~valid).sum())} unsampled texels")

    # Per-tile hue rotation (the alternating-color Escher look) is a render-time
    # choice. The OBJ export stays untinted either way -- one shared texture is the
    # point of the mesh.
    tint_mtx = None
    use_tint = escher.args.get("TILE_TINT", False) if tint is None else bool(tint)
    if use_tint:
        from escher.rendering.palette import tile_color_matrices

        hues = list(escher.args.get("PALETTE_HUES_DEG", [0.0, 120.0, -120.0]))
        tint_mtx = tile_color_matrices(escher.tiler, escher.mesh, hues)
        print(f"tinting {escher.tiler.order} tiles with hue palette {hues}")

    # Diffuse shading for the video and stills. Without it the turntable is genuinely
    # ambiguous -- a rotating textured sphere carries no shape-from-shading cue, so it
    # reads as easily as the concave inside of the ball as the convex outside.
    shade_val = float(escher.args.get("SHADE_AMBIENT", 0.55)) if shade else None
    if shade_val is not None:
        print(f"shading previews with ambient {shade_val}")

    out = Path(out_dir) if out_dir is not None else checkpoint.parent
    out.mkdir(parents=True, exist_ok=True)
    geometry_ok = export_mesh(escher, out)
    if turntable:
        render_turntable(escher, out, tint=tint_mtx, shade=shade_val)
    return {
        "iteration": iteration,
        "geometry_ok": geometry_ok,
        "obj": str(out / "tiling.obj"),
        "tiling_png": str(out / "tiling.png"),
        "turntable": str(out / "turntable.mp4") if turntable else None,
        "final_png": str(out / "final.png") if turntable else None,
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            f"usage: {sys.argv[0]} <checkpoint.pt> [TINT=1] [SHADE=0] [GUTTER=0]"
        )
    result = finalize(
        Path(sys.argv[1]),
        tint=True if "TINT=1" in sys.argv[2:] else None,
        shade="SHADE=0" not in sys.argv[2:],
        gutter="GUTTER=0" not in sys.argv[2:],
    )
    if not result["geometry_ok"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
