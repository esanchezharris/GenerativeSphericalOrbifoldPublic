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


def export_mesh(escher: SphereEscher, out_dir: Path) -> None:
    """Write the tiled sphere as OBJ + MTL + texture."""
    with torch.no_grad():
        points = escher.embedder(escher.edge_weights())
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


def render_turntable(escher: SphereEscher, out_dir: Path, n_frames: int = 120) -> None:
    with torch.no_grad():
        points = escher.embedder(escher.edge_weights())
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
                sphere, escher.texture, mv=mv, image_size=escher.args.RENDER_SIZE
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


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <checkpoint.pt>")
    checkpoint = Path(sys.argv[1])
    escher, iteration = load_run(checkpoint)
    print(f"loaded step {iteration} from {checkpoint}")

    out_dir = checkpoint.parent
    export_mesh(escher, out_dir)
    render_turntable(escher, out_dir)


if __name__ == "__main__":
    main()
