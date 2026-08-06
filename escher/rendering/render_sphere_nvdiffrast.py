r"""Rendering the tiled sphere with nvdiffrast, for score distillation.

Thin layer over :func:`escher.rendering.renderer_nvdiffrast.render_mesh_nvdiffrast`, which
already handles ``(B, V, 3)`` vertices with ``mv``/``proj`` matrices and needs no structural
change for a sphere. What this adds is the sphere-specific plumbing:

- replicate the fundamental domain by its symmetry group, **UVs included**, so every tile
  samples one shared texture;
- keep the tiling differentiable, so gradients from the rendered image reach the tile shape;
- render a batch of viewpoints per step, since one view only ever sees a hemisphere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from escher.geometry.sphere_tiler import SphericalTiler
from escher.rendering.camera import perspective, random_views

__all__ = ["TiledSphere", "build_tiled_sphere", "render_tiled_sphere"]


@dataclass
class TiledSphere:
    """A fundamental domain replicated into a closed sphere, ready to rasterise."""

    vertices: Tensor  #: ``(|G| * n, 3)``, differentiable w.r.t. the domain vertices
    faces: Tensor  #: ``(|G| * n_faces, 3)`` int32
    uv: Tensor  #: ``(|G| * n, 2)``, the domain's UVs repeated per tile
    tile_index: np.ndarray  #: ``(|G| * n_faces,)`` which group element made each face

    @property
    def n_tiles(self) -> int:
        return int(self.tile_index.max()) + 1


def build_tiled_sphere(
    points: Tensor,
    faces: np.ndarray,
    uv: np.ndarray,
    tiler: SphericalTiler,
) -> TiledSphere:
    """Replicate a domain by its group, keeping the result differentiable in ``points``.

    Args:
        points: ``(n, 3)`` domain vertices, typically straight from
            :class:`~escher.OTE.core.spherical.differentiable.SphericalEmbedder`.
        faces: ``(n_faces, 3)`` domain faces.
        uv: ``(n, 2)`` domain texture coordinates.
        tiler: the symmetry group.

    The rotations are applied as a batched matmul on ``points`` rather than by tiling a
    detached numpy array, so gradients from the rendered image flow back to the tile shape.
    """
    rotations = torch.as_tensor(
        tiler.rotations, dtype=points.dtype, device=points.device
    )
    # (G, 3, 3) x (n, 3) -> (G, n, 3) -> (G*n, 3)
    vertices = torch.einsum("gij,nj->gni", rotations, points).reshape(-1, 3)

    n = points.shape[0]
    faces_np = np.asarray(faces)
    all_faces = np.concatenate([faces_np + g * n for g in range(tiler.order)])
    tile_index = np.repeat(np.arange(tiler.order), len(faces_np))

    uv_t = torch.as_tensor(np.asarray(uv), dtype=points.dtype, device=points.device)
    return TiledSphere(
        vertices=vertices,
        faces=torch.as_tensor(all_faces, dtype=torch.int32, device=points.device),
        uv=uv_t.repeat(tiler.order, 1),
        tile_index=tile_index,
    )


def render_tiled_sphere(
    sphere: TiledSphere,
    texture: Tensor,
    n_views: int = 4,
    image_size: int = 512,
    distance: float = 3.0,
    fovy_deg: float = 40.0,
    mv: Tensor | None = None,
    glctx=None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Render ``n_views`` images of the tiled sphere.

    Args:
        texture: ``(H, W, 3)`` or ``(1, H, W, 3)`` texture map, shared by every tile.
        mv: explicit ``(B, 4, 4)`` view matrices; if omitted, a fresh spread of random views.

    Returns:
        ``(images, alpha)`` with images ``(B, H, W, 3)`` and alpha ``(B, H, W, 1)``.
    """
    from escher.rendering import renderer_nvdiffrast as renderer

    device = sphere.vertices.device
    if mv is None:
        mv = random_views(n_views, distance=distance, generator=generator)
    mv = mv.to(device)
    batch = mv.shape[0]

    proj = (
        perspective(fovy_deg=fovy_deg, aspect=1.0).to(device).unsqueeze(0).repeat(batch, 1, 1)
    )

    # nvdiffrast's kernels require contiguous inputs, and `expand` returns a stride-0 view.
    verts = sphere.vertices.unsqueeze(0).expand(batch, -1, -1).contiguous()
    uv = sphere.uv.unsqueeze(0).expand(batch, -1, -1).contiguous()

    rgba, _ = renderer.render_mesh_nvdiffrast(
        vertices=verts,
        faces=sphere.faces,
        uv=uv,
        mv=mv,
        proj=proj,
        image_size=(image_size, image_size),
        texture=texture,
        glctx=glctx,
    )
    return rgba[..., :3], rgba[..., 3:4]
