# Conventions

# In OpenGL convention, the perspective projection matrix (as implemented in, e.g., utils.projection() in our samples and glFrustum() in OpenGL) treats the view-space z as increasing towards the viewer. However, after multiplication by perspective projection matrix, the homogeneous clip-space coordinate z/w increases away from the viewer. Hence, a larger depth value in the rasterizer output tensor also corresponds to a surface further away from the viewer.

from typing import Tuple

import numpy as np
import torch
from matplotlib import image

import nvdiffrast.torch as dr

torch.concat = torch.cat


def _warmup(glctx):
    # windows workaround for https://github.com/NVlabs/nvdiffrast/issues/59
    def tensor(*args, **kwargs):
        return torch.tensor(*args, device="cuda", **kwargs)

    pos = tensor(
        [[[-0.8, -0.8, 0, 1], [0.8, -0.8, 0, 1], [-0.8, 0.8, 0, 1]]],
        dtype=torch.float32,
    )
    tri = tensor([[0, 1, 2]], dtype=torch.int32)
    dr.rasterize(glctx, pos, tri, resolution=[256, 256])


def render_mesh_nvdiffrast(
    vertices: torch.Tensor,  # B,V,3,
    faces: torch.Tensor,  # V,3,
    uv: torch.Tensor,  # V,3,
    vertices_color: torch.Tensor = None,  # V,3,
    mv: torch.Tensor = torch.eye(4)[None, ...],  # B,4,4
    proj: torch.Tensor = torch.eye(4)[None, ...],  # B,4,4
    image_size: Tuple[int, int] = (512, 512),
    texture: torch.Tensor = None,  # B,H,W,3
    glctx: dr.RasterizeCudaContext = None,
    vertex_color_mtx: torch.Tensor = None,  # V,9 or B,V,9 -- per-vertex 3x3 color matrix
    shade_ambient: float = None,  # None = unlit (training); float in [0,1] = diffuse preview
) -> torch.Tensor:  # B,H,W,4
    # Number of vertices
    vertices = vertices.to("cuda:0").float()

    if isinstance(image_size, int):
        image_size = (image_size, image_size)

    if vertices.ndim == 2:
        # add batch dimension if missing
        vertices = vertices.unsqueeze(0)

    V = vertices.shape[1]
    B = vertices.shape[0]

    if vertices.shape[-1] == 2:
        vertices = torch.cat((vertices, torch.ones(B, V, 1, device=vertices.device)), axis=-1)  # V,2 -> V,3
    # put everything on the GPU
    device = vertices.device

    faces = faces.to("cuda:0").type(torch.int32)

    if uv is not None:
        uv = uv.to("cuda:0").float()
        if uv.ndim == 2:
            # add batch dimension if missing
            uv = uv.unsqueeze(0)

    if vertices_color is not None:
        colors = vertices_color.to("cuda:0").float()

    mv = mv.to("cuda:0").float()
    proj = proj.to("cuda:0").float()
    if mv.ndim == 2:
        # add batch dimension if missing
        mv = mv.unsqueeze(0)

    if proj.ndim == 2:
        # add batch dimension if missing
        proj = proj.unsqueeze(0)

    if texture is not None:
        texture = texture.to("cuda:0").to(torch.float32).contiguous()
        if texture.ndim == 3:
            # add batch dimension if missing
            texture = texture.unsqueeze(0)

    # Change the type of faces to int32
    faces = faces.type(torch.int32)

    # Add a homogeneous coordinate to the vertices
    vert_hom = torch.cat((vertices, torch.ones(B, V, 1, device=vertices.device)), axis=-1)  # V,3 -> V,4c

    # Transform the vertices to clip space
    vertices_clip = vert_hom @ mv.transpose(-2, -1) @ proj.transpose(-2, -1)  # C,V,4

    # orthographic
    # Change of convention to OPENGL CLIP Space. Z is pointing from the viewer into the screen
    vertices_clip[:, :, 2] = -vertices_clip[:, :, 2]
    # Memory convention in the rendered image. Either
    vertices_clip[:, :, 1] = -vertices_clip[:, :, 1]

    # or
    # col = torch.flip(col, 1)

    # Check if gltctx is provided, otherwise create a new one
    if glctx is None:
        glctx = dr.RasterizeGLContext()
        # glctx = dr.RasterizeCudaContext(torch.device("cuda"))
        _warmup(glctx)

    # Rasterize data
    rast_out, rast_out_db = dr.rasterize(glctx, vertices_clip, faces, resolution=image_size, grad_db=True)  # C,H,W,4

    if texture is None:
        # interpolate depth for debugging
        # col, _ = dr.interpolate(vertices_clip[0,:,2:3].contiguous().repeat(1,3).contiguous(), rast_out, faces)  # C,H,W,3
        col, _ = dr.interpolate(colors, rast_out, faces)  # C,H,W,3
        # create alpha channel
    else:
        texc, texd = dr.interpolate(uv.float(), rast_out, faces, rast_db=rast_out_db, diff_attrs="all")
        col = dr.texture(texture, texc, texd, filter_mode="linear-mipmap-linear", max_mip_level=9)
    if vertex_color_mtx is not None:
        # Per-vertex 3x3 color matrix, applied to the sampled color BEFORE antialiasing
        # so tile borders blend already-transformed colors. When vertices are duplicated
        # per tile (faces never index across tiles) the interpolation is exact per tile.
        vcm = vertex_color_mtx.to("cuda:0").float()
        if vcm.ndim == 2:
            vcm = vcm.unsqueeze(0)
        mtx, _ = dr.interpolate(vcm.contiguous(), rast_out, faces)  # C,H,W,9
        col = torch.einsum("bhwij,bhwj->bhwi", mtx.reshape(*mtx.shape[:3], 3, 3), col)
    if shade_ambient is not None:
        # Diffuse shading, PREVIEW ONLY (never during training -- see render_tiled_sphere).
        #
        # Without it a rotating textured sphere is genuinely bistable: with no shape-from-
        # shading cue the visual system cannot tell a convex sphere turning one way from a
        # concave one turning the other, and the turntable reads as the INSIDE of the ball.
        # The geometry was never wrong -- the tiled sphere's signed solid angle is +4pi, so
        # faces are outward-oriented, and the camera orbits outside it.
        #
        # The normal is free here: these vertices lie on the unit sphere, so the outward
        # normal IS the normalised position. Lighting is fixed in VIEW space, otherwise the
        # highlight would sweep across the surface as the camera orbits and strobe.
        n_obj = torch.nn.functional.normalize(vertices, dim=-1)  # B,V,3
        n_view = torch.einsum("bij,bvj->bvi", mv[:, :3, :3], n_obj)
        n_view = torch.nn.functional.normalize(n_view, dim=-1)
        light = torch.tensor([0.35, 0.45, 0.82], device=n_view.device)
        lam = (n_view @ (light / light.norm())).clamp(min=0.0).unsqueeze(-1)  # B,V,1
        shade, _ = dr.interpolate(lam.contiguous(), rast_out, faces)  # C,H,W,1
        col = col * (shade_ambient + (1.0 - shade_ambient) * shade)
    alpha = torch.clamp(rast_out[..., -1:], max=1)  # C,H,W,1
    depth = rast_out[:, :, :, 2]  # C,H,W,1
    # if debugging with depth
    #  (col+1)/2*alpha + col*0*(1-alpha)

    # Add alpha channel
    col = torch.concat((col, alpha), dim=-1)  # C,H,W,4
    # Anti-aliasing
    col = dr.antialias(col, rast_out, vertices_clip, faces)  # C,H,W,4
    # col = col - col.min() / (col.max() - col.min())

    return col, depth  # C,H,W,4
