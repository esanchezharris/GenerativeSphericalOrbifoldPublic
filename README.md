<p align="center">
  <img src="assets/generated-tiling.png" width="640" alt="Generated Escher-style gingerbread tessellations on the sphere, next to the 2024 concept image">
</p>

<h1 align="center">Generative Spherical Orbifold</h1>

<p align="center">
  Escher-style tilings, generated from a text prompt and wrapped seamlessly around a sphere.<br>
  Emilio Sánchez-Harris &amp; <a href="https://github.com/nikhitrivedi1">Nikhil Trivedi</a>, advised by Dr.&nbsp;Crane Chen · 2024–2026
</p>

---

Give it a prompt ("gingerbread man") and it produces a closed spherical surface tiled
with that figure in the style of M.C. Escher — every tile interlocking with its
neighbors, no gaps and no overlaps, certified: the tiled mesh's signed solid angles sum
to exactly 4π with zero inverted faces.

Built on **[Generative Escher Meshes](https://github.com/thibaultgroueix/GenerativeEscherMeshes)**
(Aigerman &amp; Groueix, SIGGRAPH 2024 — [paper](https://arxiv.org/abs/2309.14564)),
extended from flat wallpaper tilings to closed spherical surfaces using
**[Spherical Orbifold Tutte Embeddings](https://github.com/noamaig/spherical_orbifolds)**
(Aigerman &amp; Lipman, SIGGRAPH 2017). The leftmost image above is the project's
original 2024 concept; the other three are actual outputs of this code — two spheres
(the dihedral and octahedral orbifolds, tile outlines carved by full-mesh deformation
through the differentiable Karcher solve) and a planar torus tiling from the same
mechanism, which also runs the original planar pipeline.

## How it works

1. **Geometry.** A fundamental domain of a spherical orbifold — a lune for the
   dihedral `(k,2,2)` groups, a kite for the platonic `(2,3,3)`, `(2,3,4)`, `(2,3,5)`
   groups — is embedded on the sphere by an orbifold Tutte solve with the Karcher
   (geodesic) Dirichlet energy, made differentiable via the implicit function theorem
   (one sparse adjoint solve per backward pass). Two parameterizations: the **edge
   weights** of the solve (full-mesh deformation, the Generative-Escher-Meshes
   mechanism — this is what produces the deeply articulated outlines above), or the
   cut **boundary points** directly. Either way, one side of the cut is free and the
   other is *generated* by the symmetry group, so tiles interlock by construction.

2. **Shape.** A clean figure silhouette is generated once with full Stable Diffusion
   denoising, aligned over the undeformed tile with its **area matched to the tile's**
   (a tile's area is pinned at `4π/|G|`, so an undersized target caps the achievable
   overlap structurally — measured, this was the binding constraint before area
   matching), and the parameters are optimized to match it with a deterministic mask
   loss. The tile silhouette is computed analytically from the projected boundary
   polygon, so the shape phase needs no renderer and no diffusion model in the loop,
   and runs in about two minutes.

3. **Texture.** With the shape frozen, one shared texture is trained by score
   distillation (SDS) against the prompt through a differentiable renderer
   (nvdiffrast); the orbifold constraints make it continuous across every tile
   boundary.

4. **Presentation.** Neighboring tiles are tinted by hue rotations about the RGB gray
   axis (a proper 3-coloring of the tiling's adjacency graph), which is what makes the
   interlocking legible — the classic Escher look.

Validity is never assumed: every run ends with the signed-solid-angle certificate
(total exactly 4π, zero flipped faces = no gaps and no overlaps), and during
optimization folding steps are projected back to the valid set.

## Running it

```bash
# 1. generate silhouette target candidates (GPU, ~1 min; CHOOSE=i re-picks)
python escher/make_target.py

# 2. carve the tile outline to the chosen target (CPU-capable, ~2 min).
#    Full-mesh weights mode (best results):
python escher/main_shape.py CONF_FILE=configs/sphere_shape_weights.yaml
python escher/main_shape.py CONF_FILE=configs/sphere_shape_weights.yaml \
    "ORBIFOLD_CONES=[2,3,4]" SHAPE_CAMERA_DISTANCE=2.4     # octahedral, 24 tiles
#    Boundary-explicit mode: python escher/main_shape.py    (sphere_shape.yaml)

# 3. train the texture on the frozen shape (GPU, ~8 min)
python escher/main_sphere.py CONF_FILE=configs/sphere_texture.yaml PARAM_MODE=weights \
    RESUME=output/sphere_shape_weights/checkpoint.pt OUTPUT_DIR=output/sphere_tex

# 4. deliverables: turntable video, textured OBJ, contact sheet
python escher/render_final.py output/sphere_tex/checkpoint.pt TINT=1

# Planar (the original wallpaper-group pipeline, same shape mechanism):
python escher/main_shape_planar.py                          # carve on the torus
python escher/main.py CONF_FILE=configs/planar_texture.yaml # GEM-style joint SDS
```

Configs live in `escher/configs/` (`sphere.yaml` base; `sphere_shape.yaml`,
`sphere_shape_weights.yaml`, `sphere_texture.yaml`, `planar_shape.yaml`,
`planar_texture.yaml` phase overlays). The test suite (`pytest tests/`, 270+ tests)
runs entirely on CPU, including both shape-phase optimizations end to end.

## Results and current limitations

Measured on the gingerbread target (soft IoU of the tile silhouette against the
area-matched target mask; perimeter relative to the undeformed tile):

| Run | IoU | Perimeter | Certificate |
|---|---|---|---|
| Sphere `(4,2,2)`, full-mesh weights | 0.797 | 1.479× | 4π at 0.0e+00, 0 folds |
| Sphere `(2,3,4)`, 24 tiles | 0.712 | 1.313× | 4π at 4.1e-10, 0 folds |
| Plane, torus | 0.751 | — | fold-free (planar Tutte guarantee) |

What is solid: the geometry is certified valid in every run, the tiles interlock by
construction with strongly articulated outlines, and the whole shape phase is
deterministic and reproducible. What is honestly not there yet: the SDS textures read
as decorated cookie tiles rather than unmistakable gingerbread figures — outline
articulation is strong, but the figure semantics *inside* each tile remain the open
gap between these renders and the concept image. Known levers, untried for compute
reasons: the upstream pipeline's full 7000-step schedule (ours ran 1000–1400), higher
shared-texture resolution, and the planar joint phase with `GLOBAL_AFFINE: false`
(with it on, the global map drifts into a shear).

## What's here

- `escher/OTE/core/spherical/` — Karcher energy with analytic gradients, projected
  L-BFGS with the reference's two-stage preconditioner schedule, and the implicit
  differentiation layer (validated against finite differences to ~1e-8).
- `escher/OTE/tilings_sphere/` — both orbifold parameterizations for the lune and the
  kite: weights-mode constraint systems (`DihedralOrbifold`, `OctahedralOrbifold`) and
  boundary-explicit ones (`BoundaryExplicitDihedral`, `BoundaryExplicitTriple`).
- `escher/geometry/` — fundamental-domain meshes (lune + kite), the spherical tiler
  for all four rotation-group families, and the signed-solid-angle certificate.
- `escher/shape_target.py` / `escher/soft_silhouette.py` — the deterministic shape
  loss: target binarization, area-matched alignment, and the analytic soft silhouette
  (the rasterizer's alpha carries no vertex gradients — measured; the analytic
  polygon does).
- `escher/main_shape.py` / `escher/main_shape_planar.py` / `escher/main_sphere.py` /
  `escher/main.py` / `escher/render_final.py` — the pipeline stages, sphere and plane.
- `escher/rendering/palette.py` — the per-tile hue rotation and the tiling adjacency
  3-coloring.
- `lbfgs-translation/` — the original MATLAB→Python solver port that seeded the
  project (superseded by `escher/OTE/core/spherical/`).

## Attribution

This is a public mirror of a collaborative research project. The planar tiling
machinery comes from Generative Escher Meshes (see `license.txt` for upstream terms);
the spherical orbifold Tutte formulation follows Aigerman &amp; Lipman's reference
implementation. The spherical extension — the differentiable Karcher solve, the
boundary-explicit parameterizations, the deterministic silhouette shape phase, and the
certified tiling pipeline — is this project's contribution.
