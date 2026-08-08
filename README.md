<p align="center">
  <img src="assets/generated-tiling.png" width="640" alt="Generated Escher-style gingerbread tessellations on the sphere, next to the 2024 concept image">
</p>

<h1 align="center">Generative Spherical Orbifold</h1>

<p align="center">
  Escher-style tilings, generated from a text prompt and wrapped seamlessly around a sphere.<br>
  Emilio Sánchez-Harris &amp; <a href="https://github.com/nikhitrivedi1">Nikhil Trivedi</a>, advised by Dr.&nbsp;Crane Chen · 2024–2026
</p>

---

Give it a prompt ("a fish with scales and fins") and it produces a closed spherical
surface tiled with that figure in the style of M.C. Escher — the tile's **outline is the
figure**, every tile interlocking with its neighbors, no gaps and no overlaps, certified:
the tiled mesh's signed solid angles sum to exactly 4π with zero inverted faces.

Built on **[Generative Escher Meshes](https://github.com/thibaultgroueix/GenerativeEscherMeshes)**
(Aigerman &amp; Groueix, SIGGRAPH 2024 — [paper](https://arxiv.org/abs/2309.14564)),
extended from flat wallpaper tilings to closed spherical surfaces using
**[Spherical Orbifold Tutte Embeddings](https://github.com/noamaig/spherical_orbifolds)**
(Aigerman &amp; Lipman, SIGGRAPH 2017). Every image above is an actual output of this
code, on the octahedral `(2,3,4)` orbifold — 24 tiles, outlines carved by full-mesh
deformation through the differentiable Karcher solve. The pipeline is prompt-driven, not
tuned to one figure: the fish and the gingerbread run differ only in their text prompt.

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

2. **Shape.** Figure silhouettes are generated once with full Stable Diffusion denoising,
   then **screened by reachability**: shapes that can tile a surface under a fixed
   symmetry group are a thin subset (every limb needs a complementary notch in a
   neighbour), and a diffusion model draws with no such constraint. So the pipeline
   carves a short trial against each candidate and keeps whichever the tiling can
   actually adopt — measured, that is worth more than the choice of figure. The winner is
   aligned over the undeformed tile with its **area matched to the tile's** (a tile's area
   is pinned at `4π/|G|`, so an undersized target caps overlap structurally) and fitted
   with a deterministic mask loss. The tile silhouette is computed analytically from the
   projected boundary polygon, so the shape phase needs no renderer and no diffusion model
   in the loop; a carve takes about two minutes and candidate screening runs in parallel
   across cores.

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

This reproduces the fish sphere above end to end (~70 min, most of it step 4):

```bash
# 1. generate silhouette candidates (GPU, ~9 min for 64)
python escher/make_target.py OUT_DIR=assets/targets/fish N=64 \
    PROMPT="a plain solid black silhouette of a fish with rounded fins and a broad tail, white background, minimal flat logo, centered, full body"

# 2. screen them by REACHABILITY and keep the best (parallel; GPU makes each
#    candidate carve ~13x faster than CPU)
python escher/main_shape.py CONF_FILE=configs/sphere_shape_weights_fish2.yaml \
    SWEEP_TARGETS=true TARGET_DIR=assets/targets/fish

# 3. carve the tile outline to the winner (GPU ~2.5 min / CPU ~26 min)
python escher/main_shape.py CONF_FILE=configs/sphere_shape_weights_fish2.yaml \
    TARGET_MASK=assets/targets/fish/target.npy OUTPUT_DIR=output/fish_shape

# 4. train the shared texture on the frozen shape (GPU; TORCH_COMPILE=true
#    measured ~18% faster with 1.6 GiB more VRAM headroom)
python escher/main_sphere.py \
    "CONF_FILE=[configs/sphere_texture.yaml,configs/sphere_texture_octa.yaml]" \
    COTANGENT_RELATIVE_WEIGHTS=true TORCH_COMPILE=true \
    RESUME=output/fish_shape/checkpoint.pt START_STEP=0 \
    PROMPT="a fish with scales and fins, flat vector illustration, solid pastel colors, simple shapes, a masterpiece" \
    OUTPUT_DIR=output/fish_tex

# 5. deliverables: shaded turntable, textured OBJ, contact sheet (the unsampled
#    60% of texels are gutter-filled at export; GUTTER=0 restores the raw look)
python escher/render_final.py output/fish_tex/checkpoint.pt TINT=1

# Optional: scrub residual background paint (deterministic, reversible) and
# measure the zero-background gate
python escher/texture_polish.py output/fish_tex/checkpoint.pt
python escher/metrics_background.py output/fish_tex/checkpoint_polished.pt

# Planar (the original wallpaper-group pipeline, same shape mechanism):
python escher/main_shape_planar.py                          # carve on the torus
python escher/main.py CONF_FILE=configs/planar_texture.yaml # GEM-style joint SDS
```

### Batch mode

The whole chain also runs unattended over many figures, seeds, and orbifold
sweeps — a two-lane scheduler overlaps CPU shape work with the exclusive GPU
texture runs, every stage runs in its own process with typed exit codes, and a
manifest + contact sheet record the night:

```bash
python -m escher.pipeline escher/configs/batch_example.yaml          # real batch
python -m escher.pipeline escher/configs/batch_smoke.yaml --dry-run  # <1 min check
```

Configs live in `escher/configs/` (`sphere.yaml` base; `sphere_shape.yaml`,
`sphere_shape_weights_fish2.yaml` (the A/B-winning carve recipe),
`sphere_texture.yaml`, `sphere_texture_octa_nobg.yaml` (zero-background levers),
`planar_shape.yaml`, `planar_texture.yaml` phase overlays). The test suite
(`pytest tests/`, 300+ tests) runs entirely on CPU, including both shape-phase
optimizations and the batch driver end to end.

## Results

Hard IoU (both masks thresholded at 0.5 — the number that corresponds to filler
on the sphere; the soft metric a perfect carve can only score ~0.93 on is gone
from the headline) of the tile silhouette against the area-matched target;
perimeter relative to the undeformed tile. All on `(2,3,4)`, 24 tiles, fish64
target, 1500 steps, unless noted.

| Run | hard IoU | sr-weighted | Perimeter | Certificate |
|---|---|---|---|---|
| Shipped carve (`output/texF_fish`) | 0.824 | 0.797 | 1.200× | 4π at 2.3e-13, 0 folds |
| + achieved-area crop fix (unconditional) | 0.835 | 0.797 | 1.184× | 0 folds |
| **+ corner/translation alignment + tau anneal** | **0.868** | **0.839** | 1.196× | 0 folds |
| Gingerbread man (soft 0.737, historical) | — | — | 1.442× | 4π at 1.2e-13, 0 folds |
| Plane, torus (soft 0.751, historical) | — | — | — | fold-free (planar Tutte) |

Background on the rendered sphere (`metrics_background.py`, 30 tinted orbit
views): the shipped fish read **1.91%** background-like; the polish scrub +
gutter fill take the same checkpoint to **0.27%** with structurally zero
background texels — no retraining.

Texture-phase speed on the current machine: 0.54 → 0.42 s/step (~22%), and peak
VRAM 9.1 → 7.5 GiB, from the frozen-solve cache + cached rasterizer context +
`TORCH_COMPILE=true`.

Four things moved the needle originally, in the order we found them (the carve
gains above came later, from measurement: the alignment POSE dominates —
scoring it by where the four immovable cone corners land, and searching
translation, was worth more than any framing or area-measure change we tried):

**The step budget.** Score distillation needs the full 7000-step schedule. At 1400 the
texture is coloured stipple; the same run at 7000 resolves into clean icing, scales and
eyes. Nothing before the timestep anneal completes predicts the final result — an earlier
run was abandoned at step 1100 over blobs that were the prompt's *candy buttons*, still
forming.

**Cotangent initialization.** The reference solver builds its system from `cotmatrix`;
our weights mode started from *uniform* weights, so `W = 0` was not the fundamental domain
we designed but a harmonic distortion of it — per-face areas spread **82×** on the kite.
Since the UV is uniform barycentric that is also the texel-density range across one tile,
i.e. an 82× spread in effective per-texel learning rate. Referencing the solve instead
(`COTANGENT_RELATIVE_WEIGHTS`) drops it to **1.4×** and cut final texture loss by 24%.

**Reachability, not figure choice.** Shapes that tile under a fixed group are a thin
subset, and a diffusion model draws with no such constraint. Screening 64 candidates and
keeping the most *reachable* one is worth **+0.052 IoU** — more than the entire
fish-vs-gingerbread gap (+0.036). Selecting on figure area instead, as the generator's own
heuristic does, picked a loser every time we checked.

**A figure the group can adopt.** Measured best IoU by prompt: fish 0.762, leaf 0.741,
gingerbread 0.726, bird 0.696, **lizard 0.659** — despite lizards being *the* Escher motif.
That is the lesson rather than a contradiction: Escher *designed* his figures around the
tiling constraint. Note also that fish at 0.762 reads unmistakably as fish while
gingerbread at 0.726 reads as a decorated star — **IoU is a weak proxy for legibility**.
What matters is where the residual error lands. A fish tolerates a fattened body; a
humanoid does not tolerate missing limbs.

## Current limitations

The gingerbread man remains the honest failure case: at its reachability ceiling the tiles
read as decorated cookies rather than figures. `configs/` keeps the alternative for that
case — a smaller `ISOLATED_DISTANCE` makes score distillation paint a recognizable figure
*inside* each tile instead of icing its border (`output/texD_figure_in_tile`), which is
less Escher-pure but more legible.

Untried levers: an area-preserving (authalic) rather than harmonic UV, texel-density
normalization of the accumulated texture gradients, and a VSD-style objective. On that
last one — lowering the guidance scale *without* switching objectives fails outright
(measured: CFG 12 collapsed the texture to flat stipple). Vanilla SDS needs high guidance
to overcome its own gradient variance; VSD is what makes low CFG viable, so the two must
move together.

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
  loss: target binarization and de-jaggying, area-matched alignment, and the analytic
  soft silhouette (the rasterizer's alpha carries no vertex gradients — measured; the
  analytic polygon does).
- `escher/geometry/cotangent_weights.py` — the reference solver's clamped cotangent
  weights, shared by both parameterizations so the two cannot drift apart again.
- `escher/main_shape.py` / `escher/main_shape_planar.py` / `escher/main_sphere.py` /
  `escher/main.py` / `escher/render_final.py` — the pipeline stages, sphere and plane.
- `escher/rendering/palette.py` — the per-tile hue rotation and the tiling adjacency
  3-coloring.
- `escher/rendering/texture_mask.py` / `escher/texture_polish.py` /
  `escher/metrics_background.py` — the zero-background toolkit: UV gutter fill
  for the ~60% of texels no triangle samples, the deterministic background
  scrub, and the render gate that measures them.
- `escher/pipeline/` — the unattended batch driver (spec → two-lane scheduler →
  manifest + contact sheet).
- `escher/pixel_solid_angle.py` — per-pixel spherical area of the shape camera
  (pixel counts are not steradians; the area matching now knows).

## Attribution

This is a public mirror of a collaborative research project. The planar tiling
machinery comes from Generative Escher Meshes (see `license.txt` for upstream terms);
the spherical orbifold Tutte formulation follows Aigerman &amp; Lipman's reference
implementation. The spherical extension — the differentiable Karcher solve, the
boundary-explicit parameterizations, the deterministic silhouette shape phase, and the
certified tiling pipeline — is this project's contribution.
