"""Text-driven spherical orbifold tilings via score distillation.

The spherical counterpart of ``main.py``. Replaces the ``main_strip_sphere.py`` stub, which
never ran -- it called an undefined ``constraints_from_args``, built a 2D square mesh, and had
no loss or renderer.

Each optimisation step:

1. map the free parameters to positive edge weights;
2. solve the spherical orbifold embedding, differentiably
   (:class:`~escher.OTE.core.spherical.differentiable.SphericalEmbedder`);
3. replicate the tile over the sphere by its symmetry group, UVs included;
4. render a batch of viewpoints;
5. take an SDS step against the text prompt.

Two parameter groups are optimised: the edge weights, which set the *tile shape*, and a
texture map shared by every tile, which sets its *appearance*. Because all ``2k`` tiles
sample the same texture and the orbifold conditions make their boundaries interlock, whatever
the diffusion model paints appears as one repeated interlocking figure.

Usage::

    python escher/main_sphere.py
    python escher/main_sphere.py PROMPT="a red crab, flat illustration" ORBIFOLD_K=6
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from escher.OTE.core.spherical.differentiable import BoundaryEmbedder, SphericalEmbedder
from escher.OTE.tilings_sphere.boundary_explicit import BoundaryExplicitDihedral
from escher.OTE.core.spherical.regularizers import (
    area_margin_loss,
    area_tail_loss,
    equal_area_loss,
    spherical_face_areas,
)
from escher.OTE.tilings_sphere import DihedralOrbifold
from escher.geometry.spherical_sanity_checks import (
    boundary_arc_ratio,
    check_covers_sphere_once,
    count_flipped_faces,
    signed_solid_angles,
)
from escher.misc.timing import PhaseTimer
from escher.rendering.camera import orbit_views, tile_centric_views
from escher.rendering.render_sphere_nvdiffrast import build_tiled_sphere, render_tiled_sphere

PATH = Path(__file__).parent.absolute()


def texture_tv(texture: torch.Tensor) -> torch.Tensor:
    """Mean squared finite differences of the texture -- the anti-"knitted" prior.

    SDS's leftover high-frequency structure reads as yarn/fabric; the target style is
    flat fills. A gentle total-variation-style term pushes toward flat regions while
    leaving edges (whose cost is localized) affordable.
    """
    dx = texture[1:, :] - texture[:-1, :]
    dy = texture[:, 1:] - texture[:, :-1]
    return dx.square().mean() + dy.square().mean()


# Shared with the planar pipeline; re-exported here because tests and older call sites
# import them from main_sphere.
from escher.guidance.schedule import annealed_max_step, arm_sds  # noqa: E402,F401


class SphereEscher:
    """Optimises a spherical Escher tiling against a text prompt."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.DEVICE)
        torch.manual_seed(args.SEED)
        np.random.seed(args.SEED)

        self.output_dir = Path(args.OUTPUT_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._init_geometry()
        self._init_parameters()
        self._init_guidance()

    # ------------------------------------------------------------------------- setup
    def _init_geometry(self):
        a = self.args
        # Rasterizer choice for the cached default context; runs here because every
        # construction path (train, shape phase, render_final) passes through
        # _init_geometry, unlike __init__.
        from escher.rendering.renderer_nvdiffrast import set_default_context_kind

        set_default_context_kind(str(a.get("RASTER_CONTEXT", "gl")))
        if a.PARAM_MODE == "boundary":
            # Free side of the cut as direct parameters; interior follows a fixed-boundary
            # cotangent solve. Dihedral (k,2,2) on the lune by default; ORBIFOLD_CONES
            # (e.g. [2,3,4] -> octahedral, 24 tiles) switches to the kite domain.
            cones = a.get("ORBIFOLD_CONES", None)
            if cones:
                from escher.OTE.tilings_sphere.boundary_explicit_triple import (
                    BoundaryExplicitTriple,
                )

                self.b_orb = BoundaryExplicitTriple.from_resolution(
                    tuple(int(c) for c in cones), n=a.KITE_N
                )
            else:
                self.b_orb = BoundaryExplicitDihedral.from_resolution(
                    k=a.ORBIFOLD_K, n_theta=a.MESH_N_THETA, n_phi=a.MESH_N_PHI
                )
            self.mesh = self.b_orb.mesh
            self.tiler = self.b_orb.tiler()
            self.embedder = BoundaryEmbedder(
                self.mesh.edges,
                self.b_orb.A,
                self.b_orb.pin_order,
                self.mesh.points,
                weights=self.b_orb.cotan_edge_weights(),
                warm_start=a.WARM_START,
            )
        else:
            cones = a.get("ORBIFOLD_CONES", None)
            if cones:
                from escher.OTE.tilings_sphere import OctahedralOrbifold

                self.orbifold = OctahedralOrbifold.from_resolution(
                    tuple(int(c) for c in cones), n=a.KITE_N
                )
            else:
                self.orbifold = DihedralOrbifold.from_resolution(
                    k=a.ORBIFOLD_K, n_theta=a.MESH_N_THETA, n_phi=a.MESH_N_PHI
                )
            self.mesh = self.orbifold.mesh
            self.tiler = self.orbifold.tiler()
            self.embedder = SphericalEmbedder(
                self.mesh.edges,
                self.orbifold.A,
                self.orbifold.b,
                self.orbifold.initial_guess(),
                warm_start=a.WARM_START,
            )
        cones_label = (
            tuple(a.ORBIFOLD_CONES)
            if a.get("ORBIFOLD_CONES", None)
            else (a.ORBIFOLD_K, 2, 2)
        )
        print(
            f"orbifold {cones_label}, mode={a.PARAM_MODE}: {self.tiler.order} tiles | "
            f"domain {self.mesh.n_verts} verts, {len(self.mesh.edges)} edges, "
            f"{len(self.mesh.faces)} faces"
        )

    def _init_parameters(self):
        a = self.args
        # Needed before ref_areas below, which solves through edge_weights().
        from escher.geometry.cotangent_weights import cotan_edge_weights

        self._cotan_weights = torch.as_tensor(
            cotan_edge_weights(self.mesh.points, self.mesh.faces, self.mesh.edges),
            dtype=torch.float64,
        )
        # Both modes report reverts (weights mode simply never increments it).
        self._n_reverts = 0
        if a.PARAM_MODE == "boundary":
            # ~35 points on the free half of the cut, raw (normalised in boundary_b)
            self.P = torch.nn.Parameter(self.b_orb.initial_free_points().clone())
            self._P_good: torch.Tensor | None = self.P.detach().clone()
            shape_group = {"params": [self.P], "lr": a.LR_BOUNDARY}
        else:
            n_edges = len(self.mesh.edges)
            # Start from zero: sigmoid(0) = 0.5 -> uniform weights -> the undeformed lune.
            self.W = torch.nn.Parameter(torch.zeros(n_edges, dtype=torch.float64))
            shape_group = {"params": [self.W], "lr": a.LR_W}
        res = a.TEXTURE_RESOLUTION
        init_color = a.get("TEXTURE_INIT_COLOR", None)
        if init_color is None:
            init = torch.full((res, res, 3), 0.5, dtype=torch.float32)
        else:
            # A flat start (e.g. gingerbread tan) instead of neutral gray: SDS then spends
            # its budget on figure detail rather than on first fighting its way out of gray.
            init = torch.tensor(list(init_color), dtype=torch.float32).repeat(res, res, 1)
        self.texture = torch.nn.Parameter(init.to(self.device))
        # group 0 is always the shape parameters; the freeze machinery zeroes its LR
        self.optimizer = torch.optim.Adam(
            [shape_group, {"params": [self.texture], "lr": a.LR_TEXTURE}]
        )

        # Reference state for the equal-area regularizer: the per-face solid angles of
        # the state the run actually STARTS in -- the uniform-weight solve -- not the raw
        # fundamental-domain mesh layout.
        #
        # These are not the same configuration, and on the kites they are far apart: the
        # solve moves vertices by up to 0.15 on the unit sphere and rescales faces by
        # 0.054x-9.27x on (2,3,5) (0.068x-5.53x on (2,3,4), 0.43x-1.46x on the lune).
        # Referencing the raw layout therefore charged a drift penalty at step 0 for
        # drift that had not happened, and the barrier spent the optimizer's budget
        # dragging the shape toward a configuration that does not solve the system.
        # Measured on (2,3,5): reg 6960 against mask 6080 at step 0, IoU pinned at 0.6169
        # and perimeter at exactly 1.0000x -- the carve could not move at all. With the
        # reference below both area terms are exactly 0 at step 0, as a drift penalty
        # should be, on every orbifold.
        self._last_info: dict | None = None
        self._frozen_at: int | None = None
        self.faces_t = torch.as_tensor(self.mesh.faces, dtype=torch.long)
        with torch.no_grad():
            self.ref_areas = spherical_face_areas(self.solve_points(), self.faces_t).detach()
        assert (self.ref_areas > 0).all(), "the uniform-weight solve must be fold-free"

    def _init_guidance(self):
        import escher.guidance.sd as sd

        a = self.args
        cfg = sd.Config(
            pretrained_model_name_or_path=a.PRETRAINED_MODEL_NAME_OR_PATH,
            guidance_scale=a.GUIDANCE_SCALE,
            half_precision_weights=a.USE_HALF_PRECISION,
            grad_clip=[0, 2.0, 8.0, 1000] if a.CLIP_GRADIENTS_IN_SDS else None,
        )
        self.guidance = sd.StableDiffusion(cfg)
        # The silhouette pass shows the model a flat solid shape, so it gets a prompt that
        # DESCRIBES a flat solid shape. Reusing the main prompt there asks for color and
        # interior detail the pass cannot express; the score's pull toward those attributes
        # leaks through the boundary antialiasing pixels as directionless pressure, which
        # grows the perimeter without shaping it -- the measured failure of the
        # no-regularizer strong-silhouette run.
        sil_prompt = a.SILHOUETTE_PROMPT or a.PROMPT
        with torch.no_grad():
            self._positive = self.guidance.get_text_embeds(a.PROMPT)
            self._sil_positive = self.guidance.get_text_embeds(sil_prompt)
            self._negative = self.guidance.get_text_embeds(a.NEGATIVE_PROMPT)
        self.text_embeds = self.text_embeds_for(a.IMAGE_BATCH_SIZE)
        del self.guidance.text_encoder
        torch.cuda.empty_cache()
        print(f'prompt: "{a.PROMPT}"')
        print(f'silhouette prompt: "{sil_prompt}"')

    def text_embeds_for(self, batch: int, silhouette: bool = False) -> torch.Tensor:
        """``[2*batch, 77, D]`` embeddings: positives then negatives.

        Cached per (pass kind, batch size): the silhouette pass has its own prompt and may
        use a different batch from the main pass, and ``train_step`` needs them to line up.
        """
        if not hasattr(self, "_embed_cache"):
            self._embed_cache: dict[tuple[bool, int], torch.Tensor] = {}
        key = (silhouette, batch)
        if key not in self._embed_cache:
            positive = self._sil_positive if silhouette else self._positive
            self._embed_cache[key] = torch.cat(
                [positive] * batch + [self._negative] * batch
            )
        return self._embed_cache[key]

    # -------------------------------------------------------------------------- step
    def edge_weights(self) -> torch.Tensor:
        r"""Map the free parameters to strictly positive Tutte weights.

        Two conventions, selected by ``COTANGENT_RELATIVE_WEIGHTS``.

        **Absolute** (upstream, and what this project shipped): ``sigmoid(W)`` affinely
        mapped into ``[r, 1-r]``, so ``W = 0`` gives 0.5 on every edge -- UNIFORM weights.
        That convention is inherited from Generative Escher Meshes, whose base mesh is a
        square grid with no prescribed shape: any positive weights give a valid planar
        embedding and upstream never cares what the identity configuration looks like.

        It does not transfer. Our fundamental domain is a specific object with prescribed
        cone angles, and the reference solver (``Solver.m``) builds its system from
        ``cotmatrix(V,T)`` -- cotangent weights, at which the base mesh is its own harmonic
        fixed point. Measured cost of the mismatch, uniform vs cotangent:

        ==============  ==================  ==================  ================
        domain          area spread         drift from mesh     UV texel stretch
        ==============  ==================  ==================  ================
        lune (4,2,2)    42.0  ->  17.9      0.395  ->  0.0008   42x  ->  17.9x
        kite (2,3,4)    82.1  ->   1.4      0.143  ->  0.0000   82x  ->   1.4x
        kite (2,3,5)   171.7  ->   1.2      0.130  ->  0.0000  172x  ->   1.2x
        ==============  ==================  ==================  ================

        So the "undeformed" tile was never the designed kite, and because the kite's UV is
        uniform barycentric, that area spread is also the texel-density variation across a
        single tile -- score distillation deposits gradient per texel in proportion to the
        screen area it covers, so an 82x density range is an 82x range in effective
        learning rate across one tile.

        **Cotangent-relative**: ``w = w_cot * (sigmoid(W) * W_RANGE + r) / 0.5``, so
        ``W = 0`` reproduces the reference configuration exactly and ``W`` modulates around
        it. This also removes a representability problem: the absolute map spans a fixed
        ratio (39:1 at ``W_RANGE`` 0.95) while the cotangent weights themselves need
        107:1 on (2,3,4) and 306:1 on (2,3,5) -- the natural configuration was literally
        outside the parameterization. Only ratios matter here, since the Karcher energy
        ``E = sum w_ij d_ij^2`` is invariant to a global scale of ``w`` (verified: uniform
        0.5 and uniform 1.0 agree to 5e-12).
        """
        r = (1.0 - self.args.W_RANGE) / 2.0
        mult = torch.special.expit(self.W) * self.args.W_RANGE + r
        if not self.args.get("COTANGENT_RELATIVE_WEIGHTS", False):
            return mult
        return self._cotan_weights * (mult / 0.5)

    @property
    def shape_param(self) -> torch.nn.Parameter:
        """The mode's shape parameter: boundary points or edge-weight logits.

        Every mode-agnostic touch of the shape parameter must go through this. The freeze
        block once referenced ``self.W`` directly and crashed run B1 at the exact moment
        the freeze condition first fired in boundary mode -- at step ~900, mid-success.
        """
        return self.P if self.args.PARAM_MODE == "boundary" else self.W

    def apply_shape_freeze(self) -> None:
        """Stop the shape from moving, exactly.

        Zeroing the gradient alone is not a freeze: Adam's momentum keeps moving the
        parameter for many steps afterwards. Zero the group's learning rate as well.
        """
        p = self.shape_param
        if p.grad is not None:
            p.grad.zero_()
        self.optimizer.param_groups[0]["lr"] = 0.0

    def reset_shape_optimizer_state(self) -> None:
        """Clear Adam's memory for the shape parameter after a fold revert.

        Rejection resets P but not the optimizer: exp_avg keeps pointing into the fold,
        so Adam re-proposes the same folding step forever -- the octahedral shape run
        measured 394 consecutive reverts with every metric frozen. Dropping the state
        makes the optimizer relearn the wall from the hinge gradient instead.
        """
        self.optimizer.state.pop(self.shape_param, None)

    def solve_points(self) -> torch.Tensor:
        """Current-mode differentiable solve: parameters -> unit-sphere vertices."""
        if self.args.PARAM_MODE == "boundary":
            return self.embedder(self.b_orb.boundary_b(self.P))
        return self.embedder(self.edge_weights())

    def ensure_valid_shape(self) -> tuple[torch.Tensor, int, bool]:
        r"""Solve; if the boundary folded the interior, revert to the last valid ``P``.

        Fixed-boundary spherical Tutte has **no bijectivity guarantee** for non-convex
        boundaries, and run B1 showed the theory biting in practice: 19 folded faces by
        step 500, 77 by the freeze -- invisible in renders, fatal to the certificate, and
        flagged only in a log line nothing watched. Freezing on first contact would either
        lock folds in or fire before any real deformation, so instead each step is
        *projected onto the valid set by rejection*: a step that folds is undone and the
        optimizer keeps sliding along fold-free directions.

        Returns ``(points, flips, reverted)``; ``flips`` counts folds in the returned
        state (0 unless even the revert target folds, which means the run is unhealthy).
        """
        points = self.solve_points()
        if self.args.PARAM_MODE != "boundary":
            return points, count_flipped_faces(
                points.detach().cpu().numpy(), self.mesh.faces
            ), False

        flips = count_flipped_faces(points.detach().cpu().numpy(), self.mesh.faces)
        reverted = False
        if flips > 0 and self._P_good is not None:
            # Backtrack toward the last valid state instead of jumping all the way
            # back: a full revert followed by a fresh lr-sized step re-folds forever on
            # domains with short chains (the octahedral run froze at 394 consecutive
            # reverts, momentum reset or not). Accepting the largest valid fraction of
            # the step turns the fold wall into a surface the optimizer can slide along.
            proposed = self.P.detach().clone()
            for alpha in (0.5, 0.25, 0.125, 0.0):
                with torch.no_grad():
                    self.P.copy_(self._P_good + alpha * (proposed - self._P_good))
                points = self.solve_points()
                flips = count_flipped_faces(
                    points.detach().cpu().numpy(), self.mesh.faces
                )
                if flips == 0:
                    break
            reverted = True
        if flips == 0:
            self._P_good = self.P.detach().clone()
        return points, flips, reverted

    def boundary_chain_regularizers(self) -> torch.Tensor:
        r"""Spacing + smoothness penalties on the free boundary chains, in plain torch.

        Direct and cheap (no solve involved): spacing keeps consecutive geodesic segment
        lengths comparable so points cannot bunch into a pseudo-collapse, and the
        second-difference term discourages jagged, self-intersection-prone curves. These
        replace the area barriers of the weight mode -- here they act on the actual failure
        surface (the curve) instead of fighting the solve.
        """
        p = self.P / self.P.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        chains = self.b_orb.anchored_chains(p)
        spacing = p.new_zeros(())
        smooth = p.new_zeros(())
        for c in chains:
            dots = (c[:-1] * c[1:]).sum(-1).clamp(-1.0, 1.0)
            seg = torch.arccos(dots)
            spacing = spacing + seg.var() / seg.mean().clamp_min(1e-9) ** 2
            smooth = smooth + (c[2:] - 2 * c[1:-1] + c[:-2]).square().sum()
        a = self.args
        return a.BOUNDARY_SPACING_WEIGHT * spacing + a.BOUNDARY_SMOOTH_WEIGHT * smooth

    def render(
        self,
        n_views: int,
        mv: torch.Tensor | None = None,
        isolated: bool = False,
        texture: torch.Tensor | None = None,
        tint: torch.Tensor | None = None,
        shade_ambient: float | None = None,
    ):
        """Render the tiling, or a single tile alone against the background.

        ``isolated=True`` renders just the fundamental domain, so the frame contains the
        tile's **silhouette**. That is what lets score distillation shape the tile outline
        into the prompt's figure -- with the full tiling every view is completely covered by
        tiles, leaving no silhouette to push on, and only the texture can respond. The planar
        pipeline gets this for free by rendering one fundamental domain.
        """
        points = self.solve_points()
        group = self.solo_tiler if isolated else self.tiler
        sphere = build_tiled_sphere(
            points.to(self.device).float(), self.mesh.faces, self.mesh.uv, group
        )
        if mv is None and self.args.TILE_CENTRIC_VIEWS:
            centers = group.tile_centers(points.detach().cpu().numpy())
            mv = tile_centric_views(
                torch.as_tensor(centers, dtype=torch.float32),
                n_views,
                distance=(
                    self.args.ISOLATED_DISTANCE if isolated else self.args.CAMERA_DISTANCE
                ),
                angular_jitter_deg=self.args.VIEW_JITTER_DEG,
            )
        images, alpha = render_tiled_sphere(
            sphere,
            self.texture if texture is None else texture,
            n_views=n_views,
            image_size=self.args.RENDER_SIZE,
            distance=self.args.CAMERA_DISTANCE,
            fovy_deg=self.args.CAMERA_FOV,
            mv=mv,
            tile_color_matrices=tint,
            # Shading is a presentation-only cue, exactly like `tint`: training passes
            # leave both off so what SDS sees is unchanged and every prior run stays
            # comparable.
            shade_ambient=shade_ambient,
        )
        return images, alpha, points

    @property
    def solid_texture(self) -> torch.Tensor:
        """A constant-colour texture, used for the silhouette pass.

        Not a parameter: it carries no gradient, which is exactly the point.
        """
        if not hasattr(self, "_solid_texture"):
            self._solid_texture = torch.full(
                (8, 8, 3),
                float(self.args.SILHOUETTE_FIGURE_VALUE),
                device=self.device,
                dtype=torch.float32,
            )
        return self._solid_texture

    def silhouette_loss(self, n_views: int) -> torch.Tensor:
        r"""SDS on a flat-coloured, untextured tile: a pure *shape* signal.

        The textured passes let the optimiser cheat. It can paint a convincing gingerbread
        man **inside** an unchanged tile, because the texture has far more capacity than the
        outline and is the easier descent direction. The tile then stays a smooth lune with a
        picture on it, which is not an Escher tiling -- in a real one the *outline itself* is
        the figure, which is what makes neighbours interlock.

        Rendering the tile as a solid colour on a contrasting background removes that escape
        route: the interior is constant, so the only thing that can change the image is the
        silhouette. nvdiffrast's ``antialias`` supplies the vertex-position gradients along
        those boundary edges, and they reach the edge weights through the implicit solve.
        The texture receives nothing from this term.

        MEASURED CAVEAT (probe, 2026-08-06): with a uniform texture this render's alpha
        AND the full composite return vertex gradients of exactly zero -- the antialias
        contribution never arrives. This pass therefore shaped nothing on its own; outline
        movement in past runs came from the textured pass's texture-slide gradients plus
        SDS noise. The deterministic shape phase (``main_shape.py``, analytic silhouette)
        replaces it as the outline driver; this term is retained for reference.
        """
        images, alpha, _ = self.render(
            n_views, isolated=True, texture=self.solid_texture
        )
        background = torch.full_like(images, float(self.args.SILHOUETTE_BACKGROUND_VALUE))
        composited = images * alpha + background * (1.0 - alpha)
        loss, _ = self.guidance.train_step(
            composited, self.text_embeds_for(n_views, silhouette=True)
        )
        return loss

    @property
    def solo_tiler(self):
        """A trivial one-element group: renders the fundamental domain by itself."""
        from escher.geometry.sphere_tiler import SphericalTiler

        if not hasattr(self, "_solo_tiler"):
            self._solo_tiler = SphericalTiler(rotations=np.eye(3)[None], cone_orders=None)
        return self._solo_tiler

    def shape_frozen(self, iteration: int) -> bool:
        """Whether the shape phase has ended, by step count or by condition.

        Fixed-step freezing assumes trajectories are reproducible; they are not. Two runs
        with identical settings and seed diverged wildly (perim 1.47 vs 1.08 at step 400)
        because CUDA nondeterminism compounds chaotically through SDS. So the freeze can
        also trigger on the *state*: once the perimeter target is reached, or once the
        spread ceiling is hit, whichever comes first. Latched: once frozen, stays frozen.
        """
        if getattr(self, "_frozen_at", None) is not None:
            return True
        a = self.args
        frozen = iteration >= a.FREEZE_SHAPE_AFTER
        if not frozen and self._last_info is not None:
            if a.FREEZE_ON_PERIM > 0 and self._last_info["boundary_ratio"] >= a.FREEZE_ON_PERIM:
                frozen = True
            if a.FREEZE_ON_SPREAD > 0 and self._last_info["area_spread"] >= a.FREEZE_ON_SPREAD:
                frozen = True
        if frozen:
            self._frozen_at = iteration
            print(f"shape frozen at step {iteration}", flush=True)
        return frozen

    def step(self, iteration: int) -> dict:
        a = self.args
        self.optimizer.zero_grad()
        if a.CLAMP_TEXTURE:
            self.texture.data.clamp_(0.0, 1.0)

        # Grad-clip arming + timestep-window anneal (escher/guidance/schedule.py).
        arm_sds(self.guidance, a, iteration)

        frozen = self.shape_frozen(iteration)

        # Validity projection BEFORE anything uses this step's geometry.
        with self.timer.phase("solve"):
            _, flips, reverted = self.ensure_valid_shape()
        if reverted:
            self._n_reverts += 1
            self.reset_shape_optimizer_state()

        # Alternate between the two framings. Isolated views give SDS a silhouette to shape
        # the tile outline with; tiled views make the texture read correctly in context.
        # The fraction sets the actual cadence (0.5 -> every 2nd step), not just on/off.
        frac = a.ISOLATED_TILE_FRACTION
        isolated = frac > 0 and iteration % max(1, round(1.0 / frac)) == 0
        with self.timer.phase("render"):
            images, alpha, points = self.render(a.IMAGE_BATCH_SIZE, isolated=isolated)

        if a.RANDOM_BACKGROUND:
            bg = torch.rand(images.shape[0], 1, 1, 3, device=self.device)
        else:
            bg = torch.ones(1, 1, 1, 3, device=self.device)
        composited = images * alpha + bg * (1.0 - alpha)

        # train_step returns (loss, sampled timestep); the timestep is diagnostic only.
        with self.timer.phase("sds"):
            loss, timestep = self.guidance.train_step(composited, self.text_embeds)

        # Shape regularizers ride the main backward: their graphs are tiny (points -> areas
        # and W -> sum of squares), so unlike a second diffusion pass they cost no VRAM.
        # W_REGULARIZATION mirrors main.py line 615; the equal-area term is the spherical
        # guard against the measured perimeter-by-degeneracy failure.
        reg = 0.0
        if not frozen:
            if a.EQUAL_AREA_WEIGHT > 0:
                reg = reg + a.EQUAL_AREA_WEIGHT * equal_area_loss(
                    points, self.faces_t, self.ref_areas
                )
            if a.AREA_TAIL_WEIGHT > 0:
                reg = reg + a.AREA_TAIL_WEIGHT * area_tail_loss(
                    points, self.faces_t, self.ref_areas, fraction=a.AREA_TAIL_FRACTION
                )
            if a.PARAM_MODE == "boundary":
                reg = reg + self.boundary_chain_regularizers()
                if a.BOUNDARY_MARGIN_WEIGHT > 0:
                    reg = reg + a.BOUNDARY_MARGIN_WEIGHT * area_margin_loss(
                        points, self.faces_t, self.ref_areas, margin=a.BOUNDARY_MARGIN
                    )
            elif a.W_REGULARIZATION > 0:
                reg = reg + a.W_REGULARIZATION * (self.W**2).sum()
        # The TV term lives on the texture's own device/dtype, so it adds to the loss
        # directly (unlike the float64 CPU shape regs, which need the .to()).
        tv_w = a.get("TEXTURE_TV_WEIGHT", 0.0)
        if tv_w > 0:
            loss = loss + tv_w * texture_tv(self.texture)
        if torch.is_tensor(reg):
            loss = loss + reg.to(loss)

        # Back-propagate this pass BEFORE building the silhouette graph. Summing the two
        # losses and calling backward once is equivalent mathematically but keeps both
        # diffusion graphs alive at the same time, which took peak VRAM from 9.1 to 11.8 GiB
        # of the 12 available -- close enough to the limit that the allocator thrashed and
        # the step rate collapsed from 0.54 to over 2.3 s. Separate backwards accumulate
        # into the same .grad buffers, so the result is identical and the first graph is
        # freed before the second is built.
        with self.timer.phase("backward"):
            loss.backward()
        total_loss = float(loss.detach())

        sil = 0.0
        use_silhouette = (
            a.SILHOUETTE_WEIGHT > 0
            and a.SILHOUETTE_EVERY > 0
            and iteration % a.SILHOUETTE_EVERY == 0
            and not frozen
        )
        if use_silhouette:
            with self.timer.phase("backward"):
                sil_loss = a.SILHOUETTE_WEIGHT * self.silhouette_loss(
                    a.SILHOUETTE_BATCH_SIZE
                )
                sil_loss.backward()
            sil = float(sil_loss.detach())
            total_loss += sil

        with self.timer.phase("opt"):
            if frozen:
                self.apply_shape_freeze()

            self.optimizer.step()

        with self.timer.phase("stats"):
            areas = np.abs(
                signed_solid_angles(points.detach().cpu().numpy(), self.mesh.faces)
            )
            area_spread = float(
                np.percentile(areas, 99) / max(np.percentile(areas, 1), 1e-30)
            )

            self._last_info = {
                "boundary_ratio": self.boundary_ratio(points),
                "area_spread": area_spread,
            }
        self.timer.tick()
        return {
            "loss": total_loss,
            "silhouette": sil,
            "flips": flips,
            "reverts": getattr(self, "_n_reverts", 0),
            "area_reg": float(reg.detach()) if torch.is_tensor(reg) else 0.0,
            "boundary_ratio": self._last_info["boundary_ratio"],
            "area_spread": area_spread,
            "frozen_at": self._frozen_at,
            "timestep": float(timestep.float().mean()),
            "energy": self.embedder.last_result.energy,
            "solver_iters": (
                self.embedder.last_result.stage1.n_iter
                + self.embedder.last_result.stage2.n_iter
            ),
            "points": points,
        }

    # ------------------------------------------------------------------- diagnostics
    TIMING_PHASES = (
        "solve",
        "render",
        "sds",
        "backward",
        "opt",
        "stats",
        "geom_check",
        "snapshot",
        "checkpoint",
    )

    @property
    def timer(self) -> PhaseTimer:
        """Lazy so the ``__new__``-built paths (shape phase, render_final) get one too."""
        t = getattr(self, "_timer", None)
        if t is None:
            t = PhaseTimer(
                enabled=bool(self.args.get("TIMING", False)),
                cuda_sync=self.device.type == "cuda",
            )
            self._timer = t
        return t

    def log_timing(self, iteration: int) -> None:
        """One row of per-phase mean ms/step over the window since the last row."""
        path = self.output_dir / "timing.csv"
        new = not path.exists()
        means = self.timer.means_ms(self.TIMING_PHASES)
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write("step," + ",".join(self.TIMING_PHASES) + ",total_ms\n")
            f.write(
                f"{iteration},"
                + ",".join(f"{m:.1f}" for m in means)
                + f",{sum(means):.1f}\n"
            )
        self.timer.reset()

    def boundary_ratio(self, points: torch.Tensor) -> float:
        """Tile perimeter relative to the undeformed lune -- see
        :func:`~escher.geometry.spherical_sanity_checks.boundary_arc_ratio`."""
        return boundary_arc_ratio(
            points.detach().cpu().numpy(),
            self.mesh.points,
            tuple(self.mesh.boundary_chains),
        )

    def log_metrics(self, iteration: int, info: dict) -> None:
        """Append one row to ``metrics.csv``.

        Written by the process itself rather than scraped from stdout: a previous run's log
        was piped through a tail-only filter at launch and the history was lost, taking the
        matched-step comparison with it.
        """
        path = self.output_dir / "metrics.csv"
        new = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if new:
                f.write(
                    "step,loss,silhouette,area_reg,karcher,"
                    "boundary_ratio,area_spread,flips,reverts,solver_iters\n"
                )
            f.write(
                f"{iteration},{info['loss']:.4f},{info['silhouette']:.4f},"
                f"{info['area_reg']:.4f},{info['energy']:.6f},"
                f"{info['boundary_ratio']:.6f},{info['area_spread']:.2f},"
                f"{info['flips']},{info['reverts']},{info['solver_iters']}\n"
            )

    def check_geometry(self, points: torch.Tensor) -> tuple[bool, str]:
        """Is the current tile still a valid, fold-free tiling?"""
        pts = points.detach().cpu().numpy()
        verts, faces, _ = self.tiler.tile_mesh(pts, self.mesh.faces)
        return check_covers_sphere_once(verts, faces)

    def save_checkpoint(self, iteration: int) -> Path:
        """Persist everything needed to re-render or resume.

        Writes a step-tagged file **and** updates ``checkpoint.pt``. Earlier runs kept only
        the latter, overwriting it each time, which made it impossible to compare two runs at
        a matched step after the fact -- the state had already been overwritten by the time
        the comparison was wanted. The state is small (edge weights plus one texture), so
        keeping every snapshot costs little.
        """
        payload = {
            "iteration": iteration,
            "texture": self.texture.detach().cpu(),
            "optimizer": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.args, resolve=True),
        }
        if self.args.PARAM_MODE == "boundary":
            payload["P"] = self.P.detach().cpu()
        else:
            payload["W"] = self.W.detach().cpu()
        tagged = self.output_dir / f"checkpoint_{iteration:06d}.pt"
        torch.save(payload, tagged)
        torch.save(payload, self.output_dir / "checkpoint.pt")
        return tagged

    def load_checkpoint(self, path: str | Path, reset_texture: bool = False) -> int:
        """``reset_texture`` is a PER-CALL choice, never read from config: the flag ends
        up saved inside the checkpoint's own config, and reading it ambiently made
        render_final load a texture-phase checkpoint as flat init blobs. Only run()'s
        cross-phase RESUME wants it (shape from the checkpoint, this run's fresh
        texture -- the shape phase never trained its texture)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            if "P" in state:
                self.P.copy_(state["P"])
            else:
                self.W.copy_(state["W"])
            if not reset_texture:
                self.texture.copy_(state["texture"].to(self.device))
        self.optimizer.load_state_dict(state["optimizer"])
        return int(state["iteration"])

    def save_snapshot(self, iteration: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # The wide panels show the deliverable look, tints included when configured; the
        # close-up must stay untinted -- it exists to show what SDS actually sees.
        tint = None
        if self.args.get("TILE_TINT", False):
            from escher.rendering.palette import tile_color_matrices

            tint = tile_color_matrices(
                self.tiler, self.mesh, list(self.args.PALETTE_HUES_DEG)
            )

        with torch.no_grad():
            # whole-sphere overview, plus one training-style close-up so both the tiling and
            # the figure the model actually sees are visible in the same snapshot
            wide = orbit_views(3, distance=self.args.PREVIEW_DISTANCE, elevation_deg=18.0)
            images, alpha, points = self.render(3, mv=wide, tint=tint)
            comp = (images * alpha + 1.0 * (1 - alpha)).clamp(0, 1).cpu().numpy()

            centers = self.tiler.tile_centers(points.detach().cpu().numpy())
            close = tile_centric_views(
                torch.as_tensor(centers, dtype=torch.float32),
                1,
                distance=self.args.CAMERA_DISTANCE,
                angular_jitter_deg=0.0,
            )
            c_img, c_alpha, _ = self.render(1, mv=close)
            close_up = (c_img * c_alpha + 1.0 * (1 - c_alpha)).clamp(0, 1).cpu().numpy()[0]

        fig, axes = plt.subplots(1, 5, figsize=(19, 4.2))
        axes[0].imshow(self.texture.detach().clamp(0, 1).cpu().numpy())
        axes[0].set_title("shared texture", fontsize=10)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        for i in range(3):
            axes[i + 1].imshow(comp[i])
            axes[i + 1].set_axis_off()
        axes[4].imshow(close_up)
        axes[4].set_axis_off()
        axes[4].set_title("what SDS sees", fontsize=10)
        fig.suptitle(f'step {iteration} — "{self.args.PROMPT}"', fontsize=12)
        fig.tight_layout()
        fig.savefig(self.output_dir / f"step_{iteration:05d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)

    # -------------------------------------------------------------------------- loop
    def run(self, resume_from: str | Path | None = None) -> None:
        """Optimise. With ``resume_from``, continue an interrupted run in place.

        Long runs are ~1 hour, so losing one to an interruption is expensive; checkpoints
        were already being written every ``VISUALIZATION_FREQ`` steps but nothing consumed
        them.
        """
        a = self.args
        first_step = 0
        if resume_from is not None:
            first_step = (
                self.load_checkpoint(
                    resume_from,
                    reset_texture=a.get("RESET_TEXTURE_ON_RESUME", False),
                )
                + 1
            )
            print(f"resuming from {resume_from} at step {first_step}")
            if first_step >= a.N_STEPS:
                print("checkpoint is already at or past N_STEPS; nothing to do")
                return

        start = time.time()
        last_print = (start, first_step)
        for iteration in range(first_step, a.N_STEPS + 1):
            info = self.step(iteration)

            if iteration % 10 == 0:
                self.log_metrics(iteration, info)
                if self.timer.enabled:
                    self.log_timing(iteration)
                now = time.time()
                per_step = (now - start) / max(iteration - first_step, 1)
                # Instantaneous (since the last print) next to cumulative: the
                # cumulative average buries a mid-run slowdown for thousands of steps.
                inst = (now - last_print[0]) / max(iteration - last_print[1], 1)
                last_print = (now, iteration)
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(
                    f"step {iteration:5d} | loss {info['loss']:9.1f} | "
                    f"sil {info['silhouette']:8.1f} | reg {info['area_reg']:8.1f} | "
                    f"karcher {info['energy']:7.4f} | "
                    f"perim {info['boundary_ratio']:6.4f}x | "
                    f"spread {info['area_spread']:6.1f} | "
                    f"flp {info['flips']:2d}/{info['reverts']:3d} | "
                    f"solver {info['solver_iters']:3d} it | "
                    f"{inst:5.2f}/{per_step:5.2f} s/step | {mem:4.1f} GiB",
                    flush=True,
                )

            if iteration % a.VISUALIZATION_FREQ == 0:
                with self.timer.phase("geom_check"):
                    ok, message = self.check_geometry(info["points"])
                if not ok:
                    print(f"  !! geometry check failed: {message}")
                with self.timer.phase("snapshot"):
                    self.save_snapshot(iteration)
                with self.timer.phase("checkpoint"):
                    self.save_checkpoint(iteration)

        self.save_checkpoint(a.N_STEPS)
        print(f"\ndone in {(time.time() - start) / 60:.1f} min -> {self.output_dir}")


def load_sphere_args(cli):
    """sphere.yaml -> each CONF_FILE overlay, left to right -> CLI.

    ``CONF_FILE`` is an OVERLAY on the base config (``sphere_texture.yaml`` holds only
    its phase's deltas), so the base always loads first, and it accepts a **list**.
    A single overlay level is a trap: an orbifold-specific config built on top of the
    texture phase gets merged onto ``sphere.yaml`` alone and silently drops every delta
    it meant to inherit -- prompt, negative prompt, guidance, TV weight, texture init.
    Measured by losing exactly those and getting back the high-frequency texture they
    exist to prevent.
    """
    conf_file = cli.pop("CONF_FILE", "configs/sphere.yaml")
    conf_files = [conf_file] if isinstance(conf_file, str) else list(conf_file)
    layers = [OmegaConf.load(PATH / "configs/sphere.yaml")]
    layers += [
        OmegaConf.load(PATH / f) for f in conf_files if f != "configs/sphere.yaml"
    ]
    return OmegaConf.merge(*layers, cli)


def main() -> None:
    cli = OmegaConf.from_cli()
    resume = cli.pop("RESUME", None)
    SphereEscher(load_sphere_args(cli)).run(resume_from=resume)


if __name__ == "__main__":
    main()
