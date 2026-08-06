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

from escher.OTE.core.spherical.differentiable import SphericalEmbedder
from escher.OTE.tilings_sphere import DihedralOrbifold
from escher.geometry.spherical_sanity_checks import check_covers_sphere_once, count_flipped_faces
from escher.rendering.camera import orbit_views, tile_centric_views
from escher.rendering.render_sphere_nvdiffrast import build_tiled_sphere, render_tiled_sphere

PATH = Path(__file__).parent.absolute()


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
        print(
            f"orbifold ({a.ORBIFOLD_K},2,2): {self.tiler.order} tiles | "
            f"domain {self.mesh.n_verts} verts, {len(self.mesh.edges)} edges, "
            f"{len(self.mesh.faces)} faces"
        )

    def _init_parameters(self):
        a = self.args
        n_edges = len(self.mesh.edges)
        # Start from zero: sigmoid(0) = 0.5, i.e. uniform weights, i.e. the undeformed lune.
        self.W = torch.nn.Parameter(torch.zeros(n_edges, dtype=torch.float64))
        res = a.TEXTURE_RESOLUTION
        self.texture = torch.nn.Parameter(
            torch.full((res, res, 3), 0.5, device=self.device, dtype=torch.float32)
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.W], "lr": a.LR_W},
                {"params": [self.texture], "lr": a.LR_TEXTURE},
            ]
        )

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
        with torch.no_grad():
            self._positive = self.guidance.get_text_embeds(a.PROMPT)
            self._negative = self.guidance.get_text_embeds(a.NEGATIVE_PROMPT)
        self.text_embeds = self.text_embeds_for(a.IMAGE_BATCH_SIZE)
        del self.guidance.text_encoder
        torch.cuda.empty_cache()
        print(f'prompt: "{a.PROMPT}"')

    def text_embeds_for(self, batch: int) -> torch.Tensor:
        """``[2*batch, 77, D]`` embeddings: positives then negatives.

        Cached per batch size, because the silhouette pass may use a different one from the
        main pass and ``train_step`` requires the two to line up.
        """
        if not hasattr(self, "_embed_cache"):
            self._embed_cache: dict[int, torch.Tensor] = {}
        if batch not in self._embed_cache:
            self._embed_cache[batch] = torch.cat(
                [self._positive] * batch + [self._negative] * batch
            )
        return self._embed_cache[batch]

    # -------------------------------------------------------------------------- step
    def edge_weights(self) -> torch.Tensor:
        """Map the free parameters into ``[r, 1-r]``, strictly positive as Tutte requires."""
        r = (1.0 - self.args.W_RANGE) / 2.0
        return torch.special.expit(self.W) * self.args.W_RANGE + r

    def render(
        self,
        n_views: int,
        mv: torch.Tensor | None = None,
        isolated: bool = False,
        texture: torch.Tensor | None = None,
    ):
        """Render the tiling, or a single tile alone against the background.

        ``isolated=True`` renders just the fundamental domain, so the frame contains the
        tile's **silhouette**. That is what lets score distillation shape the tile outline
        into the prompt's figure -- with the full tiling every view is completely covered by
        tiles, leaving no silhouette to push on, and only the texture can respond. The planar
        pipeline gets this for free by rendering one fundamental domain.
        """
        points = self.embedder(self.edge_weights())
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
        """
        images, alpha, _ = self.render(
            n_views, isolated=True, texture=self.solid_texture
        )
        background = torch.full_like(images, float(self.args.SILHOUETTE_BACKGROUND_VALUE))
        composited = images * alpha + background * (1.0 - alpha)
        loss, _ = self.guidance.train_step(
            composited, self.text_embeds_for(n_views)
        )
        return loss

    @property
    def solo_tiler(self):
        """A trivial one-element group: renders the fundamental domain by itself."""
        from escher.geometry.sphere_tiler import SphericalTiler

        if not hasattr(self, "_solo_tiler"):
            self._solo_tiler = SphericalTiler(rotations=np.eye(3)[None], cone_orders=None)
        return self._solo_tiler

    def step(self, iteration: int) -> dict:
        a = self.args
        self.optimizer.zero_grad()
        if a.CLAMP_TEXTURE:
            self.texture.data.clamp_(0.0, 1.0)

        # Alternate between the two framings. Isolated views give SDS a silhouette to shape
        # the tile outline with; tiled views make the texture read correctly in context.
        isolated = (iteration % 2 == 0) and a.ISOLATED_TILE_FRACTION > 0
        images, alpha, points = self.render(a.IMAGE_BATCH_SIZE, isolated=isolated)

        if a.RANDOM_BACKGROUND:
            bg = torch.rand(images.shape[0], 1, 1, 3, device=self.device)
        else:
            bg = torch.ones(1, 1, 1, 3, device=self.device)
        composited = images * alpha + bg * (1.0 - alpha)

        # train_step returns (loss, sampled timestep); the timestep is diagnostic only.
        loss, timestep = self.guidance.train_step(composited, self.text_embeds)

        # Back-propagate this pass BEFORE building the silhouette graph. Summing the two
        # losses and calling backward once is equivalent mathematically but keeps both
        # diffusion graphs alive at the same time, which took peak VRAM from 9.1 to 11.8 GiB
        # of the 12 available -- close enough to the limit that the allocator thrashed and
        # the step rate collapsed from 0.54 to over 2.3 s. Separate backwards accumulate
        # into the same .grad buffers, so the result is identical and the first graph is
        # freed before the second is built.
        loss.backward()
        total_loss = float(loss.detach())

        sil = 0.0
        use_silhouette = (
            a.SILHOUETTE_WEIGHT > 0
            and a.SILHOUETTE_EVERY > 0
            and iteration % a.SILHOUETTE_EVERY == 0
            and iteration < a.FREEZE_SHAPE_AFTER
        )
        if use_silhouette:
            sil_loss = a.SILHOUETTE_WEIGHT * self.silhouette_loss(a.SILHOUETTE_BATCH_SIZE)
            sil_loss.backward()
            sil = float(sil_loss.detach())
            total_loss += sil

        if iteration >= a.FREEZE_SHAPE_AFTER and self.W.grad is not None:
            self.W.grad.zero_()

        self.optimizer.step()
        return {
            "loss": total_loss,
            "silhouette": sil,
            "timestep": float(timestep.float().mean()),
            "energy": self.embedder.last_result.energy,
            "solver_iters": (
                self.embedder.last_result.stage1.n_iter
                + self.embedder.last_result.stage2.n_iter
            ),
            "points": points,
        }

    # ------------------------------------------------------------------- diagnostics
    def check_geometry(self, points: torch.Tensor) -> tuple[bool, str]:
        """Is the current tile still a valid, fold-free tiling?"""
        pts = points.detach().cpu().numpy()
        verts, faces, _ = self.tiler.tile_mesh(pts, self.mesh.faces)
        return check_covers_sphere_once(verts, faces)

    def save_checkpoint(self, iteration: int) -> Path:
        """Persist everything needed to re-render or resume.

        The optimised state is small (edge weights plus one texture), so this is cheap and
        worth doing often -- a run is ~25 minutes and snapshots alone cannot be reloaded.
        """
        path = self.output_dir / "checkpoint.pt"
        torch.save(
            {
                "iteration": iteration,
                "W": self.W.detach().cpu(),
                "texture": self.texture.detach().cpu(),
                "optimizer": self.optimizer.state_dict(),
                "config": OmegaConf.to_container(self.args, resolve=True),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> int:
        state = torch.load(path, map_location="cpu", weights_only=False)
        with torch.no_grad():
            self.W.copy_(state["W"])
            self.texture.copy_(state["texture"].to(self.device))
        self.optimizer.load_state_dict(state["optimizer"])
        return int(state["iteration"])

    def save_snapshot(self, iteration: int) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with torch.no_grad():
            # whole-sphere overview, plus one training-style close-up so both the tiling and
            # the figure the model actually sees are visible in the same snapshot
            wide = orbit_views(3, distance=self.args.PREVIEW_DISTANCE, elevation_deg=18.0)
            images, alpha, points = self.render(3, mv=wide)
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
            first_step = self.load_checkpoint(resume_from) + 1
            print(f"resuming from {resume_from} at step {first_step}")
            if first_step >= a.N_STEPS:
                print("checkpoint is already at or past N_STEPS; nothing to do")
                return

        start = time.time()
        for iteration in range(first_step, a.N_STEPS + 1):
            info = self.step(iteration)

            if iteration % 10 == 0:
                elapsed = time.time() - start
                per_step = elapsed / max(iteration, 1)
                mem = torch.cuda.max_memory_allocated() / 2**30
                sil = (
                    f"sil {info['silhouette']:9.1f} | " if info.get("silhouette") else ""
                )
                print(
                    f"step {iteration:5d} | loss {info['loss']:9.1f} | {sil}"
                    f"karcher {info['energy']:8.4f} | solver {info['solver_iters']:3d} it | "
                    f"{per_step:5.2f} s/step | {mem:4.1f} GiB"
                )

            if iteration % a.VISUALIZATION_FREQ == 0:
                ok, message = self.check_geometry(info["points"])
                if not ok:
                    print(f"  !! geometry check failed: {message}")
                self.save_snapshot(iteration)
                self.save_checkpoint(iteration)

        self.save_checkpoint(a.N_STEPS)
        print(f"\ndone in {(time.time() - start) / 60:.1f} min -> {self.output_dir}")


def main() -> None:
    cli = OmegaConf.from_cli()
    conf_file = cli.pop("CONF_FILE", "configs/sphere.yaml")
    resume = cli.pop("RESUME", None)
    args = OmegaConf.merge(OmegaConf.load(PATH / conf_file), cli)
    SphereEscher(args).run(resume_from=resume)


if __name__ == "__main__":
    main()
