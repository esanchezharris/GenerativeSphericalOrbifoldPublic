"""The SDS hygiene knobs: the grad-clip schedule and the timestep window.

The grad-clip finding, pinned here so it cannot regress silently again: ``update_step``
is the only code that arms ``grad_clip_val`` from the ``CLIP_GRADIENTS_IN_SDS`` schedule,
and nothing in this repo (or upstream ``main.py``) ever called it -- so SDS gradient
clipping was a no-op for every run of the project. ``main_sphere.step()`` now calls it
every iteration. These tests exercise the schedule arithmetic without loading weights.
"""

from __future__ import annotations

import pytest

pytest.importorskip("diffusers")

from escher.guidance.sd import Config, StableDiffusion  # noqa: E402


def bare_guidance(**cfg_kwargs) -> StableDiffusion:
    """A StableDiffusion with no weights loaded: enough for schedule arithmetic."""
    g = StableDiffusion.__new__(StableDiffusion)
    g.cfg = Config(**cfg_kwargs)
    g.num_train_timesteps = 1000
    g.grad_clip_val = None
    return g


def test_update_step_arms_the_clip_schedule():
    g = bare_guidance(grad_clip=[0, 2.0, 8.0, 1000])
    assert g.grad_clip_val is None, "the state every historical run trained in"
    g.update_step(0, 0)
    assert g.grad_clip_val == pytest.approx(2.0)
    g.update_step(0, 500)
    assert g.grad_clip_val == pytest.approx(5.0)
    g.update_step(0, 1000)
    assert g.grad_clip_val == pytest.approx(8.0)
    g.update_step(0, 5000)
    assert g.grad_clip_val == pytest.approx(8.0), "schedule clamps past its end step"


def test_update_step_without_schedule_stays_disarmed():
    g = bare_guidance(grad_clip=None)
    g.update_step(0, 123)
    assert g.grad_clip_val is None


def test_set_step_range_moves_the_sampling_window():
    g = bare_guidance()
    g.set_step_range(0.02, 0.98)
    assert (g.min_step, g.max_step) == (20, 980)
    g.set_step_range(0.02, 0.5)
    assert (g.min_step, g.max_step) == (20, 500)


def test_arm_sds_arms_clip_and_anneal():
    from omegaconf import OmegaConf

    from escher.guidance.schedule import arm_sds

    g = bare_guidance(grad_clip=[0, 2.0, 8.0, 1000])
    args = OmegaConf.create(
        {"SDS_ANNEAL_END": 800, "SDS_MAX_START": 0.98, "SDS_MAX_END": 0.5}
    )
    arm_sds(g, args, 400)
    assert g.grad_clip_val == pytest.approx(2.0 + 6.0 * 0.4)
    assert g.min_step == 20  # cfg.min_step_percent default 0.02
    assert g.max_step == int(1000 * 0.74)


def test_arm_sds_tolerates_guidance_without_step_range():
    """DeepFloyd has update_step but no set_step_range; arm_sds must not crash."""
    from omegaconf import OmegaConf

    from escher.guidance.schedule import arm_sds

    class Stub:
        armed = None

        def update_step(self, epoch, global_step):
            self.armed = global_step

    s = Stub()
    arm_sds(s, OmegaConf.create({"SDS_ANNEAL_END": 800}), 5)
    assert s.armed == 5


def test_texture_tv_prior():
    import torch

    from escher.main_sphere import texture_tv

    flat = torch.full((8, 8, 3), 0.42)
    assert texture_tv(flat).item() == pytest.approx(0.0, abs=1e-12)

    noisy = torch.rand(8, 8, 3, generator=torch.Generator().manual_seed(0))
    assert texture_tv(noisy).item() > 0.05

    grad_check = torch.rand(8, 8, 3, requires_grad=True)
    texture_tv(grad_check).backward()
    assert grad_check.grad is not None and grad_check.grad.abs().sum() > 0


def test_annealed_max_step_endpoints_and_clamp():
    from omegaconf import OmegaConf

    from escher.main_sphere import annealed_max_step

    args = OmegaConf.create(
        {"SDS_ANNEAL_END": 800, "SDS_MAX_START": 0.98, "SDS_MAX_END": 0.5}
    )
    assert annealed_max_step(args, 0) == pytest.approx(0.98)
    assert annealed_max_step(args, 400) == pytest.approx(0.74)
    assert annealed_max_step(args, 800) == pytest.approx(0.5)
    assert annealed_max_step(args, 5000) == pytest.approx(0.5), "clamped past the end"
