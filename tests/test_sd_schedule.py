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
