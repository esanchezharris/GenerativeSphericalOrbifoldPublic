# Troubleshooting

### Which Python

The project runs from a WSL venv (editable install; `pip install -e .`). Run
tests as `python -m pytest tests` from the repo root — `-m` puts the repo first
on `sys.path`. Running a SCRIPT by path does **not**: the venv's `escher.pth`
then wins and your imports silently come from wherever the editable install
points. Set `PYTHONPATH=<repo>` for script invocations from a different
checkout (worktrees especially).

### Speed expectations

Carves and candidate screens run ~13x faster with `DEVICE=cuda` (the analytic
soft mask is the hot loop); the geometry solve itself is CPU float64 either
way. The texture phase wants `TORCH_COMPILE=true` for full-length runs
(measured ~18% faster and ~1.6 GiB less peak VRAM; the first step pays a
minutes-long compile). Enable `TIMING=true` to get a per-phase breakdown in
`timing.csv` before believing any speed theory.

### The frame edge

Anything outside the shape camera's frame is invisible to the carve's loss AND
metric. `run_shape` prints the undeformed tile's margin at step 0 — if it is
small or negative, fix the framing (`SHAPE_CAMERA_FOV` / `SHAPE_RENDER_SIZE`)
before trusting any number from the run.

### A10g

Xformers does not play well with A10g and must be removed there. Moot under
torch >= 2: the code uses SDPA and never enables xformers.
