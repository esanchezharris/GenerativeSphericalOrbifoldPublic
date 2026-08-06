"""Draw a mesh on the unit sphere with matplotlib.

matplotlib's 3D axes have no depth buffer, so a ``plot_surface`` reference sphere drawn
alongside a ``Line3DCollection`` will occlude it unpredictably. Everything here instead
projects orthographically to 2D and culls the far hemisphere by hand, which gives correct
occlusion and much cleaner output.
"""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle

__all__ = ["camera_basis", "draw_sphere_edges", "draw_sphere_faces"]

_TILE_COLORS = [
    "#3d6fb4", "#e08a3c", "#4e9a5a", "#c94f4f", "#8a6bbf",
    "#8c6244", "#d183b6", "#7f7f7f", "#b5b830", "#3fa9bd",
]


def camera_basis(elev_deg: float, azim_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(right, up, view)`` unit vectors for an orthographic camera.

    ``view`` points from the origin toward the camera, so a point ``p`` is on the visible
    hemisphere when ``p @ view > 0``.
    """
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    view = np.array(
        [np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)]
    )
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(view @ world_up) > 0.999:  # looking down the pole: pick another reference
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, view)
    right /= np.linalg.norm(right)
    up = np.cross(view, right)
    return right, up, view


def _setup_axes(ax: Axes, face_color: str = "#fbfbfd") -> None:
    ax.add_patch(Circle((0, 0), 1.0, facecolor=face_color, edgecolor="#d8d8e0", lw=0.8, zorder=0))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_sphere_edges(
    ax: Axes,
    points: np.ndarray,
    edges: np.ndarray,
    elev: float = 20.0,
    azim: float = 40.0,
    color: str = "#12305c",
    linewidth: float = 0.35,
    show_hidden: bool = True,
) -> None:
    """Draw mesh edges as an orthographic projection of the sphere.

    Args:
        points: ``(n, 3)`` unit-sphere vertices.
        edges: ``(n_edges, 2)`` index pairs.
        show_hidden: if set, far-side edges are drawn very faintly instead of dropped,
            which reads as a wireframe globe.
    """
    right, up, view = camera_basis(elev, azim)
    xy = np.stack([points @ right, points @ up], axis=1)
    depth = points @ view

    segs = xy[edges]
    seg_depth = depth[edges].mean(axis=1)
    front = seg_depth > 0

    if show_hidden and (~front).any():
        ax.add_collection(
            LineCollection(segs[~front], colors=color, linewidths=linewidth * 0.7, alpha=0.10)
        )
    ax.add_collection(
        LineCollection(segs[front], colors=color, linewidths=linewidth, alpha=0.9)
    )
    _setup_axes(ax)


def draw_sphere_faces(
    ax: Axes,
    points: np.ndarray,
    faces: np.ndarray,
    tile_index: np.ndarray | None = None,
    elev: float = 20.0,
    azim: float = 40.0,
    edgecolor: str = "#ffffff",
    linewidth: float = 0.15,
) -> None:
    """Draw filled triangles, colouring each tile copy differently.

    Back-facing triangles are culled by outward normal, and the rest are painted
    back-to-front so nearer triangles win.
    """
    right, up, view = camera_basis(elev, azim)
    xy = np.stack([points @ right, points @ up], axis=1)

    tri = points[faces]  # (n_faces, 3, 3)
    centroids = tri.mean(axis=1)

    # Painter's algorithm over *all* faces, back to front. Culling the far hemisphere first
    # is tempting and faster, but triangles near the silhouette sit almost edge-on: they get
    # culled while still projecting to visible area, leaving thin background-coloured slivers
    # around the rim. Drawing everything in depth order costs 2x the polygons and has no
    # such gaps.
    idx = np.argsort(centroids @ view)

    if tile_index is None:
        colors = [_TILE_COLORS[0]] * len(idx)
    else:
        colors = [_TILE_COLORS[tile_index[i] % len(_TILE_COLORS)] for i in idx]

    ax.add_collection(
        PolyCollection(
            xy[faces[idx]], facecolors=colors, edgecolors=edgecolor, linewidths=linewidth
        )
    )
    _setup_axes(ax)
