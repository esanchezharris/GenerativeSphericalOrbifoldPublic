"""Orbifold boundary conditions for spherical tilings.

The spherical counterpart of :mod:`escher.OTE.tilings`, which holds the 17 planar wallpaper
groups. Here the symmetry groups are the finite rotation groups of the sphere.
"""

from escher.OTE.tilings_sphere.dihedral import DihedralOrbifold
from escher.OTE.tilings_sphere.sparse_system_3d import SparseSystem3D

__all__ = ["DihedralOrbifold", "SparseSystem3D"]
