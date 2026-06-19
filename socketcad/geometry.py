"""Shared geometry helpers for the socket rectification engine.

Everything in this module is plain ``numpy`` / ``trimesh`` so that the
rectification step (see :mod:`socketcad.rectify`) has the same input/output
contract a future *learned* model could satisfy:

    limb mesh + RectificationSpec  ->  rectified positive mesh

Coordinate convention used throughout the package ("canonical limb frame"):

* ``+z`` runs **distal -> proximal** along the residual limb's long axis,
  with the distal tip at ``z = 0`` and the proximal (knee) end at ``z = H``.
* ``+x`` points **anterior** (toward the tibial crest / patellar tendon).
* ``+y`` completes a right-handed frame. Whether ``+y`` is *medial* or
  *lateral* depends on whether the limb is a left or right limb, which cannot
  be recovered from an unlabelled mesh -- see :class:`LimbFrame.is_left`.

The frame is estimated heuristically (PCA long axis + an angular-radius
heuristic for the anterior direction). This is good enough for a parametric
demo but is explicitly *not* clinical-grade landmarking; its limits are
documented in ``docs/LIMITATIONS.md``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh


# --------------------------------------------------------------------------
# Frame estimation
# --------------------------------------------------------------------------
@dataclass
class LimbFrame:
    """An estimated anatomical coordinate frame for a residual-limb mesh.

    ``origin`` and ``axes`` define a rigid transform from world coordinates
    into the canonical limb frame described in the module docstring. ``axes``
    is a 3x3 matrix whose **rows** are the canonical x (anterior), y, and z
    (proximal) unit vectors expressed in world coordinates.
    """

    origin: np.ndarray  # (3,) world-space point that maps to canonical (0,0,0)
    axes: np.ndarray  # (3,3) rows = canonical [x_ant, y, z_prox] in world coords
    height: float  # limb length along the long axis (world units, mm)
    is_left: bool = True  # documented assumption; flips medial/lateral labels

    def to_canonical(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return (p - self.origin) @ self.axes.T

    def to_canonical_vectors(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors, dtype=float) @ self.axes.T

    def to_world(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float)
        return p @ self.axes + self.origin

    def to_world_vectors(self, vectors: np.ndarray) -> np.ndarray:
        return np.asarray(vectors, dtype=float) @ self.axes


def _principal_axis(vertices: np.ndarray) -> np.ndarray:
    """Return the unit long axis of a point cloud via PCA."""
    centred = vertices - vertices.mean(axis=0)
    cov = np.cov(centred.T)
    evals, evecs = np.linalg.eigh(cov)
    return evecs[:, np.argmax(evals)]


def estimate_frame(mesh: trimesh.Trimesh, is_left: bool = True) -> LimbFrame:
    """Estimate the canonical limb frame for ``mesh``.

    Steps:

    1. Long axis from PCA of the vertices.
    2. Orient it so the **proximal** (larger cross-section) end is ``+z``.
    3. Recentre the origin on the axis line at the distal tip.
    4. Anterior (``+x``) = direction *opposite* the posterior calf bulge,
       found as the angular sector with the largest mean radius across the
       mid-limb sections (calibrated against the OpenLimbTT mean shape).
    """
    V = np.asarray(mesh.vertices, dtype=float)
    axis = _principal_axis(V)

    centroid = V.mean(axis=0)
    t = (V - centroid) @ axis  # signed coordinate along the axis

    # Orient +axis toward the proximal (larger) end. Compare mean planar
    # radius of vertices near each end of the axis.
    lo_mask = t < np.quantile(t, 0.15)
    hi_mask = t > np.quantile(t, 0.85)

    def planar_radius(mask: np.ndarray) -> float:
        pts = V[mask]
        proj = pts - np.outer((pts - centroid) @ axis, axis) - centroid
        return float(np.linalg.norm(proj, axis=1).mean())

    if planar_radius(lo_mask) > planar_radius(hi_mask):
        axis = -axis
        t = -t

    # Distal tip = minimum-t end. Place the origin on the axis at that level.
    t_min, t_max = t.min(), t.max()
    height = float(t_max - t_min)
    origin = centroid + axis * t_min

    # Build provisional in-plane axes (any two perpendicular to the long axis).
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    # Anterior detection: across mid sections, find the angular direction of
    # maximum radius (the rounded posterior calf), anterior is opposite.
    rel = V - origin
    h = rel @ axis
    px = rel @ e1
    py = rel @ e2
    r = np.hypot(px, py)
    ang = np.arctan2(py, px)
    mid = (h > 0.3 * height) & (h < 0.7 * height) & (r > 1e-9)
    nb = 72
    bins = np.linspace(-np.pi, np.pi, nb + 1)
    centres = 0.5 * (bins[:-1] + bins[1:])
    idx = np.clip(np.digitize(ang[mid], bins) - 1, 0, nb - 1)
    rmid = r[mid]
    mean_r = np.array(
        [rmid[idx == i].mean() if np.any(idx == i) else 0.0 for i in range(nb)]
    )
    posterior_ang = centres[int(np.argmax(mean_r))]
    anterior_ang = posterior_ang + np.pi

    x_ant = np.cos(anterior_ang) * e1 + np.sin(anterior_ang) * e2
    x_ant /= np.linalg.norm(x_ant)
    y_axis = np.cross(axis, x_ant)  # right-handed: z cross x = y
    y_axis /= np.linalg.norm(y_axis)

    axes = np.vstack([x_ant, y_axis, axis])
    return LimbFrame(origin=origin, axes=axes, height=height, is_left=is_left)


# --------------------------------------------------------------------------
# Cylindrical coordinates & falloff
# --------------------------------------------------------------------------
def cylindrical_coords(points_canonical: np.ndarray):
    """Return (h_frac, theta, radius, radial_unit) in the canonical frame.

    * ``h_frac`` -- height as a fraction of limb length is *not* computed here
      (callers divide by ``height``); this returns raw ``z``.
    * ``theta`` -- angle in the x-y plane, 0 = anterior, measured right-handed.
    * ``radius`` -- distance from the long axis.
    * ``radial_unit`` -- (N,3) outward radial unit vectors (z component 0).
    """
    p = np.asarray(points_canonical, dtype=float)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    radius = np.hypot(x, y)
    theta = np.arctan2(y, x)
    safe = np.where(radius > 1e-9, radius, 1.0)
    radial_unit = np.column_stack([x / safe, y / safe, np.zeros_like(z)])
    return z, theta, radius, radial_unit


def angular_distance(theta: np.ndarray, theta0: float) -> np.ndarray:
    """Smallest signed angular distance from ``theta`` to ``theta0`` (radians)."""
    d = (theta - theta0 + np.pi) % (2 * np.pi) - np.pi
    return d


def gaussian_zone_weight(
    z: np.ndarray,
    theta: np.ndarray,
    height: float,
    centre_h: float,
    centre_theta: float,
    sigma_h: float,
    sigma_theta: float,
) -> np.ndarray:
    """Smooth Gaussian falloff weight in (height-fraction, angle) space.

    ``centre_h`` and ``sigma_h`` are fractions of limb length; ``centre_theta``
    and ``sigma_theta`` are in radians. Returns weights in ``[0, 1]``.
    """
    dh = (z / height) - centre_h
    dth = angular_distance(theta, centre_theta)
    return np.exp(-0.5 * ((dh / sigma_h) ** 2 + (dth / sigma_theta) ** 2))


# --------------------------------------------------------------------------
# Smoothing
# --------------------------------------------------------------------------
def laplacian_smooth(
    mesh: trimesh.Trimesh, iterations: int = 5, factor: float = 0.5
) -> trimesh.Trimesh:
    """In-place-safe Laplacian smoothing that preserves vertex correspondence.

    Uses uniform (umbrella) weights via the mesh's vertex adjacency. Open
    boundaries are left free. Returns a new mesh with identical topology.
    """
    out = mesh.copy()
    V = np.asarray(out.vertices, dtype=float)
    neighbours = out.vertex_neighbors
    for _ in range(max(0, iterations)):
        newV = V.copy()
        for i, nb in enumerate(neighbours):
            if nb:
                newV[i] = V[i] + factor * (V[nb].mean(axis=0) - V[i])
        V = newV
    out.vertices = V
    return out


def smooth_vertex_field(
    mesh: trimesh.Trimesh, field: np.ndarray, iterations: int = 3, factor: float = 0.5
) -> np.ndarray:
    """Laplacian-smooth a scalar/vector per-vertex field over the mesh graph."""
    f = np.array(field, dtype=float)
    neighbours = mesh.vertex_neighbors
    for _ in range(max(0, iterations)):
        newf = f.copy()
        for i, nb in enumerate(neighbours):
            if nb:
                newf[i] = f[i] + factor * (f[nb].mean(axis=0) - f[i])
        f = newf
    return f
