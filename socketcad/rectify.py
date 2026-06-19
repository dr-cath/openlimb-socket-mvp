"""Parametric rectification engine -- the heart of the MVP.

Contract (identical to what a future *learned* model must satisfy)::

    rectify(limb_mesh, spec)  ->  RectifyResult(rectified positive, ...)

The rectified "positive" is the modified limb shape that the socket *interior*
will match. The transform is a smooth, falloff-weighted radial displacement
per anatomical zone, plus a global volume reduction, plus any custom painted
patches. The displacement field is Laplacian-smoothed before it is applied so
zone boundaries blend without creases or self-intersection, while vertex
correspondence with the input limb is preserved (so the displacement is
directly legible as a heatmap, see :mod:`socketcad.validate`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import trimesh

from .geometry import (
    LimbFrame,
    cylindrical_coords,
    estimate_frame,
    gaussian_zone_weight,
    smooth_vertex_field,
)
from .spec import RectificationSpec, ResolvedZone


@dataclass
class RectifyResult:
    rectified: trimesh.Trimesh
    limb: trimesh.Trimesh
    frame: LimbFrame
    displacement: np.ndarray         # (N,3) world-space vertex displacement
    signed_normal_disp: np.ndarray   # (N,) displacement along limb outward normal
    zones: List[ResolvedZone]
    volume_reduction_pct: float

    def summary(self) -> dict:
        d = self.signed_normal_disp
        return {
            "n_vertices": int(len(d)),
            "max_relief_mm": float(d.max()),     # +ve = material relieved outward
            "max_carve_mm": float(-d.min()),     # +ve number = material carved in
            "mean_abs_disp_mm": float(np.abs(d).mean()),
            "volume_reduction_pct": float(self.volume_reduction_pct),
            "limb_volume_cm3": _volume_cm3(self.limb),
            "rectified_volume_cm3": _volume_cm3(self.rectified),
            "zones": [z.name for z in self.zones],
        }


def _volume_cm3(mesh: trimesh.Trimesh) -> float:
    """Volume in cm^3 for a (possibly open) limb mesh, via a capped copy."""
    try:
        from .socketize import cap_open_boundaries

        capped = cap_open_boundaries(mesh)
        if capped.is_watertight:
            return float(abs(capped.volume) / 1000.0)
    except Exception:
        pass
    return float("nan")


def rectify(
    limb: trimesh.Trimesh,
    spec: RectificationSpec,
    smooth_iterations: int = 6,
) -> RectifyResult:
    """Apply ``spec`` to ``limb`` and return the rectified positive."""
    frame = estimate_frame(limb, is_left=spec.is_left_limb)
    Vw = np.asarray(limb.vertices, dtype=float)
    Vc = frame.to_canonical(Vw)
    z, theta, radius, radial_unit = cylindrical_coords(Vc)
    H = frame.height

    disp_canonical = np.zeros_like(Vc)

    # ---- named anatomical zones ----------------------------------------
    zones = spec.resolve_zones(is_left=frame.is_left)
    for zn in zones:
        w = gaussian_zone_weight(
            z, theta, H, zn.centre_h, zn.centre_theta, zn.sigma_h, zn.sigma_theta
        )
        disp_canonical += radial_unit * (w * zn.signed_magnitude_mm)[:, None]

    # ---- global volume reduction (uniform radial shrink) ----------------
    vr_pct = spec.volume_reduction_pct()
    vr_frac = vr_pct / 100.0
    disp_canonical += radial_unit * (-vr_frac * radius)[:, None]

    # Convert zone/VR displacement to world coordinates.
    disp_world = frame.to_world_vectors(disp_canonical)

    # ---- custom painted patches (applied along surface normals) ---------
    if spec.custom_regions:
        normals = np.asarray(limb.vertex_normals, dtype=float)
        for reg in spec.custom_regions:
            point = np.asarray(reg.point, dtype=float)
            dist = np.linalg.norm(Vw - point, axis=1)
            sigma = max(reg.radius_mm, 1e-3) / 2.0
            w = np.exp(-0.5 * (dist / sigma) ** 2)
            depth = -abs(reg.depth_mm) if reg.carve else abs(reg.depth_mm)
            disp_world += normals * (w * depth)[:, None]

    # ---- smooth the displacement field, then apply ---------------------
    disp_world = smooth_vertex_field(limb, disp_world, iterations=smooth_iterations)

    rectified = limb.copy()
    rectified.vertices = Vw + disp_world

    # Signed displacement along the *original* outward normal (for heatmap):
    # +ve where the surface moved outward (relief), -ve where pushed in (carve).
    limb_normals = np.asarray(limb.vertex_normals, dtype=float)
    signed = np.einsum("ij,ij->i", disp_world, limb_normals)

    return RectifyResult(
        rectified=rectified,
        limb=limb,
        frame=frame,
        displacement=disp_world,
        signed_normal_disp=signed,
        zones=zones,
        volume_reduction_pct=vr_pct,
    )
