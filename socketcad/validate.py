"""Validation suite and the rectification displacement heatmap.

Two jobs:

* :func:`validate_socket` -- hard geometry checks (watertight, manifold,
  positive volume, wall thickness within tolerance, plausible dimensions) so
  a leaky/unprintable mesh is reported as a failure, never silently exported.
* :func:`displacement_heatmap` -- the key *legibility* output: a per-vertex
  map of how far (and which way) each point of the limb was moved to become the
  rectified positive, exported both as a vertex-coloured mesh (for the 3D
  viewer) and as a PNG (anterior/posterior renders + a cylindrical unwrap).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh

from .geometry import LimbFrame, cylindrical_coords, estimate_frame


# Plausible transtibial socket dimension bounds (mm). Generous; these only
# catch gross failures, not clinical correctness.
PLAUSIBLE_HEIGHT_MM = (60.0, 320.0)
PLAUSIBLE_MAX_DIAMETER_MM = (50.0, 180.0)


@dataclass
class ValidationReport:
    checks: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "checks": self.checks,
            "metrics": self.metrics,
            "messages": self.messages,
        }


def measure_wall_thickness(
    positive: trimesh.Trimesh,
    outer: trimesh.Trimesh,
    frame: LimbFrame,
    trimline_height_mm: float,
    n_samples: int = 600,
) -> np.ndarray:
    """Wall thickness = distance from the cavity surface to the outer surface.

    Samples points on the (cavity) positive surface below the trimline and
    above the distal pad, then takes the closest-point distance to the outer
    offset surface. This is fast and robust for thin shells (no ray casting).
    """
    from trimesh.proximity import closest_point

    pts, _ = trimesh.sample.sample_surface(positive, n_samples)
    zc = frame.to_canonical(pts)[:, 2]
    H = frame.height
    keep = (zc < min(trimline_height_mm, 0.9 * H)) & (zc > 0.18 * H)
    pts = pts[keep]
    if len(pts) == 0:
        return np.array([])
    _, dist, _ = closest_point(outer, pts)
    return dist


def validate_socket(
    socket: trimesh.Trimesh,
    positive: Optional[trimesh.Trimesh] = None,
    outer: Optional[trimesh.Trimesh] = None,
    frame: Optional[LimbFrame] = None,
    target_thickness_mm: float = 4.0,
    thickness_tol_mm: float = 2.0,
    trimline_height_mm: Optional[float] = None,
) -> ValidationReport:
    """Run the full geometry validation suite on a socket mesh."""
    rep = ValidationReport()
    c, m, msg = rep.checks, rep.metrics, rep.messages

    c["watertight"] = bool(socket.is_watertight)
    c["winding_consistent"] = bool(socket.is_winding_consistent)
    # is_volume implies watertight + consistent winding + positive volume;
    # we use it as the manifold / no-self-intersection proxy.
    c["is_volume_manifold"] = bool(socket.is_volume)
    c["no_degenerate_faces"] = bool(len(socket.faces) == int(socket.nondegenerate_faces().sum()))

    extents = np.sort(socket.extents)
    height = float(extents[-1])
    max_diam = float(extents[-2])
    m["height_mm"] = height
    m["max_cross_diameter_mm"] = max_diam
    m["n_vertices"] = int(len(socket.vertices))
    m["n_faces"] = int(len(socket.faces))
    if socket.is_watertight:
        m["volume_cm3"] = float(abs(socket.volume) / 1000.0)

    c["plausible_height"] = PLAUSIBLE_HEIGHT_MM[0] <= height <= PLAUSIBLE_HEIGHT_MM[1]
    c["plausible_diameter"] = (
        PLAUSIBLE_MAX_DIAMETER_MM[0] <= max_diam <= PLAUSIBLE_MAX_DIAMETER_MM[1]
    )

    if positive is not None and outer is not None and frame is not None:
        th = measure_wall_thickness(
            positive, outer, frame,
            trimline_height_mm if trimline_height_mm else 0.9 * frame.height,
        )
        if len(th):
            m["wall_thickness_mm"] = {
                "min": float(th.min()),
                "mean": float(th.mean()),
                "max": float(th.max()),
                "p05": float(np.percentile(th, 5)),
                "p95": float(np.percentile(th, 95)),
            }
            lo = target_thickness_mm - thickness_tol_mm
            hi = target_thickness_mm + thickness_tol_mm
            frac_in = float(np.mean((th >= lo) & (th <= hi)))
            m["wall_thickness_frac_in_tol"] = frac_in
            c["wall_thickness_in_tolerance"] = frac_in >= 0.80
            if not c["wall_thickness_in_tolerance"]:
                msg.append(
                    f"Only {frac_in*100:.0f}% of wall within {lo:.1f}-{hi:.1f} mm "
                    f"(mean {th.mean():.1f} mm)."
                )
    if not c["watertight"]:
        msg.append("Socket is not watertight (leaky) -- not printable.")
    if not c.get("is_volume_manifold", True):
        msg.append("Socket failed manifold/volume check (possible self-intersection).")
    return rep


# --------------------------------------------------------------------------
# Displacement heatmap (legibility output)
# --------------------------------------------------------------------------
def colorize_displacement(
    mesh: trimesh.Trimesh, signed_disp: np.ndarray, vmax: Optional[float] = None
) -> trimesh.Trimesh:
    """Return a copy of ``mesh`` with vertex colours encoding ``signed_disp``.

    Blue = carved in (material removed), red = relieved out (material added),
    white ~ unchanged. Symmetric colour scale.
    """
    import matplotlib
    from matplotlib.colors import Normalize

    if vmax is None:
        vmax = float(np.percentile(np.abs(signed_disp), 99)) or 1.0
    norm = Normalize(vmin=-vmax, vmax=vmax)
    colors = matplotlib.colormaps["coolwarm"](norm(signed_disp))
    out = mesh.copy()
    out.visual.vertex_colors = (colors * 255).astype(np.uint8)
    return out


def displacement_heatmap(
    result,
    out_png: Optional[str] = None,
    out_mesh: Optional[str] = None,
) -> dict:
    """Build the rectification heatmap from a :class:`RectifyResult`.

    Writes a vertex-coloured mesh (GLB/PLY, for the 3D viewer) and/or a PNG with
    anterior + posterior renders and a cylindrical unwrap. Returns metadata.
    """
    signed = result.signed_normal_disp
    vmax = float(np.percentile(np.abs(signed), 99)) or 1.0
    colored = colorize_displacement(result.rectified, signed, vmax=vmax)

    info = {"vmax_mm": vmax}
    if out_mesh:
        colored.export(out_mesh)
        info["mesh"] = out_mesh

    if out_png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize

        frame = result.frame
        Vc = frame.to_canonical(result.rectified.vertices)
        z, theta, radius, _ = cylindrical_coords(Vc)
        H = frame.height
        norm = Normalize(vmin=-vmax, vmax=vmax)

        # Subsample for the scatter plots to keep memory/time bounded.
        n = len(Vc)
        if n > 9000:
            sub = np.random.default_rng(0).choice(n, 9000, replace=False)
        else:
            sub = np.arange(n)
        Vc, z, theta, signed_s = Vc[sub], z[sub], theta[sub], signed[sub]

        fig = plt.figure(figsize=(13, 5))
        # Two 3D-ish scatter views (anterior, posterior) + unwrap.
        for k, (title, azim) in enumerate(
            [("Anterior", 0.0), ("Posterior", np.pi)]
        ):
            ax = fig.add_subplot(1, 3, k + 1, projection="3d")
            # show the half of the limb facing the viewer
            facing = np.cos(theta - azim) > -0.2
            sc = ax.scatter(
                Vc[facing, 0], Vc[facing, 1], Vc[facing, 2],
                c=signed_s[facing], cmap="coolwarm", norm=norm, s=2,
            )
            ax.set_title(f"{title} view")
            ax.set_box_aspect((1, 1, 2.2))
            ax.view_init(elev=10, azim=np.degrees(azim))
            ax.set_axis_off()

        ax = fig.add_subplot(1, 3, 3)
        sc = ax.scatter(
            np.degrees(theta), z / H, c=signed_s, cmap="coolwarm", norm=norm, s=3
        )
        ax.set_xlabel("angle from anterior (deg)")
        ax.set_ylabel("height fraction (distal->proximal)")
        ax.set_title("Cylindrical unwrap")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("displacement along normal (mm)\n(blue=carve in, red=relieve out)")
        fig.suptitle("Rectification displacement map (limb -> rectified positive)")
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
        info["png"] = out_png

    return info
