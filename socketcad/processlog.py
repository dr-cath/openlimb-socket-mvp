"""Operation history ("process log") in the style of clinical 3D-image tools
such as Materialise Mimics or Synopsys Simpleware ScanIP.

Those tools keep an ordered, human-readable list of every operation applied to
a model (import, segmentation, smoothing, boolean, etc.) so the result is
auditable and reproducible. This module reconstructs the equivalent history for
one socket design from the deterministic pipeline results, so a reviewer can see
*exactly* what the engine did, in order, with parameters and outcomes.

The log is intentionally faithful to the actual pipeline steps in
:func:`socketcad.pipeline.run_pipeline` (rectify -> socketize -> validate).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

from .rectify import RectifyResult
from .socketize import SocketResult
from .spec import RectificationSpec
from .validate import ValidationReport


# Human-readable display names for the geometric zones.
_ZONE_NAMES = {
    "patellar_tendon": "Patellar tendon",
    "medial_paratibial": "Medial paratibial",
    "lateral_paratibial": "Lateral paratibial",
    "fibular_head": "Fibular head",
    "tibial_crest": "Tibial crest",
    "distal_tibial_end": "Distal tibial end",
    "posterior_calf": "Posterior calf",
    "supracondylar": "Supracondylar",
}


@dataclass
class LogEntry:
    step: int
    category: str          # import | analyze | rectify | volume | custom | shell | cut | feature | repair | validate
    operation: str         # short title
    detail: str            # human-readable description with parameters
    params: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProcessLog:
    title: str
    entries: List[LogEntry] = field(default_factory=list)

    def add(self, category: str, operation: str, detail: str, **params) -> None:
        self.entries.append(
            LogEntry(step=len(self.entries) + 1, category=category,
                     operation=operation, detail=detail, params=params)
        )

    def as_dict(self) -> dict:
        return {"title": self.title, "entries": [e.as_dict() for e in self.entries]}

    def to_text(self) -> str:
        lines = [f"OPERATION HISTORY — {self.title}",
                 "=" * (len(self.title) + 20), ""]
        for e in self.entries:
            lines.append(f"{e.step:>2}. [{e.category}] {e.operation}")
            lines.append(f"     {e.detail}")
        return "\n".join(lines)


def build_process_log(
    limb_info: dict,
    rectify: RectifyResult,
    socket: SocketResult,
    validation: ValidationReport,
    spec: RectificationSpec,
    title: str = "Socket design",
    prompt: Optional[str] = None,
    intents: Optional[List[str]] = None,
) -> ProcessLog:
    """Reconstruct the ordered operation history for one design."""
    log = ProcessLog(title=title)

    # 1. Intent / input ----------------------------------------------------
    if prompt is not None:
        log.add("import", "Clinical intent received",
                f'Prompt: "{prompt}". Detected intents: '
                f'{", ".join(intents) if intents else "none"}.',
                prompt=prompt, intents=intents or [])

    src = limb_info.get("source", "unknown")
    ver = limb_info.get("version", "n/a")
    n = int(len(rectify.limb.vertices))
    log.add("import", "Import residual-limb mesh",
            f"Source: {src} ({ver}); {n:,} vertices; "
            f"length {rectify.frame.height:.0f} mm.",
            source=src, version=ver, vertices=n,
            length_mm=round(float(rectify.frame.height), 1))

    # 2. Frame & analysis --------------------------------------------------
    log.add("analyze", "Estimate anatomical coordinate frame",
            "PCA long axis; proximal end = larger cross-section (+z); anterior "
            "detected opposite the posterior calf bulge; "
            f"limb treated as {'LEFT' if rectify.frame.is_left else 'RIGHT'}.",
            is_left=bool(rectify.frame.is_left))

    # 3. Rectification philosophy -----------------------------------------
    log.add("rectify", "Select socket philosophy",
            f"{spec.philosophy} baseline zone pattern; snugness {spec.snugness:.2f}.",
            philosophy=spec.philosophy, snugness=spec.snugness)

    # 4. Per-zone rectifications ------------------------------------------
    for zn in rectify.zones:
        mag = zn.signed_magnitude_mm
        action = "Carve (push in)" if mag < 0 else "Build (relieve out)"
        name = _ZONE_NAMES.get(zn.name, zn.name)
        log.add("rectify", f"Rectify: {name}",
                f"{action} {abs(mag):.1f} mm, Gaussian falloff "
                f"(centre h={zn.centre_h:.2f}, σ_h={zn.sigma_h:.2f}, "
                f"σ_θ={zn.sigma_theta:.2f}).",
                zone=zn.name, action=("carve" if mag < 0 else "build"),
                magnitude_mm=round(abs(mag), 2))

    # 5. Global volume reduction ------------------------------------------
    log.add("volume", "Global volume reduction",
            f"Uniform radial shrink of {rectify.volume_reduction_pct:.1f}% "
            "(clinical 'snugness').",
            volume_reduction_pct=round(float(rectify.volume_reduction_pct), 2))

    # 6. Custom painted patches -------------------------------------------
    for reg in spec.custom_regions:
        action = "Carve (push in)" if reg.carve else "Build (relieve out)"
        log.add("custom", f"Custom patch: {reg.label}",
                f"{action} {abs(reg.depth_mm):.1f} mm over a {reg.radius_mm:.0f} mm "
                "radius around a clinician-painted point (handles atypical, "
                "person-specific sites).",
                label=reg.label, depth_mm=abs(reg.depth_mm), radius_mm=reg.radius_mm,
                carve=bool(reg.carve))

    # 7. Displacement field smoothing -------------------------------------
    d = rectify.signed_normal_disp
    log.add("rectify", "Blend & smooth displacement field",
            f"Laplacian-smoothed so zone edges blend without creases; resulting "
            f"max relief {float(d.max()):.1f} mm, max carve {float(-d.min()):.1f} mm.",
            max_relief_mm=round(float(d.max()), 2), max_carve_mm=round(float(-d.min()), 2))

    # 8. Socketization -----------------------------------------------------
    log.add("shell", "Cap proximal opening",
            "Triangulate the open proximal boundary into a watertight limb solid.")
    log.add("shell", "Offset outward to wall",
            f"Create the socket wall at {socket.wall_thickness_mm:.1f} mm uniform "
            "thickness; cavity matches the rectified limb exactly.",
            wall_thickness_mm=socket.wall_thickness_mm)
    log.add("cut", "Apply trimline",
            f"Cut at {spec.trimline_height_frac*100:.0f}% of limb length "
            f"(anterior drop {spec.trimline_anterior_drop_frac*100:.0f}%) and cap "
            f"the rim; trimline height {socket.trimline_height_mm:.0f} mm.",
            trimline_height_mm=round(float(socket.trimline_height_mm), 1))
    log.add("feature", "Add attachment boss",
            "Union a distal pylon-adapter stub — geometry only, a NON-engineered "
            "placeholder (not a load-rated fitting).")

    # 9. Repair ------------------------------------------------------------
    s = socket.summary()
    log.add("repair", "Repair for printing",
            "Remove duplicate/degenerate faces, merge vertices, fix normals.")

    # 10. Validation -------------------------------------------------------
    wt = validation.metrics.get("wall_thickness_mm", {})
    wt_txt = (f"wall {wt['mean']:.1f} mm (p05 {wt['p05']:.1f}–p95 {wt['p95']:.1f})"
              if wt else "wall n/a")
    log.add("validate", f"Validate — {'PASS' if validation.passed else 'FAIL'}",
            f"watertight={s['watertight']}, manifold={s['winding_consistent']}, "
            f"printable_volume={s['is_volume']}; {wt_txt}; "
            f"socket volume {s.get('socket_volume_cm3', float('nan')):.0f} cm³.",
            passed=bool(validation.passed))

    return log
