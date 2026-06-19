"""Design sense-check: does the rectified socket make *clinical* sense?

Two complementary, deliberately-honest checks that sit between pure geometry
validation (:mod:`socketcad.validate`) and full FEA (a documented roadmap item):

1. **Plausibility audit** -- rule-based checks against published practice
   (Dickinson et al. 2024): correct load/offload direction per zone, magnitudes
   within plausible bounds, volume reduction in the clinical band, and contact
   maintained. Each check is pass / warn / fail with a reason.

2. **Interface-pressure proxy** -- a *crude* Winkler elastic-foundation estimate
   ``pressure ≈ tissue_stiffness × interference_depth`` per vertex, where
   interference is how far the socket pushes into the limb (the inward part of
   the rectification). This is NOT FEA: it ignores shear, tissue layering, bone,
   liners, global equilibrium and large deformation. It is an interpretable
   first-order indicator of *where* pressure concentrates, clearly labelled as
   illustrative. Validated peak interface pressures require FEA or in-socket
   sensors; see docs/LIMITATIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np
import trimesh

from .rectify import RectifyResult
from .spec import RectificationSpec

# --- illustrative constants (NOT validated material properties) -------------
# Winkler foundation modulus for residual-limb soft tissue. Real values vary
# widely with site, tissue depth and indentation rate; this single number is a
# labelled placeholder so the proxy is interpretable, not a measurement.
TISSUE_STIFFNESS_KPA_PER_MM = 20.0
# Illustrative caution threshold for sustained interface pressure (kPa). Real
# tolerances are person- and site-specific; used only to flag hotspots.
HIGH_PRESSURE_KPA = 200.0

# Zones expected to be loaded (carve, push in) vs offloaded (build, relieve out).
_EXPECT_CARVE = {"patellar_tendon", "medial_paratibial", "lateral_paratibial",
                 "posterior_calf", "supracondylar"}
_EXPECT_BUILD = {"fibular_head", "tibial_crest", "distal_tibial_end"}
# Plausible per-zone magnitude ceiling (mm) before we warn (clinical local
# rectifications are typically a few mm; >~10 mm is an outlier — Dickinson 2024).
_PLAUSIBLE_ZONE_MM = 10.0


@dataclass
class Check:
    name: str
    status: str       # "pass" | "warn" | "fail"
    detail: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SenseCheck:
    checks: List[Check] = field(default_factory=list)
    pressure_stats: dict = field(default_factory=dict)

    @property
    def worst(self) -> str:
        order = {"pass": 0, "warn": 1, "fail": 2}
        return max((c.status for c in self.checks), key=lambda s: order[s], default="pass")

    @property
    def passed(self) -> bool:
        return self.worst != "fail"

    def as_dict(self) -> dict:
        return {
            "overall": self.worst,
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
            "pressure": self.pressure_stats,
        }


def _vertex_areas(mesh: trimesh.Trimesh) -> np.ndarray:
    """Per-vertex area (one third of each incident face's area)."""
    areas = np.zeros(len(mesh.vertices))
    fa = mesh.area_faces
    for k in range(3):
        np.add.at(areas, mesh.faces[:, k], fa / 3.0)
    return areas


def interface_pressure_proxy(
    rectify: RectifyResult,
    stiffness_kpa_per_mm: float = TISSUE_STIFFNESS_KPA_PER_MM,
):
    """Return (pressure_kpa per vertex, stats dict).

    Interference = inward part of the signed normal displacement (socket pushes
    into tissue). Relief (outward) -> zero pressure (a gap, no contact).
    """
    signed = rectify.signed_normal_disp           # +out (relief), -in (carve)
    interference = np.clip(-signed, 0.0, None)     # mm pushed into tissue
    pressure = stiffness_kpa_per_mm * interference  # kPa

    areas = _vertex_areas(rectify.rectified)
    total = float(areas.sum()) or 1.0
    contact = interference > 0.1                   # >0.1 mm counts as contact
    high = pressure > HIGH_PRESSURE_KPA
    stats = {
        "model": "Winkler elastic-foundation proxy (illustrative, not FEA)",
        "stiffness_kPa_per_mm": stiffness_kpa_per_mm,
        "peak_kPa": round(float(pressure.max()), 1),
        "mean_contact_kPa": round(float(pressure[contact].mean()) if contact.any() else 0.0, 1),
        "contact_area_pct": round(100.0 * float(areas[contact].sum()) / total, 1),
        "high_pressure_area_pct": round(100.0 * float(areas[high].sum()) / total, 1),
        "high_pressure_threshold_kPa": HIGH_PRESSURE_KPA,
    }
    return pressure, stats


def colorize_pressure(rectify: RectifyResult, pressure: np.ndarray,
                      vmax: Optional[float] = None) -> trimesh.Trimesh:
    """Vertex-colour the rectified mesh by proxy pressure (sequential map)."""
    import matplotlib
    from matplotlib.colors import Normalize

    if vmax is None:
        vmax = float(np.percentile(pressure, 99)) or 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    colors = matplotlib.colormaps["inferno"](norm(pressure))
    out = rectify.rectified.copy()
    out.visual.vertex_colors = (colors * 255).astype(np.uint8)
    return out


def plausibility_audit(rectify: RectifyResult, spec: RectificationSpec,
                       pressure_stats: Optional[dict] = None) -> List[Check]:
    """Rule-based checks of whether the design follows published practice."""
    checks: List[Check] = []

    # 1. Load/offload direction correct per zone.
    wrong = []
    for z in rectify.zones:
        carve = z.signed_magnitude_mm < 0
        if z.name in _EXPECT_CARVE and not carve:
            wrong.append(f"{z.name} should load (carve)")
        if z.name in _EXPECT_BUILD and carve:
            wrong.append(f"{z.name} should offload (build)")
    checks.append(Check(
        "Load/offload direction",
        "pass" if not wrong else "fail",
        "Pressure-tolerant zones loaded, sensitive zones offloaded."
        if not wrong else "; ".join(wrong)))

    # 2. Per-zone magnitudes within plausible clinical range.
    big = [f"{z.name} {abs(z.signed_magnitude_mm):.1f}mm"
           for z in rectify.zones if abs(z.signed_magnitude_mm) > _PLAUSIBLE_ZONE_MM]
    checks.append(Check(
        "Rectification magnitudes",
        "pass" if not big else "warn",
        f"All within ~{_PLAUSIBLE_ZONE_MM:.0f} mm of published local-rectification range."
        if not big else "Outlier magnitude(s): " + ", ".join(big)))

    # 3. Volume reduction within the clinical band (<10%, Dickinson 2024).
    vr = rectify.volume_reduction_pct
    checks.append(Check(
        "Volume reduction",
        "pass" if vr <= 10.0 else "warn",
        f"{vr:.1f}% — within the typical <10% band."
        if vr <= 10.0 else f"{vr:.1f}% exceeds the typical <10% band."))

    # 4. Distal-end relief present (offload the vulnerable distal tibia).
    has_distal = any(z.name == "distal_tibial_end" and z.signed_magnitude_mm > 0
                     for z in rectify.zones)
    checks.append(Check(
        "Distal-end offload",
        "pass" if has_distal else "warn",
        "Distal tibial end is relieved." if has_distal
        else "No distal-end relief — check the vulnerable distal tibia."))

    # 5. Contact maintained (the socket actually touches the limb).
    if pressure_stats:
        ca = pressure_stats.get("contact_area_pct", 0.0)
        checks.append(Check(
            "Contact maintained",
            "pass" if ca >= 30.0 else "warn",
            f"{ca:.0f}% of the surface is in load-bearing contact."))
        # 6. Pressure hotspots (proxy).
        hp = pressure_stats.get("high_pressure_area_pct", 0.0)
        peak = pressure_stats.get("peak_kPa", 0.0)
        checks.append(Check(
            "Interface pressure (proxy)",
            "pass" if hp == 0 else "warn",
            f"Peak ~{peak:.0f} kPa; no hotspots above "
            f"{pressure_stats.get('high_pressure_threshold_kPa')} kPa."
            if hp == 0 else
            f"Peak ~{peak:.0f} kPa; {hp:.1f}% of area above "
            f"{pressure_stats.get('high_pressure_threshold_kPa')} kPa (proxy estimate)."))

    return checks


def sense_check(rectify: RectifyResult, spec: RectificationSpec) -> SenseCheck:
    """Full design sense-check: pressure proxy + plausibility audit."""
    _, stats = interface_pressure_proxy(rectify)
    checks = plausibility_audit(rectify, spec, pressure_stats=stats)
    return SenseCheck(checks=checks, pressure_stats=stats)
