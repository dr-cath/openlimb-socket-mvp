"""The clinical-intent interface: :class:`RectificationSpec`.

This is the product's actual differentiator. The clinician expresses *intent*
in clinical terms (socket philosophy, snugness, named reliefs/loads, custom
relief patches); this module translates that intent into the geometric zone
displacements consumed by :mod:`socketcad.rectify`. Vertices, mm-offsets and
mesh operations are deliberately **not** part of this interface.

Default magnitudes are grounded in published transtibial rectification data
and labelled with their source. Where the literature did not give a directly
machine-readable per-zone median we use a mid-range value consistent with the
reported bounds, and say so. See ``docs/SOURCES.md``.

Key sources for the defaults below:
* Dickinson et al. (2024), *Frontiers in Rehabilitation Sciences* 5:1354069 --
  163 limb/socket pairs; zone taxonomy (PT/MP/LP/LMC carves; FH/TC/DE builds;
  gross volume reduction %), and bounds: most sockets <10% volume reduction and
  <10 mm distal-end elongation; volume-reduction 10th percentile ~1.5%; local
  rectifications larger in PTB than TSB, volume reduction larger in TSB.
* Convery & Buis (2003), *Prosthet Orthot Int* 27(3) -- PTB cast rectification
  consistency: ~2 mm mean carve in major zones; fibular-head apex location
  varied with 2.8-4.3 mm SD.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np


# --------------------------------------------------------------------------
# Custom (painted) relief region -- handles atypical cases templates can't.
# --------------------------------------------------------------------------
@dataclass
class CustomReliefRegion:
    """A clinician-painted patch: relieve/load around a surface point.

    ``point`` is a world-coordinate point on (or near) the limb surface, as
    produced by clicking in the 3D viewer. ``radius_mm`` is the patch radius;
    ``depth_mm`` is positive for relief (build outward) and the engine handles
    the sign. Set ``carve=True`` to instead push material in at the patch.
    """

    point: tuple
    radius_mm: float = 15.0
    depth_mm: float = 4.0
    carve: bool = False
    label: str = "custom"


# --------------------------------------------------------------------------
# A single resolved geometric zone (intent already translated to geometry).
# --------------------------------------------------------------------------
@dataclass
class ResolvedZone:
    name: str
    centre_h: float  # height fraction along limb (0 distal, 1 proximal)
    centre_theta: float  # angle from anterior, radians
    sigma_h: float  # falloff sigma in height fraction
    sigma_theta: float  # falloff sigma in radians
    signed_magnitude_mm: float  # +ve = build/relieve (out), -ve = carve (in)


# --------------------------------------------------------------------------
# Zone templates. Each entry: (centre_h, centre_theta, sigma_h, sigma_theta,
# direction) where direction +1 = build (relieve outward), -1 = carve (push in).
# Angles use anterior = 0; lateral structures get a side-dependent sign applied
# later. theta in radians.
# --------------------------------------------------------------------------
_ANT = 0.0
_LAT = math.pi / 2
_MED = -math.pi / 2
_POST = math.pi

_ZONE_TEMPLATES = {
    # name:            (h,    theta, sig_h, sig_th, direction, lateral?)
    "patellar_tendon":  (0.80, _ANT, 0.12, 0.45, -1, False),
    "medial_paratibial":(0.55, -0.6, 0.18, 0.35, -1, True),  # antero-medial
    "lateral_paratibial":(0.55, +0.6, 0.18, 0.35, -1, True), # antero-lateral
    "fibular_head":     (0.80, _LAT, 0.10, 0.40, +1, True),
    "tibial_crest":     (0.50, _ANT, 0.35, 0.25, +1, False),
    "distal_tibial_end":(0.10, _ANT, 0.10, 0.7, +1, False),
    "posterior_calf":   (0.55, _POST, 0.22, 0.6, -1, False),
    "supracondylar":    (0.95, _LAT, 0.06, 0.9, -1, True),
}


# Literature-anchored default magnitudes (mm) per socket philosophy. See module
# docstring for sources. PTB uses larger local rectifications; TSB uses smaller
# local rectifications with a larger overall volume reduction.
_PHILOSOPHY_DEFAULTS = {
    "PTB": {
        "patellar_tendon": 4.0,    # PT carve (relative design insensitivity; mid value)
        "medial_paratibial": 4.0,  # MP carve (larger in PTB)
        "lateral_paratibial": 4.0, # LP carve (larger in PTB)
        "fibular_head": 4.0,       # FH build/relief (apex var 2.8-4.3 mm SD)
        "tibial_crest": 3.0,       # TC build/relief
        "distal_tibial_end": 4.0,  # DE build (bounded <10 mm)
        "posterior_calf": 3.0,     # load posterior soft tissue (carve)
        "supracondylar": 2.0,      # optional suspension carve
        "volume_reduction_pct": 4.0,
    },
    "TSB": {
        "patellar_tendon": 1.0,
        "medial_paratibial": 1.5,
        "lateral_paratibial": 1.5,
        "fibular_head": 1.5,
        "tibial_crest": 1.5,
        "distal_tibial_end": 5.0,  # larger DE elongation accompanies high VR
        "posterior_calf": 1.5,
        "supracondylar": 1.0,
        "volume_reduction_pct": 7.0,
    },
}

# Hard safety/plausibility bounds (mm / %). Magnitudes are clamped to these.
MAX_ZONE_MAGNITUDE_MM = 12.0
MAX_DISTAL_END_MM = 10.0      # Frontiers 2024: majority < 10 mm
MAX_VOLUME_REDUCTION_PCT = 12.0


@dataclass
class RectificationSpec:
    """Clinical-intent parameters for one socket design.

    All magnitudes are optional; when ``None`` the philosophy default is used,
    so a clinician can work purely with toggles and the snugness slider.
    """

    philosophy: str = "PTB"  # "PTB" or "TSB"

    # Single global "snugness" control in [0, 1] -> volume reduction %.
    snugness: float = 0.5

    # Named reliefs (build outward over pressure-sensitive structures).
    relieve_fibular_head: bool = True
    relieve_distal_tibial_end: bool = True
    relieve_tibial_crest: bool = True
    fibular_head_mm: Optional[float] = None
    distal_tibial_end_mm: Optional[float] = None
    tibial_crest_mm: Optional[float] = None

    # Named loads (carve in over pressure-tolerant structures).
    load_patellar_tendon: bool = True
    load_posterior_calf: bool = True
    patellar_tendon_mm: Optional[float] = None
    posterior_calf_mm: Optional[float] = None

    # Paratibial carves (the flats either side of the tibial crest).
    carve_paratibial: bool = True
    paratibial_mm: Optional[float] = None

    # Supracondylar suspension carves (optional).
    carve_supracondylar: bool = False
    supracondylar_mm: Optional[float] = None

    # Custom painted relief/load patches (atypical cases).
    custom_regions: List[CustomReliefRegion] = field(default_factory=list)

    # Socketization controls.
    trimline_height_frac: float = 0.92   # cut height as fraction of limb length
    trimline_anterior_drop_frac: float = 0.06  # anterior wall lowered (knee flexion)
    wall_thickness_mm: float = 4.0

    # Anatomy handedness (cannot be inferred from an unlabelled mesh).
    is_left_limb: bool = True

    # ---- derived values -------------------------------------------------
    def volume_reduction_pct(self) -> float:
        base = _PHILOSOPHY_DEFAULTS[self.philosophy]["volume_reduction_pct"]
        # snugness modulates the philosophy baseline by +/-50%.
        vr = base * (0.5 + self.snugness)
        return float(min(vr, MAX_VOLUME_REDUCTION_PCT))

    def _mag(self, zone: str, override: Optional[float]) -> float:
        val = _PHILOSOPHY_DEFAULTS[self.philosophy][zone] if override is None else override
        cap = MAX_DISTAL_END_MM if zone == "distal_tibial_end" else MAX_ZONE_MAGNITUDE_MM
        return float(min(abs(val), cap))

    def resolve_zones(self, is_left: Optional[bool] = None) -> List[ResolvedZone]:
        """Translate clinical intent into a list of geometric zones."""
        if is_left is None:
            is_left = self.is_left_limb
        side = 1.0 if is_left else -1.0

        active = {
            "patellar_tendon": (self.load_patellar_tendon, self.patellar_tendon_mm),
            "medial_paratibial": (self.carve_paratibial, self.paratibial_mm),
            "lateral_paratibial": (self.carve_paratibial, self.paratibial_mm),
            "fibular_head": (self.relieve_fibular_head, self.fibular_head_mm),
            "tibial_crest": (self.relieve_tibial_crest, self.tibial_crest_mm),
            "distal_tibial_end": (self.relieve_distal_tibial_end, self.distal_tibial_end_mm),
            "posterior_calf": (self.load_posterior_calf, self.posterior_calf_mm),
            "supracondylar": (self.carve_supracondylar, self.supracondylar_mm),
        }

        zones: List[ResolvedZone] = []
        for name, (enabled, override) in active.items():
            if not enabled:
                continue
            h, theta, sig_h, sig_th, direction, lateral = _ZONE_TEMPLATES[name]
            if lateral:
                theta = theta * side
            mag = self._mag(name, override) * direction
            zones.append(
                ResolvedZone(name, h, theta, sig_h, sig_th, mag)
            )
        return zones

    def to_log_dict(self) -> dict:
        d = asdict(self)
        d["volume_reduction_pct"] = self.volume_reduction_pct()
        return d


def spec_from_dict(d: dict) -> RectificationSpec:
    """Build a spec from a plain dict (e.g. JSON from the web UI)."""
    d = dict(d)
    regions = [
        CustomReliefRegion(**r) if not isinstance(r, CustomReliefRegion) else r
        for r in d.pop("custom_regions", []) or []
    ]
    known = set(RectificationSpec.__dataclass_fields__.keys())
    clean = {k: v for k, v in d.items() if k in known}
    spec = RectificationSpec(**clean)
    spec.custom_regions = regions
    return spec
