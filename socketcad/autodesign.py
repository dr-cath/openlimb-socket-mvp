"""Auto-design: turn a plain-language clinical prompt into a few candidate
:class:`~socketcad.spec.RectificationSpec` designs that span a trade-off space.

This is intentionally a small, transparent, rule-based mapping (NOT a learned
model and NOT clinical advice). It exists so the demo can show the workflow:

    prompt + scan  ->  auto-design  ->  3 candidate sockets with trade-offs

The three candidates always span the clinically meaningful axis from
*stability-first* (PTB: pronounced local rectifications, firmer load paths) to
*total-contact comfort* (TSB: gentle local shaping, larger overall volume
reduction), with a balanced hybrid in between. Free-text keywords nudge the
candidates (e.g. "active" leans TSB; "new/oedema/vascular" leans PTB and more
distal relief; "sensitive/bony/neuroma" deepens reliefs). Every nudge that
fires is reported back so the reasoning is fully legible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .spec import RectificationSpec


@dataclass
class Candidate:
    key: str                      # short id, e.g. "stability"
    title: str                    # human title for the card
    summary: str                  # one-line description of the design intent
    pros: List[str]               # trade-off: what this design is good at
    cons: List[str]               # trade-off: what you give up
    spec: RectificationSpec
    grounding: List[dict] = field(default_factory=list)  # design-specific evidence


# Keyword -> (human reason, effect tags). Effects are applied in build_candidates.
_KEYWORDS = {
    "active": "active / higher mobility",
    "athlete": "active / higher mobility",
    "sport": "active / higher mobility",
    "k3": "active / higher mobility",
    "k4": "active / higher mobility",
    "new": "new / recent amputation",
    "recent": "new / recent amputation",
    "oedema": "oedema / swelling present",
    "edema": "oedema / swelling present",
    "swell": "oedema / swelling present",
    "vascular": "dysvascular / fragile tissue",
    "diabet": "dysvascular / fragile tissue",
    "sensitive": "sensitive / painful site",
    "pain": "sensitive / painful site",
    "bony": "bony prominence",
    "neuroma": "neuroma / focal lesion",
    "scar": "scar / graft",
    "comfort": "comfort prioritised",
}


def detect_intents(prompt: str) -> List[str]:
    """Return the de-duplicated human-readable intents found in ``prompt``."""
    p = (prompt or "").lower()
    found: List[str] = []
    for kw, reason in _KEYWORDS.items():
        if kw in p and reason not in found:
            found.append(reason)
    return found


def build_candidates(prompt: str = "", is_left_limb: bool = True):
    """Return ``(candidates, intents)``: three designs spanning stability<->comfort
    plus the list of human-readable intents detected in ``prompt``."""
    p = (prompt or "").lower()
    intents = detect_intents(prompt)

    leans_active = any(k in p for k in ("active", "athlete", "sport", "k3", "k4"))
    leans_fragile = any(k in p for k in ("new", "recent", "oedema", "edema", "swell", "vascular", "diabet"))
    sensitive = any(k in p for k in ("sensitive", "pain", "bony", "neuroma", "scar"))
    wants_comfort = "comfort" in p

    # Base snugness per candidate; nudged by intent.
    snug_stability = 0.45
    snug_balanced = 0.55
    snug_comfort = 0.70
    if leans_active or wants_comfort:
        snug_balanced += 0.1
        snug_comfort += 0.1
    if leans_fragile:
        # Looser overall for fragile/oedematous limbs early in rehab.
        snug_stability -= 0.1
        snug_balanced -= 0.1
        snug_comfort -= 0.05

    relief_boost = 1.5 if sensitive else 0.0  # mm added to named reliefs
    distal_boost = 2.0 if leans_fragile else 0.0  # extra distal-end relief (mm)

    def clamp01(x: float) -> float:
        return max(0.05, min(0.95, x))

    stability = RectificationSpec(
        philosophy="PTB",
        snugness=clamp01(snug_stability),
        is_left_limb=is_left_limb,
        relieve_fibular_head=True,
        relieve_distal_tibial_end=True,
        relieve_tibial_crest=True,
        carve_supracondylar=True,  # added suspension for security
    )
    balanced = RectificationSpec(
        philosophy="PTB",
        snugness=clamp01(snug_balanced),
        is_left_limb=is_left_limb,
    )
    comfort = RectificationSpec(
        philosophy="TSB",
        snugness=clamp01(snug_comfort),
        is_left_limb=is_left_limb,
    )

    # Apply intent-driven magnitude nudges (None means "use philosophy default").
    for s in (stability, balanced, comfort):
        if relief_boost:
            base_fh = 4.0 if s.philosophy == "PTB" else 1.5
            base_dte = 4.0 if s.philosophy == "PTB" else 5.0
            s.fibular_head_mm = base_fh + relief_boost
            s.tibial_crest_mm = (3.0 if s.philosophy == "PTB" else 1.5) + relief_boost
            s.distal_tibial_end_mm = base_dte + distal_boost
        elif distal_boost:
            s.distal_tibial_end_mm = (4.0 if s.philosophy == "PTB" else 5.0) + distal_boost

    candidates = [
        Candidate(
            key="stability",
            title="Stability-first (PTB)",
            summary="Pronounced local rectifications load pressure-tolerant areas "
                    "and firmly offload sensitive ones; added supracondylar grip.",
            pros=["Secure, well-defined load paths", "Forgiving of early volume change",
                  "Good for new / oedematous / fragile limbs"],
            cons=["Higher local pressures", "Less total-contact feel"],
            spec=stability,
            grounding=[
                {"text": "PTB applies larger local rectifications "
                         "(lateral-paratibial carve p<0.001; distal-end build p<0.05).",
                 "source": "dickinson2024"},
                {"text": "Fibular head offloaded with a build; FH build correlates "
                         "with volume reduction (ρ=0.38).", "source": "dickinson2024"},
                {"text": "A few-mm carve/relief magnitudes are clinically plausible.",
                 "source": "convery2003"},
            ],
        ),
        Candidate(
            key="balanced",
            title="Balanced hybrid",
            summary="A middle ground: moderate local shaping with a moderate overall "
                    "volume reduction.",
            pros=["Compromise of security and comfort", "Versatile default"],
            cons=["Master of neither extreme"],
            spec=balanced,
            grounding=[
                {"text": "Real socket designs lie on a PTB↔TSB spectrum; most are "
                         "hybrids rather than pure types.", "source": "dickinson2024"},
                {"text": "Moderate volume reduction with moderate local shaping sits "
                         "mid-spectrum.", "source": "dickinson2024"},
            ],
        ),
        Candidate(
            key="comfort",
            title="Total-contact comfort (TSB)",
            summary="Gentle local shaping with a larger, even volume reduction for "
                    "uniform pressure distribution.",
            pros=["Even, total-contact pressure", "Suits mature limbs / active users"],
            cons=["Needs a good liner & stable volume", "Less focal offloading"],
            spec=comfort,
            grounding=[
                {"text": "TSB uses a significantly larger gross volume reduction than "
                         "PTB (p<0.05).", "source": "dickinson2024"},
                {"text": "Local rectifications are smaller; a larger distal-end "
                         "elongation accommodates soft tissue compressed by high "
                         "volume reduction.", "source": "dickinson2024"},
            ],
        ),
    ]

    # Intent-driven grounding shared by all candidates (only when triggered).
    extra = []
    if sensitive:
        extra.append({"text": "Named reliefs deepened for sensitive/bony/neuroma "
                              "sites — person-specific offloading a 3D scan alone "
                              "cannot infer.", "source": "dickinson2024"})
    if leans_fragile:
        extra.append({"text": "Looser fit and extra distal relief for new / "
                              "oedematous / dysvascular limbs early in rehab.",
                      "source": "dickinson2024"})
    for c in candidates:
        c.grounding = c.grounding + extra

    return candidates, intents
