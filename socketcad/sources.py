"""Citations and grounding metadata, surfaced in the UI so the demo is honest
about what is evidence-based vs. a labelled assumption.

Kept in one place so the web app, docs, and any report share identical wording.
See ``docs/SOURCES.md`` for the full discussion.
"""
from __future__ import annotations

CITATIONS = [
    {
        "key": "openlimbtt",
        "label": "Sunderland, Sobey, Bramley et al. (2025) — OpenLimbTT, a "
                 "transtibial residual limb shape model. Data-Centric Engineering 6:e39",
        "doi": "10.5258/SOTON/D2895",
        "url": "https://github.com/abel-research/OpenLimbTT",
        "grounds": "Residual-limb shape model (size-normalised mean + PCA modes). "
                   "Data CC-BY-SA-4.0; fetched at runtime, not vendored.",
        "status": "verified",
    },
    {
        "key": "dickinson2024",
        "label": "Dickinson et al. (2024) — Insights into the spectrum of transtibial "
                 "prosthetic socket design. Frontiers in Rehabilitation Sciences 5:1354069",
        "doi": "10.3389/fresc.2024.1354069",
        "url": "https://doi.org/10.3389/fresc.2024.1354069",
        "grounds": "Rectification zone taxonomy (PT/MP/LP/LMC carves; FH/TC/DE builds; "
                   "volume reduction %); PTB>TSB & FH~VR relationships; bounds "
                   "(VR <10%, distal end <10 mm). 163 real limb/socket pairs.",
        "status": "verified",
    },
    {
        "key": "convery2003",
        "label": "Convery & Buis (2003) — Consistency of PTB cast rectification. "
                 "Prosthetics and Orthotics International 27(3):207–213",
        "doi": "10.1080/03093640308726683",
        "url": "https://doi.org/10.1080/03093640308726683",
        "grounds": "Order-of-magnitude sanity check only (between-prosthetist "
                   "consistency: ~2 mm major-zone difference; fibular-head apex "
                   "2.8–4.3 mm SD) — not absolute carve depths.",
        "status": "verified",
    },
]

# Honest one-line summary of what is grounded vs. interpolated.
GROUNDING_NOTE = (
    "Zone taxonomy, PTB↔TSB relationships and bounds are taken directly from the "
    "literature (verified). The exact per-zone default millimetres are mid-range "
    "values consistent with those published bounds — labelled assumptions, not "
    "measured medians. Every value in each design's operation history is computed "
    "from the model and checked against an independent recomputation in the test "
    "suite (it is not hand-written text)."
)
