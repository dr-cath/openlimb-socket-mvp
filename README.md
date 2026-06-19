# OpenLimb Socket MVP — Parametric Transtibial Socket Rectification Engine

A working **scan → socket** demo that turns a residual-limb mesh plus a small set
of **clinical-intent** parameters into a watertight, print-ready transtibial
prosthetic **socket STL** — built entirely on **open data** ([OpenLimbTT]) and
**published rectification rules**. No proprietary clinical dataset, no IRB, no
clinic partner required to stand it up.

> ⚠️ **Research/demo prototype only — NOT a cleared medical device.** No claims of
> clinical accuracy, fit, or safety. The attachment interface is a non-engineered
> placeholder. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

This is the deliberate **parametric baseline**: a future *learned* rectification
model must beat it, behind the same `limb mesh + spec -> rectified positive`
contract, and this engine doubles as that model's test harness.

---

## What it does

1. **Limb input** — sample a synthetic limb from the [OpenLimbTT] statistical
   shape model (mean / random / single-PC sweep), or upload your own STL/PLY/OBJ.
2. **Rectification** — apply clinical intent (PTB/TSB philosophy, snugness, named
   reliefs & loads, painted custom patches) as a smooth, falloff-weighted
   per-zone displacement plus a global volume reduction → a **rectified positive**.
3. **Socketization** — offset to a uniform wall thickness, cut the trimline, close
   the distal end, add a (placeholder) pylon-adapter boss → a watertight shell.
4. **Validation + heatmap** — check watertight / manifold / wall-thickness /
   plausible dimensions, and render a **displacement heatmap** so a prosthetist
   can *see* exactly where and how much material was added or removed.
5. **Interface** — a local web app for the *intent → preview → approve → export*
   loop, with session logging of every change (the spec for v2 / the learned model).

![displacement heatmap example](docs/heatmap_example.png)

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional but recommended) download the real OpenLimbTT shapes:
python scripts/fetch_openlimb.py
# ^ without this, the engine uses a clearly-labelled SYNTHETIC fallback limb.

# reproducible end-to-end demo (PTB, TSB, and an atypical custom-relief case):
python scripts/demo.py
#   -> data/out/demo_*_socket.stl, *_rectified.stl, *_limb.stl, *_heatmap.png/.glb
#   -> data/out/session_log.jsonl

# the interactive web app:
uvicorn app.main:app --reload      # open http://127.0.0.1:8000
#   /          single-design intent -> preview -> approve -> export loop
#   /compare   prompt + scan -> AUTO-DESIGN 3 candidate sockets, shown side by
#              side with trade-offs and a Mimics/ScanIP-style operation history
```

### Auto-design & compare (`/compare`)

The visual demo of the whole workflow: type the clinical intent in plain language
(e.g. *"new amputee, oedema, sensitive bony spur"*), pick a scan, and the engine
auto-designs **three candidate sockets** spanning *stability (PTB) ↔ comfort (TSB)*.
Each is shown in its own 3D viewer with its trade-offs (volume reduction, snugness,
wall thickness, printability) a downloadable STL, and a full **operation history**
— an ordered, auditable list of every step the engine applied to the mesh, in the
style of clinical 3D tools (Materialise Mimics / Simpleware ScanIP). The mapping
from prompt to designs is transparent and rule-based (`socketcad/autodesign.py`),
and the history is reconstructed faithfully from the pipeline (`socketcad/processlog.py`).

Dump sampled limbs for eyeballing:

```bash
python scripts/dump_limb.py --shape mean   -o data/out/limb_mean.stl
python scripts/dump_limb.py --shape random --seed 3 -o data/out/limb_r3.stl
python scripts/dump_limb.py --pc 0 --value 12 -o data/out/limb_pc0.stl   # sweep mode 0
```

Run the tests (hermetic — uses the synthetic limb, no network needed):

```bash
pytest -q
```

---

## Using it from Python

```python
from socketcad.limb_source import sample_limb
from socketcad.spec import RectificationSpec, CustomReliefRegion
from socketcad.pipeline import run_pipeline

limb, info = sample_limb("mean")                      # or "random" / "synthetic"
spec = RectificationSpec(
    philosophy="PTB", snugness=0.5,
    relieve_fibular_head=True, load_patellar_tendon=True,
    custom_regions=[CustomReliefRegion(point=(x, y, z), radius_mm=18, depth_mm=6)],
)
result = run_pipeline(limb, spec, out_dir="data/out", basename="my_socket")
print(result.validation.passed, result.exports["socket_stl"])
```

The rectification step is **swappable**: a future learned model only needs to
implement `rectify(limb_mesh, spec) -> RectifyResult` and the rest of the
pipeline (socketization, validation, UI) is unchanged.

---

## Architecture

```
socketcad/
  geometry.py     anatomical frame estimation, cylindrical coords, falloff, smoothing
  limb_source.py  OpenLimbTT loader (+ welding), synthetic fallback, landmarks
  spec.py         RectificationSpec — the clinical-intent interface (+ defaults)
  rectify.py      parametric zone-based rectification (the heart; swappable)
  socketize.py    shell offset, trimline, distal end, attachment stub, repair
  validate.py     geometry checks + displacement heatmap (legibility output)
  pipeline.py     end-to-end orchestration used by both CLI and web app
app/              FastAPI backend + single-page three.js viewer (intent->export)
scripts/          fetch_openlimb.py · demo.py · dump_limb.py
tests/            pytest geometry/pipeline/app assertions
docs/             SOURCES.md (provenance) · LIMITATIONS.md
```

### The clinical-intent interface (the actual differentiator)

The clinician expresses **intent in clinical terms** — not geometry. Vertices,
offsets-in-mm, and mesh ops are never exposed. `RectificationSpec` maps each
clinical control to the underlying transform with literature-grounded defaults:

- **Socket philosophy** — PTB vs TSB (sets the default zone pressure pattern).
- **Snugness** — one control → global volume reduction %.
- **Named reliefs** — fibular head, distal tibial end, tibial crest (build out).
- **Named loads** — patellar tendon, posterior calf, paratibial flats (carve in).
- **Custom relief patch** — paint a spot on the limb and set depth; this is how
  atypical cases (neuromas, scar adhesions, bony spurs) are handled — exactly
  what template/macro tools cannot do.
- **Trimline height** and **wall thickness**.

---

## What is real vs. assumed (honest grounding)

This was built "verify, don't assume". Full detail in
[`docs/SOURCES.md`](docs/SOURCES.md); in short:

- **OpenLimbTT is real and used** (verified by cloning [the repo][OpenLimbTT]):
  size-normalised mean mesh + PCA modes; `vertices = mean + Σ modeᵢ·eigenvalueᵢ`,
  scaled by intact-tibia length (mean 378.52 mm). Data is **CC-BY-SA-4.0** so it
  is **not vendored** — `scripts/fetch_openlimb.py` downloads it and the engine
  falls back to a synthetic limb if it is absent. Cite Sunderland et al. (2025).
- **Rectification zones are real**: the PT/MP/LP/LMC carves and FH/TC/DE builds
  plus gross volume reduction come from Dickinson et al. (2024, *Frontiers in
  Rehab. Sci.*, 163 limb/socket pairs). Verified bounds (e.g. most sockets
  < 10% volume reduction and < 10 mm distal elongation) are used as defaults and
  clamps. Magnitude sanity from Convery & Buis (2003).
- **Assumptions are labelled**: where the source's exact per-zone median was not
  machine-readable, a mid-range value consistent with the reported bounds is used
  and marked as such; the OpenLimbTT `LR.pkl` feasibility regression is not
  applied; left/right limb is a flag; landmarking is heuristic. See
  [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

---

## Definition of done (MVP) — status

- [x] OpenLimbTT-sampled **or** uploaded limb → watertight print-ready socket STL
      via clinical-intent controls only, in one session.
- [x] Displacement **heatmap** makes the rectification legible.
- [x] At least one **atypical case** (painted custom relief) demonstrated.
- [x] Outputs pass the validation suite (manifold, watertight, wall thickness in
      tolerance, plausible dimensions).
- [x] Docs cover how to run, real-vs-assumed, sources, OpenLimbTT licence /
      attribution, limitations, and the "not a cleared device" disclaimer.

---

## Licence & attribution

Source code: **MIT** (see [`LICENSE`](LICENSE)). OpenLimbTT data: **CC-BY-SA-4.0**
(University of Southampton; not vendored). Required third-party attribution is in
[`NOTICE`](NOTICE). Please cite Sunderland et al. (2025) for OpenLimbTT and
Dickinson et al. (2024) / Convery & Buis (2003) for the rectification
parameterisation. Full citations in [`docs/SOURCES.md`](docs/SOURCES.md).

[OpenLimbTT]: https://github.com/abel-research/OpenLimbTT
