# Sources, provenance, and what is verified vs. assumed

This file records exactly what the engine's data and default values are grounded
in, what was verified by the author of this MVP, and what is a clearly-labelled
assumption. The brief for this project insisted: *verify, don't assume.*

## 1. OpenLimbTT statistical shape model — VERIFIED

- **What it is:** an open statistical shape model (SSM) of the transtibial
  residual limb, built from MRI/CT data by the University of Southampton group
  to address the lack of accessible limb-shape data.
- **Repository:** <https://github.com/abel-research/OpenLimbTT> (verified by
  cloning). A broader model lives at <https://github.com/abel-research/openlimb>.
- **Data repository:** University of Southampton, <https://doi.org/10.5258/SOTON/D2895>.
- **Paper:** Sunderland F.E., Sobey A.J., Bramley J.L., et al. (2025),
  *OpenLimbTT, a transtibial residual limb shape model for prosthetics
  simulation and design: creating a statistical anatomic model using sparse
  data.* **Data-Centric Engineering** 6:e39. (Preprint: medRxiv
  2024.11.27.24317622.)
- **Licence:** **data CC-BY-SA-4.0**, **code MIT** (two licence files in the
  repo: `DATA-LICENSE`, `CODE-LICENSE`). We therefore do **not** vendor the data
  in this repo; `scripts/fetch_openlimb.py` downloads it on demand and copies the
  licence files alongside it. Any derivative shared publicly must keep CC-BY-SA-4.0
  attribution/share-alike for the data.
- **Format (verified by inspection):** each versioned folder
  (`version-2022-10`, `version-2023-06`, `version-2025-07`) contains:
  - a *size-normalised* mean limb STL (`Mean-Limb.stl` in 2025;
    `Mean_Limb_Shape.stl` earlier) — an **open** skin surface (open at the
    proximal end), stored as an STL triangle soup (`n_vertices == 3*n_faces`);
  - PCA modes: `Modes.npy` `(3N, k)` in 2022, `Components.npy`/`Components16.npy`
    `(k, 3N)` in 2023/2025 (2025 is `float16`, 32 modes, N=116730 vertices);
  - `LR.pkl`, a linear-regression model mapping skin-only PCA scores to full
    model scores (used by the authors to keep random samples anatomically
    feasible).
- **Sampling rule (from the authors' `Generate Random Synthetic Model.py`):**
  `vertices = mean + Σ_i mode_i * eigenvalue_i`, in size-normalised space, then
  scale by the **intact tibia length** (mean **378.52 mm**, training range
  **313.05–466.34 mm** for v2025-07).
- **How this MVP uses it:**
  - We weld the triangle soup into a connected manifold (mapping the modes to the
    welded vertices) so the mesh can be smoothed/socketised.
  - We sample the mean, random draws within the published per-mode eigenvalue
    ranges (transcribed into `socketcad/limb_source.py`), and single-PC sweeps.
  - We do **not** currently apply the `LR.pkl` feasibility regression (loading a
    pickled sklearn model is version-fragile); direct mode sampling within the
    published ranges is a documented, reasonable simplification for a demo. This
    is a labelled assumption.

## 2. Rectification zones and default magnitudes — VERIFIED taxonomy, MIXED magnitudes

The **zone taxonomy** and several quantitative bounds come from a peer-reviewed
analysis of real CAD/CAM socket designs:

- Dickinson A., et al. (2024), *Insights into the spectrum of transtibial
  prosthetic socket design from expert clinicians and their digital records.*
  **Frontiers in Rehabilitation Sciences** 5:1354069.
  DOI: <https://doi.org/10.3389/fresc.2024.1354069> (retrieved in full via
  PubMed Central, PMC11272479).
  - **163 residual-limb/socket pairs** from 134 individuals (142 PTB, 21 TSB).
  - Named rectification "design features" (this is exactly our zone set):
    - **carves** (push material in): patellar tendon (PT), medial & lateral
      paratibial (MP, LP), lateral–medial supracondylar (LMC);
    - **builds** (relieve material out): fibular head (FH), tibial crest (TC),
      distal end elongation (DE);
    - **gross volume reduction (VR, %)** across the limb.
  - Verified bounds we use for defaults/clamps:
    - the great majority of sockets used **< 10% volume reduction** and
      **< 10 mm distal-end elongation**; VR 10th-percentile ≈ **1.5%**;
    - local rectifications were **larger in PTB** than TSB; **VR larger in TSB**;
    - PT carve depth was statistically *independent* of other rectifications
      (so its exact value is relatively design-insensitive);
    - FH build correlates positively with VR; MP and LP carves correlate strongly
      with each other.
  - NOTE: Table 3's exact per-zone median magnitudes were not machine-readable
    from the retrieved full text (only DE 90th-percentile = 10 mm and VR
    10th-percentile = 1.5% came through cleanly). Where an exact median was not
    recovered we used a **mid-range value consistent with the reported bounds and
    the PTB>TSB / FH~VR relationships**, and labelled it as such in
    `socketcad/spec.py`.

- Convery P., Buis A.W.P. (2003), *Measurement of the consistency of
  patellar-tendon-bearing cast rectification.* **Prosthetics and Orthotics
  International** 27(3):207–213. (PubMed 14727701.)
  - Major-zone carve mean differences between prosthetists ≈ **2 mm** (SD ±1 mm);
  - fibular-head apex location varied with **2.8–4.3 mm SD**.
  - Used to sanity-check the order of magnitude of our PT carve and FH build
    defaults (a few mm).

### Default magnitudes actually used (see `socketcad/spec.py`)

| Zone (type)              | PTB default | TSB default | Grounding |
|--------------------------|------------:|------------:|-----------|
| Patellar tendon (carve)  | 4 mm | 1 mm | a few mm (Convery 2003); design-insensitive (Dickinson 2024) |
| Medial paratibial (carve)| 4 mm | 1.5 mm | larger in PTB (Dickinson 2024) |
| Lateral paratibial (carve)| 4 mm | 1.5 mm | larger in PTB (Dickinson 2024) |
| Fibular head (build)     | 4 mm | 1.5 mm | 2.8–4.3 mm SD (Convery 2003); FH~VR (Dickinson 2024) |
| Tibial crest (build)     | 3 mm | 1.5 mm | mid-range; offload bony ridge |
| Distal end (build)       | 4 mm | 5 mm | < 10 mm; larger DE with high VR (Dickinson 2024) |
| Posterior calf (carve/load)| 3 mm | 1.5 mm | mid-range PTB loading |
| Supracondylar (carve)    | 2 mm | 1 mm | optional/“suspension” feature (Dickinson 2024) |
| Volume reduction (%)     | ~4% | ~7% | < 10% typical; larger for TSB (Dickinson 2024) |

Hard clamps: per-zone ≤ 12 mm, distal end ≤ 10 mm, volume reduction ≤ 12%.

## 3. Datasets that are NOT used (and why)

- Proprietary scan→socket datasets (e.g. ~118-patient OIM/Twente NL; the
  ~163-pair UK set behind Dickinson 2024) are **not openly available**. This MVP
  does not use or assume access to them. They are relevant only as the future
  "real pairs" a clinical partner might provide to train a learned model.

## 4. Attribution summary

If you use or redistribute outputs derived from OpenLimbTT data, you must
attribute it under CC-BY-SA-4.0 and cite Sunderland et al. (2025). The
rectification defaults are informed by Dickinson et al. (2024) and Convery &
Buis (2003); please cite them if you reuse the parameterisation.
