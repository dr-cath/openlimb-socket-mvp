# Limitations and explicit assumptions

This is a **research/demo prototype**, not a cleared medical device, and not a
validated clinical tool. The points below are deliberately honest so the
engine's outputs are not over-trusted.

## Not a medical device
- No regulatory clearance. No claims of clinical accuracy, fit, comfort, or
  safety. Any real patient use must happen only inside an equipped institution
  under that institution's own point-of-care responsibility — out of scope here.
- The attachment interface ("pylon-adapter boss") is **geometry only, a
  non-engineered placeholder**. It is not load-rated and must not be used to
  bear weight.

## Design sense-check & interface-pressure proxy
- The "Design sense-check" combines a **rule-based plausibility audit** (against
  Dickinson et al. 2024 practice: load/offload direction, magnitude bounds,
  volume-reduction band, distal-end offload, contact) with an **interface-pressure
  proxy**.
- The pressure proxy is a **Winkler elastic-foundation estimate**
  (`pressure ≈ tissue_stiffness × interference_depth`) with an *illustrative*
  stiffness constant. It **ignores shear, tissue layering, bone, liners, global
  equilibrium and large deformation**, and is **not** a substitute for FEA or
  in-socket pressure measurement. Treat the kPa numbers as relative indicators
  of *where* pressure concentrates, not validated magnitudes.
## In-process FEA (linear-elastic interface-pressure estimate)
- `socketcad/fea.py` **actually runs** a finite-element analysis (no external
  solver): it tetrahedralises the OpenLimbTT limb, stiffens the **bone** elements
  to a near-rigid core, imposes the socket's rectification as an inward skin
  displacement, and solves linear elasticity (scipy sparse) for the reaction
  **interface pressure**. Run it with `python scripts/run_fea.py`.
- **This is a linear, small-strain proxy.** Honest scope:
  - **No** hyperelasticity, **no** frictional contact mechanics, **no** liner,
    **no** large deformation, **no** muscle/fat/skin layering.
  - Material constants are **literature-level placeholders** (soft tissue
    E≈0.2 MPa, ν 0.40; bone E≈100 MPa).
  - Robust statistics (**median / p90 of loaded nodes**) are meaningful and the
    model **discriminates** designs sensibly (e.g. higher total-contact pressure
    for high-volume-reduction TSB). The **raw peak near bone is a linear-FEM
    singularity** (thin/zero tissue over bone, e.g. tibial crest), winsorised for
    display and **not** a validated magnitude.
  - Not validated against measured in-socket pressures. Do not treat the kPa
    values as clinically accurate.
- **Roadmap — rigorous simulator model (nonlinear contact FEA):** export the
  socket+soft-tissue+bone with cited hyperelastic materials, a liner and
  donning/stance boundary conditions to a solver deck (FEBio/Abaqus), validated
  against published interface pressures. Not implemented yet.

## Landmarking / coordinate frame (heuristic)
- The limb's long axis is found by PCA; the proximal end is identified as the
  larger cross-section; the anterior direction is inferred as opposite the
  rounded posterior calf bulge (calibrated against the OpenLimbTT mean shape).
  These are geometric heuristics, **not anatomical landmark detection**. They
  can be wrong on atypical or non-OpenLimbTT geometry.
- **Left vs. right** limb cannot be recovered from an unlabelled mesh, so the
  medial/lateral assignment is a flag (`is_left_limb`, default left). If reliefs
  appear on the wrong side, flip it.

## Rectification model (parametric, not learned)
- Zones are smooth Gaussian falloffs in (height, angle) space applied as radial
  displacements, plus a uniform volume reduction, plus optional painted patches.
  This is an honest **baseline**; it does not capture prosthetist judgement,
  tissue tolerance, soft-tissue mechanics, or load transfer. A future *learned*
  model must beat it (same `limb + spec -> rectified positive` contract).
- Default magnitudes are literature-anchored where possible (see
  `docs/SOURCES.md`); some are mid-range values consistent with reported bounds
  rather than exact published medians, because the source table's per-zone
  medians were not machine-readable.

## OpenLimbTT usage
- We sample modes directly within the published per-mode eigenvalue ranges and do
  **not** apply the authors' `LR.pkl` anatomical-feasibility regression, so
  extreme multi-mode combinations could in principle be less anatomically
  plausible than the authors' own random generator. The mean and modest samples
  are fine.
- The model is size-normalised; we scale by a single intact-tibia length (default
  378.52 mm). Per-subject scaling is a single uniform factor.

## Socketisation geometry
- The outer wall is a uniform normal offset of the cavity surface; in strongly
  concave regions a normal offset can self-intersect. The validation suite checks
  watertightness, winding consistency and `is_volume` (a manifold proxy) and wall
  thickness, but **does not** robustly detect all geometric self-intersections.
  Treat `is_volume` as "manifold/printable proxy", not a self-intersection proof.
- The trimline is a single (optionally tilted) cutting plane; it does not yet
  follow the true scalloped PTB trimline (high anterior over patella, relieved
  posterior for knee flexion) beyond a simple anterior drop.
- Reported socket "volume" is the watertight mesh volume; it can be inflated by
  minor offset overlaps near the wide proximal rim. Wall-thickness sampling is
  the more reliable printability metric.

## Validation thresholds
- Plausibility bounds (height, diameter, wall thickness tolerance) are generous
  and only catch gross failures, not clinical correctness.
