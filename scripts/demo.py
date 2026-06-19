#!/usr/bin/env python3
"""Reproducible end-to-end demo: limb -> intent -> socket STL (+ heatmap).

Produces three socket designs from one limb to show the engine's range:

1. ``ptb``      -- a patellar-tendon-bearing design (pronounced local rectifications)
2. ``tsb``      -- a total-surface-bearing design (small reliefs, larger volume reduction)
3. ``atypical`` -- a PTB design PLUS a custom painted relief patch (e.g. a neuroma
                   or bony spur), demonstrating handling that template/macro tools
                   cannot do.

Outputs (STL sockets, rectified positives, raw limbs, heatmaps) go to
``data/out/`` and a session log is written to ``data/out/session_log.jsonl``.

Run::

    python scripts/fetch_openlimb.py   # optional: real OpenLimbTT limbs
    python scripts/demo.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from socketcad.limb_source import estimate_landmarks, sample_limb  # noqa: E402
from socketcad.geometry import estimate_frame  # noqa: E402
from socketcad.pipeline import run_pipeline  # noqa: E402
from socketcad.spec import CustomReliefRegion, RectificationSpec  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "out")


def _surface_point(mesh, frame, height_frac, angle_rad):
    """A point on the limb surface near (height_frac, angle_from_anterior)."""
    from socketcad.geometry import angular_distance, cylindrical_coords

    Vc = frame.to_canonical(mesh.vertices)
    z, theta, _, _ = cylindrical_coords(Vc)
    cost = ((z / frame.height - height_frac) / 0.06) ** 2 + (
        angular_distance(theta, angle_rad) / 0.3
    ) ** 2
    return mesh.vertices[int(np.argmin(cost))]


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    mesh, info = sample_limb("mean")
    frame = estimate_frame(mesh)
    print(f"Limb source: {info['source']} ({info.get('version', 'n/a')}), "
          f"length {frame.height:.0f} mm, {len(mesh.vertices)} vertices")

    # An atypical, person-specific relief: simulate a sensitive spot on the
    # anterolateral distal third (e.g. a neuroma / bony spur).
    spot = _surface_point(mesh, frame, height_frac=0.3, angle_rad=np.deg2rad(60))

    cases = {
        "ptb": RectificationSpec(philosophy="PTB", snugness=0.5),
        "tsb": RectificationSpec(philosophy="TSB", snugness=0.7),
        "atypical": RectificationSpec(
            philosophy="PTB", snugness=0.5,
            custom_regions=[CustomReliefRegion(
                point=tuple(float(x) for x in spot), radius_mm=18.0,
                depth_mm=6.0, carve=False, label="neuroma_relief")],
        ),
    }

    log_path = os.path.join(OUT, "session_log.jsonl")
    rows = []
    with open(log_path, "a") as log:
        for name, spec in cases.items():
            t = time.time()
            pr = run_pipeline(mesh, spec, out_dir=OUT, basename=f"demo_{name}")
            dt = time.time() - t
            rep = pr.validation
            rows.append((name, rep.passed, pr.socket.summary(), rep.metrics))
            entry = {
                "event": "demo_generate", "case": name, "elapsed_s": round(dt, 1),
                "source": info["source"], "spec": spec.to_log_dict(),
                "validation_passed": rep.passed,
                "rectify": pr.rectify.summary(), "exports": pr.exports,
            }
            log.write(json.dumps(entry, default=str) + "\n")
            print(f"\n[{name}] {dt:.1f}s  validation_passed={rep.passed}")
            print(f"   socket: {json.dumps(pr.socket.summary(), default=str)}")
            wt = rep.metrics.get("wall_thickness_mm", {})
            if wt:
                print(f"   wall thickness mean={wt['mean']:.1f} mm "
                      f"(p05={wt['p05']:.1f}, p95={wt['p95']:.1f}), "
                      f"in-tol={rep.metrics.get('wall_thickness_frac_in_tol', 0)*100:.0f}%")
            print(f"   exports: {pr.exports.get('socket_stl')}")

    print(f"\nSession log appended to {log_path}")
    all_pass = all(r[1] for r in rows)
    print(f"ALL CASES VALID: {all_pass}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
