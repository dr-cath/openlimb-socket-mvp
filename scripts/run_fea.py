#!/usr/bin/env python3
"""Run the in-process linear-elastic FEM interface-pressure estimate.

Requires the OpenLimbTT skin + bone surfaces (FEA needs the internal bone
geometry, which the synthetic fallback does not have)::

    python scripts/fetch_openlimb.py     # gets skin + bone too
    python scripts/run_fea.py --philosophy PTB

Outputs a summary, a coloured pressure mesh (PLY) and a PNG to data/out/.

HONEST SCOPE: linear, small-strain proxy with placeholder material constants.
Robust median/percentile pressures are meaningful; the raw peak near bone is a
linear-FEM singularity, not a validated magnitude. The rigorous path is a
nonlinear contact solver (FEBio/Abaqus). See docs/LIMITATIONS.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from socketcad import fea  # noqa: E402
from socketcad.autodesign import build_candidates  # noqa: E402
from socketcad.limb_source import sample_limb  # noqa: E402
from socketcad.rectify import rectify  # noqa: E402
from socketcad.spec import RectificationSpec  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "out")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--philosophy", default="PTB", choices=["PTB", "TSB"])
    ap.add_argument("--snugness", type=float, default=0.5)
    ap.add_argument("--target-faces", type=int, default=2000)
    args = ap.parse_args()

    geom = fea.load_openlimb_fea_geometry()
    if geom is None:
        print("ERROR: OpenLimbTT skin+bone not found. Run scripts/fetch_openlimb.py "
              "first (FEA needs the bone geometry).", file=sys.stderr)
        return 1
    skin_solid, bone = geom

    limb, _ = sample_limb("mean")
    spec = RectificationSpec(philosophy=args.philosophy, snugness=args.snugness)
    rec = rectify(limb, spec)

    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    res = fea.run_limb_fea(skin_solid, bone, fea.interference_from_rectify(rec),
                           target_faces=args.target_faces)
    print(f"FEA solved in {time.time() - t:.1f}s")
    print(json.dumps(res.summary(), indent=2))

    # Coloured pressure point mesh for inspection.
    import matplotlib
    from matplotlib.colors import Normalize
    norm = Normalize(0, float(np.percentile(res.pressure_kpa, 95)) or 1.0)
    colors = (matplotlib.colormaps["inferno"](norm(res.pressure_kpa)) * 255).astype(np.uint8)
    import trimesh
    pc = trimesh.PointCloud(res.skin_points, colors=colors)
    ply = os.path.join(OUT, f"fea_pressure_{args.philosophy}.ply")
    pc.export(ply)
    print(f"  -> {ply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
