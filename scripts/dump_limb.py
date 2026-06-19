#!/usr/bin/env python3
"""Dump sampled limb meshes to STL for eyeballing.

Examples::

    python scripts/dump_limb.py --shape mean -o data/out/limb_mean.stl
    python scripts/dump_limb.py --shape random --seed 3 -o data/out/limb_r3.stl
    python scripts/dump_limb.py --pc 0 --value 12 -o data/out/limb_pc0.stl
    python scripts/dump_limb.py --shape synthetic -o data/out/limb_synth.stl
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from socketcad.limb_source import sample_limb, estimate_landmarks  # noqa: E402
from socketcad.geometry import estimate_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="mean",
                    choices=["mean", "random", "components", "synthetic"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pc", type=int, default=None,
                    help="perturb a single principal component (shape=components)")
    ap.add_argument("--value", type=float, default=10.0,
                    help="eigenvalue for the --pc mode")
    ap.add_argument("-o", "--out", default="data/out/limb.stl")
    args = ap.parse_args()

    components = None
    shape = args.shape
    if args.pc is not None:
        shape = "components"
        components = np.zeros(args.pc + 1)
        components[args.pc] = args.value

    mesh, info = sample_limb(shape, components=components, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    mesh.export(args.out)

    frame = estimate_frame(mesh)
    lm = estimate_landmarks(mesh, frame)
    print(f"source={info['source']} shape={info.get('shape')} "
          f"verts={len(mesh.vertices)} extent_mm={np.round(mesh.extents, 1)}")
    print(f"frame: limb length {frame.height:.1f} mm")
    print(f"wrote {args.out}")
    print("landmarks (world mm):")
    for k, v in lm.points.items():
        print(f"  {k:18s} {np.round(v, 1)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
