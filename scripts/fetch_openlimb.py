#!/usr/bin/env python3
"""Download the OpenLimbTT statistical shape model into ``data/openlimb/``.

OpenLimbTT data is licensed CC-BY-SA-4.0 (code MIT) by the University of
Southampton group (Sunderland et al.). We deliberately do NOT vendor it in this
repository -- run this script to fetch it on demand. See ``docs/SOURCES.md``
for the full citation and licensing/attribution requirements.

Usage::

    python scripts/fetch_openlimb.py                 # latest version
    python scripts/fetch_openlimb.py --version version-2023-06
    python scripts/fetch_openlimb.py --keep-clone    # keep the full clone

If you have no network access, the engine still runs using the clearly-labelled
synthetic fallback limb (see socketcad/limb_source.py).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPO = "https://github.com/abel-research/OpenLimbTT.git"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(HERE, "data", "openlimb")

# Files needed per version (mean size-normalised shape + PCA modes, plus the
# skin + bone full-size surfaces used by the linear-FEM interface-pressure path).
WANTED = ("Mean-Limb.stl", "Mean_Limb_Shape.stl", "Components16.npy",
          "Components.npy", "Modes.npy", "LR.pkl",
          "Mean-Skin-Full-Size.stl", "Mean-Bones-Full-Size.stl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="version-2025-07",
                    help="OpenLimbTT version folder to fetch (default: version-2025-07)")
    ap.add_argument("--keep-clone", action="store_true",
                    help="keep the full git clone under data/openlimb/OpenLimbTT")
    args = ap.parse_args()

    os.makedirs(DEST, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="openlimbtt_")
    clone_dir = os.path.join(tmp, "OpenLimbTT")
    print(f"Cloning {REPO} ...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO, clone_dir], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: clone failed ({e}). No network? The engine will use the "
              f"synthetic fallback limb.", file=sys.stderr)
        return 1

    src_ver = os.path.join(clone_dir, args.version)
    if not os.path.isdir(src_ver):
        avail = [d for d in os.listdir(clone_dir) if d.startswith("version-")]
        print(f"ERROR: version '{args.version}' not found. Available: {avail}",
              file=sys.stderr)
        return 2

    out_ver = os.path.join(DEST, args.version)
    os.makedirs(out_ver, exist_ok=True)
    copied = []
    for name in os.listdir(src_ver):
        if name in WANTED:
            shutil.copy2(os.path.join(src_ver, name), os.path.join(out_ver, name))
            copied.append(name)
    # Preserve attribution/licence alongside the data.
    for lic in ("DATA-LICENSE", "CODE-LICENSE", "README.md"):
        p = os.path.join(clone_dir, lic)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_ver, f"OpenLimbTT-{lic}"))

    if args.keep_clone:
        dst = os.path.join(DEST, "OpenLimbTT")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(clone_dir, dst)

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Fetched {args.version}: {copied}\n  -> {out_ver}")
    print("Remember: OpenLimbTT data is CC-BY-SA-4.0; attribute per docs/SOURCES.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
