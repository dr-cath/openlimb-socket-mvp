"""End-to-end orchestration: limb mesh + spec -> validated, exported socket.

This is the single call the CLI demo and the web app both use, so the
intent->preview->approve->export loop is identical everywhere.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import trimesh

from .rectify import RectifyResult, rectify
from .socketize import SocketResult, socketize
from .spec import RectificationSpec
from .validate import ValidationReport, displacement_heatmap, validate_socket


@dataclass
class PipelineResult:
    rectify: RectifyResult
    socket: SocketResult
    validation: ValidationReport
    exports: dict = field(default_factory=dict)
    heatmap: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rectify": self.rectify.summary(),
            "socket": self.socket.summary(),
            "validation": self.validation.to_dict(),
            "exports": self.exports,
            "heatmap": self.heatmap,
        }


def run_pipeline(
    limb: trimesh.Trimesh,
    spec: RectificationSpec,
    out_dir: Optional[str] = None,
    basename: str = "socket",
    export: bool = True,
    make_heatmap: bool = True,
) -> PipelineResult:
    """Run rectify -> socketize -> validate (-> heatmap, export)."""
    rec = rectify(limb, spec)
    sock = socketize(
        rec.rectified,
        frame=rec.frame,
        wall_thickness_mm=spec.wall_thickness_mm,
        trimline_height_frac=spec.trimline_height_frac,
        trimline_anterior_drop_frac=spec.trimline_anterior_drop_frac,
        is_left_limb=spec.is_left_limb,
    )
    report = validate_socket(
        sock.socket,
        positive=sock.positive,
        outer=sock.outer,
        frame=sock.frame,
        target_thickness_mm=spec.wall_thickness_mm,
        trimline_height_mm=sock.trimline_height_mm,
    )

    exports: dict = {}
    heatmap: dict = {}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if make_heatmap and out_dir:
        heatmap = displacement_heatmap(
            rec,
            out_png=os.path.join(out_dir, f"{basename}_heatmap.png"),
            out_mesh=os.path.join(out_dir, f"{basename}_heatmap.glb"),
        )

    if export and out_dir:
        paths = {
            "socket_stl": os.path.join(out_dir, f"{basename}_socket.stl"),
            "rectified_stl": os.path.join(out_dir, f"{basename}_rectified.stl"),
            "limb_stl": os.path.join(out_dir, f"{basename}_limb.stl"),
        }
        sock.socket.export(paths["socket_stl"])
        rec.rectified.export(paths["rectified_stl"])
        rec.limb.export(paths["limb_stl"])
        exports = paths

    return PipelineResult(
        rectify=rec, socket=sock, validation=report, exports=exports, heatmap=heatmap
    )
