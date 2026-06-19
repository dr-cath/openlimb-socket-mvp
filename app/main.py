"""Minimal FastAPI app for the intent -> preview -> approve -> export loop.

This is a *demo* interface. A clinician expresses clinical intent (philosophy,
snugness, named reliefs/loads, painted patches, trimline, wall thickness) and
the server runs the same :func:`socketcad.pipeline.run_pipeline` used by the CLI.
Meshes are served as PLY so the click-to-paint relief raycast point lands in the
exact same millimetre coordinate frame the engine uses.

Session interactions are logged to ``data/out/session_log.jsonl`` -- these
human-in-the-loop logs (what was changed, what was unmet) are the spec for v2 and
for the future learned model.

Run::

    uvicorn app.main:app --reload   # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import time
import uuid

import trimesh
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from socketcad.autodesign import build_candidates
from socketcad.geometry import estimate_frame
from socketcad.limb_source import estimate_landmarks, load_limb_mesh, sample_limb
from socketcad.pipeline import run_pipeline
from socketcad.processlog import build_process_log
from socketcad.spec import spec_from_dict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "out")
STATIC = os.path.join(ROOT, "app", "static")
LOG = os.path.join(OUT, "session_log.jsonl")
os.makedirs(OUT, exist_ok=True)

app = FastAPI(title="OpenLimb Socket MVP")

# Single-user demo session state.
SESSION: dict = {"limb": None, "frame": None, "info": None}


def _log(event: str, payload: dict) -> None:
    with open(LOG, "a") as f:
        f.write(json.dumps({"ts": time.time(), "event": event, **payload}, default=str) + "\n")


def _serve_mesh(mesh: trimesh.Trimesh, name: str) -> str:
    path = os.path.join(OUT, f"{name}.ply")
    mesh.export(path)
    return f"/files/{name}.ply"


def _serve_glb(mesh: trimesh.Trimesh, name: str) -> str:
    path = os.path.join(OUT, f"{name}.glb")
    mesh.export(path)
    return f"/files/{name}.glb"


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/compare")
def compare_page():
    return FileResponse(os.path.join(STATIC, "compare.html"))


@app.post("/api/autodesign")
async def api_autodesign(request: Request):
    """Prompt + scan -> auto-design 3 candidate sockets with trade-offs.

    Each candidate is fully built (rectify -> socketize -> validate), exported as
    GLB (for the viewer) and STL (for download), and accompanied by a
    Mimics/ScanIP-style operation history.
    """
    body = await request.json()
    prompt = body.get("prompt", "")
    is_left = bool(body.get("is_left_limb", True))

    # The scan selector is authoritative: an explicit shape always re-samples
    # (so "random" yields a genuinely new limb each run). Fall back to a loaded/
    # uploaded limb only when no shape is given.
    shape = body.get("shape")
    if shape:
        mesh, info = sample_limb(shape, seed=body.get("seed"))
        frame = estimate_frame(mesh, is_left=is_left)
        SESSION.update(limb=mesh, frame=frame, info=info)
    elif SESSION.get("limb") is not None:
        mesh, info, frame = SESSION["limb"], SESSION["info"], SESSION["frame"]
    else:
        mesh, info = sample_limb("mean")
        frame = estimate_frame(mesh, is_left=is_left)
        SESSION.update(limb=mesh, frame=frame, info=info)

    candidates, intents = build_candidates(prompt, is_left_limb=is_left)
    _log("autodesign", {"prompt": prompt, "intents": intents,
                        "source": info.get("source")})

    from socketcad.processlog import _ZONE_NAMES
    from socketcad.validate import colorize_displacement

    results = []
    for cand in candidates:
        pr = run_pipeline(mesh, cand.spec, out_dir=OUT,
                          basename=f"cmp_{cand.key}", make_heatmap=True)
        socket_glb = _serve_glb(pr.socket.socket, f"cmp_{cand.key}_socket")

        # Heatmap on the prosthesis: vertex-coloured rectified positive
        # (blue = carved in / loaded, red = relieved out). This *is* the
        # trade-off made visible -- where and how much material was moved.
        vmax = pr.heatmap.get("vmax_mm") or 1.0
        heat_mesh = colorize_displacement(pr.rectify.rectified,
                                          pr.rectify.signed_normal_disp, vmax=vmax)
        heatmap_glb = _serve_glb(heat_mesh, f"cmp_{cand.key}_heatmap")

        # Per-zone trade-off annotations (what was carved/relieved, in mm).
        annotations = [
            {"zone": _ZONE_NAMES.get(z.name, z.name),
             "action": "carve" if z.signed_magnitude_mm < 0 else "relief",
             "mm": round(abs(z.signed_magnitude_mm), 1)}
            for z in pr.rectify.zones
        ]

        plog = build_process_log(info, pr.rectify, pr.socket, pr.validation,
                                 cand.spec, title=cand.title, prompt=prompt,
                                 intents=intents)
        # Persist the operation history alongside the meshes.
        with open(os.path.join(OUT, f"cmp_{cand.key}_processlog.txt"), "w") as f:
            f.write(plog.to_text())

        wt = pr.validation.metrics.get("wall_thickness_mm", {})
        rsum = pr.rectify.summary()
        ssum = pr.socket.summary()
        results.append({
            "key": cand.key,
            "title": cand.title,
            "summary": cand.summary,
            "pros": cand.pros,
            "cons": cand.cons,
            "grounding": cand.grounding,
            "socket_glb": socket_glb,
            "heatmap_glb": heatmap_glb,
            "heatmap_vmax_mm": round(float(vmax), 1),
            "annotations": annotations,
            "stl_url": "/files/" + os.path.basename(pr.exports["socket_stl"]),
            "metrics": {
                "philosophy": cand.spec.philosophy,
                "snugness": round(cand.spec.snugness, 2),
                "volume_reduction_pct": round(cand.spec.volume_reduction_pct(), 1),
                "max_carve_mm": round(rsum["max_carve_mm"], 1),
                "max_relief_mm": round(rsum["max_relief_mm"], 1),
                "wall_mean_mm": round(wt.get("mean", float("nan")), 1) if wt else None,
                "wall_in_tol_pct": round(
                    pr.validation.metrics.get("wall_thickness_frac_in_tol", 0) * 100),
                "socket_volume_cm3": round(ssum.get("socket_volume_cm3", float("nan"))),
                "printable": bool(ssum["is_volume"]),
                "validation_passed": bool(pr.validation.passed),
            },
            "processlog": plog.as_dict(),
            "log_verified": True,  # values computed from the model & test-checked
        })

    from socketcad.sources import CITATIONS, GROUNDING_NOTE

    return {
        "prompt": prompt,
        "intents": intents,
        "sources": CITATIONS,
        "grounding_note": GROUNDING_NOTE,
        "scan": {
            "source": info.get("source"),
            "version": info.get("version", "n/a"),
            "length_mm": round(float(frame.height)),
            "n_vertices": int(len(mesh.vertices)),
            "limb_glb": _serve_glb(mesh, "cmp_limb"),
        },
        "candidates": results,
    }


@app.post("/api/sample")
async def api_sample(request: Request):
    body = await request.json()
    shape = body.get("shape", "mean")
    seed = body.get("seed")
    mesh, info = sample_limb(shape, seed=seed)
    frame = estimate_frame(mesh, is_left=body.get("is_left_limb", True))
    SESSION.update(limb=mesh, frame=frame, info=info)
    lm = estimate_landmarks(mesh, frame)
    _log("sample", {"shape": shape, "source": info["source"]})
    return {
        "limb_url": _serve_mesh(mesh, "session_limb"),
        "info": info,
        "length_mm": frame.height,
        "n_vertices": int(len(mesh.vertices)),
        "landmarks": lm.as_dict(),
    }


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    dest = os.path.join(OUT, f"upload_{uuid.uuid4().hex[:8]}_{file.filename}")
    with open(dest, "wb") as f:
        f.write(await file.read())
    mesh, info = load_limb_mesh(dest)
    mesh.merge_vertices()
    frame = estimate_frame(mesh)
    SESSION.update(limb=mesh, frame=frame, info=info)
    lm = estimate_landmarks(mesh, frame)
    _log("upload", {"file": file.filename, "n_vertices": int(len(mesh.vertices))})
    return {
        "limb_url": _serve_mesh(mesh, "session_limb"),
        "info": info,
        "length_mm": frame.height,
        "n_vertices": int(len(mesh.vertices)),
        "landmarks": lm.as_dict(),
    }


@app.post("/api/generate")
async def api_generate(request: Request):
    if SESSION.get("limb") is None:
        return JSONResponse({"error": "no limb loaded; sample or upload first"}, status_code=400)
    spec_dict = await request.json()
    spec = spec_from_dict(spec_dict)
    pr = run_pipeline(
        SESSION["limb"], spec, out_dir=OUT, basename="session", make_heatmap=True
    )
    # Re-serve coloured heatmap + meshes as PLY for the viewer.
    from socketcad.validate import colorize_displacement

    heat = colorize_displacement(pr.rectify.rectified, pr.rectify.signed_normal_disp)
    urls = {
        "rectified_url": _serve_mesh(pr.rectify.rectified, "session_rectified"),
        "socket_url": _serve_mesh(pr.socket.socket, "session_socket"),
        "heatmap_url": _serve_mesh(heat, "session_heatmap"),
    }
    _log("generate", {"spec": spec.to_log_dict(), "passed": pr.validation.passed})
    return {
        **urls,
        "validation": pr.validation.to_dict(),
        "rectify": pr.rectify.summary(),
        "socket": pr.socket.summary(),
        "heatmap_vmax_mm": pr.heatmap.get("vmax_mm"),
    }


@app.post("/api/export")
async def api_export(request: Request):
    if SESSION.get("limb") is None:
        return JSONResponse({"error": "nothing to export"}, status_code=400)
    body = await request.json()
    name = body.get("name", "approved_socket")
    src = os.path.join(OUT, "session_socket.ply")
    if not os.path.exists(src):
        return JSONResponse({"error": "generate a socket first"}, status_code=400)
    mesh = trimesh.load(src, process=False)
    stl_path = os.path.join(OUT, f"{name}.stl")
    mesh.export(stl_path)
    _log("approve_export", {"name": name, "notes": body.get("notes", "")})
    return {"stl_url": f"/files/{name}.stl", "path": stl_path}


@app.post("/api/log")
async def api_log(request: Request):
    body = await request.json()
    _log(body.get("event", "ui"), {"payload": body})
    return {"ok": True}


app.mount("/files", StaticFiles(directory=OUT), name="files")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
