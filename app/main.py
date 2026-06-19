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

from socketcad.geometry import estimate_frame
from socketcad.limb_source import estimate_landmarks, load_limb_mesh, sample_limb
from socketcad.pipeline import run_pipeline
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


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


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
