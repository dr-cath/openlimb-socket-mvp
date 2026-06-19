"""Tests for prompt -> candidates auto-design, the process log, and the
auto-design API endpoint. Hermetic: uses the synthetic limb fallback when
OpenLimbTT data is absent."""
from __future__ import annotations

import numpy as np

from socketcad.autodesign import build_candidates, detect_intents
from socketcad.limb_source import sample_limb
from socketcad.pipeline import run_pipeline
from socketcad.processlog import build_process_log


def test_detect_intents_finds_keywords():
    intents = detect_intents("New amputee, dysvascular, sensitive bony spur")
    assert "new / recent amputation" in intents
    assert "dysvascular / fragile tissue" in intents
    assert "bony prominence" in intents


def test_build_candidates_spans_stability_to_comfort():
    cands, intents = build_candidates("active, comfortable total contact")
    keys = [c.key for c in cands]
    assert keys == ["stability", "balanced", "comfort"]
    assert cands[0].spec.philosophy == "PTB"
    assert cands[-1].spec.philosophy == "TSB"
    # Comfort design should reduce volume more than the stability design.
    assert cands[-1].spec.volume_reduction_pct() > cands[0].spec.volume_reduction_pct()


def test_process_log_is_ordered_and_faithful():
    mesh, info = sample_limb("synthetic", seed=1)
    cands, intents = build_candidates("bony fibular head", is_left_limb=True)
    spec = cands[0].spec
    pr = run_pipeline(mesh, spec, out_dir=None, export=False, make_heatmap=False)
    log = build_process_log(info, pr.rectify, pr.socket, pr.validation, spec,
                            title=cands[0].title, prompt="bony fibular head",
                            intents=intents)
    steps = [e.step for e in log.entries]
    assert steps == list(range(1, len(steps) + 1))  # strictly ordered
    cats = {e.category for e in log.entries}
    # The history must cover the whole pipeline.
    for required in ("import", "analyze", "rectify", "volume", "shell", "cut", "validate"):
        assert required in cats
    # A per-zone rectify entry exists for each resolved zone.
    rectify_ops = [e for e in log.entries if e.category == "rectify" and e.operation.startswith("Rectify:")]
    assert len(rectify_ops) == len(pr.rectify.zones)
    assert log.to_text().startswith("OPERATION HISTORY")


def test_autodesign_endpoint_returns_three_designs(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.post("/api/autodesign", json={"prompt": "comfort, active",
                                             "shape": "synthetic"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["candidates"]) == 3
    for c in data["candidates"]:
        assert c["socket_glb"].endswith(".glb")
        assert c["stl_url"].endswith(".stl")
        assert c["processlog"]["entries"]
        assert "volume_reduction_pct" in c["metrics"]
