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


def test_process_log_values_match_recomputation():
    """Every numeric value in the operation log must equal an independently
    recomputed value -- proves the log reflects the real pipeline, not text."""
    mesh, info = sample_limb("synthetic", seed=2)
    cands, intents = build_candidates("sensitive distal end", is_left_limb=True)
    spec = cands[0].spec
    pr = run_pipeline(mesh, spec, out_dir=None, export=False, make_heatmap=False)
    log = build_process_log(info, pr.rectify, pr.socket, pr.validation, spec)
    params = {e.operation: e.params for e in log.entries}

    # Import / volume / displacement / validation.
    assert params["Import residual-limb mesh"]["vertices"] == len(pr.rectify.limb.vertices)
    assert params["Global volume reduction"]["volume_reduction_pct"] == round(
        float(spec.volume_reduction_pct()), 2)
    be = params["Blend & smooth displacement field"]
    assert be["max_relief_mm"] == round(float(pr.rectify.signed_normal_disp.max()), 2)
    assert be["max_carve_mm"] == round(float(-pr.rectify.signed_normal_disp.min()), 2)
    validate_key = next(k for k in params if k.startswith("Validate"))
    assert params[validate_key]["passed"] == bool(pr.validation.passed)

    # Per-zone magnitudes vs an independent spec.resolve_zones().
    zones_truth = {z.name: round(abs(z.signed_magnitude_mm), 2)
                   for z in spec.resolve_zones(is_left=pr.rectify.frame.is_left)}
    logged = {e.params["zone"]: round(e.params["magnitude_mm"], 2)
              for e in log.entries if e.params.get("zone")}
    assert logged == zones_truth


def test_random_sampling_is_diverse():
    """Random draws should differ from the mean and from each other (when
    OpenLimbTT data is present); with the synthetic fallback they still vary."""
    a, ia = sample_limb("random", seed=0)
    b, ib = sample_limb("random", seed=1)
    if ia["source"] == "openlimb-tt" and a.vertices.shape == b.vertices.shape:
        import numpy as np
        assert np.linalg.norm(a.vertices - b.vertices, axis=1).mean() > 1.0


def test_sense_check_flags_direction_and_pressure():
    from socketcad.sensecheck import sense_check, interface_pressure_proxy
    mesh, info = sample_limb("synthetic", seed=3)
    cands, _ = build_candidates("standard", is_left_limb=True)
    pr = run_pipeline(mesh, cands[0].spec, out_dir=None, export=False, make_heatmap=False)
    sc = sense_check(pr.rectify, cands[0].spec)
    names = {c.name: c.status for c in sc.checks}
    # Default PTB design loads tolerant zones / offloads sensitive ones correctly.
    assert names["Load/offload direction"] == "pass"
    # Pressure proxy is non-negative and reports contact.
    pressure, stats = interface_pressure_proxy(pr.rectify)
    assert (pressure >= 0).all()
    assert stats["peak_kPa"] >= 0 and 0 <= stats["contact_area_pct"] <= 100
    assert sc.as_dict()["overall"] in {"pass", "warn", "fail"}


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
