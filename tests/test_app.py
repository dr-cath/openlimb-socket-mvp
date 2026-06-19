import pytest

pytest.importorskip("httpx")  # FastAPI TestClient needs httpx

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

client = TestClient(app)


def test_sample_generate_export_loop():
    # sample a synthetic limb (hermetic)
    r = client.post("/api/sample", json={"shape": "synthetic"})
    assert r.status_code == 200
    data = r.json()
    assert data["n_vertices"] > 0
    assert data["limb_url"].endswith(".ply")

    # generate a socket from clinical intent
    spec = {"philosophy": "PTB", "snugness": 0.5, "wall_thickness_mm": 4.0}
    r = client.post("/api/generate", json=spec)
    assert r.status_code == 200
    g = r.json()
    assert "socket_url" in g and "heatmap_url" in g
    assert g["validation"]["passed"] is True

    # approve & export
    r = client.post("/api/export", json={"name": "test_socket"})
    assert r.status_code == 200
    assert r.json()["stl_url"].endswith(".stl")


def test_generate_requires_limb():
    from app.main import SESSION

    SESSION["limb"] = None
    r = client.post("/api/generate", json={"philosophy": "PTB"})
    assert r.status_code == 400
