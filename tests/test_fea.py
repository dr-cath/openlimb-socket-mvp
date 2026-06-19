"""Tests for the linear-elastic FEM core. Unit tests are hermetic; the full
solve is skipped unless OpenLimbTT skin+bone geometry and tetgen are present."""
from __future__ import annotations

import numpy as np
import pytest

from socketcad import fea


def test_constitutive_matrix_symmetric():
    D = fea._D(0.2, 0.4)
    assert D.shape == (6, 6)
    assert np.allclose(D, D.T)


def test_unit_tet_volume():
    # Reference tet with volume 1/6.
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    _, V = fea._tet_B_V(coords)
    assert abs(V - 1.0 / 6.0) < 1e-9


def test_stiffness_symmetric_and_rigid_body_nullspace():
    # Two-tet mesh; K must be symmetric and have ~zero energy for translation.
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], float)
    elems = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    is_bone = np.zeros(len(elems), bool)
    K = fea.assemble_K(nodes, elems, is_bone)
    assert (K - K.T).nnz == 0 or np.allclose((K - K.T).toarray(), 0)
    # Uniform translation -> zero internal force.
    t = np.tile([1.0, 0.0, 0.0], len(nodes))
    assert np.allclose(K @ t, 0, atol=1e-8)


@pytest.mark.skipif(fea.load_openlimb_fea_geometry() is None,
                    reason="OpenLimbTT skin+bone not present (run fetch_openlimb)")
def test_fea_end_to_end_pressure_is_sane():
    from socketcad.limb_source import sample_limb
    from socketcad.rectify import rectify
    from socketcad.spec import RectificationSpec

    skin_solid, bone = fea.load_openlimb_fea_geometry()
    limb, _ = sample_limb("mean")
    rec = rectify(limb, RectificationSpec(philosophy="PTB"))
    res = fea.run_limb_fea(skin_solid, bone, fea.interference_from_rectify(rec),
                           target_faces=1500)
    s = res.summary()
    assert (res.pressure_kpa >= 0).all()
    assert res.is_bone.sum() > 0                      # bone core identified
    assert 10.0 < s["median_contact_kPa"] < 1000.0   # physically plausible band


@pytest.mark.skipif(fea.load_openlimb_fea_geometry() is None,
                    reason="OpenLimbTT skin+bone not present (run fetch_openlimb)")
def test_fea_endpoint():
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    r = client.post("/api/fea", json={"prompt": "standard", "key": "stability"})
    assert r.status_code == 200
    d = r.json()
    assert d["fea_glb"].endswith(".glb")
    assert d["summary"]["median_contact_kPa"] > 0
