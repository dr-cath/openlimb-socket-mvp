import numpy as np

from socketcad.limb_source import (
    estimate_landmarks,
    generate_synthetic_limb,
    sample_limb,
)


def test_synthetic_limb_topology():
    m = generate_synthetic_limb(seed=1)
    assert len(m.vertices) > 1000
    # connected (shared vertices): average neighbour count well above 0
    assert np.mean([len(n) for n in m.vertex_neighbors]) > 3
    # open at proximal end (residual-limb scan contract): not watertight as-is
    assert not m.is_watertight


def test_sample_limb_fallback():
    mesh, info = sample_limb("synthetic", seed=2)
    assert info["source"] == "synthetic-fallback"
    assert mesh.vertices.shape[1] == 3


def test_landmarks_present_and_ordered(limb, frame):
    lm = estimate_landmarks(limb, frame)
    for key in ["patellar_tendon", "tibial_end_distal", "fibular_head", "posterior_calf"]:
        assert key in lm.points
    # distal tibial end should sit lower than the patellar tendon
    pt = frame.to_canonical(lm.points["patellar_tendon"])[2]
    de = frame.to_canonical(lm.points["tibial_end_distal"])[2]
    assert de < pt


def test_openlimb_sampling_shapes(openlimb):
    mean = openlimb.sample()
    assert mean.vertices.shape[1] == 3
    # welded mesh: vertices far fewer than the triangle-soup 3*n_faces
    assert len(mean.vertices) < 3 * len(mean.faces)
    # a perturbed sample differs from the mean
    comp = np.zeros(openlimb.n_modes)
    comp[0] = openlimb.eigen_ranges[0, 1]
    pert = openlimb.sample(components=comp)
    assert not np.allclose(mean.vertices, pert.vertices)
