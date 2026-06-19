import numpy as np

from socketcad.limb_source import estimate_landmarks
from socketcad.rectify import rectify
from socketcad.spec import CustomReliefRegion, RectificationSpec


def test_rectify_preserves_vertex_count(limb):
    res = rectify(limb, RectificationSpec(philosophy="PTB"))
    assert len(res.rectified.vertices) == len(limb.vertices)
    assert np.isfinite(res.displacement).all()


def test_rectify_directions(limb):
    """Patellar tendon should carve in (-), fibular head relieve out (+)."""
    res = rectify(limb, RectificationSpec(philosophy="PTB"))
    lm = estimate_landmarks(limb, res.frame)

    def nearest_disp(point):
        i = int(np.argmin(np.linalg.norm(limb.vertices - point, axis=1)))
        return res.signed_normal_disp[i]

    assert nearest_disp(lm.points["patellar_tendon"]) < 0.0
    assert nearest_disp(lm.points["fibular_head"]) > 0.0


def test_volume_reduction_shrinks(limb):
    res = rectify(limb, RectificationSpec(philosophy="TSB", snugness=0.8))
    s = res.summary()
    assert s["rectified_volume_cm3"] < s["limb_volume_cm3"]


def test_custom_region_creates_local_relief(limb):
    res0 = rectify(limb, RectificationSpec(load_patellar_tendon=False,
                                           load_posterior_calf=False,
                                           carve_paratibial=False,
                                           relieve_fibular_head=False,
                                           relieve_distal_tibial_end=False,
                                           relieve_tibial_crest=False,
                                           snugness=0.0, philosophy="PTB"))
    # place a relief at a specific limb vertex
    target = limb.vertices[len(limb.vertices) // 2]
    spec = RectificationSpec(load_patellar_tendon=False, load_posterior_calf=False,
                             carve_paratibial=False, relieve_fibular_head=False,
                             relieve_distal_tibial_end=False, relieve_tibial_crest=False,
                             snugness=0.0, philosophy="PTB",
                             custom_regions=[CustomReliefRegion(point=tuple(target),
                                                                radius_mm=20, depth_mm=6)])
    res = rectify(limb, spec)
    i = len(limb.vertices) // 2
    # the painted relief moves the surface outward there
    assert res.signed_normal_disp[i] > res0.signed_normal_disp[i] + 1.0


def test_spec_clamps_distal_end():
    spec = RectificationSpec(distal_tibial_end_mm=999)
    zones = {z.name: z for z in spec.resolve_zones()}
    assert abs(zones["distal_tibial_end"].signed_magnitude_mm) <= 10.0
