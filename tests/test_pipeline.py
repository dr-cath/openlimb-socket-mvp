import os

from socketcad.pipeline import run_pipeline
from socketcad.spec import RectificationSpec
from socketcad.validate import validate_socket


def test_pipeline_end_to_end(limb, tmp_path):
    spec = RectificationSpec(philosophy="PTB", snugness=0.5)
    pr = run_pipeline(limb, spec, out_dir=str(tmp_path), basename="t", make_heatmap=True)
    # validation passes
    assert pr.validation.passed, pr.validation.to_dict()
    # all expected artifacts written
    for key in ["socket_stl", "rectified_stl", "limb_stl"]:
        assert os.path.exists(pr.exports[key])
    assert os.path.exists(pr.heatmap["png"])
    assert os.path.exists(pr.heatmap["mesh"])


def test_validation_flags_bad_socket(limb):
    import trimesh

    # an open box is not watertight -> should fail
    box = trimesh.creation.box((50, 50, 100))
    box.faces = box.faces[:-2]  # punch a hole
    rep = validate_socket(box)
    assert not rep.passed
    assert not rep.checks["watertight"]
