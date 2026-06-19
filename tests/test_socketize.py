import numpy as np

from socketcad.rectify import rectify
from socketcad.socketize import socketize
from socketcad.spec import RectificationSpec


def _socket(limb):
    res = rectify(limb, RectificationSpec(philosophy="PTB"))
    return socketize(res.rectified, res.frame, wall_thickness_mm=4.0)


def test_socket_is_watertight_manifold(limb):
    sr = _socket(limb)
    assert sr.socket.is_watertight
    assert sr.socket.is_winding_consistent
    assert sr.socket.is_volume  # manifold / positive-volume proxy


def test_socket_is_hollow(limb):
    sr = _socket(limb)
    fr = sr.frame
    cavity_centre = fr.to_world(np.array([0.0, 0.0, 0.5 * fr.height]))
    assert not bool(sr.socket.contains([cavity_centre])[0])


def test_socket_wall_thickness_near_target(limb):
    sr = _socket(limb)
    from socketcad.validate import measure_wall_thickness

    th = measure_wall_thickness(sr.positive, sr.outer, sr.frame, sr.trimline_height_mm)
    assert len(th) > 0
    assert abs(np.median(th) - 4.0) < 1.5


def test_socket_dimensions_plausible(limb):
    sr = _socket(limb)
    ext = np.sort(sr.socket.extents)
    assert 60 < ext[-1] < 320       # height
    assert 50 < ext[-2] < 180       # max diameter
