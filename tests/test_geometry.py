import numpy as np

from socketcad.geometry import (
    cylindrical_coords,
    estimate_frame,
    gaussian_zone_weight,
    laplacian_smooth,
)


def test_frame_is_right_handed_and_sized(limb, frame):
    assert frame.height > 0
    # orthonormal, right-handed rotation
    assert np.allclose(frame.axes @ frame.axes.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(frame.axes), 1.0, atol=1e-6)


def test_proximal_is_larger_cross_section(limb, frame):
    Vc = frame.to_canonical(limb.vertices)
    z = Vc[:, 2]
    H = frame.height
    r = np.hypot(Vc[:, 0], Vc[:, 1])
    distal = r[z < 0.2 * H].mean()
    proximal = r[z > 0.8 * H].mean()
    assert proximal > distal  # knee end is wider than the stump tip


def test_frame_roundtrip(limb, frame):
    pts = limb.vertices[:50]
    back = frame.to_world(frame.to_canonical(pts))
    assert np.allclose(pts, back, atol=1e-6)


def test_cylindrical_radial_unit(limb, frame):
    Vc = frame.to_canonical(limb.vertices)
    _, _, radius, radial = cylindrical_coords(Vc)
    norms = np.linalg.norm(radial, axis=1)
    # radial unit vectors are unit length where radius is non-trivial
    assert np.allclose(norms[radius > 1.0], 1.0, atol=1e-6)
    assert np.all(radial[:, 2] == 0.0)


def test_gaussian_zone_weight_peaks_at_centre():
    z = np.array([50.0])
    theta = np.array([0.0])
    w = gaussian_zone_weight(z, theta, 100.0, 0.5, 0.0, 0.1, 0.3)
    assert np.isclose(w[0], 1.0, atol=1e-6)
    far = gaussian_zone_weight(np.array([95.0]), np.array([np.pi]), 100.0, 0.5, 0.0, 0.1, 0.3)
    assert far[0] < 0.1


def test_laplacian_smooth_preserves_topology(limb):
    out = laplacian_smooth(limb, iterations=2)
    assert len(out.vertices) == len(limb.vertices)
    assert len(out.faces) == len(limb.faces)
