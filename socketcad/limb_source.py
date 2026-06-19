"""Limb input: OpenLimbTT statistical-shape sampling, a synthetic fallback,
arbitrary-mesh loading, and heuristic anatomical landmarks.

OpenLimbTT (Sunderland et al., 2025, CC-BY-SA-4.0) ships a *size-normalised*
mean residual-limb mesh plus PCA modes. A synthetic limb is reconstructed as::

    vertices = mean + Σ_i  mode_i * eigenvalue_i        (size-normalised)
    vertices *= intact_tibia_length                      (scale to mm)

The mean mesh is **open at the proximal end** (it is a scan-like skin surface)
and vertex order is significant -- it must align element-wise with the modes,
so meshes are always loaded with ``process=False`` (no vertex merging).

We deliberately do **not** vendor the OpenLimbTT data (CC-BY-SA-4.0, and large);
``scripts/fetch_openlimb.py`` downloads it on demand. When the data is absent
the engine falls back to :func:`generate_synthetic_limb`, a clearly-labelled
procedural limb, so the whole pipeline still runs hermetically (e.g. in CI).

See ``docs/SOURCES.md`` for provenance, licensing and attribution.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh

from .geometry import LimbFrame, estimate_frame

# Average intact (unamputated) tibia length of the OpenLimbTT training set, in
# mm; used to scale the size-normalised model. Source: OpenLimbTT v2025-07
# generator script header (range 313.05-466.34 mm, mean 378.52 mm).
DEFAULT_TIBIA_LENGTH_MM = 378.52

# Default location to look for downloaded OpenLimbTT assets.
DEFAULT_DATA_DIR = os.environ.get(
    "OPENLIMB_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "openlimb"),
)


# --------------------------------------------------------------------------
# OpenLimbTT statistical shape model
# --------------------------------------------------------------------------
@dataclass
class OpenLimbTTModel:
    """Loaded OpenLimbTT mean mesh + PCA modes.

    ``modes`` is stored as ``(n_modes, 3 * n_vertices)`` regardless of the
    on-disk layout (the 2022 release stores ``Modes.npy`` transposed relative
    to the 2023/2025 ``Components*.npy``; both are normalised here).
    """

    mean_vertices: np.ndarray  # (N, 3) size-normalised
    faces: np.ndarray  # (M, 3)
    modes: np.ndarray  # (k, 3N)
    eigen_ranges: Optional[np.ndarray] = None  # (k, 2) min/max from training
    version: str = "unknown"

    @property
    def n_modes(self) -> int:
        return self.modes.shape[0]

    @property
    def n_vertices(self) -> int:
        return self.mean_vertices.shape[0]

    def sample(
        self,
        components: Optional[np.ndarray] = None,
        scale: float = DEFAULT_TIBIA_LENGTH_MM,
    ) -> trimesh.Trimesh:
        """Return a limb mesh for the given mode eigenvalues (``components``).

        ``components=None`` returns the mean shape. Missing/extra components
        are zero-padded/truncated to ``n_modes``.
        """
        verts = self.mean_vertices.astype(float).copy()
        if components is not None:
            c = np.zeros(self.n_modes)
            comp = np.asarray(components, dtype=float).ravel()
            c[: min(len(comp), self.n_modes)] = comp[: self.n_modes]
            delta = (c @ self.modes).reshape(self.n_vertices, 3)
            verts = verts + delta
        verts = verts * float(scale)
        # mean/faces are already welded at load time, so the mesh is connected.
        return trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

    def random_components(self, rng: np.random.Generator, n_active: int = 8) -> np.ndarray:
        """Draw eigenvalues uniformly within the training-data ranges.

        Only the first ``n_active`` modes are perturbed; later OpenLimbTT modes
        are noted by its authors as possibly biased by the small training set.
        """
        c = np.zeros(self.n_modes)
        k = min(n_active, self.n_modes)
        if self.eigen_ranges is not None:
            lo, hi = self.eigen_ranges[:k, 0], self.eigen_ranges[:k, 1]
        else:  # fall back to a modest symmetric range
            lo, hi = -np.ones(k), np.ones(k)
        c[:k] = rng.uniform(lo, hi)
        return c


# Eigenvalue ranges for the first modes, transcribed from the OpenLimbTT
# v2025-07 "Generate Random Synthetic Model.py" generator (min/max observed in
# the training data). Used to draw plausible random limbs.
_V2025_EIGEN_RANGES = np.array(
    [
        [-17.04, 24.25], [-10.38, 16.42], [-8.41, 10.04], [-6.42, 11.66],
        [-4.73, 5.21], [-4 / 3, 4.38], [-4.01, 4.01], [-2.55, 2.68],
        [-2.26, 2.5], [-1.97, 2.49], [-1.54, 2.2], [-0.96, 2.48],
    ]
)


def _load_modes(path: str) -> np.ndarray:
    """Load a modes/components array as ``(n_modes, 3N)`` float64."""
    arr = np.load(path).astype(np.float64)
    # 2022 'Modes.npy' is (3N, k); 2023/2025 'Components*.npy' is (k, 3N).
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    return arr


def find_openlimb_assets(data_dir: str = DEFAULT_DATA_DIR) -> Optional[dict]:
    """Locate a usable (mean mesh, modes) pair under ``data_dir``.

    Searches recursively so that either a flat download or a cloned
    ``OpenLimbTT/version-*`` tree works. Returns ``None`` if nothing matches.
    """
    if not os.path.isdir(data_dir):
        return None
    mesh_candidates = []
    for pat in ("Mean-Limb.stl", "Mean_Limb_Shape.stl", "*ean*imb*.stl"):
        mesh_candidates += glob.glob(os.path.join(data_dir, "**", pat), recursive=True)
    mode_candidates = (
        glob.glob(os.path.join(data_dir, "**", "Components*.npy"), recursive=True)
        + glob.glob(os.path.join(data_dir, "**", "Modes*.npy"), recursive=True)
    )
    if not mesh_candidates or not mode_candidates:
        return None

    # Prefer matching mean+modes that share a directory and vertex count.
    for mesh_path in sorted(mesh_candidates):
        m = trimesh.load(mesh_path, process=False)
        n3 = m.vertices.size
        same_dir = [p for p in mode_candidates if os.path.dirname(p) == os.path.dirname(mesh_path)]
        for mode_path in sorted(same_dir) + sorted(mode_candidates):
            modes = _load_modes(mode_path)
            if modes.shape[1] == n3:
                return {"mesh": mesh_path, "modes": mode_path}
    return None


def _weld_soup(vertices: np.ndarray, faces: np.ndarray, modes: np.ndarray, decimals: int = 6):
    """Weld a triangle-soup mesh (and its per-vertex modes) into a connected mesh.

    OpenLimbTT meshes are stored as STL triangle soups: every triangle has its
    own 3 vertices, so ``n_vertices == 3 * n_faces`` and nothing is topologically
    connected. PCA modes are aligned element-wise to this soup ordering. We map
    coincident soup vertices to a single unique vertex (taking a representative),
    remap the faces, and reduce the modes to the unique vertices so that the
    mode math stays exactly aligned but the resulting mesh is a true manifold.
    """
    rounded = np.round(vertices, decimals)
    _, inverse = np.unique(rounded, axis=0, return_inverse=True)
    labels = inverse.ravel()
    # Map each unique label -> its first-occurrence (representative) soup index.
    n_unique = int(labels.max()) + 1
    rep = np.empty(n_unique, dtype=np.int64)
    seen = np.zeros(n_unique, dtype=bool)
    for soup_i, lab in enumerate(labels):
        if not seen[lab]:
            rep[lab] = soup_i
            seen[lab] = True
    mean_unique = vertices[rep]
    faces_unique = labels[faces]
    k = modes.shape[0]
    modes3 = modes.reshape(k, vertices.shape[0], 3)
    modes_unique = modes3[:, rep, :].reshape(k, 3 * n_unique)
    return mean_unique, faces_unique, modes_unique


def load_openlimb_model(data_dir: str = DEFAULT_DATA_DIR) -> Optional[OpenLimbTTModel]:
    """Load OpenLimbTT from ``data_dir`` if present, else ``None``."""
    assets = find_openlimb_assets(data_dir)
    if assets is None:
        return None
    mesh = trimesh.load(assets["mesh"], process=False)
    modes = _load_modes(assets["modes"])
    mean_v, faces, modes = _weld_soup(
        np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces), modes
    )
    k = modes.shape[0]
    ranges = _V2025_EIGEN_RANGES[: min(k, len(_V2025_EIGEN_RANGES))]
    if len(ranges) < k:  # pad unknown ranges conservatively
        pad = np.tile([[-1.0, 1.0]], (k - len(ranges), 1))
        ranges = np.vstack([ranges, pad])
    version = os.path.basename(os.path.dirname(assets["mesh"]))
    return OpenLimbTTModel(
        mean_vertices=mean_v,
        faces=faces,
        modes=modes,
        eigen_ranges=ranges,
        version=version,
    )


# --------------------------------------------------------------------------
# Synthetic fallback limb (hermetic; clearly labelled as NOT real anatomy)
# --------------------------------------------------------------------------
def generate_synthetic_limb(
    length_mm: float = 150.0,
    proximal_radius_mm: float = 45.0,
    distal_radius_mm: float = 33.0,
    calf_bulge_mm: float = 13.0,
    n_theta: int = 96,
    n_height: int = 80,
    seed: Optional[int] = None,
) -> trimesh.Trimesh:
    """Procedural residual-limb-like mesh: tapered, rounded distal tip,
    posterior calf bulge, slight anterior flattening (tibial-crest side).

    This is a deliberately simple stand-in used when OpenLimbTT data is not
    available. It is NOT anatomically derived. Output matches the real model's
    topology contract: closed distal end, **open** proximal end.
    """
    rng = np.random.default_rng(seed)
    theta_post = np.pi  # posterior calf placed at +/-180 deg (anterior = 0)

    hs = np.linspace(0.0, 1.0, n_height)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)

    # Random but reproducible mild asymmetry so samples differ.
    jitter = 1.0 + 0.05 * rng.standard_normal()

    rings = []
    for h in hs[1:]:  # ring 0 is the apex point, added separately
        base = distal_radius_mm + (proximal_radius_mm - distal_radius_mm) * h
        bulge = calf_bulge_mm * np.sin(np.pi * h) * 0.5 * (
            1.0 + np.cos(thetas - theta_post)
        )
        crest = 0.18 * base * np.exp(-((thetas) ** 2) / (2 * 0.35 ** 2))  # anterior dip
        r = (base + bulge - crest) * jitter
        # Round the distal tip hemispherically over the first 18% of length.
        h_round = 0.18
        if h < h_round:
            r = r * np.sqrt(max(1e-6, 1.0 - (1.0 - h / h_round) ** 2))
        z = length_mm * h
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        rings.append(np.column_stack([x, y, np.full_like(x, z)]))

    verts = [np.array([[0.0, 0.0, 0.0]])]  # apex (distal tip)
    verts += rings
    V = np.vstack(verts)

    faces = []
    # Fan from apex (index 0) to first ring.
    first = 1
    for j in range(n_theta):
        a = first + j
        b = first + (j + 1) % n_theta
        faces.append([0, b, a])
    # Quad strips between successive rings.
    n_rings = len(rings)
    for i in range(n_rings - 1):
        base_a = 1 + i * n_theta
        base_b = 1 + (i + 1) * n_theta
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            v00 = base_a + j
            v01 = base_a + j1
            v10 = base_b + j
            v11 = base_b + j1
            faces.append([v00, v01, v11])
            faces.append([v00, v11, v10])
    mesh = trimesh.Trimesh(vertices=V, faces=np.array(faces), process=False)
    mesh.fix_normals()
    return mesh


# --------------------------------------------------------------------------
# Unified entry points
# --------------------------------------------------------------------------
def sample_limb(
    shape: str = "mean",
    components: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    scale: float = DEFAULT_TIBIA_LENGTH_MM,
    data_dir: str = DEFAULT_DATA_DIR,
):
    """Sample a residual-limb mesh.

    Parameters
    ----------
    shape : {"mean", "random", "components", "synthetic"}
        ``mean``/``random``/``components`` use OpenLimbTT if available and
        otherwise fall back to the synthetic generator. ``synthetic`` forces
        the procedural limb.
    components : array-like, optional
        Mode eigenvalues for ``shape="components"``.

    Returns
    -------
    (mesh, info) : (trimesh.Trimesh, dict)
        ``info`` records the source actually used and parameters, for logging
        and provenance.
    """
    rng = np.random.default_rng(seed)
    if shape != "synthetic":
        model = load_openlimb_model(data_dir)
        if model is not None:
            if shape == "mean":
                comp = None
            elif shape == "random":
                comp = model.random_components(rng)
            elif shape == "components":
                comp = components
            else:
                raise ValueError(f"unknown shape '{shape}'")
            mesh = model.sample(components=comp, scale=scale)
            info = {
                "source": "openlimb-tt",
                "version": model.version,
                "shape": shape,
                "components": None if comp is None else np.asarray(comp).tolist(),
                "scale_mm": scale,
            }
            return mesh, info

    # Synthetic fallback.
    mesh = generate_synthetic_limb(seed=int(rng.integers(0, 2**31 - 1)))
    info = {
        "source": "synthetic-fallback",
        "shape": "synthetic",
        "note": "OpenLimbTT data not found; using procedural limb (NOT anatomy).",
    }
    return mesh, info


def load_limb_mesh(path: str):
    """Load an arbitrary user-supplied limb mesh (STL/PLY/OBJ).

    Vertices are *not* merged (``process=False``) to keep behaviour consistent
    with the statistical model loader.
    """
    mesh = trimesh.load(path, process=False, force="mesh")
    info = {"source": "uploaded", "path": os.path.basename(path)}
    return mesh, info


# --------------------------------------------------------------------------
# Heuristic landmarks
# --------------------------------------------------------------------------
@dataclass
class Landmarks:
    """Estimated anatomical landmark points, in **world** coordinates."""

    frame: LimbFrame
    points: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: np.asarray(v).tolist() for k, v in self.points.items()}


# Nominal (height-fraction, angle-from-anterior) locations of key structures.
# Angles assume anterior = 0; medial/lateral side handedness is governed by
# LimbFrame.is_left. Values are geometric defaults for the demo, not measured.
_LANDMARK_TARGETS = {
    "patellar_tendon": (0.82, 0.0),          # anterior, high
    "tibial_crest": (0.5, 0.0),              # anterior ridge, mid
    "tibial_end_distal": (0.12, 0.0),        # anterior-distal tip
    "fibular_head": (0.80, +np.pi / 2),      # lateral, high
    "posterior_calf": (0.55, np.pi),         # posterior bulge, mid
}


def estimate_landmarks(mesh: trimesh.Trimesh, frame: Optional[LimbFrame] = None) -> Landmarks:
    """Estimate landmark surface points by nearest (height, angle) matching."""
    from .geometry import angular_distance, cylindrical_coords

    if frame is None:
        frame = estimate_frame(mesh)
    Vc = frame.to_canonical(mesh.vertices)
    z, theta, radius, _ = cylindrical_coords(Vc)
    H = frame.height

    pts = {}
    side = 1.0 if frame.is_left else -1.0
    for name, (hf, ang) in _LANDMARK_TARGETS.items():
        target_ang = ang * side if name == "fibular_head" else ang
        dh = (z / H) - hf
        dth = angular_distance(theta, target_ang)
        # Prefer outer-surface vertices near the target (height, angle).
        cost = (dh / 0.08) ** 2 + (dth / 0.4) ** 2 - 0.001 * radius
        idx = int(np.argmin(cost))
        pts[name] = frame.to_world(Vc[idx])
    return Landmarks(frame=frame, points=pts)
