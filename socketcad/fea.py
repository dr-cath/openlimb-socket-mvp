"""Self-contained linear-elastic FEM for a first-order interface-pressure estimate.

This actually *runs* a finite-element analysis in-process (no external solver):
the residual limb is tetrahedralised, bone elements are stiffened to act as a
near-rigid core, the socket's rectification is imposed as an inward displacement
of the skin, and we solve linear elasticity (4-node constant-strain tetrahedra,
scipy sparse) for the reaction pressure at the skin -- an estimate of the
socket-limb **interface pressure**.

HONEST SCOPE -- this is a *linear, small-strain* model. It does NOT include
hyperelasticity, frictional contact mechanics, a liner, large deformation, or
tissue layering, and the material constants are literature-level placeholders.
It is a genuine continuum FEM (far beyond the Winkler proxy) but is NOT a
validated clinical prediction. The rigorous path is a nonlinear contact solver
(FEBio/Abaqus) fed by :func:`socketcad.simulator.export_febio` -- see
docs/LIMITATIONS.md.

Material defaults (placeholders, cited order-of-magnitude):
- bulk soft tissue: E ~ 0.20 MPa, nu 0.45 (near-incompressible, linear proxy);
- bone: E ~ 100 MPa (>=500x soft -> effectively rigid core), nu 0.30.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve

E_SOFT_MPA = 0.20
NU_SOFT = 0.40   # reduced from ~0.49 to limit volumetric locking in linear tets
E_BONE_MPA = 100.0
NU_BONE = 0.30


@dataclass
class FEAResult:
    nodes: np.ndarray              # (N,3) mm
    elems: np.ndarray             # (M,4) tetra
    is_bone: np.ndarray           # (M,) bool
    skin_nodes: np.ndarray        # indices of skin-surface nodes
    pressure_kpa: np.ndarray      # (len(skin_nodes),) interface pressure (smoothed)
    skin_points: np.ndarray       # (len(skin_nodes),3) coords
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        p = self.pressure_kpa
        loaded = p[p > 1.0]
        return {
            "solver": "linear-elastic FEM (4-node tetra, scipy) — small-strain proxy",
            "n_nodes": int(len(self.nodes)),
            "n_tets": int(len(self.elems)),
            "n_bone_tets": int(self.is_bone.sum()),
            # robust headline statistics (peak is singularity-limited, reported separately)
            "median_contact_kPa": round(float(np.median(loaded)) if loaded.size else 0.0, 1),
            "p90_contact_kPa": round(float(np.percentile(loaded, 90)) if loaded.size else 0.0, 1),
            "p99_kPa": round(float(np.percentile(p, 99)), 1),
            "raw_peak_kPa_singularity": round(float(p.max()), 1),
            "loaded_area_nodes_pct": round(100.0 * float(np.mean(p > 1.0)), 1),
            **self.meta,
        }


# --------------------------------------------------------------------------
def tetmesh(skin_solid: trimesh.Trimesh, target_faces: int = 2500,
            maxvolume: Optional[float] = None):
    """Tetrahedralise a watertight limb solid; returns (nodes, elems)."""
    import tetgen

    surf = skin_solid
    if target_faces and len(surf.faces) > target_faces:
        try:
            import fast_simplification
            v, f = fast_simplification.simplify(surf.vertices, surf.faces,
                                                target_count=target_faces)
            surf = trimesh.Trimesh(v, f, process=True)
        except Exception:
            surf = skin_solid
    surf.fill_holes(); surf.fix_normals()
    if maxvolume is None:
        maxvolume = float(surf.volume) / 8000.0  # ~8k tets target
    tg = tetgen.TetGen(np.asarray(surf.vertices), np.asarray(surf.faces))
    tg.tetrahedralize(order=1, mindihedral=20, minratio=1.5, maxvolume=maxvolume)
    return np.asarray(tg.node, float), np.asarray(tg.elem, int)


def classify_bone(nodes, elems, bone: trimesh.Trimesh, pitch: float = 3.0) -> np.ndarray:
    """Mark tets whose centroid lies inside the bone (voxel occupancy)."""
    b = bone.copy(); b.merge_vertices()
    vg = b.voxelized(pitch=pitch).fill()
    c = nodes[elems].mean(axis=1)
    try:
        return np.asarray(vg.is_filled(c), bool)
    except Exception:
        idx = vg.points_to_indices(c)
        shp = np.array(vg.matrix.shape)
        ok = np.all((idx >= 0) & (idx < shp), axis=1)
        out = np.zeros(len(c), bool)
        ii = idx[ok]
        out[ok] = vg.matrix[ii[:, 0], ii[:, 1], ii[:, 2]]
        return out


def _D(E: float, nu: float) -> np.ndarray:
    """Isotropic linear-elastic constitutive matrix (6x6, Voigt)."""
    c = E / ((1 + nu) * (1 - 2 * nu))
    D = np.zeros((6, 6))
    D[:3, :3] = c * nu
    D[0, 0] = D[1, 1] = D[2, 2] = c * (1 - nu)
    D[3, 3] = D[4, 4] = D[5, 5] = c * (1 - 2 * nu) / 2
    return D


def _tet_B_V(coords: np.ndarray):
    """Strain-displacement B (6x12) and volume V for one linear tet."""
    M = np.hstack([np.ones((4, 1)), coords])      # 4x4
    vol = np.linalg.det(M) / 6.0
    grads = np.linalg.inv(M)[1:, :]               # 3x4 (dN/dx,dy,dz)
    B = np.zeros((6, 12))
    for i in range(4):
        bx, by, bz = grads[:, i]
        B[0, 3 * i] = bx; B[1, 3 * i + 1] = by; B[2, 3 * i + 2] = bz
        B[3, 3 * i] = by; B[3, 3 * i + 1] = bx
        B[4, 3 * i + 1] = bz; B[4, 3 * i + 2] = by
        B[5, 3 * i] = bz; B[5, 3 * i + 2] = bx
    return B, abs(vol)


def assemble_K(nodes, elems, is_bone):
    """Assemble the global sparse stiffness matrix (3N x 3N)."""
    N = len(nodes)
    D_soft, D_bone = _D(E_SOFT_MPA, NU_SOFT), _D(E_BONE_MPA, NU_BONE)
    rows, cols, vals = [], [], []
    for e, tet in enumerate(elems):
        B, V = _tet_B_V(nodes[tet])
        Ke = (B.T @ (D_bone if is_bone[e] else D_soft) @ B) * V
        dofs = np.array([3 * n + d for n in tet for d in range(3)])
        rows.append(np.repeat(dofs, 12)); cols.append(np.tile(dofs, 12))
        vals.append(Ke.ravel())
    K = sparse.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(3 * N, 3 * N)).tocsr()
    return K


def _boundary_nodes(elems, n_nodes):
    """Surface nodes of the tet mesh (nodes on faces used by one tet)."""
    faces = elems[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]].reshape(-1, 3)
    key = np.sort(faces, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    surf = faces[idx[cnt == 1]]
    return np.unique(surf), surf


def run_limb_fea(skin_solid: trimesh.Trimesh, bone: trimesh.Trimesh,
                 interference_fn, target_faces: int = 2200) -> FEAResult:
    """Solve interface pressure for a socket's rectification.

    ``interference_fn(points) -> mm`` returns the inward push (>=0) the socket
    imposes on the skin at each surface point (the rectification interference).
    """
    nodes, elems = tetmesh(skin_solid, target_faces=target_faces)
    is_bone = classify_bone(nodes, elems, bone)
    surf_nodes, surf_faces = _boundary_nodes(elems, len(nodes))

    # Outward normals at surface nodes (area-weighted from boundary faces).
    fn = np.cross(nodes[surf_faces[:, 1]] - nodes[surf_faces[:, 0]],
                  nodes[surf_faces[:, 2]] - nodes[surf_faces[:, 0]])
    node_normal = np.zeros((len(nodes), 3))
    for k in range(3):
        np.add.at(node_normal, surf_faces[:, k], fn)
    nn = np.linalg.norm(node_normal, axis=1)
    node_normal[nn > 0] /= nn[nn > 0, None]

    bone_nodes = np.unique(elems[is_bone]) if is_bone.any() else np.array([], int)
    # Skin = surface nodes that are not part of the bone core.
    skin = np.setdiff1d(surf_nodes, bone_nodes)

    interference = np.clip(interference_fn(nodes[skin]), 0.0, None)  # mm inward
    u_skin = -node_normal[skin] * interference[:, None]              # inward disp

    N = len(nodes)
    K = assemble_K(nodes, elems, is_bone)

    # Dirichlet: bone nodes fixed (0), skin nodes prescribed (u_skin).
    presc = np.zeros(3 * N)
    fixed = np.zeros(3 * N, bool)
    for n in bone_nodes:
        fixed[3 * n:3 * n + 3] = True
    for j, n in enumerate(skin):
        d = slice(3 * n, 3 * n + 3)
        fixed[d] = True
        presc[d] = u_skin[j]
    free = ~fixed

    u = presc.copy()
    rhs = -(K[free][:, fixed] @ presc[fixed])
    u[free] = spsolve(K[free][:, free].tocsc(), rhs)

    # Reaction at skin nodes -> interface pressure (normal reaction / nodal area).
    R = K @ u
    area = np.zeros(len(nodes))
    fa = 0.5 * np.linalg.norm(fn, axis=1)
    for k in range(3):
        np.add.at(area, surf_faces[:, k], fa / 3.0)
    pressure = np.zeros(len(skin))
    for j, n in enumerate(skin):
        rn = R[3 * n:3 * n + 3] @ (-node_normal[n])   # inward reaction (N)
        a = area[n] if area[n] > 1e-6 else 1e-6
        pressure[j] = max(0.0, rn / a) * 1000.0        # MPa -> kPa

    # Linear-FEM stress singularities occur where bone is ~0 mm under skin
    # (e.g. tibial crest). Winsorise to the 99th percentile and smooth over the
    # surface (k-NN average) so the reported field is a robust spatial estimate.
    from scipy.spatial import cKDTree

    cap = np.percentile(pressure, 99)
    pressure = np.clip(pressure, 0.0, cap)
    pts = nodes[skin]
    if len(pts) > 8:
        tree = cKDTree(pts)
        _, nbr = tree.query(pts, k=8)
        pressure = pressure[nbr].mean(axis=1)

    return FEAResult(nodes=nodes, elems=elems, is_bone=is_bone, skin_nodes=skin,
                     pressure_kpa=pressure, skin_points=nodes[skin],
                     meta={"E_soft_MPa": E_SOFT_MPA, "E_bone_MPa": E_BONE_MPA,
                           "nu_soft": NU_SOFT})


def load_openlimb_fea_geometry(data_dir: Optional[str] = None):
    """Find OpenLimbTT skin + bone full-size surfaces for FEA.

    Returns ``(skin_solid, bone)`` (skin capped watertight) or ``None`` if the
    files are not present (they are fetched by ``scripts/fetch_openlimb.py``).
    """
    import glob
    import os
    from .limb_source import DEFAULT_DATA_DIR
    from .socketize import cap_open_boundaries

    data_dir = data_dir or DEFAULT_DATA_DIR
    skin = glob.glob(os.path.join(data_dir, "**", "*Skin*Full*.stl"), recursive=True)
    bone = glob.glob(os.path.join(data_dir, "**", "*Bone*Full*.stl"), recursive=True)
    if not skin or not bone:
        return None
    s = trimesh.load(sorted(skin)[0]); s.merge_vertices()
    b = trimesh.load(sorted(bone)[0])
    return cap_open_boundaries(s), b


def interference_from_rectify(rectify_result):
    """Return an interference_fn sampling the socket's inward push by nearest
    rectified-limb vertex (interference = inward part of signed normal disp)."""
    from scipy.spatial import cKDTree

    pts = np.asarray(rectify_result.limb.vertices)
    inward = np.clip(-rectify_result.signed_normal_disp, 0.0, None)
    tree = cKDTree(pts)

    def fn(query):
        _, idx = tree.query(np.asarray(query))
        return inward[idx]

    return fn
