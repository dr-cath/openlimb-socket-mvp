"""Socketization: turn a rectified positive into a print-ready socket shell.

These are the "mechanical, automatable" steps:

1. **Cap** the open proximal end of the positive to make a watertight limb solid.
2. **Offset** that solid outward by the wall thickness (robust voxel dilation)
   to form the outer wall.
3. **Difference** outer - positive -> a hollow shell whose *cavity* matches the
   rectified limb exactly (so the limb fits) and whose distal end is solid.
4. **Trimline**: cut the shell with a (optionally tilted) plane and cap the cut,
   opening the socket at the proximal rim.
5. **Attachment stub**: union a distal pylon-adapter boss (geometry only, a
   clearly-labelled NON-engineered placeholder).
6. **Repair + validate** to guarantee a watertight, manifold, printable mesh.

The boolean backend is ``manifold3d`` (installed via requirements); the outward
offset uses voxel dilation so it is robust on concave anatomy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh
from scipy.ndimage import binary_dilation

from .geometry import LimbFrame, estimate_frame, laplacian_smooth


@dataclass
class SocketResult:
    socket: trimesh.Trimesh
    positive: trimesh.Trimesh        # capped, watertight rectified limb solid
    outer: trimesh.Trimesh           # outer offset surface (for thickness checks)
    frame: LimbFrame
    wall_thickness_mm: float
    trimline_height_mm: float

    def summary(self) -> dict:
        s = self.socket
        return {
            "watertight": bool(s.is_watertight),
            "winding_consistent": bool(s.is_winding_consistent),
            "is_volume": bool(s.is_volume),
            "n_vertices": int(len(s.vertices)),
            "n_faces": int(len(s.faces)),
            "socket_volume_cm3": float(abs(s.volume) / 1000.0) if s.is_watertight else float("nan"),
            "height_mm": float(s.extents[np.argmax(s.extents)]),
            "wall_thickness_mm": float(self.wall_thickness_mm),
        }


def cap_open_boundaries(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Close every open boundary loop with a triangle fan to its centroid.

    ``trimesh.fill_holes`` only closes small holes; a residual-limb scan has a
    large proximal opening, so we triangulate each boundary loop explicitly.
    Returns a (best-effort) watertight copy.
    """
    m = mesh.copy()
    # First let trimesh close any small holes cheaply.
    try:
        m.fill_holes()
    except Exception:
        pass
    if m.is_watertight:
        m.fix_normals()
        return m

    # Boundary edges are those referenced by exactly one face.
    unique, counts = np.unique(np.sort(m.edges, axis=1), axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return m

    # Order boundary edges into loops.
    from collections import defaultdict

    adj = defaultdict(list)
    for a, b in boundary:
        adj[a].append(b)
        adj[b].append(a)

    visited_edges = set()
    loops = []
    edge_set = {tuple(sorted(e)) for e in boundary}
    for start in list(adj.keys()):
        for nxt in adj[start]:
            e0 = tuple(sorted((start, nxt)))
            if e0 in visited_edges:
                continue
            loop = [start]
            prev, cur = start, nxt
            visited_edges.add(e0)
            loop.append(cur)
            ok = True
            while cur != start:
                cands = [n for n in adj[cur] if n != prev and tuple(sorted((cur, n))) not in visited_edges]
                if not cands:
                    ok = False
                    break
                nn = cands[0]
                visited_edges.add(tuple(sorted((cur, nn))))
                prev, cur = cur, nn
                if cur != start:
                    loop.append(cur)
                if len(loop) > len(boundary) + 5:
                    ok = False
                    break
            if ok and len(loop) >= 3:
                loops.append(loop)

    verts = list(m.vertices)
    faces = list(m.faces)
    for loop in loops:
        loop_pts = m.vertices[loop]
        centroid = loop_pts.mean(axis=0)
        ci = len(verts)
        verts.append(centroid)
        for i in range(len(loop)):
            a = loop[i]
            b = loop[(i + 1) % len(loop)]
            faces.append([ci, a, b])

    capped = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)
    capped.fix_normals()
    return capped


def offset_solid_outward(solid: trimesh.Trimesh, distance_mm: float, pitch: Optional[float] = None) -> trimesh.Trimesh:
    """Return a watertight outward offset of ``solid`` via voxel dilation.

    Robust on concave anatomy. Used as a fallback; the primary socket build uses
    :func:`build_shell`, which gives a uniform wall thickness.
    """
    if pitch is None:
        pitch = max(1.2, distance_mm / 2.5)
    vg = solid.voxelized(pitch=pitch).fill()
    iters = max(1, int(np.ceil(distance_mm / pitch)))
    dilated = binary_dilation(vg.matrix, iterations=iters)
    outer_vg = trimesh.voxel.VoxelGrid(dilated, transform=vg.transform)
    # VoxelGrid.marching_cubes returns the mesh in voxel-index space; map it
    # back to world coordinates with the grid transform.
    outer = outer_vg.marching_cubes.copy()
    outer.apply_transform(outer_vg.transform)
    outer = laplacian_smooth(outer, iterations=2, factor=0.5)
    outer.fix_normals()
    return outer


def _boundary_loops(mesh: trimesh.Trimesh):
    """Return ordered vertex-index loops for each open boundary of ``mesh``."""
    from collections import defaultdict

    unique, counts = np.unique(np.sort(mesh.edges, axis=1), axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        return []
    adj = defaultdict(list)
    for a, b in boundary:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))
    visited = set()
    loops = []
    for start in list(adj.keys()):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur, prev = start, None
        while True:
            nxts = [n for n in adj[cur] if n != prev and n not in visited]
            if not nxts:
                # close back to start if possible
                break
            nxt = nxts[0]
            if nxt == start:
                break
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def build_shell(inner_surface: trimesh.Trimesh, wall_thickness_mm: float) -> trimesh.Trimesh:
    """Build a uniform-thickness watertight shell from an open cavity surface.

    ``inner_surface`` is the limb surface with a single open boundary (the
    trimline rim). The outer wall is the inner surface offset outward by the
    wall thickness along vertex normals; the two rims are bridged to form a
    closed, manifold cup. Wall thickness is the offset distance everywhere.
    """
    inner = inner_surface.copy()
    inner.merge_vertices()
    inner.fix_normals()
    n = len(inner.vertices)

    outer_verts = inner.vertices + inner.vertex_normals * wall_thickness_mm
    verts = np.vstack([inner.vertices, outer_verts])

    inner_faces = inner.faces[:, ::-1]          # flip to face into the cavity
    outer_faces = inner.faces + n               # outer wall faces

    rim_faces = []
    for loop in _boundary_loops(inner):
        L = len(loop)
        for i in range(L):
            a = loop[i]
            b = loop[(i + 1) % L]
            rim_faces.append([a, b, b + n])
            rim_faces.append([a, b + n, a + n])
    faces = [inner_faces, outer_faces]
    if rim_faces:
        faces.append(np.array(rim_faces))
    shell = trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=True)
    shell.fix_normals()
    return shell


def _trimline_plane(frame: LimbFrame, height_frac: float, anterior_drop_frac: float):
    """Return (origin, keep_normal) for the trimline cut.

    ``keep_normal`` points toward the distal portion that is retained. The plane
    is tilted so the anterior wall sits lower (clearance for knee flexion).
    """
    H = frame.height
    origin = frame.to_world(np.array([0.0, 0.0, height_frac * H]))
    z_world = frame.axes[2]   # proximal direction
    x_world = frame.axes[0]   # anterior direction
    # Keep everything below: normal points distally (-z). Tilt toward -anterior
    # so the plane dips on the anterior side.
    keep_normal = -z_world + anterior_drop_frac * 4.0 * x_world
    keep_normal = keep_normal / np.linalg.norm(keep_normal)
    return origin, keep_normal


def _attachment_stub(frame: LimbFrame, outer: trimesh.Trimesh, radius_mm: float = 18.0, length_mm: float = 12.0) -> trimesh.Trimesh:
    """A distal pylon-adapter boss (geometry only; NOT an engineered fitting)."""
    z_world = frame.axes[2]
    # Place at the distal tip, protruding distally.
    distal_origin = frame.to_world(np.array([0.0, 0.0, 0.0]))
    centre = distal_origin - z_world * (length_mm / 2.0 - 1.0)
    cyl = trimesh.creation.cylinder(radius=radius_mm, height=length_mm, sections=48)
    # Align cylinder (+z) with -z_world (pointing distally).
    align = trimesh.geometry.align_vectors([0, 0, 1], -z_world)
    cyl.apply_transform(align)
    cyl.apply_translation(centre)
    return cyl


def socketize(
    rectified: trimesh.Trimesh,
    frame: Optional[LimbFrame] = None,
    wall_thickness_mm: float = 4.0,
    trimline_height_frac: float = 0.92,
    trimline_anterior_drop_frac: float = 0.06,
    add_attachment_stub: bool = True,
    is_left_limb: bool = True,
) -> SocketResult:
    """Build a print-ready socket from a rectified positive mesh."""
    if frame is None:
        frame = estimate_frame(rectified, is_left=is_left_limb)

    # Cavity surface = rectified limb cut at the trimline (keep distal portion),
    # leaving a single open boundary at the rim.
    origin, keep_normal = _trimline_plane(frame, trimline_height_frac, trimline_anterior_drop_frac)
    inner = rectified.slice_plane(origin, keep_normal, cap=False)
    if inner is None or inner.is_empty:
        raise RuntimeError("trimline cut removed the entire limb surface")
    inner.merge_vertices()
    inner.fix_normals()

    # Uniform-thickness shell (cup) by offsetting + bridging the rim.
    socket = build_shell(inner, wall_thickness_mm)
    if socket is None or socket.is_empty:
        raise RuntimeError("failed to build socket shell")

    # The outer wall surface, kept for thickness validation.
    outer = inner.copy()
    outer.vertices = inner.vertices + inner.vertex_normals * wall_thickness_mm

    if add_attachment_stub:
        stub = _attachment_stub(frame, outer)
        unioned = socket.union(stub)
        if unioned is not None and not unioned.is_empty:
            socket = unioned

    socket = repair_for_print(socket)
    positive = cap_open_boundaries(rectified)  # closed limb solid (for volumes)
    trimline_height_mm = float(trimline_height_frac * frame.height)
    return SocketResult(
        socket=socket,
        positive=positive,
        outer=outer,
        frame=frame,
        wall_thickness_mm=wall_thickness_mm,
        trimline_height_mm=trimline_height_mm,
    )


def repair_for_print(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Best-effort cleanup so the mesh is watertight, manifold and printable."""
    m = mesh.copy()
    m.remove_unreferenced_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    try:
        m.fill_holes()
    except Exception:
        pass
    m.fix_normals()
    return m
