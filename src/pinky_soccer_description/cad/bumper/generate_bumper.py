#!/usr/bin/env python3
"""
Generate a bumper from a parameter file.

    python3 generate_bumper.py bumper_v1.yaml

Steps
  1. Turn the parameters into boxes/cones/arcs to add and subtract (base_link frame, mm).
  2. Build the solid in Blender (blender_csg.py) and export it.
  3. Check it against the official robot meshes: fitted, and while snapping on with the
     side walls bending (cantilever) at 0.1 mm steps.
  4. Compute mass, centre of mass and inertia.
  5. Write
       print/<name>_print.stl                       mm, in the print orientation
       ../../meshes/visual/<name>.stl               m, bumper_link frame (URDF visual)
       ../../meshes/collision/<name>_collision.stl  m, bumper_link frame (URDF collision)
       ../../config/<name>_params.yaml              URDF parameters for urdf/bumper.urdf.xacro
       print/<name>_report.txt

Needs: Blender (>= 4.2, `blender` on PATH or --blender), Python packages numpy, pyyaml,
trimesh (pip, same as pinky_mujoco) and a sourced workspace (for xacro and the meshes).
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent.parent                    # src/pinky_soccer_description
BASE_LINK_HEIGHT = 28.0                     # mm, base_footprint -> base_link


# ----------------------------------------------------------------------------- geometry
def box(x, y, z):
    return {'type': 'box', 'x': sorted(x), 'y': sorted(y), 'z': sorted(z)}


def cone_x(x, r0, r1, y, z):
    return {'type': 'cone', 'x': list(x), 'r0': r0, 'r1': r1, 'y': y, 'z': z}


def arc(cx, cy, r_in, r_out, phi0, phi1, z, group=None):
    return {'type': 'arc', 'cx': cx, 'cy': cy, 'r_in': r_in, 'r_out': r_out,
            'phi0': phi0, 'phi1': phi1, 'z': sorted(z), 'group': group}


def hook_numbers(h):
    """Return the inner/outer hook radius and the hook z range."""
    r_in = h['standoff_radius'] + h['radial_clearance']
    r_out = r_in + h['hook_thickness']
    z = (h['plate_gap_bottom_z'] + h['height_clearance'] / 2,
         h['plate_gap_top_z'] - h['height_clearance'] / 2)
    # the hook tip reaches y = standoff_y - r_in cos(wrap); it must clear the standoff
    return r_in, r_out, z


def poly(pts, z, group=None):
    """Convex prism; pts counter-clockwise in xy."""
    return {'type': 'poly', 'pts': [list(map(float, q)) for q in pts], 'z': sorted(z),
            'group': group}


def corner_piece(corner, a, b, r, z, n=16, group=None):
    """
    Region between a 90 deg corner and its fillet arc (r), as a polygon prism.

    a, b: unit vectors along the two faces leaving the corner. Subtract the piece to round
    a convex corner of the solid, add it to fill a concave corner.
    """
    P, a, b = np.array(corner, float), np.array(a, float), np.array(b, float)
    centre = P + r * (a + b)
    start, end = -b, -a                              # directions from the arc centre
    t0, t1 = math.atan2(start[1], start[0]), math.atan2(end[1], end[0])
    sweep = (t1 - t0 + math.pi) % (2 * math.pi) - math.pi   # shortest way, +-90 deg
    arc_pts = [centre + r * np.array([math.cos(t0 + sweep * i / n), math.sin(t0 + sweep * i / n)])
               for i in range(n + 1)]
    e = 0.05  # overshoot past both faces so the boolean leaves no zero-thickness sliver
    pts = [P - e * (a + b), P + r * a - e * b] + arc_pts + [P + r * b - e * a]
    area = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))
    if area < 0:
        pts = pts[::-1]
    return poly(pts, z, group)


def build_spec(p):
    fp, sw, h, us = p['front_plate'], p['side_walls'], p['hooks'], p['ultrasonic_holes']
    fil = p.get('fillets', {})
    front_x = fp['back_x'] + fp['thickness']
    W = fp['half_width']
    r_in, r_out, hz = hook_numbers(h)
    full_z = (fp['bottom_z'], fp['top_z'])
    cut_z = (full_z[0] - 1, full_z[1] + 1)
    adds = [box((fp['back_x'], front_x), (-W, W), full_z)]
    cuts = []
    notes = []
    cr = fp.get('corner_radius', 0.0)
    for s in (1, -1):
        group = 'hook+' if s > 0 else 'hook-'
        cy = s * h['standoff_y']
        y_arm_out = cy - s * r_in                         # arm face next to the standoff
        y_arm_in = cy - s * (r_in + h['arm_thickness'])
        y_wall_in = s * (W - sw['thickness'])
        step_x1 = sw['step_x'] + sw['step_length']

        # the step front face (x = step_x1) runs from the arm to the wall; if its two
        # fillets do not fit, shrink both so they meet tangentially (smooth S, no cusp)
        r_step_in, r_step_wall = fil.get('step_inner', 0.0), fil.get('step_wall', 0.0)
        face = abs(y_wall_in - y_arm_in)
        if r_step_in + r_step_wall > face:
            k = face / (r_step_in + r_step_wall)
            r_step_in, r_step_wall = r_step_in * k, r_step_wall * k
            if s > 0:
                notes.append(f'step fillets scaled to {r_step_in:.2f} / {r_step_wall:.2f} mm '
                             f'to fit the {face:.2f} mm step face')

        def piece(corner, a, b, r, z, sign=s):
            return corner_piece((corner[0], corner[1]), (a[0], sign * a[1]),
                                (b[0], sign * b[1]), r, z, group=group)

        # side wall along the outside, from the step into the front plate (flexes)
        wall = box((sw['step_x'], front_x - max(cr, fp.get('edge_chamfer', 0.0)) - 0.01),
                   (s * W, y_wall_in), full_z)
        # step turning inwards; its back face bears on both plate edges at step_x
        step = box((sw['step_x'], step_x1), (s * W, y_arm_in), full_z)
        # short arm between the plates along the inner side of the standoff
        arm = box((h['standoff_x'] - 0.2, step_x1), (y_arm_out, y_arm_in), hz)
        for prim in (wall, step, arm):
            prim['group'] = group
            adds.append(prim)
        # hook arc behind the standoff, starting at its inner side; it stops one tip radius
        # short and a full round cap closes it, so the hook reaches exactly wrap_deg
        r_mid, r_cap = (r_in + r_out) / 2, (r_out - r_in) / 2
        wrap = h['wrap_deg'] - math.degrees(r_cap / r_mid)
        end = math.radians(-90 - wrap) if s > 0 else math.radians(90 + wrap)
        centre = (h['standoff_x'] + r_mid * math.cos(end), cy + r_mid * math.sin(end))
        cap = [(centre[0] + r_cap * math.cos(t), centre[1] + r_cap * math.sin(t))
               for t in np.linspace(0, 2 * math.pi, 48, endpoint=False)]
        adds.append(poly(cap, hz, group))
        phi = (-90 - wrap, -90) if s > 0 else (90, 90 + wrap)
        adds.append(arc(h['standoff_x'], cy, r_in, r_out, phi[0], phi[1], hz, group))

        # rounded bends: fill concave corners, cut convex ones
        if fil.get('wall_root'):
            adds.append(piece((fp['back_x'], y_wall_in), (-1, 0), (0, -1), fil['wall_root'],
                              full_z))
        if r_step_wall:
            adds.append(piece((step_x1, y_wall_in), (0, -1), (1, 0), r_step_wall, full_z))
        # fill in front of the standoff, between the arm, the step and the outer face.
        # Inner face: standoff circle (r_in) from straight inside round to the tangent
        # point, then a convex arc (rf) tangent to it and to the outer face y = W.
        rf = h['front_wrap_fillet']
        c1 = np.array([h['standoff_x'], h['standoff_y']])
        dy = W - rf - c1[1]
        dx = math.sqrt((r_in + rf) ** 2 - dy ** 2)
        c2 = c1 + np.array([dx, dy])
        t = math.atan2(dy, dx)                         # tangent direction from c1 to c2
        n = 24
        ring1 = [c1 + r_in * np.array([math.cos(a), math.sin(a)])
                 for a in np.linspace(-math.pi / 2, t, n)]
        ring2 = [c2 + rf * np.array([math.cos(a), math.sin(a)])
                 for a in np.linspace(t + math.pi, math.pi / 2, n)]
        x_in = c2[0] + 0.05                            # inside the step (x >= step_x)
        y_low = h['standoff_y'] - r_in - h['arm_thickness'] / 2   # inside the arm
        pts = [(h['standoff_x'], y_low), *map(tuple, ring1), *map(tuple, ring2[1:]),
               (x_in, W), (x_in, y_low)]
        pts = [(x, s * y) for x, y in pts]
        area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
        adds.append(poly(pts if area > 0 else pts[::-1], hz, group))
        # the same convex arc trims the step corner over the full height, so the outer
        # face is one continuous S curve from top to bottom
        cuts.append(piece((c2[0] - rf, s * W), (1, 0), (0, -1), rf, cut_z))
        if cr > 0:
            cuts.append(piece((front_x, s * W), (0, -1), (-1, 0), cr, cut_z))
        if r_step_in:
            cuts.append(piece((step_x1, y_arm_in), (-1, 0), (0, 1), r_step_in, cut_z))
    holes = []
    r0 = us['can_radius'] + us['clearance']
    x_end = front_x + 1.0
    r1 = r0 + (x_end - us['can_front_x']) * math.tan(math.radians(us['cone_half_angle_deg']))
    for yc in us['centers_y']:
        # the cylinder runs 0.3 mm into the cone so the two cuts share no face
        holes.append(cone_x((fp['back_x'] - 1.0, us['can_front_x'] + 0.3), r0, r0, yc,
                            us['center_z']))
        holes.append(cone_x((us['can_front_x'], x_end), r0, r1, yc, us['center_z']))
    return {'add': adds, 'cut': cuts, 'subtract': holes,
            'final_chamfer': fp.get('edge_chamfer', 0.0), 'notes': notes}


def inside(points, spec, margin=0.0):
    """Exact CSG membership (ignores the edge chamfer). margin > 0 grows the solid."""
    def in_box(b, m):
        q = points
        return np.all([(q[:, i] >= b[k][0] - m) & (q[:, i] <= b[k][1] + m)
                       for i, k in enumerate('xyz')], axis=0)

    def in_cone(c, m):
        x0, x1 = c['x']
        t = np.clip((points[:, 0] - x0) / (x1 - x0), 0, 1)
        r = c['r0'] + (c['r1'] - c['r0']) * t
        d = np.hypot(points[:, 1] - c['y'], points[:, 2] - c['z'])
        return (points[:, 0] >= x0 - m) & (points[:, 0] <= x1 + m) & (d <= r + m)

    def in_arc(a, m):
        q = points
        d = np.hypot(q[:, 0] - a['cx'], q[:, 1] - a['cy'])
        ang = np.degrees(np.arctan2(q[:, 1] - a['cy'], q[:, 0] - a['cx']))
        lo, hi = a['phi0'], a['phi1']
        in_ang = (((ang - lo) % 360) <= (hi - lo)) | (d < 1e-9)
        return ((d >= a['r_in'] - m) & (d <= a['r_out'] + m) & in_ang
                & (q[:, 2] >= a['z'][0] - m) & (q[:, 2] <= a['z'][1] + m))

    def in_poly(pg, m):
        # even-odd rule; the margin is applied in z only (fillet pieces are small)
        q = points
        pts = np.array(pg['pts'])
        odd = np.zeros(len(q), bool)
        for a, b in zip(pts, np.roll(pts, -1, axis=0)):
            crosses = (a[1] > q[:, 1]) != (b[1] > q[:, 1])
            with np.errstate(divide='ignore', invalid='ignore'):
                x_at = a[0] + (q[:, 1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            odd ^= crosses & (q[:, 0] < x_at)
        return odd & (q[:, 2] >= pg['z'][0] - m) & (q[:, 2] <= pg['z'][1] + m)

    test = {'box': in_box, 'cone': in_cone, 'arc': in_arc, 'poly': in_poly}
    solid = np.any([test[a['type']](a, margin) for a in spec['add']], axis=0)
    for cut in spec['cut'] + spec['subtract']:
        solid &= ~test[cut['type']](cut, -margin)
    return solid


# ----------------------------------------------------------------------------- robot
def robot_surface_points(per_mm2=4.0):
    """Sample the official visual meshes (bumper disabled), base_link frame, mm."""
    from ament_index_python.packages import get_package_share_directory as share
    from trimesh.transformations import euler_matrix, translation_matrix
    urdf = subprocess.run(
        ['xacro', str(Path(share('pinky_soccer_description')) / 'urdf/soccer_robot.urdf.xacro'),
         'bumper_enabled:=false'], capture_output=True, text=True, check=True).stdout
    root = ET.fromstring(urdf)
    parent = {j.find('child').get('link'): j for j in root.findall('joint')}

    def origin(el):
        o = el.find('origin')
        xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
        rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
        return translation_matrix(xyz) @ euler_matrix(*rpy, 'sxyz')

    def to_base_link(name):
        T = np.eye(4)
        while name != 'base_link' and name in parent:
            j = parent[name]
            T = origin(j) @ T
            name = j.find('parent').get('link')
        return T

    points, labels = [], []
    for link in root.findall('link'):
        if link.get('name') == 'base_footprint':
            continue
        for vis in link.findall('visual'):
            g = vis.find('geometry')[0]
            if g.tag != 'mesh':
                continue
            pkg, rel = g.get('filename')[len('package://'):].split('/', 1)
            scene = trimesh.load(str(Path(share(pkg)) / rel), force='scene')
            T = to_base_link(link.get('name')) @ origin(vis)
            for node in scene.graph.nodes_geometry:
                Tn, gname = scene.graph[node]
                m = scene.geometry[gname].copy()
                m.apply_transform(T @ Tn)
                n = int(np.clip(m.area * 1e6 * per_mm2, 500, 3_000_000))
                p, _ = trimesh.sample.sample_surface(m, n, seed=0)
                points.append(p * 1000.0)
                labels += [f"{link.get('name')}/{node}"] * len(p)
    return np.vstack(points), np.array(labels)


def check_clearance(spec, p):
    """Return (problems, tip flex, report lines): fitted, snapping on, near parts."""
    pts, labels = robot_surface_points()

    def describe(mask):
        out = []
        for lab in np.unique(labels[mask]):
            q = pts[mask & (labels == lab)]
            out.append(f'    {lab}: {len(q)} pts x[{q[:, 0].min():.1f},{q[:, 0].max():.1f}] '
                       f'|y|[{np.abs(q[:, 1]).min():.1f},{np.abs(q[:, 1]).max():.1f}] '
                       f'z[{q[:, 2].min():.1f},{q[:, 2].max():.1f}]')
        return out

    lines = [f'robot surface points sampled: {len(pts)}']
    pen = inside(pts, spec, margin=-0.05)
    lines.append(f'fitted: robot points inside the bumper (> 0.05 mm): {int(pen.sum())}')
    lines += describe(pen)

    # while snapping: the side wall bends (cantilever) and the hook block turns with it
    tip, hits = snap_simulation(p, spec, pts, labels)
    lines.append(f'snapping (bent side wall, 0.1 mm steps): max hook tip flex {tip:.2f} mm; '
                 f'robot parts hit on the way: {len(hits)}')
    for lab, (n, dx, t, x0, x1, y0, y1, z0, z1) in hits.items():
        lines.append(f'    {lab}: up to {n} pts ({dx:.1f} mm before seated, tip flex {t:.2f}) '
                     f'x[{x0:.1f},{x1:.1f}] |y|[{abs(y0):.1f},{abs(y1):.1f}] '
                     f'z[{z0:.1f},{z1:.1f}]')
    flexed = len(hits)
    for m in (0.3, 1.0):
        lines.append(f'within {m} mm of the fitted bumper:')
        lines += describe(inside(pts, spec, margin=m) & ~pen)
    return int(pen.sum() + flexed), tip, lines


def bend_pose(p, k):
    """
    Return (pivot x, pivot y, end shift, rotation) of the hook block on a bent side wall.

    The side wall is a cantilever from the front plate to the step end (x_e); the step,
    arm and hook form a rigid block on its end. The standoff pushes the hook near the
    standoff axis, a force at distance a behind x_e: the wall end moves inwards by
    k * (L^3/3 + a L^2/2) and turns by k * (L^2/2 + a L) (beam theory, EI folded into k).
    """
    fp, sw, h = p['front_plate'], p['side_walls'], p['hooks']
    x_e = sw['step_x'] + sw['step_length']
    length = fp['back_x'] - x_e
    a = x_e - h['standoff_x']
    shift = k * (length ** 3 / 3 + a * length ** 2 / 2)
    theta = k * (length ** 2 / 2 + a * length)
    pivot_y = fp['half_width'] - sw['thickness'] / 2
    return x_e, pivot_y, shift, theta


def inside_bent(points, spec, p, k, dx, side):
    """
    Membership of points in one side's hook block, bent by k and slid dx forward.

    Points are mapped back into the undeformed frame (inverse rigid motion), then tested
    against that side's primitives only.
    """
    x_e, pivot_y, shift, theta = bend_pose(p, k)
    py = side * pivot_y
    q = points.copy()
    q[:, 0] -= dx
    # inverse of: rotate about (x_e, py) by side * theta, then move inwards by shift
    q[:, 1] += side * shift
    c, s_ = math.cos(-side * theta), math.sin(-side * theta)
    rx, ry = q[:, 0] - x_e, q[:, 1] - py
    q[:, 0], q[:, 1] = x_e + c * rx - s_ * ry, py + s_ * rx + c * ry
    group = 'hook+' if side > 0 else 'hook-'
    sub = {'add': [a for a in spec['add'] if a.get('group') == group],
           'cut': [c for c in spec['cut'] if c.get('group') == group],
           'subtract': []}
    return inside(q, sub)


def snap_simulation(p, spec, pts=None, labels=None, travel=10.0):
    """
    Slide the bumper on and bend the side wall as little as needed at each step.

    Returns (max hook tip flex in mm, {robot part: worst hit}).
    """
    h = p['hooks']
    radius = np.sqrt(np.linspace(0, 1, 30))[:, None] * h['standoff_radius']
    angle = np.linspace(0, 2 * np.pi, 72, endpoint=False)[None, :]
    zmid = (h['plate_gap_bottom_z'] + h['plate_gap_top_z']) / 2
    disc = np.column_stack([(h['standoff_x'] + radius * np.cos(angle)).ravel(),
                            (h['standoff_y'] + radius * np.sin(angle)).ravel(),
                            np.full(radius.size * angle.size, zmid)])
    x_e = p['side_walls']['step_x'] + p['side_walls']['step_length']
    if pts is not None:
        near = ((pts[:, 0] < x_e + travel + 1) & (pts[:, 1] > 15)
                & (np.hypot(pts[:, 0] - h['standoff_x'], pts[:, 1] - h['standoff_y'])
                   > h['standoff_radius'] + 0.05))
        robot, robot_labels = pts[near], labels[near]
    _, _, shift1, theta1 = bend_pose(p, 1.0)
    per_k = shift1 + theta1 * (x_e - h['standoff_x'])   # hook tip flex per unit k
    worst, hits = 0.0, {}
    for dx in np.arange(0.0, travel + 1e-9, 0.1):
        for tip in np.arange(0.0, 4.0 + 1e-9, 0.05):
            k = tip / per_k
            if inside_bent(disc, spec, p, k, dx, 1).any():
                continue
            worst = max(worst, tip)
            if pts is not None and tip > 0:
                bad = inside_bent(robot, spec, p, k, dx, 1)
                for lab in np.unique(robot_labels[bad]):
                    q = robot[bad & (robot_labels == lab)]
                    old = hits.get(lab)
                    if old is None or len(q) > old[0]:
                        hits[lab] = (len(q), dx, tip, q[:, 0].min(), q[:, 0].max(),
                                     q[:, 1].min(), q[:, 1].max(), q[:, 2].min(),
                                     q[:, 2].max())
            break
        else:
            return float('inf'), hits
    return worst, hits


def snap_numbers(p, tip):
    """Return the side wall root strain for a hook tip flex (same beam model as bend_pose)."""
    h, sw = p['hooks'], p['side_walls']
    x_e = sw['step_x'] + sw['step_length']
    _, _, shift1, theta1 = bend_pose(p, 1.0)
    k = tip / (shift1 + theta1 * (x_e - h['standoff_x']))
    length = p['front_plate']['back_x'] - x_e
    curvature = k * (length + x_e - h['standoff_x'])           # at the wall root
    strain = curvature * sw['thickness'] / 2
    stress = strain * p['material'].get('flex_modulus_mpa', 3500)
    return (f'snap: hook tip flexes {tip:.2f} mm; side wall root strain {strain * 100:.2f} % '
            f'(~{stress:.0f} MPa); PLA snap fits are usually kept below ~2 %')


# ----------------------------------------------------------------------------- outputs
def write_params(path, p, name, com_mm, mass, inertia, extents_mm):
    rel = 'package://pinky_soccer_description/meshes'
    data = {'bumper': {
        'width': round(float(extents_mm[1]) / 1000, 5),
        'depth': round(float(extents_mm[0]) / 1000, 5),
        'height': round(float(extents_mm[2]) / 1000, 5),
        'offset_x': round(float(com_mm[0]) / 1000, 5),
        'offset_y': round(float(com_mm[1]) / 1000, 5),
        'offset_z': round(float(com_mm[2]) / 1000, 5),
        'mass': round(float(mass), 5),
        'inertia': {k: float(f'{v:.6g}') for k, v in inertia.items()},
        'visual_mesh': f'{rel}/visual/{name}.stl',
        'collision_mesh': f'{rel}/collision/{name}_collision.stl',
        'mesh_scale': [1.0, 1.0, 1.0],
        'color': [0.10, 0.45, 0.90, 1.0],
    }}
    header = (f'# GENERATED by cad/bumper/generate_bumper.py from cad/bumper/{name}.yaml.\n'
              '# Do not edit by hand; change the CAD parameters and regenerate.\n'
              '# width/depth/height: bounding box. offset: centre of mass (bumper_link origin)\n'
              '# in base_link. mass/inertia: from the mesh volume, PLA density x fill fraction.\n')
    path.write_text(header + yaml.safe_dump(data, sort_keys=False))


def build_mesh(spec, blender):
    with tempfile.TemporaryDirectory() as tmp:
        spec_path, raw = Path(tmp) / 'spec.json', Path(tmp) / 'raw.stl'
        spec_path.write_text(json.dumps(spec))
        out = subprocess.run([blender, '-b', '--factory-startup', '--python',
                              str(HERE / 'blender_csg.py'), '--', str(spec_path), str(raw)],
                             capture_output=True, text=True)
        if 'BLENDER_CSG_OK' not in out.stdout:
            sys.exit('Blender failed:\n' + out.stdout[-3000:] + out.stderr[-3000:])
        return trimesh.load(raw, force='mesh')


def mesh_quality(mesh):
    """Return bodies, watertight, short edges (< 0.05 mm) and slivers (height < 0.02 mm)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    adj = mesh.face_adjacency
    graph = coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(len(mesh.faces),) * 2)
    bodies = connected_components(graph, directed=False)[0]
    e = mesh.edges_unique
    short = int((np.linalg.norm(mesh.vertices[e[:, 0]] - mesh.vertices[e[:, 1]], axis=1)
                 < 0.05).sum())
    tri = mesh.triangles
    longest = np.max([np.linalg.norm(tri[:, i] - tri[:, (i + 1) % 3], axis=1)
                      for i in range(3)], axis=0)
    slivers = int((2 * mesh.area_faces / np.maximum(longest, 1e-12) < 0.02).sum())
    return {'bodies': int(bodies), 'watertight': bool(mesh.is_watertight),
            'short_edges': short, 'slivers': slivers}


def mesh_ok(mesh):
    q = mesh_quality(mesh)
    return q['watertight'] and q['bodies'] == 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('params', type=Path)
    ap.add_argument('--blender', default=shutil.which('blender') or '/snap/bin/blender')
    ap.add_argument('--skip-check', action='store_true', help='skip the robot clearance check')
    args = ap.parse_args()
    p = yaml.safe_load(args.params.read_text())
    name = p['name']
    spec = build_spec(p)

    notes = list(spec.get('notes', []))
    mesh = build_mesh(spec, args.blender)
    if not mesh_ok(mesh) and spec.get('final_chamfer'):
        notes.append('final chamfer broke the mesh; built without it')
        mesh = build_mesh(dict(spec, final_chamfer=0.0), args.blender)
    if not mesh_ok(mesh):
        sys.exit('generated mesh is not a single watertight body')
    material = p['material']
    density = material['density_g_cm3'] * material['fill_fraction'] * 1e-6  # kg/mm^3
    mesh.density = density
    mass = mesh.volume * density
    com = mesh.center_mass
    tensor = mesh.moment_inertia * 1e-6                         # kg m^2 about the COM
    inertia = {'ixx': tensor[0, 0], 'iyy': tensor[1, 1], 'izz': tensor[2, 2],
               'ixy': tensor[0, 1], 'ixz': tensor[0, 2], 'iyz': tensor[1, 2]}
    lo, hi = mesh.bounds

    q = mesh_quality(mesh)
    report = [f'{name}', f'faces {len(mesh.faces)}, watertight {q["watertight"]}, '
              f'bodies {q["bodies"]}, edges < 0.05 mm {q["short_edges"]}, '
              f'slivers {q["slivers"]}', *notes,
              f'volume {mesh.volume / 1000:.2f} cm^3 -> mass {mass * 1000:.1f} g '
              f'(density {material["density_g_cm3"]} x fill {material["fill_fraction"]})',
              f'bounds base_link mm x[{lo[0]:.2f},{hi[0]:.2f}] y[{lo[1]:.2f},{hi[1]:.2f}] '
              f'z[{lo[2]:.2f},{hi[2]:.2f}]',
              f'ground clearance {lo[2] + BASE_LINK_HEIGHT:.2f} mm, '
              f'top {hi[2] + BASE_LINK_HEIGHT:.2f} mm above ground',
              f'centre of mass base_link mm {np.round(com, 2).tolist()}',
              'inertia about COM kg m^2 ' + ', '.join(f'{k} {v:.3e}' for k, v in inertia.items())]
    penetrations = 0
    if args.skip_check:
        tip, _ = snap_simulation(p, spec)
        report.append(snap_numbers(p, tip))
    else:
        penetrations, tip, lines = check_clearance(spec, p)
        report.append(snap_numbers(p, tip))
        report += ['', 'clearance check against pinky_description visual meshes'] + lines

    # URDF meshes: bumper_link frame (origin = COM), metres
    visual = mesh.copy()
    visual.apply_translation(-com)
    visual.apply_scale(0.001)
    fp = p['front_plate']
    plate = trimesh.creation.box(
        extents=[fp['thickness'], 2 * fp['half_width'], fp['top_z'] - fp['bottom_z']])
    plate.apply_translation([fp['back_x'] + fp['thickness'] / 2, 0,
                             (fp['top_z'] + fp['bottom_z']) / 2])
    plate.apply_translation(-com)
    plate.apply_scale(0.001)
    (PKG / 'meshes/visual').mkdir(parents=True, exist_ok=True)
    (PKG / 'meshes/collision').mkdir(parents=True, exist_ok=True)
    visual.export(PKG / f'meshes/visual/{name}.stl')
    plate.export(PKG / f'meshes/collision/{name}_collision.stl')

    # print file, mm. upright: bottom face on the bed, so the hook arms bend along the
    # filament lines (supports needed under the hooks and inside the ultrasonic holes).
    # front_down: front face on the bed, no supports, but the arms bend across layers.
    printable = mesh.copy()
    orientation = p.get('print', {}).get('orientation', 'upright')
    if orientation == 'front_down':
        printable.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    elif orientation != 'upright':
        sys.exit(f'unknown print orientation {orientation}')
    plo, phi = printable.bounds
    printable.apply_translation([-(plo[0] + phi[0]) / 2, -(plo[1] + phi[1]) / 2, -plo[2]])
    (HERE / 'print').mkdir(exist_ok=True)
    printable.export(HERE / f'print/{name}_print.stl')
    report.append(f'\nprint ({orientation}) size mm: {np.round(printable.extents, 1).tolist()}')

    write_params(PKG / f'config/{name}_params.yaml', p, name, com, mass, inertia, hi - lo)
    (HERE / f'print/{name}_report.txt').write_text('\n'.join(report) + '\n')
    print('\n'.join(report))
    if penetrations:
        sys.exit(f'{penetrations} robot surface points penetrate the bumper')


if __name__ == '__main__':
    os.environ.setdefault('PYTHONHASHSEED', '0')
    main()
