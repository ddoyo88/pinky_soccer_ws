"""
Blender (bpy) side of generate_bumper.py: build a solid from boxes/cones and export STL.

Run by generate_bumper.py as
    blender -b --factory-startup --python blender_csg.py -- <spec.json> <out.stl>
One Blender unit = 1 mm. The spec lists primitives to add (union) and to subtract.
"""
import json
import math
import sys

import bpy


def make_box(p):
    (x0, x1), (y0, y1), (z0, z1) = p['x'], p['y'], p['z']
    bpy.ops.mesh.primitive_cube_add(size=1, location=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    obj = bpy.context.object
    obj.scale = (x1 - x0, y1 - y0, z1 - z0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return obj


def make_cone_x(p):
    """Make a truncated cone along +x: radius r0 at x0, r1 at x1."""
    x0, x1 = p['x']
    bpy.ops.mesh.primitive_cone_add(vertices=p.get('segments', 96), radius1=p['r0'],
                                    radius2=p['r1'], depth=x1 - x0,
                                    location=((x0 + x1) / 2, p['y'], p['z']),
                                    rotation=(0.0, math.pi / 2, 0.0))  # local +z -> +x
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return bpy.context.object


def make_arc(p):
    """Make an annular sector (r_in..r_out, phi0..phi1 deg) around (cx, cy), extruded in z."""
    n = p.get('segments', 48)
    z0, z1 = p['z']
    phis = [math.radians(p['phi0'] + (p['phi1'] - p['phi0']) * i / n) for i in range(n + 1)]
    ring = [(p['cx'] + r * math.cos(a), p['cy'] + r * math.sin(a))
            for r in (p['r_in'], p['r_out']) for a in phis]
    verts = [(x, y, z) for z in (z0, z1) for (x, y) in ring]
    m = n + 1                       # points per arc
    top = 2 * m                     # offset of the z1 layer

    def v(layer, radius, i):
        return layer * top + radius * m + i

    faces = []
    for i in range(n):
        faces.append((v(0, 0, i), v(0, 1, i), v(0, 1, i + 1), v(0, 0, i + 1)))  # bottom
        faces.append((v(1, 0, i), v(1, 0, i + 1), v(1, 1, i + 1), v(1, 1, i)))  # top
        faces.append((v(0, 0, i), v(0, 0, i + 1), v(1, 0, i + 1), v(1, 0, i)))  # inner
        faces.append((v(0, 1, i), v(1, 1, i), v(1, 1, i + 1), v(0, 1, i + 1)))  # outer
    for i in (0, n):                                                           # end caps
        cap = (v(0, 0, i), v(1, 0, i), v(1, 1, i), v(0, 1, i))
        faces.append(cap if i == 0 else cap[::-1])
    mesh = bpy.data.meshes.new('arc')
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new('arc', mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    return obj


def make_poly(p):
    """Make a prism from a counter-clockwise xy polygon, extruded over z."""
    pts = p['pts']
    n = len(pts)
    z0, z1 = p['z']
    verts = [(x, y, z0) for x, y in pts] + [(x, y, z1) for x, y in pts]
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    faces += [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    mesh = bpy.data.meshes.new('poly')
    mesh.from_pydata(verts, [], faces)
    mesh.validate()
    obj = bpy.data.objects.new('poly', mesh)
    bpy.context.collection.objects.link(obj)
    return obj


def make(p):
    return {'box': make_box, 'cone': make_cone_x, 'arc': make_arc,
            'poly': make_poly}[p['type']](p)


def boolean(target, tool, operation):
    mod = target.modifiers.new('csg', 'BOOLEAN')
    mod.operation = operation
    mod.object = tool
    mod.solver = 'EXACT'
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(tool, do_unlink=True)


def main():
    spec_path, out_path = sys.argv[sys.argv.index('--') + 1:][:2]
    with open(spec_path) as f:
        spec = json.load(f)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    adds = [make(p) for p in spec['add']]
    body = adds[0]
    for tool in adds[1:]:
        boolean(body, tool, 'UNION')

    for p in spec.get('cut', []):
        boolean(body, make(p), 'DIFFERENCE')

    for p in spec.get('subtract', []):
        boolean(body, make(p), 'DIFFERENCE')

    # Clean up boolean leftovers: merge near-duplicate vertices, drop degenerate faces and
    # dissolve coplanar splits so no slivers or zero-area faces remain.
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.002)
    bpy.ops.mesh.dissolve_degenerate(threshold=0.002)
    bpy.ops.mesh.dissolve_limited(angle_limit=math.radians(0.5))
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Final chamfer on sharp edges only, after every other operation.
    chamfer = spec.get('final_chamfer', 0.0)
    if chamfer > 0:
        mod = body.modifiers.new('chamfer', 'BEVEL')
        mod.width = chamfer
        mod.segments = 1
        mod.limit_method = 'ANGLE'
        mod.angle_limit = math.radians(spec.get('chamfer_angle_deg', 50))
        mod.use_clamp_overlap = True
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.002)
        bpy.ops.mesh.dissolve_degenerate(threshold=0.002)
        bpy.ops.object.mode_set(mode='OBJECT')

    # Even triangulation (no long thin fans across the large flat faces).
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')

    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    bpy.ops.wm.stl_export(filepath=out_path, export_selected_objects=True,
                          ascii_format=False, apply_modifiers=True, global_scale=1.0)
    print('BLENDER_CSG_OK', out_path, len(body.data.polygons), 'faces')


main()
