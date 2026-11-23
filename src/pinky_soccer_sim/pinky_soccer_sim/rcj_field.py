"""RoboCupJunior Soccer field geometry (shared by the Gazebo world and the Nav2 map).

Vendored from cse1598/pinky-soccer  src/pinky_soccer_sim/pinky_soccer_sim/rcj_field.py (Apache-2.0):
`scene()` and `stl()` are copied unchanged; only the YAML loading / MJCF export were dropped.
Dimensions: https://robocup-junior.github.io/soccer-rules/master/field_specification.html (2026-06-03)

Frame: field centre, floor top z = 0, +X = blue goal, -X = yellow goal.
"""
import math
import struct

import numpy as np

GREEN = (0.035, 0.35, 0.075, 1)
BLACK = (0.025, 0.025, 0.025, 1)
WHITE = (0.95, 0.95, 0.95, 1)
BLUE = (0.08, 0.3, 0.95, 1)
YELLOW = (1, 0.85, 0.04, 1)

# config/rcj_field.yaml of the source repo
DEFAULT_CFG = {
    'playing_length': 2.19, 'playing_width': 1.58, 'outer_strip': 0.12, 'wall_height': 0.22,
    'goal_width': 0.60, 'goal_height': 0.10, 'goal_depth': 0.074,
    'ramp_base': 0.10, 'ramp_rise': 0.02, 'line_width': 0.02,
    'penalty_depth': 0.25, 'penalty_width': 0.80, 'penalty_radius': 0.15,
    'center_diameter': 0.60, 'neutral_diameter': 0.01, 'neutral_distance_from_end': 0.45,
    # construction / rendering / physics assumptions (not measured)
    'wall_thickness': 0.01, 'floor_thickness': 0.02, 'thin_black_line': 0.002,
    'friction': 0.8, 'arc_segments': 128,
}


def vector(values):
    """Write deterministic SI vectors with enough precision for both engines."""
    return ' '.join(f'{float(v):.10g}' for v in values)


def scene(cfg):
    """Return shared visual and collision primitives, including mitered wedge meshes."""
    items = []

    def box(name, pos, size, color, collision=True, yaw=0):
        items.append({'name': name, 'kind': 'box', 'pos': list(pos), 'size': list(size),
                      'color': color, 'collision': collision, 'yaw': yaw})

    def line(name, start, end, width, color):
        dx, dy = end[0]-start[0], end[1]-start[1]
        box(name, [(start[0]+end[0])/2, (start[1]+end[1])/2, .0003],
            [math.hypot(dx, dy), width, .0002], color, False, math.atan2(dy, dx))

    def arc(name, cx, cy, radius, begin, end, width, color, sign=1):
        count = max(4, round(cfg['arc_segments'] * abs(end-begin)/(2*math.pi)))
        angles = np.linspace(begin, end, count+1)
        points = [(sign*(cx+radius*math.cos(a)), cy+radius*math.sin(a)) for a in angles]
        for i, (p, q) in enumerate(zip(points, points[1:])):
            line(f'{name}_{i:03d}', p, q, width, color)

    length, width = cfg['playing_length'], cfg['playing_width']
    hx, hy = length/2+cfg['outer_strip'], width/2+cfg['outer_strip']
    wall, height = cfg['wall_thickness'], cfg['wall_height']
    box('carpet', [0, 0, -cfg['floor_thickness']/2],
        [2*hx, 2*hy, cfg['floor_thickness']], GREEN)
    for sign in (-1, 1):
        box(f'end_wall_{sign}', [sign*(hx+wall/2), 0, height/2],
            [wall, 2*hy+2*wall, height], BLACK)
        box(f'side_wall_{sign}', [0, sign*(hy+wall/2), height/2],
            [2*hx, wall, height], BLACK)
    goal_half = cfg['goal_width']/2
    for sign, color, name in [(1, BLUE, 'blue'), (-1, YELLOW, 'yellow')]:
        front, depth, gh = length/2, cfg['goal_depth'], cfg['goal_height']
        box(f'{name}_goal_back', [sign*(front+depth+wall/2), 0, gh/2],
            [wall, cfg['goal_width']+2*wall, gh], color)
        for side in (-1, 1):
            box(f'{name}_goal_side_{side}',
                [sign*(front+depth/2-cfg['line_width']/2), side*(goal_half+wall/2), gh/2],
                [depth+cfg['line_width'], wall, gh], color)
        box(f'{name}_goal_roof',
            [sign*(front+(depth+wall-cfg['line_width'])/2), 0, gh+wall/2],
            [depth+wall+cfg['line_width'], cfg['goal_width']+2*wall, wall], color)
        box(f'{name}_goal_floor', [sign*(front+depth/2), 0, .00015],
            [depth, cfg['goal_width'], .0001], color, False)

    # Convex wedges meet on 45 degree corner seams; no overlaps or corner steps.
    ramp, rise = cfg['ramp_base'], cfg['ramp_rise']

    def wedge(name, points, elevations):
        vertices = [[x, y, -.001] for x, y in points]
        vertices += [[x, y, z] for (x, y), z in zip(points, elevations)]
        items.append({'name': name, 'kind': 'mesh', 'vertices': vertices, 'pos': [0, 0, 0],
                      'color': GREEN, 'collision': True, 'yaw': 0})

    for sign in (-1, 1) if ramp > 0 and rise > 0 else ():      # flat boards: no ramps
        wedge(f'side_ramp_{sign}',
              [(-hx, sign*hy), (hx, sign*hy),
               (hx-ramp, sign*(hy-ramp)), (-hx+ramp, sign*(hy-ramp))],
              [rise, rise, 0, 0])
        for side in (-1, 1):
            gap = goal_half+wall
            wedge(f'end_ramp_{sign}_{side}',
                  [(sign*hx, side*gap), (sign*hx, side*hy),
                   (sign*(hx-ramp), side*(hy-ramp)), (sign*(hx-ramp), side*gap)],
                  [rise, rise, 0, 0])

    lw = cfg['line_width']
    # The OUTSIDE edge of the white rectangle defines the playing field.
    for sign in (-1, 1):
        line(f'boundary_side_{sign}', (-length/2, sign*(width/2-lw/2)),
             (length/2, sign*(width/2-lw/2)), lw, WHITE)
        line(f'boundary_end_{sign}', (sign*(length/2-lw/2), -width/2),
             (sign*(length/2-lw/2), width/2), lw, WHITE)
        radius = cfg['penalty_radius']
        cy = cfg['penalty_width']/2-radius
        cx = length/2-cfg['penalty_depth']+radius
        for side in (-1, 1):
            line(f'penalty_side_{sign}_{side}',
                 (sign*length/2, side*(cfg['penalty_width']/2-lw/2)),
                 (sign*cx, side*(cfg['penalty_width']/2-lw/2)), lw, WHITE)
            arc(f'penalty_arc_{sign}_{side}', cx, side*cy, radius-lw/2,
                math.pi/2 if side == 1 else math.pi,
                math.pi if side == 1 else 3*math.pi/2, lw, WHITE, sign)
        line(f'penalty_front_{sign}', (sign*(length/2-cfg['penalty_depth']+lw/2), -cy),
             (sign*(length/2-cfg['penalty_depth']+lw/2), cy), lw, WHITE)
    arc('center_circle', 0, 0, cfg['center_diameter']/2, 0, 2*math.pi,
        cfg['thin_black_line'], BLACK)
    for i, (x, y) in enumerate([(0, 0)] + [
            (sx*(length/2-cfg['neutral_distance_from_end']), sy*cfg['penalty_width']/2)
            for sx in (-1, 1) for sy in (-1, 1)]):
        items.append({'name': f'neutral_{i}', 'kind': 'cylinder', 'pos': [x, y, .0005],
                      'radius': cfg['neutral_diameter']/2, 'length': .0002,
                      'color': BLACK, 'collision': False, 'yaw': 0})
    return items


def stl(path, vertices):
    """Write outward-wound triangles of a convex eight-vertex wedge."""
    from scipy.spatial import ConvexHull
    vertices = np.asarray(vertices)
    hull = ConvexHull(vertices)
    data = bytearray(b'Generated RCJ convex wedge'.ljust(80, b'\0'))
    data.extend(struct.pack('<I', len(hull.simplices)))
    for indices, equation in zip(hull.simplices, hull.equations):
        a, b, c = vertices[indices]
        if np.dot(np.cross(b-a, c-a), equation[:3]) < 0:
            b, c = c, b
        data.extend(struct.pack('<12fH', *equation[:3], *a, *b, *c, 0))
    path.write_bytes(data)
