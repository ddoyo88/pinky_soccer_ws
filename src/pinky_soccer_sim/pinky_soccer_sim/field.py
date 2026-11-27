"""Parametric soccer field.

Coordinate system (world == map when the generated map + AMCL are used):
    origin  = field centre
    +X      = field length                  goals are on the X ends
    +Y      = field width
    goal    = opening of `goal_width` centred on y=0, goal line at x = +-length/2
    yellow team attacks the +X goal, blue team attacks the -X goal

Styles (field_style launch argument):
    hardboard (default) the team's four-board training field, config/field_hardboard.yaml: playing area
                   1.66 x 1.06 m, 8 cm outer strip, flat (no ramps), 0.20 m walls, goals 0.40 x 0.05 x 0.07 m
    rcj            RoboCupJunior Soccer field, config/field_rcj.yaml (geometry from rcj_field.py): length/width are the PLAYING
                   area (2.19 x 1.58); a 0.12 m outer strip with 2 cm ramps runs to the 0.22 m black walls;
                   goals (0.60 wide, 0.074 deep, 0.10 high) stand inside the walls, mouth on the goal line.
    box            the original simple field: walls directly on the field edge, goal rails + backstop.
"""
from dataclasses import dataclass
import os
from typing import Optional

import yaml

from pinky_soccer_sim import rcj_field


def _config_dir():
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory('pinky_soccer_sim'), 'config')
    except Exception:                                  # not installed: source tree
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config')


def field_config(style):
    """Numeric entries of config/field_<style>.yaml (the single source of truth for the field)."""
    with open(os.path.join(_config_dir(), f'field_{style}.yaml')) as f:
        cfg = yaml.safe_load(f)
    return {k: v for k, v in cfg.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


BOX_DEFAULTS = {'playing_length': 2.4, 'playing_width': 1.8, 'goal_width': 0.6, 'wall_height': 0.18,
                'outer_strip': 0.0}


def default_dims(style='hardboard'):
    """(playing_length, playing_width, goal_width, outer_strip) of a field style."""
    c = BOX_DEFAULTS if style == 'box' else field_config(style)
    return c['playing_length'], c['playing_width'], c['goal_width'], c['outer_strip']


def resolve_dims(style, length='auto', width='auto', goal_width='auto'):
    """Launch-argument helper: 'auto' (or '') -> value from the field YAML.  Returns floats
    (length, width, goal_width, outer_strip)."""
    d = default_dims(style)

    def pick(v, default):
        return default if str(v).strip() in ('', 'auto') else float(v)
    return pick(length, d[0]), pick(width, d[1]), pick(goal_width, d[2]), d[3]


@dataclass
class FieldSpec:
    length: Optional[float] = None  # None -> from config/field_<style>.yaml
    width: Optional[float] = None
    goal_width: Optional[float] = None
    goal_depth: float = 0.15        # box style only: goal side-wall extension
    wall_height: Optional[float] = None   # must be higher than the Pinky lidar plane (rplidar_link = 0.125 m)
    wall_thickness: float = 0.02    # box style only (rcj uses 0.01)
    ball_radius: float = 0.02135    # golf ball d = 42.67 mm
    ball_mass: float = 0.0459
    ball_damping: float = 0.05      # rolling resistance [N s/m]; 0 = the ball rolls forever
    world_name: str = 'soccer'
    style: str = 'hardboard'

    def __post_init__(self):
        base = BOX_DEFAULTS if self.style == 'box' else field_config(self.style)
        self.length = base['playing_length'] if self.length is None else self.length
        self.width = base['playing_width'] if self.width is None else self.width
        self.goal_width = base['goal_width'] if self.goal_width is None else self.goal_width
        self.wall_height = base['wall_height'] if self.wall_height is None else self.wall_height

    @property
    def half_l(self):
        return self.length / 2.0

    @property
    def half_w(self):
        return self.width / 2.0

    def rcj_cfg(self):
        cfg = dict(rcj_field.DEFAULT_CFG)
        cfg.update(field_config(self.style))
        cfg.update(playing_length=self.length, playing_width=self.width,
                   goal_width=self.goal_width, wall_height=self.wall_height)
        return cfg


# ---------------------------------------------------------------- box style
def wall_boxes(s: FieldSpec):
    """Return [(name, cx, cy, sx, sy, team)] axis-aligned wall boxes.  team in {None,'yellow','blue'}."""
    t, L, W, gw, gd = s.wall_thickness, s.length, s.width, s.goal_width, s.goal_depth
    boxes = []
    for sgn, nm in ((1, 'left'), (-1, 'right')):
        boxes.append((f'side_{nm}', 0.0, sgn * (W / 2 + t / 2), L + 2 * t, t, None))
    # End walls leave the goal mouth open; goal rails extend inward and close against a backstop.
    seg = W / 2 + t - gw / 2
    for e, nm, team in ((1, 'plus', 'blue'), (-1, 'minus', 'yellow')):
        for sgn, side in ((1, 'l'), (-1, 'r')):
            boxes.append((f'end_{nm}_{side}', e * (L / 2 + t / 2), sgn * (gw / 2 + seg / 2),
                          t, seg, None))
            side_wall_length = 2 * gd + t
            side_wall_x = e * (L / 2 + t / 2)
            boxes.append((f'goal_side_{nm}_{side}', side_wall_x, sgn * (gw / 2 + t / 2),
                          side_wall_length, t, team))
        boxes.append((f'goal_back_{nm}', e * (L / 2 + gd + t / 2), 0.0,
                      t, gw + 2 * t, team))
    return boxes


def _color(team):
    if team == 'yellow':
        return '1.00 0.85 0.04 1'
    if team == 'blue':
        return '0.15 0.25 0.90 1'
    return '0.30 0.30 0.36 1'


def _box_field_model(s: FieldSpec):
    t = s.wall_thickness
    gx = s.length + 2 * (t + s.goal_depth + 0.10)
    gy = s.width + 2 * (t + 0.10)
    walls = []
    for name, cx, cy, sx, sy, team in wall_boxes(s):
        walls.append(f'''
      <collision name="{name}_col">
        <pose>{cx:.4f} {cy:.4f} {s.wall_height / 2:.4f} 0 0 0</pose>
        <geometry><box><size>{sx:.4f} {sy:.4f} {s.wall_height:.4f}</size></box></geometry>
      </collision>
      <visual name="{name}_vis">
        <pose>{cx:.4f} {cy:.4f} {s.wall_height / 2:.4f} 0 0 0</pose>
        <geometry><box><size>{sx:.4f} {sy:.4f} {s.wall_height:.4f}</size></box></geometry>
        <material><ambient>{_color(team)}</ambient><diffuse>{_color(team)}</diffuse></material>
      </visual>''')
    return f'''
    <model name="field">
      <static>true</static>
      <link name="floor">
        <collision name="floor_col">
          <pose>0 0 -0.005 0 0 0</pose>
          <geometry><box><size>{gx:.3f} {gy:.3f} 0.01</size></box></geometry>
          <surface><friction><ode><mu>0.9</mu><mu2>0.9</mu2></ode></friction></surface>
        </collision>
        <visual name="floor_vis">
          <pose>0 0 -0.005 0 0 0</pose>
          <geometry><box><size>{gx:.3f} {gy:.3f} 0.01</size></box></geometry>
          <material><ambient>0.10 0.45 0.15 1</ambient><diffuse>0.10 0.45 0.15 1</diffuse></material>
        </visual>
      </link>
      <link name="walls">{''.join(walls)}
      </link>
    </model>'''


# ---------------------------------------------------------------- rcj style
def _rcj_field_model(s: FieldSpec, mesh_dir):
    cfg = s.rcj_cfg()
    os.makedirs(mesh_dir, exist_ok=True)
    from pathlib import Path
    v = rcj_field.vector
    out = []
    for it in rcj_field.scene(cfg):
        if it['kind'] == 'mesh':
            mesh_path = os.path.abspath(os.path.join(mesh_dir, f'{it["name"]}.stl'))
            rcj_field.stl(Path(mesh_path), it['vertices'])
            geom = f'<mesh><uri>file://{mesh_path}</uri></mesh>'
        elif it['kind'] == 'box':
            geom = f'<box><size>{v(it["size"])}</size></box>'
        else:
            geom = f'<cylinder><radius>{it["radius"]}</radius><length>{it["length"]}</length></cylinder>'
        pose = v([*it['pos'], 0, 0, it['yaw']])
        if it['collision']:
            mu = cfg['friction']
            out.append(f'''
        <collision name="{it["name"]}_col">
          <pose>{pose}</pose><geometry>{geom}</geometry>
          <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>
        </collision>''')
        c = v(it['color'])
        out.append(f'''
        <visual name="{it["name"]}_vis">
          <pose>{pose}</pose><geometry>{geom}</geometry>
          <material><ambient>{c}</ambient><diffuse>{c}</diffuse><specular>0 0 0 1</specular></material>
        </visual>''')
    return f'''
    <model name="rcj_field">
      <static>true</static>
      <link name="field">{''.join(out)}
      </link>
    </model>'''


def rcj_obstacle_boxes(s: FieldSpec):
    """Axis-aligned (name, cx, cy, sx, sy) of everything the robot cannot drive through: walls and goal
    back/side panels.  Carpet, ramps (drivable) and goal roofs (would fill the goal mouth) are left out."""
    boxes = []
    for it in rcj_field.scene(s.rcj_cfg()):
        n = it['name']
        if it['kind'] == 'box' and it['collision'] and ('wall' in n or n.endswith('_back') or '_side_' in n
                                                         or n.endswith('_flag')):
            boxes.append((n, it['pos'][0], it['pos'][1], it['size'][0], it['size'][1]))
    return boxes


def outer_half_extents(s: FieldSpec):
    """Half extents of the area enclosed by the outer walls (incl. wall thickness)."""
    if s.style != 'box':
        c = s.rcj_cfg()
        return (s.half_l + c['outer_strip'] + c['wall_thickness'],
                s.half_w + c['outer_strip'] + c['wall_thickness'])
    t = s.wall_thickness
    return s.half_l + t + s.goal_depth, s.half_w + t


# ---------------------------------------------------------------- world
def _pose_publisher(freq=50):
    """Publishes this model's world pose on /model/<name>/pose with child_frame_id = model name
    (the world-level dynamic_pose/info loses the names in ros_gz_bridge)."""
    return f'''
      <plugin filename="gz-sim-pose-publisher-system" name="gz::sim::systems::PosePublisher">
        <publish_link_pose>false</publish_link_pose>
        <publish_sensor_pose>false</publish_sensor_pose>
        <publish_collision_pose>false</publish_collision_pose>
        <publish_visual_pose>false</publish_visual_pose>
        <publish_nested_model_pose>false</publish_nested_model_pose>
        <publish_model_pose>true</publish_model_pose>
        <use_pose_vector_msg>true</use_pose_vector_msg>
        <static_publisher>false</static_publisher>
        <update_frequency>{freq}</update_frequency>
      </plugin>'''


def _rolling_resistance(c):
    """gz-sim ignores <velocity_decay> and friction (mu) only resists sliding, so a rolling ball never loses
    energy.  The Hydrodynamics system with only a linear damping term (negative = damping, Fossen notation)
    gives F = -c * v on the ball, like carpet rolling resistance: time constant ~ 1.4 m / c."""
    if c <= 0:
        return ''
    return f'''
      <plugin filename="gz-sim-hydrodynamics-system" name="gz::sim::systems::Hydrodynamics">
        <link_name>ball_link</link_name>
        <xU>{-c}</xU><yV>{-c}</yV><zW>{-c}</zW>
        <disable_coriolis>true</disable_coriolis>
        <disable_added_mass>true</disable_added_mass>
      </plugin>'''


def build_world_sdf(s: FieldSpec, mesh_dir='/tmp/pinky_soccer/rcj_meshes', step=0.004) -> str:
    """step: physics step [s].  Gazebo publishes /clock every step and every use_sim_time node wakes up for
    it, so 1 ms (1 kHz) saturates a laptop CPU; 4 ms keeps DiffDrive / ball contacts stable."""
    field = _box_field_model(s) if s.style == 'box' else _rcj_field_model(s, mesh_dir)
    r, m = s.ball_radius, s.ball_mass
    ball_i = 0.4 * m * r * r
    return f'''<?xml version="1.0"?>
<!-- AUTO-GENERATED by pinky_soccer_sim.field  ({s.style} field {s.length} x {s.width} m, goal {s.goal_width} m) -->
<sdf version="1.9">
  <world name="{s.world_name}">
    <physics name="fast_step" type="ignored">
      <max_step_size>{step}</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <gravity>0 0 -9.81</gravity>
    <scene><ambient>0.55 0.55 0.55 1</ambient><background>0.12 0.12 0.14 1</background></scene>

    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 5 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>-0.3 0.2 -1.0</direction>
    </light>
{field}

    <!-- orange golf ball (the real match ball) -->
    <model name="golf_ball">
      <pose>0 0 {r + 0.002:.4f} 0 0 0</pose>
      <link name="ball_link">
        <inertial>
          <mass>{m}</mass>
          <inertia><ixx>{ball_i:.3e}</ixx><iyy>{ball_i:.3e}</iyy><izz>{ball_i:.3e}</izz>
                   <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
        </inertial>
        <collision name="ball_col">
          <geometry><sphere><radius>{r}</radius></sphere></geometry>
          <surface>
            <friction><ode><mu>0.4</mu><mu2>0.4</mu2></ode></friction>
            <bounce><restitution_coefficient>0.3</restitution_coefficient></bounce>
          </surface>
        </collision>
        <visual name="ball_vis">
          <geometry><sphere><radius>{r}</radius></sphere></geometry>
          <material><ambient>1.0 0.45 0.05 1</ambient><diffuse>1.0 0.45 0.05 1</diffuse><specular>0.3 0.3 0.3 1</specular></material>
        </visual>
      </link>{_pose_publisher()}{_rolling_resistance(s.ball_damping)}
    </model>
  </world>
</sdf>
'''


# ---------------------------------------------------------------- Nav2 map
def build_map(s: FieldSpec, resolution=0.02, margin=0.20):
    """Rasterise the field into a Nav2 occupancy map.  Returns (width_px, height_px, origin_xy, pixels)."""
    hx, hy = outer_half_extents(s)
    x_ext, y_ext = hx + margin, hy + margin
    wpx = int(round(2 * x_ext / resolution))
    hpx = int(round(2 * y_ext / resolution))
    ox, oy = -x_ext, -y_ext
    if s.style != 'box':
        boxes = rcj_obstacle_boxes(s)
        c = s.rcj_cfg()
        free_x, free_y = s.half_l + c['outer_strip'], s.half_w + c['outer_strip']
    else:
        boxes = [(n, cx, cy, sx, sy) for n, cx, cy, sx, sy, _t in wall_boxes(s)]
        free_x, free_y = s.half_l, s.half_w
    half = resolution / 2.0
    pix = bytearray(wpx * hpx)
    for row in range(hpx):
        y = oy + (hpx - 1 - row + 0.5) * resolution
        for col in range(wpx):
            x = ox + (col + 0.5) * resolution
            val = 205                                   # unknown
            if abs(x) <= free_x and abs(y) <= free_y:
                val = 254                               # free
            for _, cx, cy, sx, sy in boxes:
                if abs(x - cx) <= sx / 2 + half and abs(y - cy) <= sy / 2 + half:
                    val = 0                             # wall
                    break
            pix[row * wpx + col] = val
    return wpx, hpx, (ox, oy), bytes(pix)


def write_map(s: FieldSpec, out_prefix: str, resolution=0.02):
    wpx, hpx, (ox, oy), pix = build_map(s, resolution)
    pgm = out_prefix + '.pgm'
    with open(pgm, 'wb') as f:
        f.write(f'P5\n{wpx} {hpx}\n255\n'.encode())
        f.write(pix)
    with open(out_prefix + '.yaml', 'w') as f:
        f.write(f'image: {os.path.basename(pgm)}\nmode: trinary\nresolution: {resolution}\n'
                f'origin: [{ox:.3f}, {oy:.3f}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
    return out_prefix + '.yaml'
