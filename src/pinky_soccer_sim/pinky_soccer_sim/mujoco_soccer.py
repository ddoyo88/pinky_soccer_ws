#!/usr/bin/env python3
"""MuJoCo soccer simulator: the field from config/field_<style>.yaml, the orange golf ball and TWO Pinky Pro
robots (yellow / blue), each publishing what the real robot publishes, under its own namespace and in its own
ROS domain - exactly the real two-robot layout on one PC:

    yellow (domain 37):  /yellow/cmd_vel  /yellow/odom  /yellow/tf  /yellow/scan  /yellow/joint_states
                         /yellow/imu  /yellow/camera/image_raw  /yellow/camera/camera_info   (+ /clock)
    blue   (domain 38):  the same under /blue/
    referee (domain 37): /gz/dynamic_pose (ground truth, same message as the Gazebo bridge)
                         /sim/set_pose    (geometry_msgs/PoseStamped, header.frame_id = model name) to place
                                          the ball / lift a robot - the Gazebo set_pose service equivalent

The robot model is the official pinky_mujoco conversion of our URDF (bumper + team LED), placed twice with
prefixes 'yellow/' and 'blue/'.  Wheels are velocity motors with the real Dynamixel ramp (pinky_mujoco
config/mujoco.yaml), odometry is integrated from the wheel encoders, the lidar is ray-cast at its real height
(it sees walls and the other robot, not the 4 cm ball - like the real RPLIDAR), cameras are rendered offscreen.

    ros2 run pinky_soccer_sim mujoco_soccer -- --viewer          (normally started by sim_duel.launch.py)
"""
import argparse
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, Imu, JointState, LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from pinky_soccer_sim import rcj_field
from pinky_soccer_sim.field import FieldSpec, resolve_dims

BALL_R, BALL_M = 0.02135, 0.0459
# collision bits: 1 = world (walls, goals, carpet), 2 = robots, 4 = ball
WORLD_BITS, ROBOT_BITS, BALL_BITS = (1, 2 | 4), (2, 1 | 2 | 4), (4, 1 | 2)
LIDAR_GROUP = {'yellow': 1, 'blue': 3}         # each robot's collision shapes; the other robot's lidar sees them
JOINTS = ['l_wheel_joint', 'r_wheel_joint', 'caster_rotate_joint', 'caster_wheel_joint']


def stamp(t):
    s = int(t)
    return Time(sec=s, nanosec=int((t - s) * 1e9))


def yaw_quat(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]           # MuJoCo w x y z


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


# ------------------------------------------------------------------------------------------------ model
def build_scene(out_dir, style, cam_tilt, poses, ball_roll=0.00035, markings=True, hull_verts=0, visuals=True):
    """Field + ball + yellow/blue Pinky -> compiled MjModel.  markings=False leaves out the painted lines (visual
    only).  hull_verts > 0 caps the collision hulls of the robot meshes (up to 11k vertices: the physics of
    several processes then fights over the CPU cache and does not scale - RL training uses 64).  visuals=False
    drops every non-colliding geom and unused mesh (RL training: nothing is rendered)."""
    from ament_index_python.packages import get_package_share_directory
    from pinky_mujoco.model import build_model
    from pinky_soccer_sim.urdf_tools import build_urdf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'empty.yaml').write_text('boxes: []\n')
    share = get_package_share_directory('pinky_mujoco')
    scenes = {}
    for team in ('yellow', 'blue'):     # one conversion per team: the LED colour lives in the URDF
        urdf = build_urdf(namespace='', cam_tilt_deg=cam_tilt, team=team, camera=(640, 360, 15), odom_tf=False)
        scenes[team] = build_model(share, out / team, urdf=urdf, world=out / 'empty.yaml', simple_render=True)

    spec = mujoco.MjSpec.from_file(str(scenes['yellow']))      # options, defaults, light from the official model
    spec.delete(spec.body('base_link'))
    for g in list(spec.worldbody.geoms):
        if g.name == 'floor':
            g.rgba = [0.3, 0.3, 0.32, 1]                            # room floor around the field
            g.pos = [0, 0, -0.021]

    # --- field (same boxes as the Gazebo world)
    L, W, G, strip = resolve_dims(style)
    cfg = FieldSpec(length=L, width=W, goal_width=G, style=style).rcj_cfg()
    for i, o in enumerate(rcj_field.scene(cfg)):
        if not markings and not o['collision']:
            continue
        if o['kind'] == 'cylinder':                                 # neutral spots
            shape, size = mujoco.mjtGeom.mjGEOM_CYLINDER, [o['radius'], o['length'] / 2, 0]
        else:
            shape, size = mujoco.mjtGeom.mjGEOM_BOX, [s / 2 for s in o['size']]
        g = spec.worldbody.add_geom(name=f'field_{i}_{o["name"]}', type=shape, size=size, pos=list(o['pos']),
                                    quat=yaw_quat(o.get('yaw', 0.0)), rgba=list(o['color']))
        if o['collision']:
            g.contype, g.conaffinity, g.group = WORLD_BITS[0], WORLD_BITS[1], 0
            g.friction = [float(cfg.get('friction', 0.8)), 0.005, 0.0001]
        else:
            g.contype, g.conaffinity, g.group = 0, 0, 2
            g.pos = [o['pos'][0], o['pos'][1], o['pos'][2] + 0.0005]   # lines just above the carpet

    # --- ball: rolling friction gives the slow-down of a golf ball on carpet
    #     decel ~ ball_roll * g / r * 5/7: 0.0003 m -> ~0.1 m/s^2, a 0.5 m/s push rolls ~1.3 m
    ball = spec.worldbody.add_body(name='golf_ball', pos=[0, 0, BALL_R + 0.002])
    ball.add_freejoint(name='ball_joint')
    ball.explicitinertial = True                     # the official model compiles with inertiafromgeom=false
    ball.mass = BALL_M
    ball.inertia = [0.4 * BALL_M * BALL_R ** 2] * 3    # (nearly) solid sphere
    ball.ipos = [0.0, 0.0, 0.0]                        # centre of mass = ball centre (else it took the body
    ball.iquat = [1.0, 0.0, 0.0, 0.0]                  # height: a lopsided ball that wobbled instead of rolling)
    bg = ball.add_geom(name='ball', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL_R, 0, 0],
                       rgba=[1.0, 0.45, 0.05, 1], condim=6, friction=[0.5, 0.005, ball_roll])
    # group 0 = drawn by cameras/viewer; the lidar plane (15.7 cm) is far above a 4 cm ball anyway
    bg.contype, bg.conaffinity, bg.group = BALL_BITS[0], BALL_BITS[1], 0

    # --- robots
    for team, (x, y, yaw) in poses.items():
        child = mujoco.MjSpec.from_file(str(scenes[team]))
        frame = spec.worldbody.add_frame(pos=[x, y, 0.03], quat=yaw_quat(yaw))
        frame.attach_body(child.body('base_link'), f'{team}/', '')
        # team LED (WS2812 strip at the rear top): the official conversion drops the URDF colour -> glowing
        # team-coloured material on the lamp's visual geoms
        rgba = {'yellow': [1.0, 0.82, 0.0, 1], 'blue': [0.05, 0.35, 1.0, 1]}[team]
        spec.add_material(name=f'{team}_led', rgba=rgba, emission=0.8)
        for g in spec.body(f'{team}/robot_lamp').geoms:
            if g.contype == 0:
                g.material = f'{team}_led'
                g.rgba = rgba
        # the official model has no collision shape on the RPLIDAR head, so the other robot's lidar (same
        # height) would look straight over it; the real C1 head is a 55.6 mm cylinder
        head = spec.body(f'{team}/rplidar_link').add_geom(
            name=f'{team}/rplidar_head_collision', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.028, 0.02, 0],
            rgba=[0.1, 0.1, 0.1, 1])        # opaque: rays skip alpha-0 geoms (hidden groups anyway)
        head.contype, head.conaffinity, head.group = ROBOT_BITS[0], ROBOT_BITS[1], LIDAR_GROUP[team]
    if not visuals:
        for b in spec.bodies:
            for g in list(b.geoms):
                if g.contype == 0 and g.conaffinity == 0:
                    spec.delete(g)
        used = {g.meshname for b in spec.bodies for g in b.geoms if g.meshname}
        for mesh in list(spec.meshes):
            if mesh.name not in used:
                spec.delete(mesh)
    if hull_verts:
        for mesh in spec.meshes:
            mesh.maxhullvert = int(hull_verts)
    compiled = spec.compile()                                  # panel pose from a first compile (mesh recentring)
    for team in SCREEN_IMAGES if visuals else ():
        _add_screen(spec, team, compiled)
    model = spec.compile()
    # detailed robot meshes (visual only, 742k vertices) -> geom group 4: the viewer draws them, the robot
    # cameras skip them (a camera frame took 22 ms with them, 9 ms without; field lines stay in group 2)
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_group[i] == 2 and \
                model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0:
            model.geom_group[i] = 4
    # robot collision shapes: collide with each other too, and sit in a per-team group for the lidars
    for i in range(model.ngeom):
        name = model.geom(i).name
        team = name.split('/', 1)[0] if '/' in name else None
        if team in LIDAR_GROUP and model.geom_contype[i] == 2:
            model.geom_conaffinity[i] = ROBOT_BITS[1]
            model.geom_group[i] = LIDAR_GROUP[team]
    return model, (L, W, G, strip)


SCREEN_IMAGES = {                                                 # pinky_soccer_sim/media; flag = normal, goal = celebration
    'blue': {'flag': 'taegukgi.png', 'goal': 'son.png'},
    'yellow': {'flag': 'portugal.png', 'goal': 'ronaldo.png'},
}


def _media(name):
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('pinky_soccer_sim'), 'media', name)


def _add_screen(spec, team, compiled):
    """The Pinky's front LCD (57 x 41 mm, landscape) as a thin textured box over the model's panel, with one
    material per picture; SoccerSim switches the material while the robot celebrates."""
    name = f'{team}/screen_mount_visual_0_3'
    panel = next(g for g in spec.body(f'{team}/screen_mount').geoms if g.name == name)
    gid = compiled.geom(name).id                       # spec pos = mesh file origin; compiled = recentred panel
    for key, file in SCREEN_IMAGES[team].items():
        path = _media(file)
        if not os.path.isfile(path):                   # player photos are not in the repository: flag instead
            path = _media(SCREEN_IMAGES[team]['flag'])
        spec.add_texture(name=f'{team}_screen_{key}', type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
        mat = spec.add_material(name=f'{team}_screen_{key}', emission=0.6)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f'{team}_screen_{key}'
    # panel frame: local x = outward normal, y = up the screen (41 mm), z = across (57 mm)
    q = compiled.geom_quat[gid].copy()
    normal = np.zeros(3)
    mujoco.mju_rotVecQuat(normal, np.array([1.0, 0.0, 0.0]), q)
    pos = compiled.geom_pos[gid] + normal * 0.0008             # local +x is the side facing out
    # MuJoCo puts a 2D texture on a box through its local x-y plane: box x = across the panel (panel z),
    # box y = up the panel (panel y; checked with a rendered test image), box z = outward normal (panel x)
    rot = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])      # columns: box axes in panel frame
    qr = np.zeros(4)
    mujoco.mju_mat2Quat(qr, rot.flatten())
    qbox = np.zeros(4)
    mujoco.mju_mulQuat(qbox, q, qr)
    panel.rgba = [0, 0, 0, 0]                                  # the grey panel would cover the picture
    scr = spec.body(f'{team}/screen_mount').add_geom(
        name=f'{team}/screen_image', type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0283, 0.0203, 0.0003],
        pos=pos.tolist(), quat=qbox.tolist(), material=f'{team}_screen_flag', rgba=[1, 1, 1, 1])
    scr.contype, scr.conaffinity, scr.group = 0, 0, 2


# ------------------------------------------------------------------------------------------------ robots
class Robot:
    """One Pinky in the shared simulation: wheel control, encoder odometry, lidar, camera, its ROS topics."""

    def __init__(self, sim, team, node, settings, camera):
        m = sim.model
        self.sim, self.team, self.node, self.s = sim, team, node, settings
        p = f'{team}/'
        self.ctrl = [m.actuator(f'{p}l_motor').id, m.actuator(f'{p}r_motor').id]
        self.radius = float(m.geom(f'{p}l_wheel_collision_0').size[0])
        self.track = float(abs(m.body(f'{p}l_wheel').pos[1] + m.geom(f'{p}l_wheel_collision_0').pos[1]
                               - m.body(f'{p}r_wheel').pos[1] - m.geom(f'{p}r_wheel_collision_0').pos[1]))
        self.q = [m.joint(p + j).qposadr[0] for j in JOINTS]
        self.v = [m.joint(p + j).dofadr[0] for j in JOINTS]
        free = m.joint(f'{p}floating_base')
        self.free_q, self.free_v = free.qposadr[0], free.dofadr[0]
        self.lidar = m.site(f'{p}lidar').id
        self.lidar_body = m.site_bodyid[self.lidar]
        self.cam_name = f'{p}front_camera'
        samples = settings['lidar']['samples']
        self.angles = np.linspace(-math.pi, math.pi, samples, endpoint=False)
        self.rays = np.column_stack((np.cos(self.angles), np.sin(self.angles), np.zeros(samples)))
        group = np.zeros(6, dtype=np.uint8)
        group[0] = 1
        group[LIDAR_GROUP['blue' if team == 'yellow' else 'yellow']] = 1       # walls + the OTHER robot
        self.ray_group = group
        self.odom = np.zeros(3)
        self.twist = np.zeros(2)
        self.prev_wheels = sim.data.qpos[self.q[:2]].copy()
        self.prev_yaw = self._true_yaw()
        self.cmd, self.cmd_t = (0.0, 0.0), -math.inf
        node.create_subscription(Twist, f'/{team}/cmd_vel', self._cmd, 10)
        self.odom_pub = node.create_publisher(Odometry, f'/{team}/odom', 10)
        self.joint_pub = node.create_publisher(JointState, f'/{team}/joint_states', 10)
        self.scan_pub = node.create_publisher(LaserScan, f'/{team}/scan', qos_profile_sensor_data)
        self.imu_pub = node.create_publisher(Imu, f'/{team}/imu', 10)
        self.tf_pub = node.create_publisher(TFMessage, f'/{team}/tf', 10)
        self.camera = camera
        if camera:
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
            self.img_pub = node.create_publisher(Image, f'/{team}/camera/image_raw', qos)
            self.info_pub = node.create_publisher(CameraInfo, f'/{team}/camera/camera_info', qos)
            w, h = camera['width'], camera['height']
            fovy = math.radians(float(m.cam(self.cam_name).fovy[0]))
            f = (h / 2) / math.tan(fovy / 2)
            self.info = CameraInfo(width=w, height=h, distortion_model='plumb_bob', d=[0.0] * 5,
                                   k=[f, 0.0, w / 2, 0.0, f, h / 2, 0.0, 0.0, 1.0],
                                   r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                   p=[f, 0.0, w / 2, 0.0, 0.0, f, h / 2, 0.0, 0.0, 0.0, 1.0, 0.0])
            self.info.header.frame_id = 'front_camera_link'

    def _cmd(self, msg):
        c = self.s['control']
        if math.isfinite(msg.linear.x) and math.isfinite(msg.angular.z):
            self.cmd = (max(-c['max_linear'], min(c['max_linear'], msg.linear.x)),
                        max(-c['max_angular'], min(c['max_angular'], msg.angular.z)))
            self.cmd_t = time.monotonic()

    def wheel_targets(self):
        v, w = self.cmd if time.monotonic() - self.cmd_t < self.s['control']['command_timeout'] else (0.0, 0.0)
        left = (v - w * self.track / 2) / self.radius
        right = (v + w * self.track / 2) / self.radius
        mx = self.s['motors']['max_speed']
        return np.clip([left, right], -mx, mx)

    def apply(self, target, limit):
        d = self.sim.data
        d.ctrl[self.ctrl] = d.ctrl[self.ctrl] + np.clip(target - d.ctrl[self.ctrl], -limit, limit)

    def _true_yaw(self):
        return quat_yaw(self.sim.data.qpos[self.free_q + 3:self.free_q + 7])

    def integrate(self, dt):
        """Gyro odometry: distance from the wheel encoders, heading change from the IMU (BNO055 on the real
        Pinky).  Wheel-only heading drifts badly when the robots shove each other or spin on the spot (wheel
        slip) and AMCL then flips to the mirror image of the point-symmetric field."""
        wheels = self.sim.data.qpos[self.q[:2]]
        dl, dr = (wheels - self.prev_wheels) * self.radius
        self.prev_wheels = wheels.copy()
        yaw = self._true_yaw()
        dist, ang = (dl + dr) / 2, math.atan2(math.sin(yaw - self.prev_yaw), math.cos(yaw - self.prev_yaw))
        self.prev_yaw = yaw
        th = self.odom[2] + ang / 2
        self.odom += [dist * math.cos(th), dist * math.sin(th), ang]
        self.twist[:] = [dist / dt, ang / dt]

    def publish_state(self, now):
        d = self.sim.data
        x, y, th = self.odom
        q = [math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)]
        o = Odometry()
        o.header.stamp, o.header.frame_id, o.child_frame_id = now, 'odom', 'base_footprint'
        o.pose.pose.position.x, o.pose.pose.position.y = float(x), float(y)
        o.pose.pose.orientation.w, o.pose.pose.orientation.z = q[0], q[3]
        o.twist.twist.linear.x, o.twist.twist.angular.z = map(float, self.twist)
        for i, var in enumerate([0.0025, 0.0025, 1e6, 1e6, 1e6, 0.01]):
            o.pose.covariance[7 * i] = var
            o.twist.covariance[7 * i] = var
        self.odom_pub.publish(o)
        t = TransformStamped()
        t.header = o.header
        t.child_frame_id = 'base_footprint'
        t.transform.translation.x, t.transform.translation.y = float(x), float(y)
        t.transform.rotation = o.pose.pose.orientation
        self.tf_pub.publish(TFMessage(transforms=[t]))
        js = JointState()
        js.header.stamp = now
        js.name = JOINTS
        js.position = d.qpos[self.q].tolist()
        js.velocity = d.qvel[self.v].tolist()
        self.joint_pub.publish(js)
        imu = Imu()
        imu.header.stamp, imu.header.frame_id = now, 'imu_link'
        bq = d.qpos[self.free_q + 3:self.free_q + 7]
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = map(float, bq)
        imu.angular_velocity.z = float(d.qvel[self.free_v + 5])            # body-frame yaw rate
        imu.orientation_covariance[0] = imu.angular_velocity_covariance[0] = 0.001
        self.imu_pub.publish(imu)

    def publish_scan(self, now):
        m, d = self.sim.model, self.sim.data
        rot = d.site_xmat[self.lidar].reshape(3, 3)
        rays = np.ascontiguousarray(self.rays @ rot.T)
        n = len(rays)
        dist = np.empty(n, dtype=np.float64)
        gid = np.empty(n, dtype=np.int32)
        mujoco.mj_multiRay(m, d, d.site_xpos[self.lidar], rays.ravel(), self.ray_group, True,
                           self.lidar_body, gid, dist, None, n, self.s['lidar']['range_max'])
        lo, hi = self.s['lidar']['range_min'], self.s['lidar']['range_max']
        dist[(dist < lo) | (dist > hi)] = np.inf
        s = LaserScan()
        s.header.stamp, s.header.frame_id = now, 'rplidar_link'
        s.angle_min, s.angle_max = float(self.angles[0]), float(self.angles[-1])
        s.angle_increment = float(self.angles[1] - self.angles[0])
        s.scan_time, s.range_min, s.range_max = 1.0 / self.s['rates']['scan_hz'], float(lo), float(hi)
        s.ranges = dist.astype('float32').tolist()
        self.scan_pub.publish(s)

    def publish_image(self, renderer, option, now):
        renderer.update_scene(self.sim.data, camera=self.cam_name, scene_option=option)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
            renderer.scene.flags[flag] = 0
        rgb = renderer.render()
        self.last_rgb = rgb                              # for the match video (video_hud)
        img = Image()
        img.header.stamp, img.header.frame_id = now, 'front_camera_link'
        img.height, img.width, img.encoding, img.step = rgb.shape[0], rgb.shape[1], 'rgb8', rgb.shape[1] * 3
        img.data = rgb.tobytes()
        self.img_pub.publish(img)
        self.info.header.stamp = now
        self.info_pub.publish(self.info)


# ------------------------------------------------------------------------------------------------ simulator
class SoccerSim:
    def __init__(self, model, domains, settings, camera):
        self.model, self.data = model, mujoco.MjData(model)
        mujoco.mj_forward(model, self.data)
        self.s = settings
        self.nodes, self.execs = {}, []
        for team, dom in domains.items():
            ctx = Context()
            rclpy.init(context=ctx, domain_id=dom)
            node = Node(f'mujoco_soccer_{team}', context=ctx)
            ex = SingleThreadedExecutor(context=ctx)
            ex.add_node(node)
            self.nodes[team], self.execs = node, self.execs + [ex]
        self.robots = {t: Robot(self, t, self.nodes[t], settings, camera) for t in domains}
        self.clock_pubs = [n.create_publisher(Clock, '/clock', 10) for n in self.nodes.values()]
        ref = self.nodes['yellow']                                     # referee lives in the yellow domain
        self.gt_pub = ref.create_publisher(TFMessage, '/gz/dynamic_pose', 10)
        ref.create_subscription(PoseStamped, '/sim/set_pose', self._set_pose, 10)
        self.pending_poses = []
        # match information for the recorded video (--record): score, attack clock, referee messages
        self.hud_info = {'score': (0, 0), 'clock': '', 'events': [], 'phase': ''}

        def _score(m):
            import re
            g = re.fullmatch(r'YELLOW\s+(\d+)\s*-\s*(\d+)\s+BLUE', m.data.strip())
            if g:
                self.hud_info['score'] = (int(g.group(1)), int(g.group(2)))

        def _event(m):
            self.hud_info['events'] = (self.hud_info['events'] + [m.data])[-2:]
        ref.create_subscription(String, '/game/score', _score, 10)
        ref.create_subscription(String, '/game/attack_clock', lambda m: self.hud_info.__setitem__('clock', m.data), 10)
        ref.create_subscription(String, '/game/event', _event, 10)
        ref.create_subscription(String, '/game/phase', lambda m: self.hud_info.__setitem__('phase', m.data.strip()), 10)
        ref.create_subscription(String, '/game/clock', lambda m: self.hud_info.__setitem__('time', m.data.strip()), 10)
        # front LCD pictures: flag normally, the goal picture while the robot celebrates its goal
        self.screens = {}
        for team in self.robots:
            try:
                gid = model.geom(f'{team}/screen_image').id
            except KeyError:
                continue
            mats = {k: model.material(f'{team}_screen_{k}').id for k in SCREEN_IMAGES.get(team, {})}
            self.screens[team] = (gid, mats)
            self.nodes[team].create_subscription(
                String, f'/{team}/game/state', lambda m, t=team: self._screen(t, m.data), 10)
        self.models = {'golf_ball': model.joint('ball_joint'),
                       'pinky_yellow': model.joint('yellow/floating_base'),
                       'pinky_blue': model.joint('blue/floating_base')}
        r = settings['rates']
        self.period = 1.0 / r['bridge_hz']
        self.steps = round(self.period / model.opt.timestep)
        self.state_ticks = round(r['bridge_hz'] / r['state_hz'])
        self.scan_ticks = round(r['bridge_hz'] / r['scan_hz'])
        self.cam_period = 1.0 / camera['fps'] if camera else None
        self.last_cam = -math.inf
        self.renderer, self.option = None, None
        self.camera = camera
        self.tick = 0
        ref.get_logger().info(
            f'MuJoCo soccer ready: {", ".join(f"{t} -> domain {d}" for t, d in domains.items())}, '
            f'step {model.opt.timestep * 1000:.0f} ms, camera {camera["width"]}x{camera["height"]} @ '
            f'{camera["fps"]} Hz' if camera else 'no cameras')

    def _screen(self, team, state):
        gid, mats = self.screens[team]
        self.model.geom_matid[gid] = mats['goal' if state.startswith('CELEBRATE') else 'flag']

    def _set_pose(self, msg):
        self.pending_poses.append(msg)                                 # applied between physics steps

    def _apply_poses(self):
        while self.pending_poses:
            msg = self.pending_poses.pop(0)
            j = self.models.get(msg.header.frame_id)
            if j is None:
                continue
            qa, va = j.qposadr[0], j.dofadr[0]
            p, o = msg.pose.position, msg.pose.orientation
            self.data.qpos[qa:qa + 7] = [p.x, p.y, p.z, o.w, o.x, o.y, o.z]
            self.data.qvel[va:va + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def spin_ros(self):
        for ex in self.execs:
            ex.spin_once(timeout_sec=0.0)

    def advance(self):
        self._apply_poses()
        limit = self.s['motors'].get('wheel_accel', 1e9) * self.model.opt.timestep
        targets = {t: r.wheel_targets() for t, r in self.robots.items()}
        T = self.prof = getattr(self, 'prof', {})
        t0 = time.perf_counter()
        for _ in range(self.steps):
            for t, r in self.robots.items():
                r.apply(targets[t], limit)
            mujoco.mj_step(self.model, self.data)
        t1 = time.perf_counter()
        T['physics'] = T.get('physics', 0.0) + t1 - t0
        for r in self.robots.values():
            r.integrate(self.period)
        now = stamp(self.data.time)
        for p in self.clock_pubs:
            p.publish(Clock(clock=now))
        self.tick += 1
        if self.tick % self.state_ticks == 0:
            for r in self.robots.values():
                r.publish_state(now)
            self.publish_truth(now)
        if self.tick % self.scan_ticks == 0:
            for r in self.robots.values():
                r.publish_scan(now)
        t2 = time.perf_counter()
        T['publish'] = T.get('publish', 0.0) + t2 - t1
        if self.camera and self.data.time - self.last_cam >= self.cam_period - 1e-9:
            self.last_cam = self.data.time
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, self.camera['height'], self.camera['width'])
                self.option = mujoco.MjvOption()
                self.option.geomgroup[1] = self.option.geomgroup[3] = 0      # hide collision shapes
                self.option.geomgroup[4] = 0                                  # and the robot meshes
            for r in self.robots.values():
                r.publish_image(self.renderer, self.option, now)
            T['camera'] = T.get('camera', 0.0) + time.perf_counter() - t2

    def publish_truth(self, now):
        msg = TFMessage()
        for name, j in self.models.items():
            q = self.data.qpos[j.qposadr[0]:j.qposadr[0] + 7]
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = now, 'soccer', name
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = map(float, q[:3])
            t.transform.rotation.w, t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z = \
                map(float, q[3:7])
            msg.transforms.append(t)
        self.gt_pub.publish(msg)

    def shutdown(self):
        for n in self.nodes.values():
            ctx = n.context
            n.destroy_node()
            if ctx.ok():
                rclpy.shutdown(context=ctx)


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--viewer', action='store_true')
    ap.add_argument('--no-camera', action='store_true')
    # 2026-11-25: 640x360 @ 15 Hz for two cameras (22 ms a frame with the robot meshes) plus physics took longer
    # than real time - the match ran at 0.39x and the robots looked slow; 424x240 @ 10 Hz keeps up
    ap.add_argument('--cam-size', default='424x240', help='WxH of the simulated robot cameras')
    ap.add_argument('--cam-fps', type=float, default=10.0)
    ap.add_argument('--bridge-hz', type=float, default=50.0)
    ap.add_argument('--simple-robots', action='store_true', help='viewer: robots as simple shapes (faster)')
    # a video straight from the simulator, frames taken at fixed SIMULATION time steps: smooth and at the right
    # speed even when the PC is busy (a screen recorder made the sim stutter, 2026-11-25)
    ap.add_argument('--record', default='', help='write the match to this video file (.mp4)')
    ap.add_argument('--record-fps', type=float, default=20.0)
    ap.add_argument('--record-size', default='960x540')
    ap.add_argument('--record-plain', action='store_true', help='video without the match information overlay')
    ap.add_argument('--viewer-hz', type=float, default=10.0, help='viewer redraws per wall second')
    ap.add_argument('--field-style', default='hardboard')
    ap.add_argument('--yellow-domain', type=int, default=37)
    ap.add_argument('--blue-domain', type=int, default=38)
    ap.add_argument('--yellow-pose', default='auto', help='x,y,yaw (auto = -L/4, 0, 0)')
    ap.add_argument('--blue-pose', default='auto', help='x,y,yaw (auto = +L/4, 0, pi)')
    ap.add_argument('--cam-tilt', type=float, default=20.0)
    ap.add_argument('--max-linear', type=float, default=0.30)
    ap.add_argument('--ball-roll', type=float, default=0.00035, help='rolling friction [m]; larger = stops sooner')
    ap.add_argument('--output-dir', default='/tmp/pinky_soccer/mujoco')
    opt, _ = ap.parse_known_args(args)

    from pinky_mujoco.model import load_settings
    settings = load_settings(None)
    # real time (2026-11-25: the match ran at 0.47x - the robots looked slow): 50 Hz bridge ticks (10 physics
    # steps of 2 ms each, unchanged physics), odometry / TF at 25 Hz - enough for AMCL and the striker
    settings['rates']['bridge_hz'] = opt.bridge_hz
    settings['rates']['state_hz'] = min(settings['rates']['state_hz'], 20.0)
    settings['control']['max_linear'] = opt.max_linear
    settings['control']['max_angular'] = 4.0
    L = resolve_dims(opt.field_style)[0]

    def pose(text, default):
        return default if text == 'auto' else tuple(float(v) for v in text.split(','))
    poses = {'yellow': pose(opt.yellow_pose, (-L / 4, 0.0, 0.0)), 'blue': pose(opt.blue_pose, (L / 4, 0.0, math.pi))}
    model, _ = build_scene(opt.output_dir, opt.field_style, opt.cam_tilt, poses, ball_roll=opt.ball_roll)
    cw, ch = (int(v) for v in opt.cam_size.lower().split('x'))
    camera = None if opt.no_camera else {'width': cw, 'height': ch, 'fps': opt.cam_fps}
    sim = SoccerSim(model, {'yellow': opt.yellow_domain, 'blue': opt.blue_domain}, settings, camera)
    # Ctrl-C / launch shutdown (SIGINT, SIGTERM): leave the main loop normally, so the recorded video is finished
    # (rclpy's own handler only shut the ROS contexts down - the loop ran on and the .mp4 stayed unplayable)
    import signal
    stop = {'now': False}

    def _stop(*_):
        stop['now'] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    viewer = None
    try:
        if opt.viewer:
            import importlib
            viewer = importlib.import_module('mujoco.viewer').launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False)

            def place():
                viewer.cam.lookat[:] = [0, 0, 0]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.6, 90, -60
                viewer.opt.geomgroup[1] = viewer.opt.geomgroup[3] = 0
                # robot meshes in the viewer cost ~20 ms a redraw (742k vertices): --simple-robots shows the
                # collision shapes instead (the match keeps real time)
                viewer.opt.geomgroup[4] = 0 if opt.simple_robots else 1
                # both teams' collision shapes (yellow = group 1, blue = group 3: LIDAR_GROUP)
                viewer.opt.geomgroup[1] = viewer.opt.geomgroup[3] = 1 if opt.simple_robots else 0
            place()
        deadline = time.monotonic()
        rec = None
        if opt.record:
            import cv2
            rw, rh = (int(v) for v in opt.record_size.lower().split('x'))
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, rw)
            model.vis.global_.offheight = max(model.vis.global_.offheight, rh)
            rcam = mujoco.MjvCamera()
            rcam.lookat[:] = [0, 0, 0]
            rcam.distance, rcam.azimuth, rcam.elevation = 2.6, 90, -60
            ropt = mujoco.MjvOption()
            ropt.geomgroup[1] = ropt.geomgroup[3] = 0      # collision shapes off, detailed robot meshes on
            ropt.geomgroup[4] = 1
            path = os.path.expanduser(opt.record)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            rec = {'w': cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), opt.record_fps, (rw, rh)),
                   'r': mujoco.Renderer(model, rh, rw), 't': -1e9, 'cv2': cv2, 'hud': None}
            if not opt.record_plain:
                from pinky_soccer_sim.video_hud import VideoHud
                rec['hud'] = VideoHud(rw, rh)          # score, attack clock, referee messages, robot cameras
            print(f'recording the match to {path} ({rw}x{rh} @ {opt.record_fps:g} fps of simulation time)',
                  flush=True)
        prof_t, prof_sim = time.monotonic(), sim.data.time
        while (viewer is None or viewer.is_running()) and not stop['now']:
            ta = time.perf_counter()
            sim.spin_ros()
            tb = time.perf_counter()
            with viewer.lock() if viewer else nullcontext():
                sim.advance()
            tc = time.perf_counter()
            if rec is not None and sim.data.time - rec['t'] >= 1.0 / opt.record_fps - 1e-9:
                rec['t'] = sim.data.time
                rec['r'].update_scene(sim.data, camera=rcam, scene_option=ropt)
                frame = rec['r'].render()
                if rec['hud'] is not None:
                    cams = {t: getattr(r, 'last_rgb', None) for t, r in sim.robots.items()}
                    rec['w'].write(rec['hud'].compose(frame, cams, sim.hud_info))
                else:
                    rec['w'].write(rec['cv2'].cvtColor(frame, rec['cv2'].COLOR_RGB2BGR))
            if viewer and time.monotonic() - getattr(sim, 'view_t', 0.0) >= 1.0 / opt.viewer_hz:
                sim.view_t = time.monotonic()
                viewer.sync()
            P = sim.prof
            P['ros_spin'] = P.get('ros_spin', 0.0) + tb - ta
            P['advance(lock)'] = P.get('advance(lock)', 0.0) + tc - tb
            P['viewer'] = P.get('viewer', 0.0) + time.perf_counter() - tc
            if time.monotonic() - prof_t > 5.0:      # where the wall time goes (real-time factor < 1)
                wall = time.monotonic() - prof_t
                msg = (f'real-time factor {(sim.data.time - prof_sim) / wall:.2f}; per wall second: ' +
                       ', '.join(f'{k} {1000 * v / wall:.0f} ms' for k, v in P.items()))
                print(msg, flush=True)
                P.clear()
                prof_t, prof_sim = time.monotonic(), sim.data.time
                if sim.tick == 60:
                    with viewer.lock():
                        place()
            deadline += sim.period
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rec is not None:
                rec['w'].release()                        # finish the video file
                print(f'video saved: {opt.record}', flush=True)
        except Exception:
            pass
        try:
            if viewer:
                viewer.close()
            sim.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    main()
