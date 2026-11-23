#!/usr/bin/env python3
"""Camera calibration window for one soccer robot (real robot or simulation).

    ros2 run pinky_soccer_vision camera_calib --team blue                 real blue robot (domain 38)
    ros2 run pinky_soccer_vision camera_calib --team yellow               real yellow robot (domain 37)
    ros2 run pinky_soccer_vision camera_calib --team blue --ns /blue      sim_duel (namespaced topics)

Real robots: run it with the same DDS settings as real_duel (CycloneDDS + ~/cyclonedds_pc.xml); the launcher
script scripts/camera_calib.sh does that.

1) Ball colour (HSV): sliders for the orange window; 'mask' shows what the detector would see.
2) Camera pitch: put the ball straight ahead at a measured distance (from the robot centre = base_footprint;
   the bumper front is 4.9 cm ahead of it), type the distance, press 측정 -> the pitch correction that makes the
   computed distance match is found (30 frames averaged).  The robot model says the camera looks 8 deg down;
   the real mount usually differs by a few degrees, which made the ball look twice as close.
3) 저장 -> ~/.config/pinky_soccer/camera_<team>.yaml  (cam_pitch_deg + HSV).  real_duel.launch.py reads it.
4) Lens (fx, fy, distortion): checkerboard (inner corners), 수집 -> 계산; saved with 저장.
5) Wheels: '1 m 전진' drives 1 m by the robot's own odometry; measure the real distance with a tape ->
   wheel_radius = current * real / odom.  '1바퀴 회전' turns 360 deg by odometry; the real angle (IMU gyro,
   or a tape mark) -> wheel_separation.  Saved with 저장 (wheel_radius / wheel_separation); they belong to the
   ROBOT's driver: robot.launch.py ... wheel_radius:=<r> wheel_separation:=<s>.
"""
import argparse
import math
import os
import shutil
import subprocess
import threading
import time
import tkinter as tk

import cv2
import numpy as np
import rclpy
import yaml
from PIL import Image as PILImage
from PIL import ImageTk
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Imu
from tf2_msgs.msg import TFMessage

BALL_R = 0.02135
DEFAULTS = {'cam_pitch_deg': 0.0, 'hsv_h_min': 4, 'hsv_h_max': 18, 'hsv_s_min': 130, 'hsv_v_min': 80,
            'intrinsics': None, 'distortion': None, 'lens_rms': None, 'image_size': None,
            'wheel_radius': None, 'wheel_separation': None}
WHEEL_DEFAULT = {'wheel_radius': 0.027, 'wheel_separation': 0.0961}      # pinky_bringup pinky_params.yaml


def config_path(team):
    return os.path.expanduser(f'~/.config/pinky_soccer/camera_{team}.yaml')


def load_config(team):
    cfg = dict(DEFAULTS)
    try:
        with open(config_path(team)) as f:
            cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items() if k in DEFAULTS})
    except FileNotFoundError:
        pass
    return cfg


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class StaticTF:
    """Chain of static transforms (robot_state_publisher) without tf2: base_footprint -> camera frame."""

    def __init__(self):
        self.edges = {}                                  # child -> (parent, R, t)

    def update(self, msg):
        for tr in msg.transforms:
            q, p = tr.transform.rotation, tr.transform.translation
            self.edges[tr.child_frame_id.lstrip('/')] = (tr.header.frame_id.lstrip('/'),
                                                         quat_mat((q.x, q.y, q.z, q.w)),
                                                         np.array([p.x, p.y, p.z]))

    def base_to(self, frame, base='base_footprint'):
        R, t = np.eye(3), np.zeros(3)                    # pose of `frame` in `base`
        f = frame
        for _ in range(20):
            if f == base:
                return R, t
            if f not in self.edges:
                return None
            parent, Rp, tp = self.edges[f]
            R, t = Rp @ R, Rp @ t + tp
            f = parent
        return None


class CalibNode:
    def __init__(self, domain, ns, compressed):
        self.ctx = Context()
        rclpy.init(context=self.ctx, domain_id=domain)
        self.node = Node('camera_calib', context=self.ctx)
        self.lock = threading.Lock()
        self.frame, self.frame_id, self.K, self.raw_K = None, '', None, False
        self.tf = StaticTF()
        n = self.node
        if compressed:
            n.create_subscription(CompressedImage, f'{ns}/camera/image_raw/compressed', self._jpeg,
                                  qos_profile_sensor_data)
        else:
            n.create_subscription(Image, f'{ns}/camera/image_raw', self._raw, qos_profile_sensor_data)
        n.create_subscription(CameraInfo, f'{ns}/camera/camera_info', self._info, qos_profile_sensor_data)
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        static_qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                reliability=ReliabilityPolicy.RELIABLE)
        n.create_subscription(TFMessage, f'{ns}/tf_static', self.tf.update, static_qos)
        # wheel test: the robot's own odometry, the IMU gyro (truth for the turn), drive commands
        self.odom, self.imu_yaw, self._imu_t = None, 0.0, None
        n.create_subscription(Odometry, f'{ns}/odom', self._odom, qos_profile_sensor_data)
        n.create_subscription(Imu, f'{ns}/imu_raw', self._imu, qos_profile_sensor_data)
        self.cmd_pub = n.create_publisher(Twist, f'{ns}/cmd_vel', 10)
        self.ex = SingleThreadedExecutor(context=self.ctx)
        self.ex.add_node(n)
        threading.Thread(target=self.ex.spin, daemon=True).start()

    def _jpeg(self, m):
        bgr = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            with self.lock:
                self.frame, self.frame_id = bgr, m.header.frame_id
                self.stamp = time.time()

    def _raw(self, m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
        bgr = a[:, :, ::-1].copy() if m.encoding.startswith('rgb') else a.copy()
        with self.lock:
            self.frame, self.frame_id = bgr, m.header.frame_id
            self.stamp = time.time()

    def _info(self, m):
        if m.k[0] > 0:
            with self.lock:
                self.K, self.raw_K = (m.k[0], m.k[4], m.k[2], m.k[5]), True

    def _odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.odom = (p.x, p.y, yaw, time.time())

    def _imu(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        with self.lock:
            if self._imu_t is not None and 0.0 < t - self._imu_t < 0.2:
                self.imu_yaw += m.angular_velocity.z * (t - self._imu_t)
            self._imu_t = t

    def drive(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.cmd_pub.publish(t)

    def wheel_params(self):
        """The driver's current wheel_radius / wheel_separation (pinky_bringup node), defaults if unreachable."""
        out = dict(WHEEL_DEFAULT)
        try:
            from rclpy.parameter_client import AsyncParameterClient
            cli = AsyncParameterClient(self.node, 'pinky_bringup')
            if cli.wait_for_services(timeout_sec=2.0):
                fut = cli.get_parameters(list(out))
                t0 = time.time()
                while not fut.done() and time.time() - t0 < 3.0:
                    time.sleep(0.05)
                if fut.done() and fut.result() is not None:
                    for k, val in zip(list(out), fut.result().values):
                        if val.double_value > 0:
                            out[k] = val.double_value
        except Exception:                                              # noqa: BLE001
            pass
        return out

    def shutdown(self):
        self.ex.shutdown()
        self.node.destroy_node()
        if self.ctx.ok():
            rclpy.shutdown(context=self.ctx)


def detect(bgr, cfg):
    """Same HSV orange-blob detector as ball_detector (best circular blob)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (int(cfg['hsv_h_min']), int(cfg['hsv_s_min']), int(cfg['hsv_v_min'])),
                       (int(cfg['hsv_h_max']), 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    best, score = None, 0.0
    for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if area < 12:
            continue
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        circ = area / (math.pi * rad * rad + 1e-6)
        if circ >= 0.55 and area * circ > score:
            best, score = (cx, cy, rad), area * circ
    return best, mask


def ground_point(u, v, K, cam_pose, optical, pitch_deg, dist=None):
    """Ball centre on the floor in base_footprint (same maths as ball_detector._localise) or None."""
    fx, fy, cx, cy = K
    if dist is not None:                                 # checkerboard calibration: remove lens distortion
        Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        u, v = cv2.undistortPoints(np.array([[[u, v]]], dtype=float), Km, np.array(dist, dtype=float), P=Km)[0, 0]
    if optical:
        d = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    else:
        d = np.array([1.0, -(u - cx) / fx, -(v - cy) / fy])
    p = math.radians(pitch_deg)
    if p and optical:
        d = np.array([d[0], d[1] * math.cos(p) + d[2] * math.sin(p), -d[1] * math.sin(p) + d[2] * math.cos(p)])
    elif p:
        d = np.array([d[0] * math.cos(p) + d[2] * math.sin(p), d[1], -d[0] * math.sin(p) + d[2] * math.cos(p)])
    R, o = cam_pose
    d = R @ d
    if d[2] >= -1e-4:
        return None
    s = (BALL_R - o[2]) / d[2]
    return o + s * d if s > 0 else None


class CalibWindow:
    def __init__(self, ros, team, hfov):
        self.ros, self.team, self.hfov = ros, team, hfov
        self.cfg = load_config(team)
        self.samples, self.collecting = [], False
        self.root = tk.Tk()
        self.root.title(f'카메라 보정 - {"파랑" if team == "blue" else "노랑"} 로봇')
        self.root.configure(bg='#101820')
        self.root.protocol('WM_DELETE_WINDOW', self.root.destroy)
        style = {'bg': '#101820', 'fg': 'white'}
        self.canvas = tk.Label(self.root, bg='black')
        self.canvas.grid(row=0, column=0, rowspan=20, padx=8, pady=8)
        side = tk.Frame(self.root, **{'bg': '#101820'})
        side.grid(row=0, column=1, sticky='n', padx=8, pady=8)
        self.wheel_panel(style)

        tk.Label(side, text='1. 공 색 (HSV)', font=('DejaVu Sans', 11, 'bold'), **style).pack(anchor='w')
        self.vars = {}
        for key, lo, hi in (('hsv_h_min', 0, 179), ('hsv_h_max', 0, 179), ('hsv_s_min', 0, 255),
                            ('hsv_v_min', 0, 255)):
            var = tk.IntVar(value=int(self.cfg[key]))
            tk.Scale(side, from_=lo, to=hi, orient='horizontal', label=key, variable=var, length=260,
                     bg='#18242f', fg='white', highlightthickness=0, troughcolor='#31465a').pack(anchor='w')
            self.vars[key] = var
        self.show_mask = tk.BooleanVar(value=False)
        tk.Checkbutton(side, text='마스크 보기 (흰색 = 공으로 인식되는 부분)', variable=self.show_mask,
                       selectcolor='#31465a', **style).pack(anchor='w', pady=(2, 10))

        tk.Label(side, text='2. 카메라 각도', font=('DejaVu Sans', 11, 'bold'), **style).pack(anchor='w')
        tk.Label(side, text='공을 로봇 정면에 두고, 로봇 중심(바퀴 축)에서\n공 중심까지 실제 거리를 재서 입력',
                 justify='left', **style).pack(anchor='w')
        row = tk.Frame(side, bg='#101820')
        row.pack(anchor='w', pady=4)
        tk.Label(row, text='실제 거리 (m)', **style).pack(side='left')
        self.true_dist = tk.StringVar(value='0.50')
        tk.Entry(row, textvariable=self.true_dist, width=7).pack(side='left', padx=6)
        tk.Button(row, text='측정·계산', command=self.start_measure, bg='#16834a', fg='white').pack(side='left')
        self.pitch = tk.DoubleVar(value=float(self.cfg['cam_pitch_deg']))
        tk.Scale(side, from_=-15, to=15, resolution=0.1, orient='horizontal', variable=self.pitch, length=260,
                 label='각도 보정 cam_pitch_deg (−: 위로)', bg='#18242f', fg='white', highlightthickness=0,
                 troughcolor='#31465a').pack(anchor='w')
        self.info = tk.Label(side, text='', justify='left', font=('DejaVu Sans Mono', 10), **style)
        self.info.pack(anchor='w', pady=8)

        tk.Button(side, text=f'3. 저장  ({config_path(team).replace(os.path.expanduser("~"), "~")})',
                  command=self.save, bg='#3c5fc4', fg='white').pack(anchor='w', fill='x', pady=4)
        tk.Label(side, text='4. 렌즈 보정 (체커보드)', font=('DejaVu Sans', 11, 'bold'), **style).pack(anchor='w',
                                                                                              pady=(8, 0))
        row = tk.Frame(side, bg='#101820')
        row.pack(anchor='w', pady=2)
        tk.Label(row, text='안쪽 코너 가로', **style).pack(side='left')
        self.cb_cols = tk.StringVar(value='8')
        tk.Entry(row, textvariable=self.cb_cols, width=3).pack(side='left', padx=3)
        tk.Label(row, text='세로', **style).pack(side='left')
        self.cb_rows = tk.StringVar(value='6')
        tk.Entry(row, textvariable=self.cb_rows, width=3).pack(side='left', padx=3)
        tk.Label(row, text='칸(m)', **style).pack(side='left')
        self.cb_square = tk.StringVar(value='0.025')
        tk.Entry(row, textvariable=self.cb_square, width=6).pack(side='left', padx=3)
        row = tk.Frame(side, bg='#101820')
        row.pack(anchor='w', pady=2)
        self.lens_btn = tk.Button(row, text='수집 시작', command=self.lens_toggle, bg='#16834a', fg='white')
        self.lens_btn.pack(side='left')
        tk.Button(row, text='계산', command=self.lens_compute, bg='#3c5fc4', fg='white').pack(side='left', padx=6)
        tk.Button(row, text='초기화', command=self.lens_reset, bg='#5a1f1f', fg='white').pack(side='left')
        self.lens_info = tk.Label(side, text='', justify='left', font=('DejaVu Sans Mono', 10), **style)
        self.lens_info.pack(anchor='w')
        self.lens_collect, self.lens_obj, self.lens_img, self.lens_last = False, [], [], (0.0, None)
        self.lens_size = None
        self.status = tk.Label(side, text='', wraplength=300, justify='left', fg='#f5cc1f', bg='#101820')
        self.status.pack(anchor='w', pady=6)
        self.refresh()

    # ------------------------------------------------------------------ helpers
    def current_cfg(self):
        c = {k: v.get() for k, v in self.vars.items()}
        c['cam_pitch_deg'] = round(float(self.pitch.get()), 2)
        return c

    def geometry(self, bgr):
        with self.ros.lock:
            K, frame_id = self.ros.K, self.ros.frame_id
        if self.cfg.get('intrinsics'):                   # our checkerboard calibration wins
            K = tuple(self.cfg['intrinsics'])
        if K is None:                                    # no calibration published: FOV fallback
            h, w = bgr.shape[:2]
            f = (w / 2) / math.tan(math.radians(self.hfov) / 2)
            K = (f, f, w / 2, h / 2)
        pose = self.ros.tf.base_to(frame_id) if frame_id else None
        return K, frame_id, pose

    def start_measure(self):
        try:
            float(self.true_dist.get())
        except ValueError:
            self.status.configure(text='실제 거리를 숫자로 입력하세요 (m)')
            return
        self.samples, self.collecting = [], True
        self.status.configure(text='측정 중... 공과 로봇을 움직이지 마세요')

    def solve(self):
        """Pitch correction so that the averaged ball pixel lands at the measured distance."""
        u, v = np.median(np.array(self.samples), axis=0)
        target = float(self.true_dist.get())
        K, frame_id, pose = self.geometry(self.last_bgr)

        def dist(p):
            g = ground_point(u, v, K, pose, 'optical' in frame_id, p, self.cfg.get('distortion'))
            return None if g is None else float(math.hypot(g[0], g[1]))
        lo, hi = -20.0, 20.0                             # more pitch down -> ball computed closer
        for _ in range(60):
            mid = (lo + hi) / 2
            d = dist(mid)
            if d is None or d > target:
                lo = mid                                 # ray above the horizon / too far: tilt down more
            else:
                hi = mid
        p = (lo + hi) / 2
        d = dist(p)
        if d is None or abs(d - target) > 0.02:
            self.status.configure(text=f'계산 실패 (거리 {d}) - 공이 정면 바닥에 있는지 확인하세요')
            return
        self.pitch.set(round(p, 1))
        self.status.configure(text=f'각도 보정 {p:+.1f}°  (공 {len(self.samples)}프레임 평균).  저장을 누르세요.')

    def save(self):
        path = config_path(self.team)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out = self.current_cfg()
        for k in ('intrinsics', 'distortion', 'lens_rms', 'image_size', 'wheel_radius', 'wheel_separation'):
            if self.cfg.get(k) is not None:
                out[k] = self.cfg[k]
        with open(path, 'w') as f:
            yaml.safe_dump(out, f, sort_keys=False)
        txt = f'저장됨: {path}\nreal_duel을 다시 실행하면 적용됩니다.'
        if out.get('wheel_radius') or out.get('wheel_separation'):
            r = out.get('wheel_radius') or WHEEL_DEFAULT['wheel_radius']
            sep = out.get('wheel_separation') or WHEEL_DEFAULT['wheel_separation']
            txt += f'\n바퀴 값은 로봇에서: robot.launch.py ... wheel_radius:={r} wheel_separation:={sep}'
        self.status.configure(text=txt)

    # ------------------------------------------------------------------ wheels
    def wheel_panel(self, style):
        w = tk.Frame(self.root, bg='#101820')
        w.grid(row=0, column=2, sticky='n', padx=8, pady=8)
        tk.Label(w, text='5. 바퀴 보정', font=('DejaVu Sans', 11, 'bold'), **style).pack(anchor='w')
        tk.Label(w, text='앞 1.2 m, 주변 0.3 m를 비우고\n바퀴 축 중심 아래 바닥에 테이프 표시',
                 justify='left', **style).pack(anchor='w')
        row = tk.Frame(w, bg='#101820')
        row.pack(anchor='w', pady=4)
        tk.Button(row, text='1 m 전진', command=lambda: self.wheel_start('line'), bg='#16834a',
                  fg='white').pack(side='left')
        tk.Button(row, text='1바퀴 회전', command=lambda: self.wheel_start('spin'), bg='#16834a',
                  fg='white').pack(side='left', padx=6)
        tk.Button(row, text='정지', command=self.wheel_abort, bg='#a02020', fg='white').pack(side='left')
        row = tk.Frame(w, bg='#101820')
        row.pack(anchor='w', pady=4)
        tk.Label(row, text='실제 거리 (m)', **style).pack(side='left')
        self.real_dist = tk.StringVar(value='')
        tk.Entry(row, textvariable=self.real_dist, width=7).pack(side='left', padx=6)
        tk.Button(row, text='반지름 계산', command=self.wheel_radius_calc, bg='#3c5fc4',
                  fg='white').pack(side='left')
        row = tk.Frame(w, bg='#101820')
        row.pack(anchor='w', pady=4)
        tk.Label(row, text='실제 각도 (°)', **style).pack(side='left')
        self.real_angle = tk.StringVar(value='')
        tk.Entry(row, textvariable=self.real_angle, width=7).pack(side='left', padx=6)
        tk.Button(row, text='간격 계산', command=self.wheel_sep_calc, bg='#3c5fc4',
                  fg='white').pack(side='left')
        self.wheel_info = tk.Label(w, text='', justify='left', font=('DejaVu Sans Mono', 10), **style)
        self.wheel_info.pack(anchor='w', pady=6)
        self.wheel_msg = tk.Label(w, text='', wraplength=300, justify='left', fg='#f5cc1f', bg='#101820')
        self.wheel_msg.pack(anchor='w')
        self.wheel_busy, self.wheel_stop, self.wheel_cur = False, False, None
        self.wheel_odom_dist, self.wheel_odom_ang = None, None
        self.wheel_show()

    def wheel_show(self):
        cur = self.wheel_cur or {}
        r, s = self.cfg.get('wheel_radius'), self.cfg.get('wheel_separation')
        t = [f'로봇 현재값  r {cur.get("wheel_radius", "-")}  s {cur.get("wheel_separation", "-")}',
             f'보정값       r {r if r else "-"}  s {s if s else "-"}']
        if self.wheel_odom_dist is not None:
            t.append(f'전진: 오도메트리 {self.wheel_odom_dist:.3f} m')
        if self.wheel_odom_ang is not None:
            t.append(f'회전: 오도메트리 {math.degrees(self.wheel_odom_ang[0]):.1f}°  '
                     f'IMU {math.degrees(self.wheel_odom_ang[1]):.1f}°')
        self.wheel_info.configure(text='\n'.join(t))

    def wheel_abort(self):
        self.wheel_stop = True

    def wheel_start(self, kind):
        if self.wheel_busy:
            return
        with self.ros.lock:
            od = self.ros.odom
        if od is None or time.time() - od[3] > 1.0:
            self.wheel_msg.configure(text='/odom 없음: 로봇 드라이버(robot.launch.py)가 켜져 있는지 확인')
            return
        self.wheel_busy, self.wheel_stop = True, False
        self.wheel_msg.configure(text='주행 중... (정지 버튼으로 멈춤)')
        threading.Thread(target=self._wheel_run, args=(kind,), daemon=True).start()

    def _wheel_run(self, kind):
        ros = self.ros
        if self.wheel_cur is None:
            self.wheel_cur = ros.wheel_params()
        with ros.lock:
            x0, y0, yaw0, _ = ros.odom
            imu0 = ros.imu_yaw
        turned, last_yaw, t0, msg = 0.0, yaw0, time.time(), ''
        try:
            while not self.wheel_stop:
                with ros.lock:
                    x, y, yaw, ts = ros.odom
                if time.time() - ts > 0.5:
                    msg = '/odom 끊김 - 중단'
                    break
                if time.time() - t0 > 25.0:
                    msg = '시간 초과 - 중단'
                    break
                if kind == 'line':
                    d = math.hypot(x - x0, y - y0)
                    if d >= 0.995:
                        break
                    v = max(0.04, min(0.12, 0.6 * (1.0 - d) + 0.03, 0.04 + 0.3 * (time.time() - t0)))
                    err = math.atan2(math.sin(yaw - yaw0), math.cos(yaw - yaw0))
                    ros.drive(v, max(-0.3, min(0.3, -2.0 * err)))
                else:
                    turned += math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw))
                    last_yaw = yaw
                    rest = 2 * math.pi - turned
                    if rest <= 0.01:
                        break
                    ros.drive(0.0, max(0.35, min(1.2, 1.5 * rest, 0.35 + 1.5 * (time.time() - t0))))
                time.sleep(0.05)
        finally:
            for _ in range(10):
                ros.drive(0.0, 0.0)
                time.sleep(0.05)
        time.sleep(1.0)                                              # let it settle, then read the final odometry
        with ros.lock:
            x, y, yaw, _ = ros.odom
            imu = ros.imu_yaw - imu0
        if kind == 'line':
            self.wheel_odom_dist = math.hypot(x - x0, y - y0)
            text = msg or (f'완료. 오도메트리 {self.wheel_odom_dist:.3f} m.\n줄자로 시작 표시~바퀴 축 중심까지 '
                           '잰 거리를 "실제 거리"에 넣고 반지름 계산')
        else:
            turned += math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw))
            self.wheel_odom_ang = (turned, imu)
            self.root.after(0, lambda: self.real_angle.set(f'{math.degrees(imu):.1f}'))
            text = msg or (f'완료. 오도메트리 {math.degrees(turned):.1f}°, IMU {math.degrees(imu):.1f}°.\n'
                           '실제 각도에 IMU 값을 넣어 뒀습니다 (테이프로 본 값이 있으면 고쳐 쓰기) -> 간격 계산')
        self.wheel_busy = False
        self.root.after(0, lambda: (self.wheel_msg.configure(text=text), self.wheel_show()))

    def wheel_radius_calc(self):
        try:
            real = float(self.real_dist.get())
        except ValueError:
            self.wheel_msg.configure(text='실제 거리를 숫자로 (m)')
            return
        if not self.wheel_odom_dist or self.wheel_cur is None:
            self.wheel_msg.configure(text='먼저 "1 m 전진"')
            return
        r = self.wheel_cur['wheel_radius'] * real / self.wheel_odom_dist
        self.cfg['wheel_radius'] = round(r, 5)
        self.wheel_msg.configure(text=f'바퀴 반지름 {self.wheel_cur["wheel_radius"]} -> {r:.5f} m '
                                      f'({(real / self.wheel_odom_dist - 1) * 100:+.1f}%).  저장을 누르세요.')
        self.wheel_show()

    def wheel_sep_calc(self):
        try:
            real = math.radians(float(self.real_angle.get()))
        except ValueError:
            self.wheel_msg.configure(text='실제 각도를 숫자로 (°)')
            return
        if not self.wheel_odom_ang or self.wheel_cur is None or real <= 0:
            self.wheel_msg.configure(text='먼저 "1바퀴 회전"')
            return
        k = (self.cfg['wheel_radius'] / self.wheel_cur['wheel_radius']) if self.cfg.get('wheel_radius') else 1.0
        s = self.wheel_cur['wheel_separation'] * k * self.wheel_odom_ang[0] / real
        self.cfg['wheel_separation'] = round(s, 5)
        self.wheel_msg.configure(text=f'바퀴 간격 {self.wheel_cur["wheel_separation"]} -> {s:.5f} m.  저장을 누르세요.')
        self.wheel_show()

    # ------------------------------------------------------------------ checkerboard
    def board(self):
        try:
            return (int(self.cb_cols.get()), int(self.cb_rows.get())), float(self.cb_square.get())
        except ValueError:
            return None, None

    def lens_toggle(self):
        self.lens_collect = not self.lens_collect
        self.lens_btn.configure(text='수집 정지' if self.lens_collect else '수집 시작')
        if self.lens_collect:
            self.status.configure(text='체커보드를 화면 여기저기(가운데·네 구석·가까이·멀리·기울여서)에 '
                                       '보여 주세요. 초록 코너가 보이면 자동으로 모읍니다 (15장 이상).')

    def lens_reset(self):
        self.lens_obj, self.lens_img, self.lens_last = [], [], (0.0, None)
        self.cfg.update({'intrinsics': None, 'distortion': None, 'lens_rms': None, 'image_size': None})
        self.status.configure(text='렌즈 보정 초기화 (FOV 기본값 사용)')

    def lens_frame(self, bgr, img):
        size, square = self.board()
        if size is None:
            return
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ok, corners = self.find_board(gray, size)
        if not ok:
            self.auto_size(gray, size)
            cv2.putText(img, f'board {size[0]}x{size[1]} not found: whole board in view? corner count?',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            return
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        cv2.drawChessboardCorners(img, size, corners, ok)
        if not self.lens_collect:
            return
        c = corners.reshape(-1, 2)
        key = np.array([*c.mean(axis=0), np.linalg.norm(c[0] - c[-1])])     # board position + size in image
        t_last, k_last = self.lens_last
        if time.time() - t_last < 0.7:
            return
        if any(np.linalg.norm(key - np.array(k)) < 25 for k in [kk for kk in self.lens_keys()]):
            return                                         # too similar to a view we already have
        obj = np.zeros((size[0] * size[1], 3), np.float32)
        obj[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2) * square
        self.lens_obj.append(obj)
        self.lens_img.append(corners)
        self.lens_last = (time.time(), key.tolist())
        self.lens_size = (bgr.shape[1], bgr.shape[0])
        self._keys = self.lens_keys() + [key.tolist()]

    @staticmethod
    def find_board(gray, size):
        """Classic detector first (fast), then the sector-based one (robust to blur / uneven light)."""
        ok, corners = cv2.findChessboardCorners(gray, size, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if not ok and hasattr(cv2, 'findChessboardCornersSB'):
            ok, corners = cv2.findChessboardCornersSB(gray, size, cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                corners = corners.astype(np.float32)
        return ok, corners

    def auto_size(self, gray, size):
        """The typed corner count may be wrong: every ~1.5 s try the usual sizes and take the one that fits."""
        now = time.time()
        if now - getattr(self, '_auto_t', 0.0) < 1.5:
            return
        self._auto_t = now
        small = cv2.resize(gray, None, fx=0.5, fy=0.5)
        for cols in range(11, 3, -1):
            for rows in range(min(cols, 8), 2, -1):
                if (cols, rows) == tuple(size):
                    continue
                ok, _ = cv2.findChessboardCorners(small, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                  cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
                if ok:
                    self.cb_cols.set(str(cols))
                    self.cb_rows.set(str(rows))
                    self.status.configure(text=f'보드 안쪽 코너가 {cols} x {rows}로 보여서 입력값을 바꿨습니다. '
                                               f'칸 크기(m)도 실제 보드에 맞게 확인하세요.')
                    return

    def lens_keys(self):
        return getattr(self, '_keys', [])

    def lens_compute(self):
        if len(self.lens_img) < 10:
            self.status.configure(text=f'장면이 {len(self.lens_img)}장뿐입니다 - 15장 이상 모아 주세요')
            return
        rms, K, D, _, _ = cv2.calibrateCamera(self.lens_obj, self.lens_img, self.lens_size, None, None)
        self.cfg['intrinsics'] = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]
        self.cfg['distortion'] = [float(x) for x in D.ravel()[:5]]
        self.cfg['lens_rms'] = float(rms)
        self.cfg['image_size'] = list(self.lens_size)
        hfov = math.degrees(2 * math.atan(self.lens_size[0] / 2 / K[0, 0]))
        good = '좋음' if rms < 0.5 else '보통' if rms < 1.0 else '다시 하는 게 좋음'
        self.status.configure(text=f'렌즈 보정 완료: 오차 {rms:.2f} px ({good}), 화각 {hfov:.1f}°.\n'
                                   f'이제 2번(각도)을 다시 측정하고 저장하세요.')
        self.lens_collect = False
        self.lens_btn.configure(text='수집 시작')

    def lens(self):
        if shutil.which('ros2') and subprocess.run(['ros2', 'pkg', 'prefix', 'camera_calibration'],
                                                   capture_output=True).returncode == 0:
            subprocess.Popen(['ros2', 'run', 'camera_calibration', 'cameracalibrator', '--size', '8x6',
                              '--square', '0.025', '--no-service-check',
                              f'image:={self.ros_ns}/camera/image_raw', f'camera:={self.ros_ns}/camera'])
            self.status.configure(text='체커보드(내부 코너 8x6, 칸 2.5 cm)를 여러 위치·각도로 보여 주고 '
                                       'CALIBRATE → SAVE.  /tmp/calibrationdata.tar.gz 의 ost.yaml을 로봇에 '
                                       '복사해 robot.launch.py camera_info_url:=file:///home/pinky/ost.yaml')
        else:
            self.status.configure(text='ROS camera_calibration이 없습니다. 설치:\n'
                                       'sudo apt install ros-jazzy-camera-calibration')

    # ------------------------------------------------------------------ loop
    def refresh(self):
        with self.ros.lock:
            bgr = None if self.ros.frame is None else self.ros.frame.copy()
            age = time.time() - getattr(self.ros, 'stamp', 0)
        lines = []
        if bgr is None or age > 2.0:
            img = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(img, 'no camera image', (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (160, 160, 160), 2)
            lines.append('카메라 영상 없음: 로봇 연결 / 도메인 / DDS 설정 확인')
        else:
            self.last_bgr = bgr
            cfg = self.current_cfg()
            img_cb = bgr.copy()
            self.lens_frame(bgr, img_cb)
            ball, mask = detect(bgr, cfg)
            K, frame_id, pose = self.geometry(bgr)
            img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if self.show_mask.get() else img_cb
            src = '체커보드' if self.cfg.get('intrinsics') else ('camera_info' if self.ros.raw_K else f'FOV {self.hfov}°')
            lines.append(f'frame {frame_id or "-"}   K {src}')
            lines.append(f'fx {K[0]:.0f}  cx {K[2]:.0f}  cy {K[3]:.0f}')
            if pose is None:
                lines.append('TF(base_footprint→카메라) 아직 없음')
            if ball is not None:
                u, v, rad = ball
                cv2.circle(img, (int(u), int(v)), int(rad), (0, 255, 0), 2)
                cv2.drawMarker(img, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 14, 2)
                g = ground_point(u, v, K, pose, 'optical' in frame_id, cfg['cam_pitch_deg'],
                                 self.cfg.get('distortion')) if pose else None
                if g is not None:
                    txt = f'front {g[0]:.3f} m   side {g[1]:+.3f} m   dist {math.hypot(g[0], g[1]):.3f} m'
                    cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
                    cv2.putText(img, txt, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                    lines.append(f'공: 앞 {g[0]:.3f} m  옆 {g[1]:+.3f} m')
                    lines.append(f'    거리 {math.hypot(g[0], g[1]):.3f} m (보정 {cfg["cam_pitch_deg"]:+.1f}°)')
                else:
                    lines.append('공: 바닥 위치 계산 불가 (TF 없음 / 수평선 위)')
                    cv2.putText(img, 'ball: no floor position (TF?)', (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                (0, 0, 255), 2)
                if self.collecting:
                    self.samples.append((u, v))
                    if len(self.samples) >= 30:
                        self.collecting = False
                        self.solve()
            else:
                lines.append('공: 안 보임 (HSV 범위를 조절해 보세요)')
                cv2.putText(img, 'ball: not detected', (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
            cv2.line(img, (0, int(K[3])), (img.shape[1], int(K[3])), (90, 90, 90), 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(PILImage.fromarray(rgb))
        self.canvas.configure(image=photo)
        self.canvas.image = photo
        self.info.configure(text='\n'.join(lines))
        rms = self.cfg.get('lens_rms')
        self.lens_info.configure(text=f'모은 장면 {len(self.lens_img)}장' + (f'   보정 오차 {rms:.2f} px' if rms else ''))
        self.root.after(66, self.refresh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--team', default='blue', choices=['blue', 'yellow'])
    ap.add_argument('--domain', type=int, default=None, help='default: yellow 37, blue 38')
    ap.add_argument('--ns', default='', help="topic namespace ('' real robot, '/blue' sim_duel)")
    ap.add_argument('--raw', action='store_true', help='raw /camera/image_raw instead of the JPEG topic')
    ap.add_argument('--hfov', type=float, default=53.5, help='deg, when the camera publishes no calibration')
    opt, _ = ap.parse_known_args()
    domain = opt.domain if opt.domain is not None else (37 if opt.team == 'yellow' else 38)
    ros = CalibNode(domain, opt.ns.rstrip('/'), not opt.raw)
    win = CalibWindow(ros, opt.team, opt.hfov)
    win.ros_ns = opt.ns.rstrip('/')
    try:
        win.root.mainloop()
    finally:
        ros.shutdown()


if __name__ == '__main__':
    main()
