#!/usr/bin/env python3
"""Sensor view: what each robot's camera and lidar see, side by side.

    +-------------------------+-------------+
    | YELLOW camera           | YELLOW lidar|   lidar panel: robot-centred top view, robot facing UP,
    +-------------------------+-------------+   rings every 0.5 m, camera field of view as a wedge,
    | BLUE camera             | BLUE lidar  |   ground truth (ball = orange, other robot = its colour)
    +-------------------------+-------------+   drawn as hollow markers to compare with the scan.

Topics:  /camera/image_raw  /scan             (yellow, ours)
         /blue/camera/image_raw  /blue/scan   (blue, needs sim.launch.py blue_sensors:=true)
         /gz/dynamic_pose                     (ground-truth markers, optional)
Duel (--duel, sim_duel / two real robots): each robot in its own domain (yellow 37, blue 38), topics under
/<team>/: the camera panel shows /<team>/ball/detection_image (camera + detection box), and a sensor line
below it: striker state, odom speed, AMCL pose, ball seen (distance / bearing), nearest lidar obstacle.
Keys:    q / Esc closes the window.
"""
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.context import Context
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage
import tf2_ros

from pinky_soccer_sim.gz_utils import PoseTracker, yaw_from_quat

CAM_W, CAM_H, LIDAR_PX = 640, 360, 360
INFO_H = 54                                     # duel: sensor text lines under each camera
TEAM_BGR = {'yellow': (0, 210, 255), 'blue': (255, 110, 20)}
ROBOTS = (  # team, model name, image topic, scan topic, base frame
    ('yellow', 'pinky_yellow', '/camera/image_raw', '/scan', 'base_footprint'),
    ('blue', 'pinky_blue', '/blue/camera/image_raw', '/blue/scan', 'blue/base_footprint'),
)


class DuelLink:
    """One robot's topics in its own ROS domain (duel layout)."""

    def __init__(self, view, team, domain, plain=False):
        self.ctx = Context()
        rclpy.init(context=self.ctx, domain_id=domain)
        self.node = Node(f'sensor_view_{team}', context=self.ctx)
        self.ex = SingleThreadedExecutor(context=self.ctx)
        self.ex.add_node(self.node)
        n, ns = self.node, ('' if plain else f'/{team}')      # real robots: plain names in their own domain
        n.create_subscription(Image, f'{ns}/ball/detection_image', lambda m: view._img(team, m), 2)
        n.create_subscription(LaserScan, f'{ns}/scan', lambda m: view._scan(team, m), qos_profile_sensor_data)
        n.create_subscription(String, f'{ns}/game/state', lambda m: view._info(team, 'state', m.data), 10)
        n.create_subscription(Odometry, f'{ns}/odom', lambda m: view._info(
            team, 'speed', (m.twist.twist.linear.x, m.twist.twist.angular.z)), 10)
        n.create_subscription(PoseWithCovarianceStamped, f'{ns}/amcl_pose', lambda m: view._info(
            team, 'pose', (m.pose.pose.position.x, m.pose.pose.position.y,
                           math.degrees(yaw_from_quat(m.pose.pose.orientation)))), 10)
        n.create_subscription(PointStamped, f'{ns}/ball/position_base', lambda m: view._info(
            team, 'ball', (m.point.x, m.point.y)), 10)
        if team == 'yellow':                      # ground truth lives in the yellow (referee) domain
            n.create_subscription(TFMessage, '/gz/dynamic_pose', view._gt, 10)
        threading.Thread(target=self.ex.spin, daemon=True).start()


class SensorView(Node):
    def __init__(self, duel=None, plain=False):
        super().__init__('sensor_view')
        self.duel = duel                                  # {team: domain} or None
        self.declare_parameter('lidar_range', 1.6)       # half width of the lidar panel [m]
        self.declare_parameter('camera_hfov', 1.1519)    # official Pinky camera horizontal FOV [rad]
        self.lock = threading.Lock()
        self.images, self.scans = {}, {}
        self.tracker = PoseTracker()
        self.info = {'yellow': {}, 'blue': {}}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        if duel:
            self.links = [DuelLink(self, team, dom, plain) for team, dom in duel.items()]
            return
        for team, _, img_topic, scan_topic, _ in ROBOTS:
            self.create_subscription(Image, img_topic, lambda m, t=team: self._img(t, m), qos_profile_sensor_data)
            self.create_subscription(LaserScan, scan_topic, lambda m, t=team: self._scan(t, m),
                                     qos_profile_sensor_data)
        self.create_subscription(TFMessage, '/gz/dynamic_pose', self._gt, 10)

    def _img(self, team, msg):
        with self.lock:
            self.images[team] = (msg, time.time())

    def _scan(self, team, msg):
        with self.lock:
            self.scans[team] = (msg, time.time())

    def _info(self, team, key, value):
        with self.lock:
            self.info[team][key] = (value, time.time())

    def info_panel(self, team):
        """Sensor values as text (duel)."""
        panel = np.full((INFO_H, CAM_W, 3), 24, np.uint8)
        with self.lock:
            info = {k: v for k, (v, t) in self.info[team].items() if time.time() - t < 2.0}
            scan = self.scans.get(team)
        state = info.get('state', '-')
        sp = info.get('speed')
        pose = info.get('pose')
        ball = info.get('ball')
        near = '-'
        if scan is not None and time.time() - scan[1] < 2.0:
            r = np.asarray(scan[0].ranges, dtype=float)
            r = r[np.isfinite(r) & (r > scan[0].range_min)]
            near = f'{r.min():.2f} m' if len(r) else '-'
        line1 = f'state {state:<15}  v {sp[0]:+.2f} m/s  w {sp[1]:+.2f} rad/s' if sp else f'state {state}'
        line2 = (f'pose ({pose[0]:+.2f}, {pose[1]:+.2f}, {pose[2]:+4.0f} deg)' if pose else 'pose -') + \
            (f'   ball {math.hypot(*ball):.2f} m @ {math.degrees(math.atan2(ball[1], ball[0])):+3.0f} deg'
             if ball else '   ball not seen') + f'   nearest {near}'
        cv2.putText(panel, line1, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEAM_BGR[team], 1)
        cv2.putText(panel, line2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        return panel

    def _gt(self, msg):
        with self.lock:
            self.tracker.update(msg)

    # ------------------------------------------------------------------ panels
    def camera_panel(self, team):
        with self.lock:
            item = self.images.get(team)
        panel = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        if item is None or time.time() - item[1] > 2.0:
            cv2.putText(panel, f'{team.upper()} camera: no image', (20, CAM_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 160, 160), 2)
        else:
            m = item[0]
            ch = 4 if m.encoding in ('rgba8', 'bgra8') else 3
            img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, ch)[:, :, :3]
            if m.encoding.startswith('rgb'):
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            panel = cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel, (0, 0), (CAM_W - 1, CAM_H - 1), TEAM_BGR[team], 3)
        cv2.putText(panel, f'{team.upper()}  camera', (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    TEAM_BGR[team], 2)
        return panel

    def _scan_to_base(self, scan, base_frame):
        """Scan points in the robot base frame (Pinky's rplidar_link is yawed 180 deg)."""
        try:
            if self.duel:                     # each robot's TF is on its own /<team>/tf: use the fixed mount
                raise tf2_ros.TransformException('duel')
            t = self.tf_buffer.lookup_transform(base_frame, scan.header.frame_id, rclpy.time.Time())
            tx, ty, yaw = t.transform.translation.x, t.transform.translation.y, yaw_from_quat(t.transform.rotation)
        except tf2_ros.TransformException:
            tx, ty, yaw = -0.017, 0.0, math.pi                    # official mount, if TF is not there yet
        r = np.asarray(scan.ranges, dtype=float)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment
        ok = np.isfinite(r) & (r > scan.range_min) & (r < scan.range_max)
        x, y = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        c, s = math.cos(yaw), math.sin(yaw)
        return tx + c * x - s * y, ty + s * x + c * y

    def lidar_panel(self, team, model, base_frame):
        rng = self.get_parameter('lidar_range').value
        px = LIDAR_PX / (2.0 * rng)
        cx = cy = LIDAR_PX // 2
        panel = np.full((LIDAR_PX, LIDAR_PX, 3), 18, np.uint8)

        def to_px(x, y):        # base frame (x fwd, y left) -> image (robot facing up)
            return int(round(cx - y * px)), int(round(cy - x * px))

        for ring in np.arange(0.5, rng + 1e-6, 0.5):
            cv2.circle(panel, (cx, cy), int(ring * px), (55, 55, 55), 1)
            cv2.putText(panel, f'{ring:.1f}', (cx + 3, cy - int(ring * px) + 12), cv2.FONT_HERSHEY_PLAIN, 0.8,
                        (90, 90, 90), 1)
        half = self.get_parameter('camera_hfov').value / 2
        for s in (-1, 1):
            cv2.line(panel, (cx, cy), to_px(rng * 1.5 * math.cos(half), s * rng * 1.5 * math.sin(half)),
                     (80, 80, 40), 1)

        with self.lock:
            item = self.scans.get(team)
            me = self.tracker.get(model)
            others = {n: self.tracker.get(n) for n in ('golf_ball', 'pinky_yellow', 'pinky_blue') if n != model}
        if item is None or time.time() - item[1] > 2.0:
            cv2.putText(panel, 'no scan', (cx - 40, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)
        else:
            xs, ys = self._scan_to_base(item[0], base_frame)
            u = np.round(cx - ys * px).astype(int)
            v = np.round(cy - xs * px).astype(int)
            ok = (u >= 1) & (u < LIDAR_PX - 1) & (v >= 1) & (v < LIDAR_PX - 1)
            for du in (-1, 0, 1):          # 3x3 dots, vectorised
                for dv in (-1, 0, 1):
                    panel[v[ok] + dv, u[ok] + du] = (90, 255, 90)

        # ground truth of the other objects, relative to this robot (hollow markers)
        if me is not None:
            c, s = math.cos(me[3]), math.sin(me[3])
            for name, p in others.items():
                if p is None:
                    continue
                dx, dy = p[0] - me[0], p[1] - me[1]
                bx, by = c * dx + s * dy, -s * dx + c * dy
                if abs(bx) > rng or abs(by) > rng:
                    continue
                if name == 'golf_ball':
                    cv2.circle(panel, to_px(bx, by), max(3, int(0.02135 * px)), (0, 140, 255), 2)
                else:
                    col = TEAM_BGR['yellow' if name == 'pinky_yellow' else 'blue']
                    cv2.circle(panel, to_px(bx, by), int(0.06 * px), col, 1)

        # the robot itself (approx. footprint) + heading
        cv2.circle(panel, (cx, cy), int(0.06 * px), TEAM_BGR[team], 2)
        cv2.arrowedLine(panel, (cx, cy), to_px(0.14, 0.0), TEAM_BGR[team], 2, tipLength=0.35)
        cv2.rectangle(panel, (0, 0), (LIDAR_PX - 1, LIDAR_PX - 1), TEAM_BGR[team], 3)
        cv2.putText(panel, f'{team.upper()}  lidar', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEAM_BGR[team], 2)
        cv2.putText(panel, 'green=scan  hollow=truth', (10, LIDAR_PX - 10), cv2.FONT_HERSHEY_PLAIN, 0.9,
                    (150, 150, 150), 1)
        return panel

    def compose(self):
        rows = []
        for team, model, _, _, base in ROBOTS:
            lidar = self.lidar_panel(team, model, 'base_footprint' if self.duel else base)
            left = self.camera_panel(team)
            if self.duel:
                left = np.vstack([left, self.info_panel(team)])
                lidar = np.vstack([lidar, np.full((left.shape[0] - lidar.shape[0], LIDAR_PX, 3), 18, np.uint8)])
            rows.append(np.hstack([left, lidar]))
        return np.vstack(rows)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--duel', action='store_true', help='two robots, each in its own ROS domain')
    ap.add_argument('--yellow-domain', type=int, default=37)
    ap.add_argument('--blue-domain', type=int, default=38)
    ap.add_argument('--plain', action='store_true', help='duel with real robots: topics without /<team>/')
    opt, _ = ap.parse_known_args()
    rclpy.init()
    node = SensorView({'yellow': opt.yellow_domain, 'blue': opt.blue_domain} if opt.duel else None, opt.plain)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    win = 'Pinky soccer - sensor view'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, CAM_W + LIDAR_PX, 2 * (CAM_H + (INFO_H if opt.duel else 0)))
    try:
        while rclpy.ok():
            cv2.imshow(win, node.compose())
            key = cv2.waitKey(66) & 0xFF
            if key in (ord('q'), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
