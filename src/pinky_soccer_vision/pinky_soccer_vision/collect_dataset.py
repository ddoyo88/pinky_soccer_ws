#!/usr/bin/env python3
"""Auto-labelled YOLO dataset from the simulator.

Teleports the yellow robot / golf ball / blue robot to random poses (gz set_pose), grabs the camera image and
projects the ground-truth ball into it -> YOLO txt label (class 0 = golf_ball). No manual labelling.

Run with the simulation up (stage 5, with_blue:=true, ball detector not required):
  ros2 run pinky_soccer_vision collect_dataset --out ~/datasets/golfball --num 1500
"""
import argparse
import math
import os
import random
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
import tf2_ros

from pinky_soccer_sim.gz_utils import PoseTracker, gz_set_pose
from pinky_soccer_vision.ball_detector import quat_to_rot


class Collector(Node):
    def __init__(self):
        super().__init__('dataset_collector')
        self.tracker = PoseTracker()
        self.K = None
        self.img = None
        self.img_t = 0.0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(TFMessage, '/gz/dynamic_pose', self.tracker.update, 10)
        self.create_subscription(CameraInfo, '/camera/camera_info', self._info, 1)
        self.create_subscription(Image, '/camera/image_raw', self._img, qos_profile_sensor_data)

    def _info(self, m):
        self.K = (m.k[0], m.k[4], m.k[2], m.k[5], m.width, m.height)

    def _img(self, m):
        if m.encoding not in ('rgb8', 'bgr8'):
            return
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        self.img = a[:, :, ::-1].copy() if m.encoding == 'rgb8' else a.copy()   # BGR for cv2
        self.img_t = time.time()

    def label(self, ball_r):
        """YOLO label line for the current ground truth, '' if not visible, None if data missing."""
        yellow, ball = self.tracker.get('pinky_yellow'), self.tracker.get('golf_ball')
        if yellow is None or ball is None or self.K is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform('base_footprint', 'front_camera_link', rclpy.time.Time())
        except tf2_ros.TransformException:
            return None
        fx, fy, cx, cy, W, H = self.K
        yaw = yellow[3]
        R_wb = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        R_bc = quat_to_rot(t.transform.rotation)
        t_bc = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        p_w = np.array(ball[:3])
        p_b = R_wb.T @ (p_w - np.array(yellow[:3]))
        p_c = R_bc.T @ (p_b - t_bc)                       # front_camera_link: x fwd, y left, z up
        xo, yo, zo = -p_c[1], -p_c[2], p_c[0]             # optical
        if zo < 0.08:
            return ''
        u, v = fx * xo / zo + cx, fy * yo / zo + cy
        rp = fx * ball_r / zo
        x1, y1, x2, y2 = u - rp, v - rp, u + rp, v + rp
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            return ''
        if (cx2 - cx1) * (cy2 - cy1) < 0.6 * (x2 - x1) * (y2 - y1) or rp < 2.0:
            return ''
        # occluded by the blue robot?  (rough: blue closer to camera and within 0.09 m of the sight line)
        blue = self.tracker.get('pinky_blue')
        if blue is not None:
            seg = p_w[:2] - np.array(yellow[:2])
            n = np.linalg.norm(seg)
            if n > 1e-3:
                w = (np.array(blue[:2]) - np.array(yellow[:2]))
                proj = float(w @ seg) / n
                dist = abs(seg[0] * w[1] - seg[1] * w[0]) / n
                if 0 < proj < n and dist < 0.09:
                    return None
        bx, by = (cx1 + cx2) / 2 / W, (cy1 + cy2) / 2 / H
        bw, bh = (cx2 - cx1) / W, (cy2 - cy1) / H
        return f'0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--num', type=int, default=1500)
    ap.add_argument('--length', type=float, default=1.66)
    ap.add_argument('--width', type=float, default=1.06)
    ap.add_argument('--world', default='soccer')
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--negative-ratio', type=float, default=0.1)
    ap.add_argument('--settle', type=float, default=0.7, help='seconds to wait after teleport')
    args, _ = ap.parse_known_args()

    import cv2
    for sp in ('train', 'val'):
        os.makedirs(os.path.join(args.out, 'images', sp), exist_ok=True)
        os.makedirs(os.path.join(args.out, 'labels', sp), exist_ok=True)

    rclpy.init()
    node = Collector()
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    ball_r, m = 0.02135, 0.12
    hl, hw = args.length / 2 - m, args.width / 2 - m
    saved = 0
    attempts = 0
    print('waiting for camera / ground truth ...')
    while node.K is None or node.tracker.get('golf_ball') is None:
        time.sleep(0.2)

    while saved < args.num and attempts < args.num * 4:
        attempts += 1
        rx, ry, ryaw = random.uniform(-hl, hl), random.uniform(-hw, hw), random.uniform(-math.pi, math.pi)
        negative = random.random() < args.negative_ratio
        if negative:    # ball behind the robot
            d = random.uniform(0.2, 0.6)
            bx, by = rx - d * math.cos(ryaw), ry - d * math.sin(ryaw)
        else:           # ball in front, 0.12 .. 1.5 m, inside the camera cone
            d = random.uniform(0.12, 1.5)
            a = ryaw + random.uniform(-0.55, 0.55)      # camera hfov is 66 deg (+-0.58 rad)
            bx, by = rx + d * math.cos(a), ry + d * math.sin(a)
        bx, by = max(-hl, min(hl, bx)), max(-hw, min(hw, by))
        gz_set_pose(args.world, 'pinky_yellow', rx, ry, 0.03, ryaw)
        gz_set_pose(args.world, 'golf_ball', bx, by, ball_r + 0.002, 0.0)
        gz_set_pose(args.world, 'pinky_blue', random.uniform(-hl, hl), random.uniform(-hw, hw),
                    0.03, random.uniform(-math.pi, math.pi))
        time.sleep(args.settle)
        node.img = None
        t0 = time.time()
        while node.img is None and time.time() - t0 < 2.0:
            time.sleep(0.02)
        if node.img is None:
            continue
        line = node.label(ball_r)
        if line is None:
            continue
        split = 'val' if random.random() < args.val_ratio else 'train'
        stem = f'{saved:06d}'
        cv2.imwrite(os.path.join(args.out, 'images', split, stem + '.jpg'), node.img)
        with open(os.path.join(args.out, 'labels', split, stem + '.txt'), 'w') as f:
            f.write(line + ('\n' if line else ''))
        saved += 1
        if saved % 50 == 0:
            print(f'{saved}/{args.num} saved')

    with open(os.path.join(args.out, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(args.out)}\ntrain: images/train\nval: images/val\n'
                f'names:\n  0: golf_ball\n')
    print('done ->', args.out)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
