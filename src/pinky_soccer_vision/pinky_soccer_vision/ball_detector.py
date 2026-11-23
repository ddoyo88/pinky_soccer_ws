#!/usr/bin/env python3
"""Golf-ball detector: image -> ball position on the floor (base_footprint and map frames).

backend:=hsv   orange-blob baseline (orange golf ball), works immediately (no training)   [default]
backend:=yolo  Ultralytics YOLO weights (model_path), see train_yolo.py
backend:=gt    Gazebo ground truth (assumes map == world, i.e. AMCL on the generated map). For debugging
               navigation / strategy independently of perception.

Position estimate: the pixel ray through the bbox centre is intersected with the horizontal plane at the
ball-centre height (known ball radius) -> no depth camera needed.

Publishes:  /ball/position_base (PointStamped, base_footprint)   /ball/position_map (PointStamped, map)
            /ball/detection_image (Image, debug overlay)
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
import tf2_ros

from pinky_soccer_sim.gz_utils import PoseTracker, yaw_from_quat


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')
        self.declare_parameter('backend', 'hsv')
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf', 0.35)
        self.declare_parameter('ball_radius', 0.02135)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('base_height', 0.0)          # base_footprint lies on the floor
        self.declare_parameter('process_every_n', 2)        # camera is 30 Hz @1280x720 -> use every 2nd frame
        self.declare_parameter('imgsz', 640)                # YOLO inference size
        self.declare_parameter('max_range', 2.5)
        # orange golf ball, OpenCV hue 0-180: orange ~13; the yellow goal / LEDs are ~25, red ~0-3, lines white
        self.declare_parameter('hsv_h_min', 4)
        self.declare_parameter('hsv_h_max', 18)
        self.declare_parameter('hsv_s_min', 130)
        self.declare_parameter('hsv_v_min', 80)
        self.declare_parameter('min_area_px', 12)
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('compressed', False)
        # extra downward camera pitch on top of /tf.  The real Pinky's robot_description says 0 deg, the
        # camera is actually ~6 deg down (measured: floor/wall line vs lidar distance) -> real launch sets 6
        self.declare_parameter('cam_pitch_deg', 0.0)
        # camera_info without a calibration (K all zeros, as on the real robots' camera_publisher) or no
        # camera_info at all: intrinsics from this horizontal FOV and the image size (OV5647 = 53.5 deg)
        self.declare_parameter('fallback_hfov_deg', 53.5)
        # checkerboard calibration (camera_calib window): [fx, fy, cx, cy] + distortion [k1 k2 p1 p2 k3];
        # empty = camera_info / FOV fallback
        self.declare_parameter('intrinsics', [0.0])
        self.declare_parameter('distortion', [0.0])          # real robot: JPEG on /camera/image_raw/compressed

        self.backend = self.get_parameter('backend').value
        self.ball_r = self.get_parameter('ball_radius').value
        self.base_h = self.get_parameter('base_height').value
        self.base_frame = self.get_parameter('base_frame').value
        self._n = 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.K = None
        self.model = None

        self.pub_base = self.create_publisher(PointStamped, 'ball/position_base', 10)
        self.pub_map = self.create_publisher(PointStamped, 'ball/position_map', 10)
        self.pub_dbg = self.create_publisher(Image, 'ball/detection_image', 1)

        if self.backend == 'gt':
            self.tracker = PoseTracker()
            self.create_subscription(TFMessage, '/gz/dynamic_pose',
                                     lambda m: self._gt(m), 10)
            self.get_logger().info('backend=gt (ground truth, map==world assumed)')
            return

        if self.backend == 'yolo':
            from ultralytics import YOLO
            path = self.get_parameter('model_path').value
            if not path:
                raise RuntimeError("backend:=yolo needs model_path:=/path/to/ball.pt")
            self.model = YOLO(path)
            self.get_logger().info(f'YOLO model loaded: {path}')
        else:
            self.get_logger().info('backend=hsv (orange-blob baseline)')

        self.create_subscription(CameraInfo, 'camera/camera_info', self._info, 1)
        if self.get_parameter('compressed').value:
            from sensor_msgs.msg import CompressedImage
            self.create_subscription(CompressedImage, 'camera/image_raw/compressed', self._image,
                                     qos_profile_sensor_data)
        else:
            self.create_subscription(Image, 'camera/image_raw', self._image, qos_profile_sensor_data)

    # ----------------------------------------------------------------- ground-truth backend
    def _gt(self, msg):
        self.tracker.update(msg)
        ball = self.tracker.get('golf_ball')
        if ball is None:
            return
        out = PointStamped()
        out.header.frame_id = 'map'
        out.header.stamp = self.get_clock().now().to_msg()
        out.point.x, out.point.y, out.point.z = ball[0], ball[1], ball[2]
        self.pub_map.publish(out)

    # ----------------------------------------------------------------- vision backends
    def _info(self, msg):
        if getattr(self, 'K_source', '') == 'calibrated':
            return                                         # our checkerboard calibration wins
        if msg.k[0] > 0 and msg.k[4] > 0:
            self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])   # fx, fy, cx, cy
            self.K_source = 'camera_info'

    def _fallback_K(self, rgb):
        h, w = rgb.shape[:2]
        f = (w / 2) / math.tan(math.radians(self.get_parameter('fallback_hfov_deg').value) / 2)
        self.K, self.K_source = (f, f, w / 2, h / 2), 'fov'
        self.get_logger().warning(f'no camera calibration: using hfov {self.get_parameter("fallback_hfov_deg").value}'
                                  f' deg -> fx={f:.0f} for {w}x{h}')

    @staticmethod
    def _to_numpy(msg):
        buf = np.frombuffer(msg.data, np.uint8)
        if hasattr(msg, 'format'):                                       # CompressedImage (JPEG/PNG)
            import cv2
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f'cannot decode compressed image ({msg.format})')
            return bgr[:, :, ::-1]
        if msg.encoding in ('rgb8', 'bgr8'):
            img = buf.reshape(msg.height, msg.width, 3)
            return img if msg.encoding == 'rgb8' else img[:, :, ::-1]
        raise RuntimeError(f'unsupported encoding {msg.encoding}')

    def _detect_hsv(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        mask = cv2.inRange(hsv, (g('hsv_h_min'), g('hsv_s_min'), g('hsv_v_min')), (g('hsv_h_max'), 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < self.get_parameter('min_area_px').value:
                continue
            (cx, cy), rad = cv2.minEnclosingCircle(c)
            circ = area / (math.pi * rad * rad + 1e-6)
            if circ < 0.55:
                continue
            cands.append((area * circ, (cx, cy, 2 * rad, 2 * rad, circ)))
        # best first; _image takes the first one that lies on the floor (an orange thing on a desk behind the
        # wall is above the horizon and must not hide the real ball)
        return [c for _, c in sorted(cands, key=lambda t: -t[0])]   # [(cx, cy, w, h, conf)]

    def _detect_yolo(self, rgb):
        res = self.model.predict(np.ascontiguousarray(rgb), conf=self.get_parameter('conf').value,
                                 imgsz=self.get_parameter('imgsz').value, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        i = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = res.boxes.xyxy[i].tolist()
        return [((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, float(res.boxes.conf[i]))]

    def _image(self, msg):
        if self.K is None and self._n < 30:      # give camera_info a moment (~2 s) before the FOV fallback
            self._n += 1
            return
        self._n += 1
        if self._n % max(1, self.get_parameter('process_every_n').value):
            return
        try:
            rgb = self._to_numpy(msg)
        except RuntimeError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return
        intr = list(self.get_parameter('intrinsics').value)
        if len(intr) == 4 and intr[0] > 0 and getattr(self, 'K_source', '') != 'calibrated':
            self.K, self.K_source = tuple(intr), 'calibrated'
            dist = list(self.get_parameter('distortion').value)
            self.dist = np.array(dist, dtype=float) if len(dist) >= 4 else None
            self.get_logger().info(f'checkerboard intrinsics fx={intr[0]:.1f} fy={intr[1]:.1f} '
                                   f'cx={intr[2]:.1f} cy={intr[3]:.1f}')
        if self.K is None:
            self._fallback_K(rgb)
        cands = self._detect_yolo(rgb) if self.backend == 'yolo' else self._detect_hsv(rgb)
        det = None
        for c in cands[:5]:
            if self._localise(c, msg.header.frame_id, msg.header.stamp):
                det = c
                break
        if self.get_parameter('publish_debug').value:
            self._debug(rgb, det, msg)

    def _localise(self, det, cam_frame, stamp):
        fx, fy, cx, cy = self.K
        u, v = det[0], det[1]
        if getattr(self, 'dist', None) is not None and getattr(self, 'K_source', '') == 'calibrated':
            import cv2                                     # lens distortion out of the ball centre pixel
            Km = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
            u, v = cv2.undistortPoints(np.array([[[u, v]]], dtype=float), Km, self.dist, P=Km)[0, 0]
        if 'optical' in cam_frame:
            # image already in the optical frame (x right, y down, z forward; real robots' camera_publisher)
            d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        else:
            # ray in the camera link (x fwd, y left, z up; Pinky: front_camera_link) from optical (x right, y down, z fwd)
            d_cam = np.array([1.0, -(u - cx) / fx, -(v - cy) / fy])
        pitch = math.radians(self.get_parameter('cam_pitch_deg').value)
        if pitch and 'optical' in cam_frame:              # same nose-down rotation, optical axes (x right, y down)
            d_cam = np.array([d_cam[0], d_cam[1] * math.cos(pitch) + d_cam[2] * math.sin(pitch),
                              -d_cam[1] * math.sin(pitch) + d_cam[2] * math.cos(pitch)])
        elif pitch:                                         # rotate about the camera y axis (nose down)
            d_cam = np.array([d_cam[0] * math.cos(pitch) + d_cam[2] * math.sin(pitch), d_cam[1],
                              -d_cam[0] * math.sin(pitch) + d_cam[2] * math.cos(pitch)])
        try:
            t_bc = self.tf_buffer.lookup_transform(self.base_frame, cam_frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            return False
        R = quat_to_rot(t_bc.transform.rotation)
        o = np.array([t_bc.transform.translation.x, t_bc.transform.translation.y,
                      t_bc.transform.translation.z])
        d = R @ d_cam
        plane_z = self.ball_r - self.base_h              # ball-centre height in the base frame
        if d[2] >= -1e-4:
            return False
        s = (plane_z - o[2]) / d[2]
        if s <= 0:
            return False
        p = o + s * d
        if math.hypot(p[0], p[1]) > self.get_parameter('max_range').value:
            return False

        pb = PointStamped()
        pb.header.frame_id, pb.header.stamp = self.base_frame, stamp
        pb.point.x, pb.point.y, pb.point.z = float(p[0]), float(p[1]), float(p[2])
        self.pub_base.publish(pb)

        try:
            t_mb = self.tf_buffer.lookup_transform('map', self.base_frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            return True
        yaw = yaw_from_quat(t_mb.transform.rotation)
        tx, ty = t_mb.transform.translation.x, t_mb.transform.translation.y
        pm = PointStamped()
        pm.header.frame_id, pm.header.stamp = 'map', stamp
        pm.point.x = tx + math.cos(yaw) * p[0] - math.sin(yaw) * p[1]
        pm.point.y = ty + math.sin(yaw) * p[0] + math.cos(yaw) * p[1]
        pm.point.z = self.ball_r
        self.pub_map.publish(pm)
        return True

    def _debug(self, rgb, det, msg):
        import cv2
        img = np.ascontiguousarray(rgb).copy()
        if det is not None:
            cx, cy, w, h, conf = det
            cv2.rectangle(img, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)),
                          (255, 0, 0), 2)
            cv2.putText(img, f'ball {conf:.2f}', (int(cx - w / 2), max(12, int(cy - h / 2) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        out = Image()
        out.header = msg.header
        out.height, out.width, out.encoding = img.shape[0], img.shape[1], 'rgb8'
        out.step = img.shape[1] * 3
        out.data = img.tobytes()
        self.pub_dbg.publish(out)


def main():
    rclpy.init()
    node = BallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
