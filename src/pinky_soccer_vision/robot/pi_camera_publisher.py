#!/usr/bin/env python3
"""Run ON THE REAL PINKY (Raspberry Pi 5): Pi camera -> /camera/image_raw/compressed (JPEG) + /camera/camera_info.

The official pinky_bringup does not publish the camera (pinkylib only offers it to Python), but the soccer
ball detector on the PC needs it as ROS topics - the same names the simulation uses.

JPEG by default: raw 640x480 @ 15 Hz would be ~13 MB/s over Wi-Fi (run the PC detector with compressed:=true;
real_yellow.launch.py does).  compressed:=false publishes raw /camera/image_raw (wired / local use).
Camera: pinkylib.Camera (the Pinky's own library, get_frame() = BGR), else picamera2, else OpenCV /dev/video0.

Standalone on purpose (only rclpy + numpy + cv2 + pinkylib, all already on the Pinky image), copy and run:
    scp pi_camera_publisher.py pinky@192.168.4.1:~/
    ssh pinky@192.168.4.1
    python3 ~/pi_camera_publisher.py                       # 640x480 @ 15 Hz, JPEG
    python3 ~/pi_camera_publisher.py --ros-args -p hfov_deg:=62.2   # camera module 2

Do not run a pinkylib Camera() at the same time (only one process can own the camera).

camera_info: K is computed from the horizontal field of view (no calibration file needed for a first test;
for better ball distances calibrate with camera_calibration and set fx/fy/cx/cy).  The frame is the official
URDF's front_camera_link, so the detector uses the camera pose bringup already publishes on /tf.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class PiCameraPublisher(Node):
    def __init__(self):
        super().__init__('pi_camera_publisher')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('rate', 15.0)
        self.declare_parameter('hfov_deg', 66.0)        # Raspberry Pi Camera Module 3 (standard lens)
        self.declare_parameter('fx', 0.0)               # > 0: calibrated intrinsics override hfov_deg
        self.declare_parameter('fy', 0.0)
        self.declare_parameter('cx', 0.0)
        self.declare_parameter('cy', 0.0)
        self.declare_parameter('frame_id', 'front_camera_link')
        self.declare_parameter('flip', False)           # camera mounted upside down -> rotate 180 deg
        self.declare_parameter('compressed', True)
        self.declare_parameter('jpeg_quality', 80)
        w, h = self.get_parameter('width').value, self.get_parameter('height').value
        self.w, self.h = w, h
        self.cam = self._open_camera(w, h)
        fx = self.get_parameter('fx').value or (w / 2) / math.tan(math.radians(self.get_parameter('hfov_deg').value) / 2)
        fy = self.get_parameter('fy').value or fx
        cx = self.get_parameter('cx').value or w / 2
        cy = self.get_parameter('cy').value or h / 2
        self.info = CameraInfo(width=w, height=h, distortion_model='plumb_bob')
        self.info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.info.d = [0.0] * 5
        self.compressed = self.get_parameter('compressed').value
        if self.compressed:
            self.pub_img = self.create_publisher(CompressedImage, '/camera/image_raw/compressed',
                                                 qos_profile_sensor_data)
        else:
            self.pub_img = self.create_publisher(Image, '/camera/image_raw', qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        self.create_timer(1.0 / self.get_parameter('rate').value, self.tick)
        self.get_logger().info(f'{self.backend}: publishing {w}x{h} '
                               f'{"JPEG on /camera/image_raw/compressed" if self.compressed else "raw on /camera/image_raw"}'
                               f' (fx={fx:.0f}, frame {self.get_parameter("frame_id").value})')

    def _open_camera(self, w, h):
        try:
            from pinkylib import Camera                       # the Pinky's own camera wrapper
            cam = Camera()
            cam.start(width=w, height=h)
            self.backend = 'pinkylib'
            return cam
        except Exception as e:                                         # noqa: BLE001
            self.get_logger().warning(f'pinkylib camera not usable ({e}); trying picamera2')
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            cam.configure(cam.create_video_configuration(main={'size': (w, h), 'format': 'RGB888'}))
            cam.start()
            self.backend = 'picamera2'
            return cam
        except Exception as e:                                         # noqa: BLE001
            self.get_logger().warning(f'picamera2 not usable ({e}); trying OpenCV /dev/video0')
        import cv2
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if not cap.isOpened():
            raise RuntimeError('no camera: picamera2 failed and /dev/video0 cannot be opened '
                               '(is a pinkylib Camera() still holding it?)')
        self.backend = 'opencv'
        return cap

    def _grab(self):
        if self.backend == 'pinkylib':
            # get_frame() is BGR in memory (checked on the real Pinky: faces came out blue) -> RGB
            return self.cam.get_frame()[:, :, ::-1]
        if self.backend == 'picamera2':
            # picamera2 'RGB888' is stored B,G,R in memory (libcamera naming) -> swap to real RGB
            return self.cam.capture_array('main')[:, :, ::-1]
        ok, frame = self.cam.read()
        return frame[:, :, ::-1] if ok else None                         # OpenCV gives BGR

    def tick(self):
        rgb = self._grab()
        if rgb is None:
            return
        if self.get_parameter('flip').value:
            rgb = rgb[::-1, ::-1]
        rgb = np.ascontiguousarray(rgb[:, :, :3])
        stamp = self.get_clock().now().to_msg()
        frame = self.get_parameter('frame_id').value
        if self.compressed:
            import cv2
            ok, jpg = cv2.imencode('.jpg', rgb[:, :, ::-1],
                                   [int(cv2.IMWRITE_JPEG_QUALITY), self.get_parameter('jpeg_quality').value])
            if not ok:
                return
            msg = CompressedImage(format='jpeg')
            msg.data = jpg.tobytes()
        else:
            msg = Image(height=rgb.shape[0], width=rgb.shape[1], encoding='rgb8', step=rgb.shape[1] * 3)
            msg.data = rgb.tobytes()
        msg.header.stamp, msg.header.frame_id = stamp, frame
        self.info.header.stamp, self.info.header.frame_id = stamp, frame
        self.pub_img.publish(msg)
        self.pub_info.publish(self.info)

    def close(self):
        try:
            if self.backend == 'pinkylib':
                self.cam.close()
            elif self.backend == 'picamera2':
                self.cam.stop()
                self.cam.close()
            else:
                self.cam.release()
        except Exception:                                                # noqa: BLE001
            pass


def main():
    rclpy.init()
    node = PiCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.close()
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
