"""Small helpers that talk to Gazebo through the `gz` CLI (teleport / reset entities)."""
import math
import subprocess


def gz_set_pose(world, name, x, y, z, yaw=0.0, timeout_ms=1500):
    """Teleport a model via /world/<world>/set_pose (needs the UserCommands system in the world)."""
    qz, qw = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    req = (f'name: "{name}" position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}} '
           f'orientation: {{x: 0, y: 0, z: {qz:.5f}, w: {qw:.5f}}}')
    cmd = ['gz', 'service', '-s', f'/world/{world}/set_pose', '--reqtype', 'gz.msgs.Pose',
           '--reptype', 'gz.msgs.Boolean', '--timeout', str(timeout_ms), '--req', req]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_ms / 1000.0 + 2.0)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return 'true' in r.stdout.lower()


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class PoseTracker:
    """Caches latest ground-truth model poses from the bridged /gz/dynamic_pose (tf2_msgs/TFMessage)."""

    def __init__(self):
        self.poses = {}   # name -> (x, y, z, yaw, stamp_sec)

    def update(self, msg):
        for tf in msg.transforms:
            name = tf.child_frame_id.split('/')[-1]
            tr, q = tf.transform.translation, tf.transform.rotation
            self.poses[name] = (tr.x, tr.y, tr.z, yaw_from_quat(q),
                                tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9)

    def get(self, name):
        return self.poses.get(name)
