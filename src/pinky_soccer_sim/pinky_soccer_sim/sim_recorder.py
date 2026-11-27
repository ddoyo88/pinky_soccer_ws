"""Match video in its OWN process (2026-11-27): recording inside mujoco_soccer slowed the simulation to ~0.5x real
time (the detailed robot meshes are expensive to draw, and the simulator runs on one core).  This process builds
the same scene, follows the true poses the simulator publishes (/gz/dynamic_pose), draws the match from above with
the information overlay (score, clock, referee messages, both robot cameras) and writes an .mp4 whose length
follows the simulation time (/clock).

    sim_recorder --output ~/pinky_rl/videos/match.mp4 --field-style hardboard --cam-tilt 20 \
                 --yellow-domain 47 --blue-domain 48 [--fps 20] [--size 960x540] [--plain]
"""
import argparse
import math
import os
import signal
import threading
import time

import mujoco
import numpy as np
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


def main(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--field-style', default='hardboard')
    ap.add_argument('--cam-tilt', type=float, default=20.0)
    ap.add_argument('--yellow-domain', type=int, default=47)
    ap.add_argument('--blue-domain', type=int, default=48)
    ap.add_argument('--fps', type=float, default=20.0)
    ap.add_argument('--size', default='960x540')
    ap.add_argument('--plain', action='store_true', help='no information overlay')
    ap.add_argument('--output-dir', default='/tmp/pinky_soccer/recorder')
    opt, _ = ap.parse_known_args(args)

    import cv2
    from pinky_soccer_sim.mujoco_soccer import build_scene
    from pinky_soccer_sim.field import resolve_dims
    L = resolve_dims(opt.field_style)[0]
    model, _ = build_scene(opt.output_dir, opt.field_style, opt.cam_tilt,
                           {'yellow': (-L / 4, 0.0, 0.0), 'blue': (L / 4, 0.0, math.pi)})
    data = mujoco.MjData(model)
    rw, rh = (int(v) for v in opt.size.lower().split('x'))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, rw)
    model.vis.global_.offheight = max(model.vis.global_.offheight, rh)
    renderer = mujoco.Renderer(model, rh, rw)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0]
    cam.distance, cam.azimuth, cam.elevation = 2.6, 90, -60
    vopt = mujoco.MjvOption()
    vopt.geomgroup[1] = vopt.geomgroup[3] = 0        # collision shapes off, detailed robot meshes on
    vopt.geomgroup[4] = 1
    joints = {'golf_ball': model.joint('ball_joint').qposadr[0],
              'pinky_yellow': model.joint('yellow/floating_base').qposadr[0],
              'pinky_blue': model.joint('blue/floating_base').qposadr[0]}
    hud = None
    if not opt.plain:
        from pinky_soccer_sim.video_hud import VideoHud
        hud = VideoHud(rw, rh)
    path = os.path.expanduser(opt.output)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), opt.fps, (rw, rh))

    lock = threading.Lock()
    state = {'poses': {}, 'sim_t': None, 'cams': {}, 'new_pose': False}
    info = {'score': (0, 0), 'clock': '', 'events': [], 'phase': '', 'time': ''}

    def on_truth(m):
        with lock:
            for t in m.transforms:
                if t.child_frame_id in joints:
                    p, q = t.transform.translation, t.transform.rotation
                    state['poses'][t.child_frame_id] = (p.x, p.y, p.z, q.w, q.x, q.y, q.z)
            state['new_pose'] = True

    def on_clock(m):
        state['sim_t'] = m.clock.sec + m.clock.nanosec * 1e-9

    def on_image(team):
        def cb(m):
            if m.encoding == 'rgb8':
                state['cams'][team] = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
        return cb

    def on_score(m):
        import re
        g = re.fullmatch(r'YELLOW\s+(\d+)\s*-\s*(\d+)\s+BLUE', m.data.strip())
        if g:
            info['score'] = (int(g.group(1)), int(g.group(2)))

    nodes, execs, ctxs = [], [], []
    for team, dom in (('yellow', opt.yellow_domain), ('blue', opt.blue_domain)):
        ctx = Context()
        rclpy.init(context=ctx, domain_id=dom)
        node = Node(f'sim_recorder_{team}', context=ctx)
        node.create_subscription(Image, f'/{team}/camera/image_raw', on_image(team), qos_profile_sensor_data)
        if team == 'yellow':
            node.create_subscription(TFMessage, '/gz/dynamic_pose', on_truth, 10)
            node.create_subscription(Clock, '/clock', on_clock, 10)
            node.create_subscription(String, '/game/score', on_score, 10)
            node.create_subscription(String, '/game/attack_clock', lambda m: info.__setitem__('clock', m.data), 10)
            node.create_subscription(String, '/game/event',
                                     lambda m: info.__setitem__('events', (info['events'] + [m.data])[-2:]), 10)
            node.create_subscription(String, '/game/phase', lambda m: info.__setitem__('phase', m.data.strip()), 10)
            node.create_subscription(String, '/game/clock', lambda m: info.__setitem__('time', m.data.strip()), 10)
        ex = SingleThreadedExecutor(context=ctx)
        ex.add_node(node)
        threading.Thread(target=ex.spin, daemon=True).start()
        nodes.append(node), execs.append(ex), ctxs.append(ctx)

    stop = {'now': False}
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: stop.__setitem__('now', True))
    print(f'recording the match to {path} ({rw}x{rh} @ {opt.fps:g} fps of simulation time, own process)',
          flush=True)
    t0, written = None, 0
    while not stop['now']:
        sim_t = state['sim_t']
        if sim_t is None or not state['poses']:
            time.sleep(0.02)
            continue
        if t0 is None:
            t0 = sim_t
        due = int((sim_t - t0) * opt.fps) + 1 - written       # frames owed to keep the video on simulation time
        if due <= 0:
            time.sleep(0.005)
            continue
        with lock:
            for name, (x, y, z, qw, qx, qy, qz) in state['poses'].items():
                a = joints[name]
                data.qpos[a:a + 7] = [x, y, z, qw, qx, qy, qz]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=vopt)
        frame = renderer.render()
        out = hud.compose(frame, dict(state['cams']), info) if hud is not None else \
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for _ in range(min(due, 10)):                         # (a long stall: at most 10 copies)
            writer.write(out)
        written += due
    writer.release()
    print(f'video saved: {path} ({written / opt.fps:.0f} s)', flush=True)
    for ex, node, ctx in zip(execs, nodes, ctxs):
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown(context=ctx)


if __name__ == '__main__':
    main()
