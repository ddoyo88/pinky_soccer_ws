#!/usr/bin/env python3
"""Watch a sim_duel match and summarise it: referee events, phases, robot states, AMCL error vs ground truth,
time over the white boundary line, closest distance between the robots.  Sends NEW_GAME after a delay.

    source /opt/ros/jazzy/setup.bash
    python3 scripts/match_watch.py [yellow_domain 37] [blue_domain 38] [start_after_s 40] [max_s 1500]

Simulation only (uses /gz/dynamic_pose, the simulator's ground truth).  Ends after FINISHED or max_s.
"""
import math
import sys
import threading
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

YD, BD = (int(sys.argv[i]) if len(sys.argv) > i else d for i, d in ((1, 37), (2, 38)))
START_AFTER = float(sys.argv[3]) if len(sys.argv) > 3 else 40.0
DURATION = float(sys.argv[4]) if len(sys.argv) > 4 else 1500.0
L2, W2, G2 = 0.83, 0.53, 0.20                     # hardboard field: half length / width, half goal width

truth, amcl, state, phase = {}, {}, {}, ['?']
over = {'pinky_yellow': [0.0, -1.0, None], 'pinky_blue': [0.0, -1.0, None]}   # s over the line, worst, last t
close = {'min': 9.0, 'time': 0.0, 't': None}      # robot-robot centre distance (< 0.16 m: touching)
t0 = time.time()


def log(*a):
    print(f'{time.time() - t0:7.1f}', *a, flush=True)


def yaw(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def make(dom, name):
    c = Context()
    rclpy.init(context=c, domain_id=dom)
    n = Node(name, context=c)
    ex = SingleThreadedExecutor(context=c)
    ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()
    return n


ny, nb = make(YD, 'match_watch_y'), make(BD, 'match_watch_b')


def on_truth(m):
    now = time.time()
    for t in m.transforms:
        p = (t.transform.translation.x, t.transform.translation.y, yaw(t.transform.rotation))
        truth[t.child_frame_id] = p
        o = over.get(t.child_frame_id)
        if o is not None and phase[0] == 'START':
            x, y = abs(p[0]), abs(p[1])
            e = max(y - W2, (x - L2) if y > G2 + 0.05 else -1) + 0.065      # body edge past the line [m]
            if o[2] is not None and e > 0:
                o[0] += min(0.2, now - o[2])
            o[1], o[2] = max(o[1], e), now
    y, b = truth.get('pinky_yellow'), truth.get('pinky_blue')
    if y and b and phase[0] in ('START', 'PREPARE', 'GOAL'):
        d = math.hypot(y[0] - b[0], y[1] - b[1])
        close['min'] = min(close['min'], d)
        if close['t'] is not None and d < 0.16:
            close['time'] += min(0.2, now - close['t'])
            if close.get('c0') is None:
                close['c0'], close['cmin'] = now, d
                log(f"  CONTACT start {phase[0]} d {d * 100:.1f} cm yellow ({y[0]:+.2f},{y[1]:+.2f}) "
                    f"{state.get('yellow', '?')} | blue ({b[0]:+.2f},{b[1]:+.2f}) {state.get('blue', '?')}")
            close['cmin'] = min(close['cmin'], d)
        elif close.get('c0') is not None and d > 0.17:
            log(f"  CONTACT end after {now - close['c0']:.1f} s, min {close['cmin'] * 100:.1f} cm "
                f"(yellow {state.get('yellow', '?')} | blue {state.get('blue', '?')})")
            close['c0'] = None
        close['t'] = now


ny.create_subscription(TFMessage, '/gz/dynamic_pose', on_truth, 10)


def on_amcl(team):
    def f(m):
        p = m.pose.pose
        amcl[team] = (p.position.x, p.position.y, time.time())
    return f


ny.create_subscription(PoseWithCovarianceStamped, '/yellow/amcl_pose', on_amcl('yellow'), 10)
nb.create_subscription(PoseWithCovarianceStamped, '/blue/amcl_pose', on_amcl('blue'), 10)


def on_state(team):
    def f(m):
        if state.get(team) != m.data:
            log(f'  {team} state {state.get(team)} -> {m.data}')
        state[team] = m.data
    return f


ny.create_subscription(String, '/yellow/game/state', on_state('yellow'), 10)
nb.create_subscription(String, '/blue/game/state', on_state('blue'), 10)


def on_phase(m):
    if m.data != phase[0]:
        log(f'PHASE {phase[0]} -> {m.data}')
        phase[0] = m.data


ny.create_subscription(String, '/game/phase', on_phase, 10)
ny.create_subscription(String, '/game/event', lambda m: log('EVENT', m.data), 10)
ctl = ny.create_publisher(String, '/game/control', 10)

sent, last, errs = False, 0.0, {'yellow': [], 'blue': []}
while time.time() - t0 < DURATION:
    time.sleep(0.5)
    if not sent and time.time() - t0 > START_AFTER:
        ctl.publish(String(data='NEW_GAME'))
        sent = True
        log('sent NEW_GAME')
    for team in errs:
        g, a = truth.get('pinky_' + team), amcl.get(team)
        if g and a and time.time() - a[2] < 3:
            errs[team].append(math.hypot(g[0] - a[0], g[1] - a[1]))
    if time.time() - last > 10:
        last = time.time()
        parts = []
        for team in ('yellow', 'blue'):
            g, a = truth.get('pinky_' + team), amcl.get(team)
            if g and a:
                parts.append(f'{team} true({g[0]:+.2f},{g[1]:+.2f}) amcl({a[0]:+.2f},{a[1]:+.2f}) '
                             f'err {math.hypot(g[0] - a[0], g[1] - a[1]) * 100:.1f}cm')
        b = truth.get('golf_ball')
        log('POS', ' | '.join(parts), f'ball({b[0]:+.2f},{b[1]:+.2f})' if b else '')
    if phase[0] == 'FINISHED':
        time.sleep(3)
        break
for team, e in errs.items():
    if e:
        e.sort()
        log(f'{team} AMCL error: median {e[len(e) // 2] * 100:.1f} cm, 95% {e[int(len(e) * .95)] * 100:.1f} cm, '
            f'max {e[-1] * 100:.1f} cm')
for k, o in over.items():
    log(f'{k}: body over the boundary line {o[0]:.1f} s in play, worst {o[1] * 100:.1f} cm past the line')
log(f"robots: closest centre distance {close['min'] * 100:.1f} cm, "
    f"time closer than 16 cm (touching) {close['time']:.1f} s")
log('done')
